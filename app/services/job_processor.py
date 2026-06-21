import asyncio
import json # <--- ДОБАВЛЕН ДЛЯ РАБОТЫ СО СЛОВАРЯМИ
import logging
import re
from typing import Optional
from .ai_pipeline import AiFeaturePipeline
from .db_cache import DatabaseCacheManager
from .matcher import MatcherService
from .url_fetcher import fetch_all
from .enrichment import VisionProducer, WebSearchProducer, AttributeMerger, AttributeValue, Source
from .enrichment.base import ExtractionContext, TargetAttribute
from .enrichment.sources.icecat_source import IceCatSource
from ..models import ProductData, FeatureOption, ResearchMode, BatchOptions
from ..database import AsyncSessionLocal
from ..config import VAGUE_FEATURE_PATTERNS
from .. import config as _config

# ---------------------------------------------------------------------------
# New pipeline integration (feature flag USE_NEW_PIPELINE)
# ---------------------------------------------------------------------------
# When config.USE_NEW_PIPELINE is True (PROD воркер запускается с этим env),
# process_product routes through PipelineAdapter -> PipelineOrchestrator instead
# of the legacy per-feature flow. WIRING DONE: self._pipeline_adapter.run(...) is connected
# (see process_product below), schema→TargetAttribute id = caller's real attr id
# or positional index, and _values_to_legacy_format emits the v2 contract
# (filled_features + debug_info{value_id,confidence,evidence,source,...} + skipped).
#   Known Tier-2 TODOs (non-blocking):
#     - tokens_used = 0 (per-call token counter not yet threaded for billing).
#     - new path bypasses db_cache by design (avoids stale/poisoned cache);
#       product-fingerprint cache at adapter level is a future optimisation.
try:
    from .enrichment.pipeline_adapter import PipelineAdapter as _PipelineAdapter
    _PIPELINE_ADAPTER_AVAILABLE = True
except ImportError:
    _PipelineAdapter = None  # type: ignore[assignment,misc]
    _PIPELINE_ADAPTER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Pre-compile vague-feature-name regexes once at import time. Each pattern
# is wrapped with fullmatch semantics (re.fullmatch) and IGNORECASE so the
# operator can add patterns in config.py without worrying about flags.
_VAGUE_FEATURE_REGEXES = [
    re.compile(p, re.IGNORECASE) for p in VAGUE_FEATURE_PATTERNS
]


def is_vague_feature_name(name: str) -> bool:
    """
    Returns True if the feature name is a placeholder / meaningless label
    (e.g. "NewFeature", "Field 1", "X", "123"). Such tuples must be
    short-circuited BEFORE any LLM call — otherwise the model is forced to
    invent a value, which is the operator's product-403 bug.
    """
    if not name:
        return True
    trimmed = name.strip()
    if len(trimmed) < 2:
        return True
    for rx in _VAGUE_FEATURE_REGEXES:
        if rx.fullmatch(trimmed):
            return True
    return False

class JobProcessor:
    def __init__(
        self,
        pipeline: AiFeaturePipeline,
        db_cache: DatabaseCacheManager,
        matcher: MatcherService,
        global_semaphore: asyncio.Semaphore,
        vision_producer: Optional[VisionProducer] = None,
        websearch_producer: Optional[WebSearchProducer] = None,
        icecat_source: Optional["IceCatSource"] = None,
    ):
        self.pipeline = pipeline
        self.db_cache = db_cache
        self.matcher = matcher
        self.semaphore = global_semaphore
        self.vision_producer = vision_producer
        self.websearch_producer = websearch_producer
        self.icecat_source = icecat_source
        self.merger = AttributeMerger()
        # New pipeline adapter — only instantiated when the feature flag is on
        # and the import succeeded, to avoid import-time cost when unused.
        self._pipeline_adapter: Optional[_PipelineAdapter] = (
            _PipelineAdapter() if _config.USE_NEW_PIPELINE and _PIPELINE_ADAPTER_AVAILABLE else None
        )

    # ------------------------------------------------------------------
    # Enrichment helper: run basic extraction on arbitrary text (1 LLM call).
    # Only the first extraction stage of the pipeline — no deduction /
    # knowledge / web fallbacks — so each branch stays at exactly 1 call.
    # ------------------------------------------------------------------
    async def _extract_attrs_from_text(
        self,
        text: str,
        product: ProductData,
        schema: dict[str, FeatureOption],
        source: Source,
    ) -> list[AttributeValue]:
        """Run extraction-only (no deduction/knowledge/web) on *text*.

        Returns a list of AttributeValue for each feature in *schema* that
        yielded a non-None result.
        """
        attrs: list[AttributeValue] = []
        for f_name, f_schema in schema.items():
            if is_vague_feature_name(f_name):
                continue
            suffix = getattr(f_schema, "suffix", "")
            opts = getattr(f_schema, "options", [])

            # One structured_request — extraction stage only.
            try:
                full_instruction, _router_debug, matched_leaf = await self.pipeline.router.find_instruction(
                    text, f_name, unit=suffix
                )
                if matched_leaf is None:
                    continue

                TargetModel = matched_leaf.get_response_model(options=opts)
                dynamic_instruction = matched_leaf.get_instruction(
                    unit=suffix, options=opts, target_languages=product.languages
                )
                sys_msg = (
                    f"{dynamic_instruction}\n\n"
                    f"--- EXECUTION CONTEXT ---\n"
                    f"TARGET FEATURE: '{f_name}'\n"
                    f"TARGET UNIT/SUFFIX: '{suffix}'\n"
                )

                result, _tokens = await self.pipeline.llm.structured_request(
                    system_prompt=sys_msg,
                    user_text=text,
                    response_model=TargetModel,
                )

                if result and result.confidence != "Low":
                    val = self.pipeline._unpack_value(result)
                    if val is not None:
                        # Map pipeline confidence string to 0-1 float.
                        conf_str = getattr(result, "confidence", "Low")
                        conf = {"High": 0.9, "Medium": 0.6, "Low": 0.3}.get(conf_str, 0.5)
                        attrs.append(
                            AttributeValue(
                                attribute_id=f_name,
                                value=val,
                                confidence=conf,
                                source=source,
                                reasoning=getattr(result, "analysis", None),
                            )
                        )
            except Exception as exc:
                logger.warning(
                    "_extract_attrs_from_text: error on feature %r (source=%s): %s",
                    f_name,
                    source,
                    exc,
                )
                continue

        return attrs

    def _values_to_legacy_format(
        self,
        values: list[AttributeValue],
        schema: dict[str, FeatureOption],
        targets_raw: list[dict],
        marketplace: Optional[str] = None,
    ) -> dict:
        """Convert orchestrator output (list[AttributeValue]) to the legacy process_product return format.

        Contract v2 (consumed by eg_importer — отчёт/Excel строит ОН):
            {
                "product_id": int,          # caller must set this
                "filled_features": dict,    # feature_name -> value (заполненные)
                "debug_info": dict,         # feature_name -> {
                                            #   attribute_id:int, value_id:int|None,
                                            #   value_ids:list[int]|None, is_collection:bool,
                                            #   semantic_type:str|None, source:str|None,
                                            #   confidence:float, evidence:str|None,
                                            #   judge_validated:bool, ...legacy placeholders}
                "tokens_used": int,         # approximate; exact tracking is TODO Tier 2
                "is_cached": bool,          # new path never caches (TODO Tier 2)
            }
        ПРИНЦИП: этот сервис отдаёт ТОЛЬКО то, что достаётся у него — значение +
        авторитетный value_id + источник/уверенность/evidence. Подсчёт
        заполнено/не-заполнено, вёрстку Excel и выходной файл делает eg_importer
        (у него есть полный список запрошенных targets → diff по filled_features).

        Cost tracking note:
            PipelineOrchestrator does not expose a per-call token count.
            tokens_used is set to 0 here.
            TODO(Tier-2-cost-tracking): Add a token counter to ExtractionContext
            (e.g. ExtractionContext.tokens_used: int = 0) and increment it inside
            each AttributeSource.extract() call. Expose via orchestrator and sum here.
        """
        # Build int-id -> feature_name lookup (mirrors adapter's id assignment logic)
        id_to_name: dict[int, str] = {}
        for idx, raw in enumerate(targets_raw):
            raw_id = raw.get("id") or raw.get("attribute_id")
            try:
                attr_id = int(raw_id) if raw_id is not None else idx
            except (TypeError, ValueError):
                attr_id = idx
            id_to_name[attr_id] = raw.get("name", str(attr_id))

        filled_features: dict = {}
        debug_info: dict = {}

        for av in values:
            f_name = id_to_name.get(av.attribute_id, str(av.attribute_id))
            filled_features[f_name] = av.value
            debug_info[f_name] = {
                # --- CONTRACT v2 (eg_importer): authoritative + identity fields ---
                "attribute_id": av.attribute_id,   # стабильный числовой ключ (имена хрупкие)
                "value_id": av.value_id,           # словарный ID Ozon (одиночное) — нужен для импорта
                "value_ids": av.value_ids,         # словарные ID Ozon (коллекция is_collection)
                "is_collection": av.is_collection, # value — массив, не скаляр
                "semantic_type": av.semantic_type, # color|weight|brand… подсказка downstream
                # --- grounding metadata (только мы это знаем) ---
                "source": av.source.value if av.source else None,
                "confidence": av.confidence,
                "evidence": av.evidence,
                "judge_validated": av.judge_validated,
                # Placeholders matching legacy debug_info shape so callers don't break
                "router": {},
                "extraction_reasoning": av.evidence or "",
                "deduced_context": None,
                "source_urls": None,
            }

        # --- skipped-блок (contract v2): причина пустоты per НЕзаполненный target ---
        # «Знаем только мы»: платформенное поле (учётные данные продавца, заполнять
        # нельзя) vs нет данных ни в одном источнике. eg_importer пишет это в отчёт
        # как 🔒 «ваше поле» vs — «нет данных», не считая платформенное недоработкой.
        skipped = self._build_skipped(values, targets_raw, marketplace)

        return {
            "filled_features": filled_features,
            "debug_info": debug_info,
            "skipped": skipped,
            "tokens_used": 0,  # TODO(Tier-2-cost-tracking): sum from ExtractionContext.tokens_used
            "is_cached": False,  # TODO(Tier-2-cache): add product-fingerprint cache at adapter level
        }

    @staticmethod
    def _build_skipped(
        values: list[AttributeValue],
        targets_raw: list[dict],
        marketplace: Optional[str],
    ) -> dict:
        """feature_name -> {attribute_id, reason} для каждого НЕзаполненного target.

        reason:
          "platform_field" — учётное/платформенное поле (баркод/НДС/декларации/ИКПУ…),
                              сервис намеренно не заполняет → eg_importer метит 🔒.
          "no_data"        — поле извлекаемое, но ни один источник не дал значение.
        (Отброс по low-confidence на этом слое неразличим от no_data — кандидаты уже
        отсеяны пайплайном; при необходимости пробросить — Tier-2.)
        """
        try:
            if (marketplace or "").lower() == "wb":
                from app.services.enrichment.strategies.dictionaries.wb_field_classifier import (
                    is_wb_platform_field as _is_platform,
                )
            else:
                from app.services.enrichment.strategies.dictionaries.ozon_field_classifier import (
                    is_platform_field as _is_platform,
                )
        except Exception:  # pragma: no cover — classifier import optional
            _is_platform = None

        filled_ids = {av.attribute_id for av in values}
        skipped: dict = {}
        for idx, raw in enumerate(targets_raw):
            raw_id = raw.get("id") or raw.get("attribute_id")
            try:
                attr_id = int(raw_id) if raw_id is not None else idx
            except (TypeError, ValueError):
                attr_id = idx
            if attr_id in filled_ids:
                continue
            name = raw.get("name", str(attr_id))
            reason = "no_data"
            if _is_platform is not None:
                # классификатор читает name/type/is_required/values(=allowed_values)
                char = {
                    "name": name,
                    "type": raw.get("type"),
                    "is_required": raw.get("is_required"),
                    "values": raw.get("allowed_values"),
                    "popular": raw.get("popular"),
                }
                try:
                    if _is_platform(char):
                        reason = "platform_field"
                except Exception:  # pragma: no cover
                    pass
            skipped[name] = {"attribute_id": attr_id, "reason": reason}
        return skipped

    async def process_product(
        self,
        product: ProductData,
        schema: dict[str, FeatureOption],
        client_id: int,
        use_cache: bool = True,
        research_mode: ResearchMode = ResearchMode.OFF,
        options: Optional[BatchOptions] = None,
    ) -> dict:
        # ------------------------------------------------------------------
        # Feature flag: new PipelineOrchestrator path
        # ------------------------------------------------------------------
        if _config.USE_NEW_PIPELINE and self._pipeline_adapter is not None:
            logger.info(
                "USE_NEW_PIPELINE=True: routing product_id=%s through PipelineAdapter",
                product.id,
            )
            # Build targets_raw using positional index as id (mirrors adapter's fallback strategy).
            # The schema is Dict[str, FeatureOption] — no numeric ids — so positional index is
            # the stable id that both adapter and _values_to_legacy_format agree on.
            targets_raw = [
                {
                    # Real Ozon attribute_id when the caller supplies it (eg-importer);
                    # falls back to the positional index for callers that don't.
                    "id": getattr(f_schema, "id", None) or idx,
                    "name": f_name,
                    "type": getattr(f_schema, "type", "text"),
                    "allowed_values": getattr(f_schema, "options", None) or None,
                    "is_required": bool(getattr(f_schema, "is_required", False)),
                    "semantic_type": None,  # TODO Tier 2: infer from feature name via classifier
                }
                for idx, (f_name, f_schema) in enumerate(schema.items())
            ]
            av_list = await self._pipeline_adapter.run(
                product_id=product.id,
                product_name=product.name,
                product_description=product.description,
                category_id=product.category_id,
                category_path=getattr(product, "category_path", []),
                brand=getattr(product, "brand", None),
                ean=getattr(product, "ean", None),
                ozon_type_id=getattr(product, "ozon_type_id", None),
                source_urls=getattr(product, "source_urls", []),
                image_urls=getattr(product, "image_urls", []),
                targets_raw=targets_raw,
                marketplace=options.marketplace if options else None,
            )
            result_dict = self._values_to_legacy_format(
                av_list, schema, targets_raw,
                marketplace=options.marketplace if options else None,
            )
            return {
                "product_id": product.id,
                **result_dict,
            }

        result_features = {}
        total_tokens_used = 0
        is_fully_cached = True

        opts = options or BatchOptions()

        # --- Web-fetch enrichment (Stage 2) ---
        # Fetch supplier/competitor URLs in parallel and inject into description.
        # Failures are non-fatal: pipeline continues with whatever was fetched.
        fetched_content = ""
        if getattr(product, "source_urls", None):
            try:
                fetched_content = await fetch_all(product.source_urls)
            except Exception as _fetch_exc:
                logger.warning("fetch_all failed entirely, continuing without URL content: %s", _fetch_exc)

        description = product.description or ""
        if fetched_content:
            description = f"{description}\n\n=== Sourced from URLs ===\n{fetched_content}"

        # --- Parallel enrichment inject: IceCat + web_search → description ---
        # Both are text-augmentation steps (NOT separate AttributeValue branches —
        # see the broken bridge story above). They run in parallel via
        # asyncio.gather, so total wall-clock = max(icecat, web_search), not sum.
        #
        # Ordering inside the final description: IceCat comes FIRST because
        # brand-verified manufacturer specs are the highest-quality signal —
        # the per-feature extractor below will see them ahead of fuzzier web
        # snippets. Web search comes second as fallback / cross-check.
        async def _icecat_text() -> Optional[str]:
            if not self.icecat_source:
                return None
            # IceCat needs a brand. Cheapest non-LLM detection: first whitespace
            # token of the product name (works for ~95% of titles that start
            # with the brand: "Dyson V15 Detect", "Samsung Galaxy S24", ...).
            # When the heuristic misses (e.g. "Apple iPhone 15 Pro" — first token
            # IS "Apple", correct), IceCat returns [] cleanly.
            brand_token = (product.name or "").split()[:1]
            if not brand_token:
                return None
            try:
                ctx = ExtractionContext(
                    product_id=product.id,
                    product_name=product.name,
                    product_description=product.description or "",
                    category_id=product.category_id,
                    brand=brand_token[0],
                )
                # Build minimal TargetAttributes from schema (positional ids).
                # Names matter for IceCat's name→target fuzzy mapping; ids are
                # just for downstream tracking.
                targets = [
                    TargetAttribute(id=idx, name=fname, type="text")
                    for idx, fname in enumerate(schema.keys())
                ]
                avs = await self.icecat_source.extract(ctx, targets)
                if not avs:
                    return None
                # Format mapped (target_name, value) pairs as plain text.
                name_by_id = {t.id: t.name for t in targets}
                lines = [
                    f"{name_by_id[av.attribute_id]}: {av.value}"
                    for av in avs
                    if av.attribute_id in name_by_id
                ]
                if not lines:
                    return None
                logger.info(
                    "icecat inject for product_id=%s: %d features matched (brand=%s)",
                    product.id, len(lines), brand_token[0],
                )
                return "\n".join(lines)
            except Exception as exc:
                logger.warning("icecat extract failed (non-fatal): %s", exc)
                return None

        async def _websearch_text() -> Optional[str]:
            if not (opts.enable_web_search and self.websearch_producer):
                return None
            try:
                ws_text = await self.websearch_producer.produce_summary(
                    product.name,
                    brand=getattr(product, "brand", None),
                    ean=getattr(product, "ean", None),
                    languages=getattr(product, "languages", None) or ["ru"],
                )
                if ws_text:
                    logger.info(
                        "web_search inject for product_id=%s: +%d chars (~%d sources)",
                        product.id, len(ws_text), ws_text.count("https://"),
                    )
                return ws_text
            except Exception as exc:
                logger.warning("web_search produce_summary failed (non-fatal): %s", exc)
                return None

        icecat_chunk, web_chunk = await asyncio.gather(
            _icecat_text(), _websearch_text(),
        )
        if icecat_chunk:
            description = f"{description}\n\n=== IceCat brand-verified ===\n{icecat_chunk}"
        if web_chunk:
            description = f"{description}\n\n=== Web search ===\n{web_chunk}"

        info = f"Title: {product.name}\nDescription: {description[:50000]}"
        context_hash = f"{product.name} {product.description}"

        # ==============================================================
        # ENRICHMENT BRANCHES (parallel, error-isolated)
        # Each branch: producer → plain text → extraction (1 LLM call).
        # Results are merged per-attribute by highest confidence / source
        # priority before the main per-feature description branch runs.
        # ==============================================================
        enrichment_branch_tasks = []

        if opts.enable_vision and self.vision_producer and getattr(product, "image_urls", None):
            async def _vision_branch(prod=product, sc=schema):
                vision_text = await self.vision_producer.produce_description(
                    prod.image_urls, prod.name
                )
                if not vision_text:
                    return []
                return await self._extract_attrs_from_text(vision_text, prod, sc, Source.VISION)
            enrichment_branch_tasks.append(_vision_branch())

        # Collect enrichment attrs — branch failures are warned, not raised.
        enrichment_attrs: list[AttributeValue] = []
        if enrichment_branch_tasks:
            branch_results = await asyncio.gather(*enrichment_branch_tasks, return_exceptions=True)
            for br in branch_results:
                if isinstance(br, Exception):
                    logger.warning("Enrichment branch failed (isolated): %s", br)
                elif isinstance(br, list):
                    enrichment_attrs.extend(br)

        # Build a lookup: attribute_id → best enrichment AttributeValue.
        # Used below to possibly override description-branch results.
        enrichment_best: dict[str, AttributeValue] = {
            av.attribute_id: av for av in self.merger.merge([enrichment_attrs])
        }

        async def process_feature(f_name, f_schema):
            async with self.semaphore:
                async with AsyncSessionLocal() as session:
                    existing = product.context.existing_features
                    if isinstance(existing, dict) and f_name in existing:
                        return (f_name, None, 0, True, {}, "Skipped: Already exists", None, None, None)

                    # Early reject: vague/placeholder feature names ("NewFeature",
                    # "Field 1", "X", purely numeric, etc.) can't be answered
                    # without hallucinating. Skip the LLM entirely.
                    if is_vague_feature_name(f_name):
                        logger.info(f"Skipped vague feature name: {f_name!r}")
                        return (
                            f_name, None, 0, False, {},
                            f"⛔ EARLY REJECT: vague/placeholder feature name '{f_name}'.",
                            None, None, None,
                        )

                    if use_cache:
                        cached_val = await self.db_cache.get_cached_value(
                            session, client_id, product.id, f_name, context_hash
                        )
                        if cached_val:
                            debug_reason = "Cached" if cached_val != "__EMPTY__" else "Cached Empty"
                            val_to_return = None
                            if cached_val != "__EMPTY__":
                                # 👇 БЕЗОПАСНО ПАРСИМ JSON, ЕСЛИ ЭТО СЛОВАРЬ 👇
                                try:
                                    val_to_return = json.loads(cached_val)
                                except json.JSONDecodeError:
                                    val_to_return = cached_val
                            return (f_name, val_to_return, 0, True, {}, debug_reason, None, "cache", None)

                    suffix = getattr(f_schema, 'suffix', "")
                    f_opts = getattr(f_schema, 'options', [])

                    # If the operator configured a prompt_hint for this feature (stored in
                    # admin_rules.settings and forwarded in the BatchFillRequest schema),
                    # prepend it to the product text as a constraint block. The hint is
                    # injected BEFORE the main product info so the LLM sees it as a
                    # high-priority instruction before any extraction routing begins.
                    prompt_hint = getattr(f_schema, 'prompt_hint', None)
                    effective_product_text = info
                    if prompt_hint:
                        effective_product_text = (
                            f"--- OPERATOR CONSTRAINT ---\n{prompt_hint}\n\n"
                            f"--- PRODUCT INFO ---\n{info}"
                        )

                    # 👇 ПЕРЕДАЕМ product.languages + research_mode 👇
                    result_data = await self.pipeline.extract_feature(
                        product_text=effective_product_text,
                        feature_name=f_name,
                        suffix=suffix,
                        options=f_opts,
                        target_languages=product.languages,
                        research_mode=research_mode
                    )

                    raw_val = result_data["value"]
                    tokens = result_data["tokens"]
                    router_debug = result_data.get("router_debug", {})
                    extraction_reasoning = result_data.get("extraction_reasoning", "")
                    deduced_context = result_data.get("deduced_context", None)
                    source = result_data.get("source", None)
                    source_urls = result_data.get("source_urls", None)

                    # 👇 СЕРИАЛИЗУЕМ СЛОВАРЬ В СТРОКУ ПЕРЕД КЕШОМ 👇
                    if raw_val is not None:
                        val_to_save = json.dumps(raw_val, ensure_ascii=False) if isinstance(raw_val, dict) else str(raw_val)
                    else:
                        val_to_save = "__EMPTY__"

                    await self.db_cache.set_cached_value(
                        session, client_id, product.id, f_name, context_hash, val_to_save
                    )

                    return (f_name, raw_val, tokens, False, router_debug, extraction_reasoning, deduced_context, source, source_urls)

        tasks = []
        for f_name, f_schema in schema.items():
            tasks.append(process_feature(f_name, f_schema))

        results = await asyncio.gather(*tasks)
        debug_info = {}

        for res in results:
            if not res: continue

            f_name, val, tokens, from_cache, r_debug, e_reason, d_context, source, source_urls = res

            if not from_cache:
                total_tokens_used += tokens
                is_fully_cached = False

            if val and str(val).lower() == "none":
                val = None

            if val:
                result_features[f_name] = val

            debug_info[f_name] = {
                "router": r_debug,
                "extraction_reasoning": e_reason,
                "deduced_context": d_context,
                "source": source,
                "source_urls": source_urls,
            }

        # Merge enrichment attrs (vision / web_search) into description results.
        # Enrichment wins only when description branch returned nothing for a feature
        # OR when enrichment has strictly higher confidence (AttributeMerger rules).
        for f_name, enrich_av in enrichment_best.items():
            if f_name not in result_features:
                # Description branch missed it — use enrichment value.
                result_features[f_name] = enrich_av.value
                debug_info.setdefault(f_name, {})["source"] = enrich_av.source.value
                debug_info[f_name]["extraction_reasoning"] = (
                    f"[ENRICHMENT branch={enrich_av.source.value} "
                    f"conf={enrich_av.confidence:.2f}]: {enrich_av.reasoning or ''}"
                )
            # (If description already filled it, we keep description result — it has
            # priority per Source.DESCRIPTION > all others in AttributeMerger.)

        final_result = {}
        for f_name, val in result_features.items():
            f_schema = schema.get(f_name)

            if f_schema and f_schema.type == 'select' and getattr(f_schema, 'options', None):
                matched = self.matcher.find_best_match(val, f_schema.options)
                if matched:
                    final_result[f_name] = matched
            else:
                final_result[f_name] = val

        return {
            "product_id": product.id,
            "filled_features": final_result,
            "debug_info": debug_info,
            "tokens_used": total_tokens_used,
            "is_cached": is_fully_cached
        }