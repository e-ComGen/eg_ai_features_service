"""WebSearchSource — извлекает characteristics через поиск в интернете.

Steps:
1. Serper search → top organic results
2. LLM summary → текст о товаре из найденных страниц
3. Extraction LLM → AttributeValue list

Step 0 (apparel): if targets include Ozon attr 4604 (Состав материала) or 4496
(Материал), run the adaptive multi-site composition harvester (harvest_composition)
BEFORE the LLM step. This fills the "apparel data-desert" gap using a curated pool
of retail sites (kixbox/sneakerhead/basketshop/brandshop/street-beat/blankstyle via
plain httpx; sportmaster/lamoda via real-Chrome BrowserFetcher). Falls back to the
legacy WebSearchProducer.mine_composition if harvest_composition yields nothing.
Composition values are emitted directly — no LLM call needed for them.

Самый дорогой source. Применять последним когда другие не дали достаточно
информации. CostPredictor (отдельный класс) решает стоит ли запускать.

Spec: docs/architecture/pipeline.md, section "Stage 4 / WebSearchSource".
"""
import atexit
import logging
import os
from typing import Optional
from pydantic import BaseModel, Field, AliasChoices, model_validator
from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager, get_openai_strict_manager
from app.services.enrichment.websearch_producer import WebSearchProducer
from app.services.enrichment.judges.websearch_judge import WebSearchJudge
from app.services.enrichment.prompt_router import (
    format_target_line, build_meta_guidance,
    build_already_filled_block, filter_already_filled_targets,
)
from app.services.enrichment.strategies.base import MarketplaceStrategy
from app.services.enrichment.strategies.default_strategy import DefaultStrategy

logger = logging.getLogger(__name__)

# Ozon attribute IDs for fabric composition (hardcoded by Ozon spec, not by us)
_ATTR_SOSTAV_MATERIALA = 4604   # free-text "Состав материала"
_ATTR_MATERIAL = 4496           # enum "Материал"

# Confidence cap for composition sourced from open shops (below WB/Ozon card
# sources at 0.90, but meaningful signal — brand-verified page + two-signal rule)
_COMPOSITION_CONFIDENCE = 0.72


class _WebExtractedAttr(BaseModel):
    model_config = {"populate_by_name": True}
    attribute_id: int = Field(..., validation_alias=AliasChoices("attribute_id", "id"))
    value: str | int | float | bool | list[str | int | float | bool] = Field(..., validation_alias=AliasChoices("value", "attribute_value", "extracted_value"))
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    source_url: Optional[str] = None
    evidence: Optional[str] = Field(None, max_length=200)


class _WebExtractionResponse(BaseModel):
    extracted: list[_WebExtractedAttr]

    @model_validator(mode="before")
    @classmethod
    def _drop_null_values(cls, data):
        if isinstance(data, dict) and isinstance(data.get("extracted"), list):
            def _value_of(e):
                if isinstance(e, dict):
                    return e.get("value")
                # Also accept already-constructed _WebExtractedAttr instances
                return getattr(e, "value", None)

            data["extracted"] = [
                e for e in data["extracted"]
                if _value_of(e) is not None
            ]
        return data


class WebSearchSource(AttributeSource):
    def __init__(
        self,
        websearch_producer: Optional[WebSearchProducer] = None,
        extraction_manager: Optional[StructuredLlmManager] = None,
        strategy: Optional[MarketplaceStrategy] = None,
        browser_fetcher=None,  # injected BrowserFetcher for DI/testing; None → lazy-created
    ):
        self._search = websearch_producer or WebSearchProducer()
        self._extractor = extraction_manager or get_main_manager()
        self._judge = WebSearchJudge()
        self._strategy: MarketplaceStrategy = strategy or DefaultStrategy()
        # Cache summary per product_id чтобы не повторять search
        self._summary_cache: dict[int, Optional[str]] = {}

        # Shared BrowserFetcher for composition harvesting — one instance per
        # WebSearchSource, lazily created on first browser-site need.
        # Caller may inject one (testing / pipeline-level sharing).
        # If WE created it, close it on GC / atexit.
        self._browser_fetcher = browser_fetcher
        self._browser_fetcher_owned = False  # True only when we created it

        if browser_fetcher is not None:
            # Injected from outside — caller owns lifecycle, we never close it
            self._browser_fetcher_owned = False

    @property
    def source_type(self) -> Source:
        return Source.WEB_SEARCH

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим всегда если есть осмысленный product_name (search query).
        Реальное решение запускать ли — за CostPredictor (отдельный stage)."""
        return bool(context.product_name) and len(context.product_name) >= 5

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: list[AttributeValue] | None = None,
    ) -> list[AttributeValue]:
        if not targets or not context.product_name:
            return []

        # Убираем уже заполненные attrs из targets чтобы не тратить токены
        effective_targets = filter_already_filled_targets(targets, already_filled or [])
        if not effective_targets:
            return []

        # Step 0: composition mining (apparel data-desert).
        # Run BEFORE the LLM path — no LLM call needed for composition.
        # Only fires when 4604 or 4496 are among the unfilled targets.
        composition_avs = await self._mine_composition_if_needed(
            context, effective_targets, already_filled or []
        )

        # Step 1+2: search + summary (cached per product).
        # MPN передаётся первым в search query — точный код производителя имеет
        # наивысший signal-to-noise (Vision может обогатить context.mpn из фото).
        if context.product_id not in self._summary_cache:
            # Dual-lang search ON by default in the pipeline: ru + en in parallel
            # (EN manufacturer pages hold authoritative specs, RU pages local
            # variants; producer concatenates both into ONE extraction call).
            # context.languages wins if set; else WEBSEARCH_LANGS env (default
            # "ru,en"). Set WEBSEARCH_LANGS=ru to disable EN cheaply.
            languages = context.languages
            if languages is None:
                languages = [
                    lang.strip()
                    for lang in os.environ.get("WEBSEARCH_LANGS", "ru,en").split(",")
                    if lang.strip()
                ]
            summary = await self._search.produce_summary(
                product_name=context.product_name,
                brand=context.brand,
                ean=context.ean,
                mpn=context.mpn,
                languages=languages,
            )
            self._summary_cache[context.product_id] = summary
            # WebSearchProducer делает 1 LLM call внутри + Serper search
            context.llm_calls_so_far += 1
        else:
            summary = self._summary_cache[context.product_id]

        if not summary:
            return []

        # Step 2.5: OEM spec-pair harvest — verbatim spec extraction from raw page text.
        # Fires ONLY when OEM_SPEC_HARVEST_ENABLED=1 (default OFF). No LLM call needed.
        # Reads the raw page text (trafilatura output before LLM summarisation) that
        # WebSearchProducer stored in _last_page_text during produce_summary.
        # Falls back silently to [] when the flag is off or no page text was cached.
        oem_spec_avs: list[AttributeValue] = []
        try:
            from app.services.enrichment.sources.oem_spec_harvest import (
                harvest_spec_pairs,
                OEM_SPEC_HARVEST_ENABLED,
            )
            if OEM_SPEC_HARVEST_ENABLED:
                raw_page = getattr(self._search, "_last_page_text", None)
                if raw_page:
                    oem_spec_avs = await harvest_spec_pairs(
                        raw_page, effective_targets, already_filled or [], context
                    )
        except Exception as exc:
            logger.warning(
                "WebSearchSource: oem_spec_harvest failed (non-fatal): %s", exc
            )

        # Step 3: extraction from summary с type-aware подсказками
        # Батчинг: разбиваем targets на чанки по CHUNK_SIZE, context не дублируем
        CHUNK_SIZE = 30
        already_preamble, already_rule = build_already_filled_block(already_filled or [])

        system_prompt = (
            "You extract product characteristics from a summary of web search results. "
            "Prefer values from authoritative sources (manufacturer site, well-known retailers). "
            "Include source URL if mentioned in the summary. Evidence should be a brief quote. "
            "If the target has is_collection=true, return a JSON array of values; otherwise a single scalar."
            + build_meta_guidance()
            + already_rule
        )

        context_prefix = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n\n"
            f"Web search summary:\n{summary}\n\n"
            + already_preamble
        )

        chunks = [
            effective_targets[i: i + CHUNK_SIZE]
            for i in range(0, len(effective_targets), CHUNK_SIZE)
        ]

        target_by_id = {t.id: t for t in targets}
        all_extracted: list[_WebExtractedAttr] = []

        for chunk in chunks:
            targets_block = "\n".join([format_target_line(t) for t in chunk])
            user_text = (
                context_prefix
                + f"Target attributes:\n{targets_block}\n\n"
                f"Return JSON with 'extracted' list of {{attribute_id, value, confidence, source_url, evidence}}."
            )

            response_model = self._strategy.build_response_model(_WebExtractionResponse, chunk)
            # Маршрутизация: enum-heavy модели → OpenAI strict mode для token-level enforcement
            extractor = self._extractor
            if getattr(response_model, "__has_enum_constraints__", False):
                strict = get_openai_strict_manager()
                if strict is not None:
                    extractor = strict
            parsed, _tokens = await extractor.structured_request(
                system_prompt=system_prompt,
                user_text=user_text,
                response_model=response_model,
            )
            context.llm_calls_so_far += 1
            if parsed is not None:
                all_extracted.extend(parsed.extracted)

        # Дедуп + защита от кросс-чанк галлюцинаций: оставляем только id, которые
        # реально были в effective_targets, первое вхождение на id.
        _eff_ids = {t.id for t in effective_targets}
        _seen: set[int] = set()
        all_extracted = [
            a for a in all_extracted
            if a.attribute_id in _eff_ids
            and not (a.attribute_id in _seen or _seen.add(a.attribute_id))
        ]

        llm_avs = [
            AttributeValue(
                attribute_id=a.attribute_id,
                value=a.value,
                confidence=a.confidence,
                source=Source.WEB_SEARCH,
                evidence=f"[{a.source_url}] {a.evidence}" if a.source_url else a.evidence,
                semantic_type=target_by_id[a.attribute_id].semantic_type
                              if a.attribute_id in target_by_id else None,
                is_collection=target_by_id[a.attribute_id].is_collection
                              if a.attribute_id in target_by_id else False,
            )
            for a in all_extracted
        ]

        # Merge: composition_avs first (deterministic), then oem_spec_avs (verbatim,
        # source=DESCRIPTION, higher priority in merge), then LLM results.
        # Ordering matters: deterministic fills come first so the merger can protect
        # them from lower-confidence LLM overwrites via _merge_winner / card-protection.
        return composition_avs + oem_spec_avs + llm_avs

    # ------------------------------------------------------------------
    # BrowserFetcher lifecycle helpers
    # ------------------------------------------------------------------

    def _get_browser_fetcher(self):
        """Return the shared BrowserFetcher, lazily creating it once.

        We own the instance only when we create it here (not when injected).
        Registered with atexit to ensure cleanup on interpreter exit.
        """
        if self._browser_fetcher is None:
            from app.services.providers.browser_fetcher import BrowserFetcher
            self._browser_fetcher = BrowserFetcher()
            self._browser_fetcher_owned = True
            logger.info("WebSearchSource: created shared BrowserFetcher (owned)")

            # Schedule cleanup on interpreter exit (fire-and-forget; best effort)
            def _atexit_close():
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed():
                        loop.run_until_complete(self._close_browser_fetcher())
                except Exception:
                    pass

            atexit.register(_atexit_close)
        return self._browser_fetcher

    async def _close_browser_fetcher(self) -> None:
        """Close the BrowserFetcher if we own it. Safe to call multiple times."""
        if self._browser_fetcher_owned and self._browser_fetcher is not None:
            try:
                await self._browser_fetcher.close()
            except Exception as exc:
                logger.debug("WebSearchSource: BrowserFetcher close error: %s", exc)
            finally:
                self._browser_fetcher = None
                self._browser_fetcher_owned = False

    # ------------------------------------------------------------------
    # Composition mining helper
    # ------------------------------------------------------------------

    async def _mine_composition_if_needed(
        self,
        context: ExtractionContext,
        effective_targets: list[TargetAttribute],
        already_filled: list[AttributeValue],
    ) -> list[AttributeValue]:
        """Mine fabric composition from raw pages and emit Состав/Материал AVs.

        Integration point for the adaptive multi-site harvester:
          1. harvest_composition (PREFERRED): tries curated pool of retail sites
             (open httpx: kixbox/sneakerhead/basketshop/brandshop/street-beat/
             blankstyle; real-Chrome: sportmaster/lamoda) with SPA auto-escalation.
             Returns a rich dict {composition, material, source_url, site, evidence}.
          2. mine_composition (FALLBACK): legacy Serper search + httpx + LLM recall.
             Runs only when harvest_composition yields nothing.

        Gating (cost-aware):
          - Only fires when 4604 or 4496 are in the unfilled targets (apparel check).
          - Respects already_filled (skips if both fields are already filled).
          - BrowserFetcher (real-Chrome) is only created on first browser-site need
            and shared across all calls on this WebSearchSource instance.

        Returns [] when:
        - Neither 4604 nor 4496 is in unfilled targets.
        - The producer doesn't support composition mining (no Serper).
        - Nothing passes the two-signal filter + brand-verification gate.
        """
        target_ids = {t.id for t in effective_targets}
        wants_sostav = _ATTR_SOSTAV_MATERIALA in target_ids
        wants_material = _ATTR_MATERIAL in target_ids
        if not wants_sostav and not wants_material:
            return []

        # Already filled check (don't mine if already emitted by another source)
        filled_ids = {av.attribute_id for av in already_filled}
        if _ATTR_SOSTAV_MATERIALA in filled_ids and not wants_material:
            return []
        if _ATTR_MATERIAL in filled_ids and not wants_sostav:
            return []

        # Brand is required by harvest_composition for the brand-verification gate.
        # Without it the harvester accepts any page → false-positives. Fall through
        # to mine_composition (which also works without brand, but uses LLM recall).
        product_name = context.product_name or ""
        brand = context.brand or ""

        compositions: list[str] = []
        harvest_source_url: Optional[str] = None
        harvest_evidence: Optional[str] = None

        # ------------------------------------------------------------------
        # Path A: adaptive multi-site harvester (harvest_composition)
        # Preferred: curated pool, SPA escalation, brand-gate, deterministic.
        # Cost: open-httpx sites = cheap (<<$0.001); browser sites = ~$0.01-0.03
        #       per product (Chrome launch + 2 pages). Only fires when open sites
        #       fail AND Serper found a browser-site URL.
        # ------------------------------------------------------------------
        if product_name and brand:
            try:
                from app.services.enrichment.sources.multisite_composition import (
                    harvest_composition,
                )
                # Inject the shared BrowserFetcher so Chrome is not relaunched
                # per product — harvest_composition will NOT close it (we own it).
                bf = self._browser_fetcher  # may be None; passed for reuse if exists
                result = await harvest_composition(
                    product_name=product_name,
                    brand=brand,
                    browser_fetcher=bf,
                )
                if result is not None:
                    compositions = [result["composition"]]
                    harvest_source_url = result.get("source_url")
                    harvest_evidence = (
                        f"[{result.get('site', '')}] {result.get('evidence', '')}"
                        f" (route={result.get('route', 'open')})"
                    )
                    logger.info(
                        "WebSearchSource: harvest_composition HIT site=%r "
                        "composition=%r route=%s",
                        result.get("site"), result["composition"][:80],
                        result.get("route", "open"),
                    )
            except Exception as exc:
                logger.warning(
                    "WebSearchSource: harvest_composition raised: %s — falling back "
                    "to mine_composition",
                    exc,
                )

        # ------------------------------------------------------------------
        # Path B: legacy mine_composition fallback
        # Runs when: harvest_composition found nothing OR brand was empty.
        # Cost: 2-3 Serper calls + httpx + optional 1 LLM call.
        # ------------------------------------------------------------------
        if not compositions:
            raw_provider = getattr(self._extractor, "_provider", self._extractor)
            compositions = await self._search.mine_composition(
                product_name=product_name,
                brand=brand or None,
                llm_provider=raw_provider,
                llm_calls_budget=10,      # generous per-product cap
                llm_calls_so_far=context.llm_calls_so_far,
            )
            if compositions:
                logger.info(
                    "WebSearchSource: mine_composition fallback HIT: %r",
                    compositions,
                )

        if not compositions:
            return []

        from app.services.enrichment.composition_extractor import (
            normalize_material_en_ru,
            primary_material,
        )
        from app.services.enrichment.strategies.dictionaries.ozon_loader import resolve_value_id

        avs: list[AttributeValue] = []
        composed_str = "; ".join(compositions)

        # Build evidence string: prefer rich harvester evidence, fall back to legacy tag.
        av_evidence_sostav = (
            harvest_evidence
            if harvest_evidence
            else "composition_extractor: two-signal rule on fetched page HTML"
        )
        av_evidence_material_suffix = (
            f" via {harvest_source_url}" if harvest_source_url else ""
        )

        # 4604 — free-text "Состав материала"
        if wants_sostav and _ATTR_SOSTAV_MATERIALA not in filled_ids:
            avs.append(AttributeValue(
                attribute_id=_ATTR_SOSTAV_MATERIALA,
                value=composed_str,
                confidence=_COMPOSITION_CONFIDENCE,
                source=Source.WEB_SEARCH,
                evidence=av_evidence_sostav,
            ))
            logger.info(
                "WebSearchSource: emitting Состав материала(4604)=%r (conf=%.2f)",
                composed_str[:80], _COMPOSITION_CONFIDENCE,
            )

        # 4496 — enum "Материал": resolve dominant material to dict value_id
        if wants_material and _ATTR_MATERIAL not in filled_ids:
            dom_mat = primary_material(compositions)
            if dom_mat:
                # Attempt resolve; only emit if we get a value_id (fail-closed)
                type_id = getattr(context, "ozon_type_id", None)
                value_id = None
                if type_id is not None:
                    try:
                        value_id = resolve_value_id(
                            context.category_id, type_id, _ATTR_MATERIAL, dom_mat
                        )
                    except Exception as exc:
                        logger.debug(
                            "WebSearchSource: resolve_value_id(4496, %r) failed: %s", dom_mat, exc
                        )
                if value_id is not None:
                    av = AttributeValue(
                        attribute_id=_ATTR_MATERIAL,
                        value=dom_mat,
                        confidence=_COMPOSITION_CONFIDENCE,
                        source=Source.WEB_SEARCH,
                        evidence=(
                            f"composition_extractor: dominant material={dom_mat}"
                            + av_evidence_material_suffix
                        ),
                        value_id=value_id,
                    )
                    avs.append(av)
                    logger.info(
                        "WebSearchSource: emitting Материал(4496)=%r value_id=%d",
                        dom_mat, value_id,
                    )
                else:
                    logger.debug(
                        "WebSearchSource: Материал(4496) %r could not be resolved — skipped",
                        dom_mat,
                    )

        return avs

    def get_judge(self) -> LlmJudge:
        return self._judge
