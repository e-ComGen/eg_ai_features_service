"""WebSearchSource — извлекает characteristics через поиск в интернете.

Steps:
1. Serper search → top organic results
2. LLM summary → текст о товаре из найденных страниц
3. Extraction LLM → AttributeValue list

Step 0 (apparel): if targets include Ozon attr 4604 (Состав материала) or 4496
(Материал), run domain-agnostic composition mining on fetched page HTML BEFORE
the LLM step. This fills the "apparel data-desert" gap (WB/Ozon have no card but
open shops like kixbox.ru do carry composition). Composition values are emitted
directly — no LLM call needed for them.

Самый дорогой source. Применять последним когда другие не дали достаточно
информации. CostPredictor (отдельный класс) решает стоит ли запускать.

Spec: docs/architecture/pipeline.md, section "Stage 4 / WebSearchSource".
"""
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
    ):
        self._search = websearch_producer or WebSearchProducer()
        self._extractor = extraction_manager or get_main_manager()
        self._judge = WebSearchJudge()
        self._strategy: MarketplaceStrategy = strategy or DefaultStrategy()
        # Cache summary per product_id чтобы не повторять search
        self._summary_cache: dict[int, Optional[str]] = {}

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

        # Merge: composition_avs first (deterministic, no LLM), then LLM results.
        # LLM results that overlap 4604/4496 are kept as well (merger will pick best).
        return composition_avs + llm_avs

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

        # Pass LLM provider and budget so mine_composition can make ONE LLM call
        # if regex found nothing. Budget: respect existing per-product cap.
        # We count a potential LLM call inside mine_composition against the budget.
        raw_provider = getattr(self._extractor, "_provider", self._extractor)
        compositions = await self._search.mine_composition(
            product_name=context.product_name,
            brand=context.brand,
            llm_provider=raw_provider,
            llm_calls_budget=10,      # generous per-product cap
            llm_calls_so_far=context.llm_calls_so_far,
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

        # 4604 — free-text "Состав материала"
        if wants_sostav and _ATTR_SOSTAV_MATERIALA not in filled_ids:
            avs.append(AttributeValue(
                attribute_id=_ATTR_SOSTAV_MATERIALA,
                value=composed_str,
                confidence=_COMPOSITION_CONFIDENCE,
                source=Source.WEB_SEARCH,
                evidence="composition_extractor: two-signal rule on fetched page HTML",
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
                        evidence=f"composition_extractor: dominant material={dom_mat}",
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
