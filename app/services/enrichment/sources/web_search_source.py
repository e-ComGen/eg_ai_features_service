"""WebSearchSource — извлекает characteristics через поиск в интернете.

Steps:
1. Serper search → top organic results
2. LLM summary → текст о товаре из найденных страниц
3. Extraction LLM → AttributeValue list

Самый дорогой source. Применять последним когда другие не дали достаточно
информации. CostPredictor (отдельный класс) решает стоит ли запускать.

Spec: docs/architecture/pipeline.md, section "Stage 4 / WebSearchSource".
"""
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
            data["extracted"] = [
                e for e in data["extracted"]
                if isinstance(e, dict) and e.get("value") is not None
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

        # Step 1+2: search + summary (cached per product).
        # MPN передаётся первым в search query — точный код производителя имеет
        # наивысший signal-to-noise (Vision может обогатить context.mpn из фото).
        if context.product_id not in self._summary_cache:
            summary = await self._search.produce_summary(
                product_name=context.product_name,
                brand=context.brand,
                ean=context.ean,
                mpn=context.mpn,
            )
            self._summary_cache[context.product_id] = summary
            # WebSearchProducer делает 1 LLM call внутри + Serper search
            context.llm_calls_so_far += 1
        else:
            summary = self._summary_cache[context.product_id]

        if not summary:
            return []

        # Step 3: extraction from summary с type-aware подсказками
        targets_block = "\n".join([format_target_line(t) for t in effective_targets])
        already_preamble, already_rule = build_already_filled_block(already_filled or [])

        system_prompt = (
            "You extract product characteristics from a summary of web search results. "
            "Prefer values from authoritative sources (manufacturer site, well-known retailers). "
            "Include source URL if mentioned in the summary. Evidence should be a brief quote. "
            "If the target has is_collection=true, return a JSON array of values; otherwise a single scalar."
            + build_meta_guidance()
            + already_rule
        )
        user_text = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n\n"
            f"Web search summary:\n{summary}\n\n"
            + already_preamble
            + f"Target attributes:\n{targets_block}\n\n"
            f"Return JSON with 'extracted' list of {{attribute_id, value, confidence, source_url, evidence}}."
        )

        response_model = self._strategy.build_response_model(_WebExtractionResponse, targets)
        # Маршрутизация: enum-heavy модели → OpenAI strict mode для token-level enforcement
        extractor = self._extractor
        if getattr(response_model, "__has_enum_constraints__", False):
            strict = get_openai_strict_manager()
            if strict is not None:
                extractor = strict
        parsed, tokens = await extractor.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=response_model,
        )
        if parsed is None:
            return []
        context.llm_calls_so_far += 1

        target_by_id = {t.id: t for t in targets}
        return [
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
            for a in parsed.extracted
        ]

    def get_judge(self) -> LlmJudge:
        return self._judge
