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
from pydantic import BaseModel, Field
from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager
from app.services.enrichment.websearch_producer import WebSearchProducer
from app.services.enrichment.judges.websearch_judge import WebSearchJudge


class _WebExtractedAttr(BaseModel):
    attribute_id: int
    value: str | int | float | bool
    confidence: float = Field(ge=0.0, le=1.0)
    source_url: Optional[str] = None
    evidence: Optional[str] = Field(None, max_length=200)


class _WebExtractionResponse(BaseModel):
    extracted: list[_WebExtractedAttr]


class WebSearchSource(AttributeSource):
    def __init__(
        self,
        websearch_producer: Optional[WebSearchProducer] = None,
        extraction_manager: Optional[StructuredLlmManager] = None,
    ):
        self._search = websearch_producer or WebSearchProducer()
        self._extractor = extraction_manager or get_main_manager()
        self._judge = WebSearchJudge()
        # Cache summary per product_id чтобы не повторять search
        self._summary_cache: dict[int, Optional[str]] = {}

    @property
    def source_type(self) -> Source:
        return Source.WEB_SEARCH

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим всегда если есть осмысленный product_name (search query).
        Реальное решение запускать ли — за CostPredictor (отдельный stage)."""
        return bool(context.product_name) and len(context.product_name) >= 5

    async def extract(self, context: ExtractionContext, targets: list[TargetAttribute]) -> list[AttributeValue]:
        if not targets or not context.product_name:
            return []

        # Step 1+2: search + summary (cached per product)
        if context.product_id not in self._summary_cache:
            summary = await self._search.produce_summary(
                product_name=context.product_name,
                brand=context.brand,
                ean=context.ean,
            )
            self._summary_cache[context.product_id] = summary
            # WebSearchProducer делает 1 LLM call внутри + Serper search
            context.llm_calls_so_far += 1
        else:
            summary = self._summary_cache[context.product_id]

        if not summary:
            return []

        # Step 3: extraction from summary
        targets_block = "\n".join([
            f"- id={t.id}, name={t.name!r}, type={t.type}" +
            (f", allowed={t.allowed_values}" if t.allowed_values else "")
            for t in targets
        ])

        system_prompt = (
            "You extract product characteristics from a summary of web search results. "
            "Prefer values from authoritative sources (manufacturer site, well-known retailers). "
            "Include source URL if mentioned in the summary. Evidence should be a brief quote."
        )
        user_text = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n\n"
            f"Web search summary:\n{summary}\n\n"
            f"Target attributes:\n{targets_block}\n\n"
            f"Return JSON with 'extracted' list of {{attribute_id, value, confidence, source_url, evidence}}."
        )

        parsed, tokens = await self._extractor.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=_WebExtractionResponse,
        )
        if parsed is None:
            return []
        context.llm_calls_so_far += 1

        return [
            AttributeValue(
                attribute_id=a.attribute_id,
                value=a.value,
                confidence=a.confidence,
                source=Source.WEB_SEARCH,
                evidence=f"[{a.source_url}] {a.evidence}" if a.source_url else a.evidence,
            )
            for a in parsed.extracted
        ]

    def get_judge(self) -> LlmJudge:
        return self._judge
