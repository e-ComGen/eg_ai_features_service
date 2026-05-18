"""DescriptionSource — извлекает характеристики из текста описания товара.

Один из 4 sources в cost-aware pipeline. Самый дешёвый: 1 LLM call, не требует
внешних данных кроме product.description. Самый надёжный (highest priority
при merge) — описание привязано к конкретному товару от селлера.

Spec: docs/architecture/pipeline.md, section "Stage 0 / DescriptionSource".
"""
from typing import Optional
from pydantic import BaseModel, Field, AliasChoices
from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager
from app.services.enrichment.judges.description_judge import DescriptionJudge


class _ExtractedAttr(BaseModel):
    """Schema для LLM structured output."""
    model_config = {"populate_by_name": True}
    attribute_id: int = Field(..., validation_alias=AliasChoices("attribute_id", "id"))
    value: str | int | float | bool = Field(..., validation_alias=AliasChoices("value", "attribute_value", "extracted_value"))
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: Optional[str] = None  # цитата из description


class _ExtractionResponse(BaseModel):
    """Top-level response — list of extracted attributes."""
    extracted: list[_ExtractedAttr]


class DescriptionSource(AttributeSource):
    """Извлекает characteristics из product.description через 1 LLM call.

    Если description пустой/слишком короткий — возвращает [] (is_applicable=False).
    """

    def __init__(self, llm_manager: Optional[StructuredLlmManager] = None):
        self._llm = llm_manager or get_main_manager()
        self._judge = DescriptionJudge()

    @property
    def source_type(self) -> Source:
        return Source.DESCRIPTION

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """DescriptionSource применим если есть осмысленное описание."""
        return bool(context.product_description) and len(context.product_description.strip()) >= 10

    async def extract(self, context: ExtractionContext, targets: list[TargetAttribute]) -> list[AttributeValue]:
        if not context.product_description or len(targets) == 0:
            return []

        # Build prompt
        targets_block = "\n".join([
            f"- id={t.id}, name={t.name!r}, type={t.type}" +
            (f", allowed={t.allowed_values}" if t.allowed_values else "")
            for t in targets
        ])

        system_prompt = (
            "You extract product characteristics from product description text. "
            "For each target attribute, find its value in the description if mentioned. "
            "Provide confidence (0-1) based on how clearly the value is stated. "
            "If attribute is not mentioned, do NOT include it in the response. "
            "Provide a short evidence quote (max 100 chars) from the description."
        )
        user_text = (
            f"Product name: {context.product_name}\n"
            f"Category: {' / '.join(context.category_path) or context.category_id}\n\n"
            f"Description:\n{context.product_description}\n\n"
            f"Target attributes:\n{targets_block}\n\n"
            f"Return JSON with field 'extracted' = list of {{attribute_id, value, confidence, evidence}}."
        )

        parsed, tokens = await self._llm.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=_ExtractionResponse,
        )
        if parsed is None:
            return []

        context.llm_calls_so_far += 1
        # cost tracking — точное число cost нужно из manager, пока 0 (TODO в следующем шаге)

        target_by_id = {t.id: t for t in targets}
        return [
            AttributeValue(
                attribute_id=a.attribute_id,
                value=a.value,
                confidence=a.confidence,
                source=Source.DESCRIPTION,
                evidence=a.evidence,
                semantic_type=target_by_id[a.attribute_id].semantic_type
                              if a.attribute_id in target_by_id else None,
            )
            for a in parsed.extracted
        ]

    def get_judge(self) -> LlmJudge:
        return self._judge
