"""DescriptionSource — извлекает характеристики из текста описания товара.

Один из 4 sources в cost-aware pipeline. Самый дешёвый: 1 LLM call, не требует
внешних данных кроме product.description. Самый надёжный (highest priority
при merge) — описание привязано к конкретному товару от селлера.

Spec: docs/architecture/pipeline.md, section "Stage 0 / DescriptionSource".
"""
from typing import Optional
from pydantic import BaseModel, Field, AliasChoices, model_validator
from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager, get_openai_strict_manager
from app.services.enrichment.judges.description_judge import DescriptionJudge
from app.services.enrichment.prompt_router import (
    format_target_line, build_meta_guidance,
    build_already_filled_block, filter_already_filled_targets,
)
from app.services.enrichment.strategies.base import MarketplaceStrategy
from app.services.enrichment.strategies.default_strategy import DefaultStrategy


class _ExtractedAttr(BaseModel):
    """Schema для LLM structured output."""
    model_config = {"populate_by_name": True}
    attribute_id: int = Field(..., validation_alias=AliasChoices("attribute_id", "id"))
    value: str | int | float | bool | list[str | int | float | bool] = Field(..., validation_alias=AliasChoices("value", "attribute_value", "extracted_value"))
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: Optional[str] = None  # цитата из description


class _ExtractionResponse(BaseModel):
    """Top-level response — list of extracted attributes."""
    extracted: list[_ExtractedAttr]

    @model_validator(mode="before")
    @classmethod
    def _drop_null_values(cls, data):
        if isinstance(data, dict) and isinstance(data.get("extracted"), list):
            def _value_of(e):
                if isinstance(e, dict):
                    return e.get("value")
                # Also accept already-constructed _ExtractedAttr instances
                return getattr(e, "value", None)

            data["extracted"] = [
                e for e in data["extracted"]
                if _value_of(e) is not None
            ]
        return data


class DescriptionSource(AttributeSource):
    """Извлекает characteristics из product.description через 1 LLM call.

    Если description пустой/слишком короткий — возвращает [] (is_applicable=False).
    """

    def __init__(
        self,
        llm_manager: Optional[StructuredLlmManager] = None,
        strategy: Optional[MarketplaceStrategy] = None,
    ):
        self._llm = llm_manager or get_main_manager()
        self._judge = DescriptionJudge()
        self._strategy: MarketplaceStrategy = strategy or DefaultStrategy()

    @property
    def source_type(self) -> Source:
        return Source.DESCRIPTION

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """DescriptionSource применим если есть осмысленное описание."""
        return bool(context.product_description) and len(context.product_description.strip()) >= 10

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: list[AttributeValue] | None = None,
    ) -> list[AttributeValue]:
        if not context.product_description or len(targets) == 0:
            return []

        # Убираем уже заполненные attrs из targets чтобы не тратить токены
        effective_targets = filter_already_filled_targets(targets, already_filled or [])
        if not effective_targets:
            return []

        # Build prompt с type-aware подсказками
        targets_block = "\n".join([format_target_line(t) for t in effective_targets])
        already_preamble, already_rule = build_already_filled_block(already_filled or [])

        system_prompt = (
            "You extract product characteristics from product description text. "
            "For each target attribute, find its value in the description if mentioned. "
            "Provide confidence (0-1) based on how clearly the value is stated. "
            "If attribute is not mentioned, do NOT include it in the response. "
            "Provide a short evidence quote (max 100 chars) from the description. "
            "If the target has is_collection=true, return a JSON array of values; otherwise a single scalar."
            + build_meta_guidance()
            + already_rule
        )
        user_text = (
            f"Product name: {context.product_name}\n"
            f"Category: {' / '.join(context.category_path) or context.category_id}\n\n"
            f"Description:\n{context.product_description}\n\n"
            + already_preamble
            + f"Target attributes:\n{targets_block}\n\n"
            f"Return JSON with field 'extracted' = list of {{attribute_id, value, confidence, evidence}}."
        )

        response_model = self._strategy.build_response_model(_ExtractionResponse, targets)
        # Маршрутизация: enum-heavy модели → OpenAI strict mode для token-level enforcement
        llm = self._llm
        if getattr(response_model, "__has_enum_constraints__", False):
            strict = get_openai_strict_manager()
            if strict is not None:
                llm = strict
        parsed, tokens = await llm.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=response_model,
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
                is_collection=target_by_id[a.attribute_id].is_collection
                              if a.attribute_id in target_by_id else False,
            )
            for a in parsed.extracted
        ]

    def get_judge(self) -> LlmJudge:
        return self._judge
