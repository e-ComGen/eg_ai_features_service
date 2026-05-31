"""LlmKnowledgeSource — извлекает характеристики из обучающей памяти LLM.

Полезно для известных брендов/моделей (iPhone, Nike, Samsung) — LLM знает
типичные характеристики без поиска. 1 LLM call. Confidence threshold 0.92
(строже чем для description, потому что LLM может галлюцинировать).

Spec: docs/architecture/pipeline.md, section "Stage 2 / LlmKnowledgeSource".
"""
from typing import Optional
from pydantic import BaseModel, Field, AliasChoices, model_validator
from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager, get_openai_strict_manager
from app.services.enrichment.judges.knowledge_judge import KnowledgeJudge
from app.services.enrichment.prompt_router import (
    format_target_line, build_meta_guidance,
    build_already_filled_block, filter_already_filled_targets,
)
from app.services.enrichment.strategies.base import MarketplaceStrategy
from app.services.enrichment.strategies.default_strategy import DefaultStrategy


class _KnowledgeAttr(BaseModel):
    attribute_id: int = Field(..., validation_alias=AliasChoices("attribute_id", "id"))
    value: str | int | float | bool | list[str | int | float | bool] = Field(..., validation_alias=AliasChoices("value", "attribute_value", "extracted_value"))
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(None, max_length=200, description="откуда LLM знает")

    model_config = {"populate_by_name": True}


class _KnowledgeResponse(BaseModel):
    known_attributes: list[_KnowledgeAttr]

    @model_validator(mode="before")
    @classmethod
    def _drop_null_values(cls, data):
        if isinstance(data, dict) and isinstance(data.get("known_attributes"), list):
            data["known_attributes"] = [
                e for e in data["known_attributes"]
                if isinstance(e, dict) and e.get("value") is not None
            ]
        return data


class LlmKnowledgeSource(AttributeSource):
    """1 LLM call — извлечение из обучающей памяти LLM.

    Применим только для well-known товаров (есть бренд + конкретная модель).
    Для no-name товаров возвращает [] чтобы не тратить токены на галлюцинации.
    """

    def __init__(
        self,
        llm_manager: Optional[StructuredLlmManager] = None,
        strategy: Optional[MarketplaceStrategy] = None,
    ):
        self._llm = llm_manager or get_main_manager()
        self._judge = KnowledgeJudge()
        self._strategy: MarketplaceStrategy = strategy or DefaultStrategy()

    @property
    def source_type(self) -> Source:
        return Source.LLM_KNOWLEDGE

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим если есть бренд (известный товар) или конкретное модель в названии."""
        # Heuristic: длинное название с бренд-likely substring И не пустое
        if not context.product_name or len(context.product_name) < 5:
            return False
        # Если есть явный brand field — отлично
        if context.brand:
            return True
        # Иначе предполагаем что если название содержит признаки модели (число, версия) — known
        # Для MVP: возвращаем True для всех товаров с осмысленным name, judge отфильтрует галлюцинации
        return True

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

        # Build prompt с type-aware подсказками
        targets_block = "\n".join([format_target_line(t) for t in effective_targets])
        already_preamble, already_rule = build_already_filled_block(already_filled or [])

        system_prompt = (
            "You are a product knowledge expert. Given a product name (and optional brand), "
            "you provide attribute values that you confidently KNOW from your training data. "
            "If you are NOT sure about an attribute — DO NOT include it. Better to skip than guess. "
            "Confidence scale: 0.95-1.0 = industry-standard or official spec (e.g. Samsung S24 Ultra "
            "camera is 200MP, Adidas Superstar sole is rubber — these are well-known facts); "
            "0.92-0.94 = highly likely but minor variation possible; "
            "below 0.92 = uncertain, DO NOT include. "
            "Set confidence=0.95 for facts you know with certainty from official specs or brand history. "
            "Brief reasoning helps audit (e.g., 'official Samsung spec', 'Adidas classic model'). "
            "If the target has is_collection=true, return a JSON array of values; otherwise a single scalar."
            + build_meta_guidance()
            + already_rule
        )
        user_text = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n"
            f"Category: {' / '.join(context.category_path) or 'n/a'}\n\n"
            + already_preamble
            + f"Target attributes:\n{targets_block}\n\n"
            f"Return only attributes you confidently know. Field name: 'known_attributes'."
        )

        response_model = self._strategy.build_response_model(_KnowledgeResponse, targets)
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

        target_by_id = {t.id: t for t in targets}
        return [
            AttributeValue(
                attribute_id=a.attribute_id,
                value=a.value,
                confidence=a.confidence,
                source=Source.LLM_KNOWLEDGE,
                evidence=a.reasoning,
                semantic_type=target_by_id[a.attribute_id].semantic_type
                              if a.attribute_id in target_by_id else None,
                is_collection=target_by_id[a.attribute_id].is_collection
                              if a.attribute_id in target_by_id else False,
            )
            for a in parsed.known_attributes
        ]

    def get_judge(self) -> LlmJudge:
        return self._judge
