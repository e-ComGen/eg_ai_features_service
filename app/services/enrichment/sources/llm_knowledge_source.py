"""LlmKnowledgeSource — извлекает характеристики из обучающей памяти LLM.

Полезно для известных брендов/моделей (iPhone, Nike, Samsung) — LLM знает
типичные характеристики без поиска. 1 LLM call. Confidence threshold 0.92
(строже чем для description, потому что LLM может галлюцинировать).

Spec: docs/architecture/pipeline.md, section "Stage 2 / LlmKnowledgeSource".
"""
from typing import Optional
from pydantic import BaseModel, Field, AliasChoices
from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager
from app.services.enrichment.judges.knowledge_judge import KnowledgeJudge


class _KnowledgeAttr(BaseModel):
    attribute_id: int = Field(..., validation_alias=AliasChoices("attribute_id", "id"))
    value: str | int | float | bool = Field(..., validation_alias=AliasChoices("value", "attribute_value", "extracted_value"))
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(None, max_length=200, description="откуда LLM знает")

    model_config = {"populate_by_name": True}


class _KnowledgeResponse(BaseModel):
    known_attributes: list[_KnowledgeAttr]


class LlmKnowledgeSource(AttributeSource):
    """1 LLM call — извлечение из обучающей памяти LLM.

    Применим только для well-known товаров (есть бренд + конкретная модель).
    Для no-name товаров возвращает [] чтобы не тратить токены на галлюцинации.
    """

    def __init__(self, llm_manager: Optional[StructuredLlmManager] = None):
        self._llm = llm_manager or get_main_manager()
        self._judge = KnowledgeJudge()

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

    async def extract(self, context: ExtractionContext, targets: list[TargetAttribute]) -> list[AttributeValue]:
        if not targets or not context.product_name:
            return []

        targets_block = "\n".join([
            f"- id={t.id}, name={t.name!r}, type={t.type}" +
            (f", allowed={t.allowed_values}" if t.allowed_values else "")
            for t in targets
        ])

        system_prompt = (
            "You are a product knowledge expert. Given a product name (and optional brand), "
            "you provide attribute values that you confidently KNOW from your training data. "
            "If you are NOT sure about an attribute — DO NOT include it. Better to skip than guess. "
            "Confidence scale: 0.95-1.0 = industry-standard or official spec (e.g. Samsung S24 Ultra "
            "camera is 200MP, Adidas Superstar sole is rubber — these are well-known facts); "
            "0.92-0.94 = highly likely but minor variation possible; "
            "below 0.92 = uncertain, DO NOT include. "
            "Set confidence=0.95 for facts you know with certainty from official specs or brand history. "
            "Brief reasoning helps audit (e.g., 'official Samsung spec', 'Adidas classic model')."
        )
        user_text = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n"
            f"Category: {' / '.join(context.category_path) or 'n/a'}\n\n"
            f"Target attributes:\n{targets_block}\n\n"
            f"Return only attributes you confidently know. Field name: 'known_attributes'."
        )

        parsed, tokens = await self._llm.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=_KnowledgeResponse,
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
            )
            for a in parsed.known_attributes
        ]

    def get_judge(self) -> LlmJudge:
        return self._judge
