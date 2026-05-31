"""VisionJudge — проверяет что вывод обоснован тем что реально видно на фото.

Per-source: знает specific failure mode Vision — over-inference (LLM может
догадываться о невидимых свойствах из контекста, не из самого фото).
"""
import logging
from typing import Optional
from pydantic import BaseModel, Field
from app.services.enrichment.base import LlmJudge, AttributeValue, ExtractionContext, Source
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager

logger = logging.getLogger(__name__)


class _VisionVerdict(BaseModel):
    valid: bool
    reason: str = Field(max_length=500)


class VisionJudge(LlmJudge):
    source = Source.VISION

    def __init__(self, llm_manager: Optional[StructuredLlmManager] = None):
        self._llm = llm_manager or get_main_manager()
        # Lookup table: attribute_id -> allowed_values (set by VisionSource before judge calls).
        # Lets the judge auto-accept enum values that match allowed list verbatim
        # без второго LLM-вызова (fixes «enum extracted but rejected as not-literal»).
        self._allowed_values: dict[int, list[str]] = {}

    def register_allowed_values(self, mapping: dict[int, list[str]]) -> None:
        """Регистрирует allowed_values для enum-targets текущей extraction batch.

        VisionSource вызывает это в extract() перед тем как pipeline дёрнет
        ConfidenceAwareJudgeWrapper.maybe_validate(). Старые записи мержатся,
        не очищаются — judge может быть переиспользован для разных продуктов.
        """
        if mapping:
            self._allowed_values.update(mapping)

    @staticmethod
    def _matches_allowed(value: object, allowed: list[str]) -> bool:
        """Case-insensitive match сравнение value против allowed list."""
        if not allowed:
            return False
        if isinstance(value, list):
            # is_collection: каждый элемент должен соответствовать allowed
            allowed_lower = {str(a).strip().lower() for a in allowed}
            return all(
                isinstance(item, (str, int, float, bool))
                and str(item).strip().lower() in allowed_lower
                for item in value
            ) and len(value) > 0
        allowed_lower = {str(a).strip().lower() for a in allowed}
        return str(value).strip().lower() in allowed_lower

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        """Проверка: цитата evidence из vision-description реально визуальная (не догадка).

        Fast-path: если target — enum (есть allowed_values) И extracted value
        матчится одному из allowed (case-insensitive) → accept автоматом без LLM.
        Это unlock enum fills которые currently rejected as «not literal-from-image».
        """
        allowed = self._allowed_values.get(value.attribute_id)
        if allowed and self._matches_allowed(value.value, allowed):
            logger.debug(
                "[VisionJudge] enum fast-accept attr=%d value=%r (matches allowed)",
                value.attribute_id, value.value,
            )
            return True

        if not value.evidence:
            return False

        verdict, _ = await self._llm.structured_request(
            system_prompt=(
                "You verify if a visual attribute was actually OBSERVED on product photos, "
                "not inferred from product name or category. "
                "Evidence is valid if it describes (a) what is visually seen on the product "
                "(color, shape, material appearance), OR (b) text/logos transcribed verbatim "
                "from packaging, labels, or stickers visible in the photos. "
                "Reject only if the evidence is generic, contradicts the photos, or sounds "
                "like an assumption drawn purely from the product name/category."
            ),
            user_text=(
                f"Product: {context.product_name}\n"
                f"Visual attribute value: {value.value!r}\n"
                f"Cited visual evidence: {value.evidence!r}\n\n"
                f"Is this evidence clearly visual (not inferred)?"
            ),
            response_model=_VisionVerdict,
        )
        return bool(verdict and verdict.valid)
