"""VisionJudge — проверяет что вывод обоснован тем что реально видно на фото.

Per-source: знает specific failure mode Vision — over-inference (LLM может
догадываться о невидимых свойствах из контекста, не из самого фото).
"""
from typing import Optional
from pydantic import BaseModel, Field
from app.services.enrichment.base import LlmJudge, AttributeValue, ExtractionContext, Source
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager


class _VisionVerdict(BaseModel):
    valid: bool
    reason: str = Field(max_length=200)


class VisionJudge(LlmJudge):
    source = Source.VISION

    def __init__(self, llm_manager: Optional[StructuredLlmManager] = None):
        self._llm = llm_manager or get_main_manager()

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        """Проверка: цитата evidence из vision-description реально визуальная (не догадка)."""
        if not value.evidence:
            return False

        verdict, _ = await self._llm.structured_request(
            system_prompt=(
                "You verify if a visual attribute was actually OBSERVED on product photos, "
                "not inferred from product name or category. The evidence should describe what "
                "is visually seen (color, shape, material appearance), not deduced. "
                "Reject if the evidence is generic or sounds like assumption."
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
