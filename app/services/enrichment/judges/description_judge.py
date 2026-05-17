"""DescriptionJudge — проверяет что extracted value реально упомянуто в description.

Per-source judge: знает specific failure mode DescriptionSource —
hallucination (LLM может придумать значение которого нет в тексте).
"""
from pydantic import BaseModel, Field
from app.services.enrichment.base import LlmJudge, AttributeValue, ExtractionContext, Source
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager
from typing import Optional


class _JudgeVerdict(BaseModel):
    valid: bool
    reason: str = Field(max_length=200)


class DescriptionJudge(LlmJudge):
    """Validate: значение действительно есть в description text?"""
    source = Source.DESCRIPTION

    def __init__(self, llm_manager: Optional[StructuredLlmManager] = None):
        self._llm = llm_manager or get_main_manager()

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        if not context.product_description:
            return False

        verdict, _ = await self._llm.structured_request(
            system_prompt=(
                "You are a strict fact-checker. Given a product description and an extracted "
                "attribute value, decide if the value is actually supported by the description. "
                "Return valid=false if the value is a hallucination (not in text)."
            ),
            user_text=(
                f"Description:\n{context.product_description}\n\n"
                f"Extracted attribute: id={value.attribute_id}, value={value.value!r}\n"
                f"Evidence cited: {value.evidence!r}\n\n"
                f"Is this value really supported by the description?"
            ),
            response_model=_JudgeVerdict,
        )
        return bool(verdict and verdict.valid)
