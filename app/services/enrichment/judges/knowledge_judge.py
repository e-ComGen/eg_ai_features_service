"""KnowledgeJudge — проверяет что значение реально соответствует known fact о товаре,
не галлюцинация. Использует second LLM call для cross-check.

Per-source judge: знает specific failure mode LLM Knowledge — confidence inflation
(LLM может уверенно выдать неправильное значение для нишевого товара).
"""
from typing import Optional
from pydantic import BaseModel, Field
from app.services.enrichment.base import LlmJudge, AttributeValue, ExtractionContext, Source
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager


class _KnowledgeVerdict(BaseModel):
    valid: bool
    reason: str = Field(max_length=200)


class KnowledgeJudge(LlmJudge):
    source = Source.LLM_KNOWLEDGE

    def __init__(self, llm_manager: Optional[StructuredLlmManager] = None):
        self._llm = llm_manager or get_main_manager()

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        verdict, _ = await self._llm.structured_request(
            system_prompt=(
                "You are a fact-checker for product specifications. Given a product name and "
                "a claimed attribute value, decide if this value is a well-established fact about "
                "this specific product, or potentially a hallucination. Be strict — when in doubt, "
                "return valid=false. Famous products (iPhone, popular Nike models) have well-known "
                "specs; obscure items often invite hallucination."
            ),
            user_text=(
                f"Product: {context.product_name}\n"
                f"Brand: {context.brand or 'unknown'}\n"
                f"Claimed attribute: id={value.attribute_id}, value={value.value!r}\n"
                f"Reasoning given: {value.evidence!r}\n\n"
                f"Is this a well-established fact about this product?"
            ),
            response_model=_KnowledgeVerdict,
        )
        return bool(verdict and verdict.valid)
