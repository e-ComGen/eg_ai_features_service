"""WebSearchJudge — проверяет credibility источника + цитату.

Per-source: знает specific failure mode Web Search — unreliable sources
(блог-копия может противоречить производителю), outdated info, marketing claims.
"""
from typing import Optional
from pydantic import BaseModel, Field
from app.services.enrichment.base import LlmJudge, AttributeValue, ExtractionContext, Source
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager


class _WebVerdict(BaseModel):
    valid: bool
    reason: str = Field(max_length=200)


class WebSearchJudge(LlmJudge):
    source = Source.WEB_SEARCH

    def __init__(self, llm_manager: Optional[StructuredLlmManager] = None):
        self._llm = llm_manager or get_main_manager()

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        """Validate: источник в evidence credible, цитата подтверждает value."""
        if not value.evidence:
            return False

        verdict, _ = await self._llm.structured_request(
            system_prompt=(
                "You validate web search-extracted product attributes. Check: "
                "1) Source mentioned in evidence is authoritative (manufacturer, major retailer) "
                "not just a random blog or forum. "
                "2) Evidence quote actually supports the claimed value, not tangentially related. "
                "3) Information looks current, not outdated marketing."
            ),
            user_text=(
                f"Product: {context.product_name}\n"
                f"Claimed value: {value.value!r}\n"
                f"Evidence (with source URL): {value.evidence!r}\n\n"
                f"Is this a credible web finding?"
            ),
            response_model=_WebVerdict,
        )
        return bool(verdict and verdict.valid)
