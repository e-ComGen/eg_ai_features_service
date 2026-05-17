"""Unit tests for WebSearchJudge.

All tests use mocked LLM — no real API calls.
"""

import pytest
from unittest.mock import AsyncMock

from app.services.enrichment.base import (
    AttributeValue, ExtractionContext, Source,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.enrichment.judges.websearch_judge import WebSearchJudge, _WebVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context() -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name="ACME Widget Pro",
        category_id=10,
        brand="ACME",
    )


def _make_value(evidence: str | None = "[https://acme.com] weight 500g") -> AttributeValue:
    return AttributeValue(
        attribute_id=42,
        value="500g",
        confidence=0.9,
        source=Source.WEB_SEARCH,
        evidence=evidence,
    )


def _make_judge(verdict: bool = True, llm_raises: bool = False) -> tuple[WebSearchJudge, AsyncMock]:
    mock_llm = AsyncMock(spec=StructuredLlmManager)
    if llm_raises:
        mock_llm.structured_request = AsyncMock(side_effect=Exception("LLM error"))
    else:
        mock_llm.structured_request = AsyncMock(
            return_value=(_WebVerdict(valid=verdict, reason="ok"), 50)
        )
    judge = WebSearchJudge(llm_manager=mock_llm)
    return judge, mock_llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# 1. validate returns False immediately when evidence is None/empty
@pytest.mark.asyncio
async def test_validate_rejects_without_evidence():
    judge, mock_llm = _make_judge()
    ctx = _make_context()
    value = _make_value(evidence=None)

    result = await judge.validate(value, ctx)

    assert result is False
    mock_llm.structured_request.assert_not_called()


# 2. validate calls the LLM when evidence is present
@pytest.mark.asyncio
async def test_validate_calls_llm():
    judge, mock_llm = _make_judge(verdict=True)
    ctx = _make_context()
    value = _make_value()

    await judge.validate(value, ctx)

    mock_llm.structured_request.assert_awaited_once()
    call_kwargs = mock_llm.structured_request.call_args.kwargs
    assert "evidence" in call_kwargs["user_text"].lower() or "Evidence" in call_kwargs["user_text"]


# 3. validate returns LLM verdict (True / False)
@pytest.mark.asyncio
async def test_validate_returns_llm_verdict():
    judge_true, _ = _make_judge(verdict=True)
    judge_false, _ = _make_judge(verdict=False)
    ctx = _make_context()
    value = _make_value()

    assert await judge_true.validate(value, ctx) is True
    assert await judge_false.validate(value, ctx) is False


# 4. validate returns False when LLM returns (None, 0)
@pytest.mark.asyncio
async def test_validate_returns_false_on_llm_failure():
    mock_llm = AsyncMock(spec=StructuredLlmManager)
    mock_llm.structured_request = AsyncMock(return_value=(None, 0))
    judge = WebSearchJudge(llm_manager=mock_llm)
    ctx = _make_context()
    value = _make_value()

    result = await judge.validate(value, ctx)

    assert result is False
