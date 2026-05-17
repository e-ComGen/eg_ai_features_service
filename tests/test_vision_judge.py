"""Unit tests for VisionJudge.

All tests use mocked LLM — no real API calls.
"""
import pytest
from unittest.mock import AsyncMock

from app.services.enrichment.base import AttributeValue, ExtractionContext, Source
from app.services.enrichment.judges.vision_judge import VisionJudge, _VisionVerdict
from app.services.providers.structured_adapter import StructuredLlmManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context() -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name="Test Product",
        category_id=10,
    )


def _make_value(evidence: str | None = "clearly red surface visible on photo") -> AttributeValue:
    return AttributeValue(
        attribute_id=101,
        value="red",
        confidence=0.9,
        source=Source.VISION,
        evidence=evidence,
    )


def _make_judge(verdict: bool | None = True) -> tuple[VisionJudge, AsyncMock]:
    mock_llm = AsyncMock(spec=StructuredLlmManager)
    if verdict is None:
        # Simulate LLM failure
        mock_llm.structured_request = AsyncMock(return_value=(None, 0))
    else:
        mock_llm.structured_request = AsyncMock(
            return_value=(_VisionVerdict(valid=verdict, reason="test reason"), 50)
        )
    judge = VisionJudge(llm_manager=mock_llm)
    return judge, mock_llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_rejects_without_evidence():
    """value.evidence=None → returns False without making any LLM call."""
    judge, mock_llm = _make_judge()
    ctx = _make_context()
    value = _make_value(evidence=None)

    result = await judge.validate(value, ctx)

    assert result is False
    mock_llm.structured_request.assert_not_called()


@pytest.mark.asyncio
async def test_validate_calls_llm_with_evidence():
    """When evidence is present, LLM is called once."""
    judge, mock_llm = _make_judge(verdict=True)
    ctx = _make_context()
    value = _make_value(evidence="bright red surface clearly visible")

    await judge.validate(value, ctx)

    mock_llm.structured_request.assert_called_once()
    call_kwargs = mock_llm.structured_request.call_args.kwargs
    # Evidence and value must appear in the user_text passed to the LLM
    assert "bright red surface clearly visible" in call_kwargs["user_text"]
    assert call_kwargs["response_model"] is _VisionVerdict


@pytest.mark.asyncio
async def test_validate_returns_llm_verdict():
    """Returns True when LLM verdict is valid=True, False when valid=False."""
    ctx = _make_context()
    value = _make_value()

    judge_true, _ = _make_judge(verdict=True)
    assert await judge_true.validate(value, ctx) is True

    judge_false, _ = _make_judge(verdict=False)
    assert await judge_false.validate(value, ctx) is False


@pytest.mark.asyncio
async def test_validate_returns_false_on_llm_failure():
    """Returns False when LLM returns None (parse error / API failure)."""
    judge, _ = _make_judge(verdict=None)
    ctx = _make_context()
    value = _make_value()

    result = await judge.validate(value, ctx)

    assert result is False
