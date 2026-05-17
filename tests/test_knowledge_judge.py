"""Tests for KnowledgeJudge (Step C).

All tests use AsyncMock to avoid real LLM calls.
"""
import pytest
from unittest.mock import AsyncMock

from app.services.enrichment.base import (
    ExtractionContext, AttributeValue, Source,
)
from app.services.enrichment.judges.knowledge_judge import KnowledgeJudge, _KnowledgeVerdict
from app.services.providers.structured_adapter import StructuredLlmManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**kwargs) -> ExtractionContext:
    defaults = dict(product_id=1, product_name="Nike Air Max 90", category_id=5, brand="Nike")
    defaults.update(kwargs)
    return ExtractionContext(**defaults)


def _make_value(attr_id: int = 1, value: str = "white", confidence: float = 0.95) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=confidence,
        source=Source.LLM_KNOWLEDGE,
        evidence="classic colorway",
    )


def _make_llm_mock(verdict=None, tokens=50) -> AsyncMock:
    mock = AsyncMock(spec=StructuredLlmManager)
    mock.structured_request.return_value = (verdict, tokens)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_calls_llm_with_product_context():
    """validate() must call structured_request with product name and value in user_text."""
    verdict = _KnowledgeVerdict(valid=True, reason="well known fact")
    mock_llm = _make_llm_mock(verdict=verdict)
    judge = KnowledgeJudge(llm_manager=mock_llm)
    ctx = _make_context(product_name="Nike Air Max 90", brand="Nike")
    val = _make_value(value="white")

    await judge.validate(val, ctx)

    mock_llm.structured_request.assert_called_once()
    call_kwargs = mock_llm.structured_request.call_args
    user_text = call_kwargs[1].get("user_text") or call_kwargs[0][1]
    assert "Nike Air Max 90" in user_text
    assert "white" in user_text


@pytest.mark.asyncio
async def test_validate_returns_llm_verdict():
    """validate() must return True when LLM says valid=True, False when valid=False."""
    judge_true = KnowledgeJudge(llm_manager=_make_llm_mock(
        verdict=_KnowledgeVerdict(valid=True, reason="confirmed")
    ))
    judge_false = KnowledgeJudge(llm_manager=_make_llm_mock(
        verdict=_KnowledgeVerdict(valid=False, reason="unverified")
    ))
    ctx = _make_context()
    val = _make_value()

    assert await judge_true.validate(val, ctx) is True
    assert await judge_false.validate(val, ctx) is False


@pytest.mark.asyncio
async def test_validate_returns_false_on_llm_failure():
    """validate() must return False when LLM returns None (network/parse error)."""
    judge = KnowledgeJudge(llm_manager=_make_llm_mock(verdict=None))
    ctx = _make_context()
    val = _make_value()

    result = await judge.validate(val, ctx)

    assert result is False
