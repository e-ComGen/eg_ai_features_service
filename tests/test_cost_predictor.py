"""Tests for CostPredictor (Step F).

All tests use AsyncMock to avoid real LLM calls.
"""
import pytest
from unittest.mock import AsyncMock

from app.services.enrichment.base import (
    ExtractionContext, TargetAttribute,
)
from app.services.enrichment.intelligence.cost_predictor import (
    CostPredictor, _WorthVerdict,
)
from app.services.providers.structured_adapter import StructuredLlmManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**kwargs) -> ExtractionContext:
    defaults = dict(product_id=1, product_name="iPhone 15 Pro", category_id=10)
    defaults.update(kwargs)
    return ExtractionContext(**defaults)


def _make_target(attr_id: int = 1, name: str = "Weight", type_: str = "numeric") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=type_)


def _make_llm_mock(parsed=None, tokens=50) -> AsyncMock:
    mock = AsyncMock(spec=StructuredLlmManager)
    mock.structured_request.return_value = (parsed, tokens)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_is_web_search_worth_returns_false_if_no_targets():
    """Returns False immediately when targets list is empty."""
    predictor = CostPredictor(llm_manager=_make_llm_mock())
    ctx = _make_context()
    result = await predictor.is_web_search_worth(ctx, [])
    assert result is False


@pytest.mark.asyncio
async def test_is_web_search_worth_returns_false_if_over_budget():
    """Returns False without calling LLM when cost_so_far > 70% of max_cost."""
    mock_llm = _make_llm_mock()
    predictor = CostPredictor(llm_manager=mock_llm)
    # cost_so_far = 0.08, max_cost = 0.10 → 0.08 > 0.07 → over 70%
    ctx = _make_context(cost_so_far_usd=0.08, max_cost_usd=0.10)
    result = await predictor.is_web_search_worth(ctx, [_make_target()])
    assert result is False
    mock_llm.structured_request.assert_not_called()


@pytest.mark.asyncio
async def test_is_web_search_worth_calls_llm_with_product_context():
    """LLM structured_request is called with product name and attributes in user_text."""
    verdict = _WorthVerdict(worth_it=True, confidence=0.9, reason="well-known brand")
    mock_llm = _make_llm_mock(parsed=verdict)
    predictor = CostPredictor(llm_manager=mock_llm)

    ctx = _make_context(product_name="Nike Air Max 90", brand="Nike")
    targets = [_make_target(1, "Size"), _make_target(2, "Weight")]
    await predictor.is_web_search_worth(ctx, targets)

    mock_llm.structured_request.assert_called_once()
    call_args = mock_llm.structured_request.call_args
    user_text = call_args.kwargs.get("user_text") or call_args[0][1]
    assert "Nike Air Max 90" in user_text
    assert "Size" in user_text


@pytest.mark.asyncio
async def test_is_web_search_worth_returns_llm_verdict():
    """Returns the LLM verdict's worth_it value."""
    for worth in (True, False):
        verdict = _WorthVerdict(worth_it=worth, confidence=0.8, reason="test")
        mock_llm = _make_llm_mock(parsed=verdict)
        predictor = CostPredictor(llm_manager=mock_llm)
        ctx = _make_context()
        result = await predictor.is_web_search_worth(ctx, [_make_target()])
        assert result is worth


@pytest.mark.asyncio
async def test_is_web_search_worth_default_true_on_llm_failure():
    """Returns True (don't block) when LLM returns None."""
    mock_llm = _make_llm_mock(parsed=None)
    predictor = CostPredictor(llm_manager=mock_llm)
    ctx = _make_context()
    result = await predictor.is_web_search_worth(ctx, [_make_target()])
    assert result is True


@pytest.mark.asyncio
async def test_is_web_search_worth_increments_llm_calls():
    """context.llm_calls_so_far is incremented by 1 after a successful LLM call."""
    verdict = _WorthVerdict(worth_it=True, confidence=0.95, reason="known product")
    mock_llm = _make_llm_mock(parsed=verdict)
    predictor = CostPredictor(llm_manager=mock_llm)

    ctx = _make_context()
    assert ctx.llm_calls_so_far == 0
    await predictor.is_web_search_worth(ctx, [_make_target()])
    assert ctx.llm_calls_so_far == 1
