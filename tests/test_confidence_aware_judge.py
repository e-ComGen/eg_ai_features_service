"""Tests for ConfidenceAwareJudgeWrapper (Step H).

8 tests covering: skip on high confidence, call judge on low confidence,
validated/rejected outcomes, llm_calls tracking, stats, source delegation.
"""
import pytest
from unittest.mock import AsyncMock

from app.services.enrichment.base import Source, AttributeValue, ExtractionContext, LlmJudge
from app.services.enrichment.confidence_aware_judge import ConfidenceAwareJudgeWrapper


def _make_value(confidence: float, source=Source.DESCRIPTION):
    return AttributeValue(
        attribute_id=1, value="red", confidence=confidence, source=source
    )


def _make_judge(source=Source.DESCRIPTION):
    j = AsyncMock(spec=LlmJudge)
    j.source = source
    return j


def _make_ctx():
    return ExtractionContext(product_id=1, product_name="x", category_id=1)


# Source.DESCRIPTION threshold is 0.95
HIGH_CONFIDENCE = 0.96
LOW_CONFIDENCE = 0.50


@pytest.mark.asyncio
async def test_skip_judge_for_high_confidence_value():
    judge = _make_judge()
    wrapper = ConfidenceAwareJudgeWrapper(judge)
    value = _make_value(HIGH_CONFIDENCE)
    ctx = _make_ctx()

    result = await wrapper.maybe_validate(value, ctx)

    assert result is value
    judge.validate.assert_not_called()
    assert wrapper.stats["skipped"] == 1


@pytest.mark.asyncio
async def test_call_judge_for_low_confidence_value():
    judge = _make_judge()
    judge.validate.return_value = True
    wrapper = ConfidenceAwareJudgeWrapper(judge)
    value = _make_value(LOW_CONFIDENCE)
    ctx = _make_ctx()

    await wrapper.maybe_validate(value, ctx)

    judge.validate.assert_called_once_with(value, ctx)


@pytest.mark.asyncio
async def test_returns_validated_value_when_judge_accepts():
    judge = _make_judge()
    judge.validate.return_value = True
    wrapper = ConfidenceAwareJudgeWrapper(judge)
    value = _make_value(LOW_CONFIDENCE)
    ctx = _make_ctx()

    result = await wrapper.maybe_validate(value, ctx)

    assert result is not None
    assert result.judge_validated is True
    assert wrapper.stats["validated"] == 1


@pytest.mark.asyncio
async def test_returns_none_when_judge_rejects():
    judge = _make_judge()
    judge.validate.return_value = False
    wrapper = ConfidenceAwareJudgeWrapper(judge)
    value = _make_value(LOW_CONFIDENCE)
    ctx = _make_ctx()

    result = await wrapper.maybe_validate(value, ctx)

    assert result is None
    assert wrapper.stats["invalidated"] == 1


@pytest.mark.asyncio
async def test_increments_llm_calls_in_context_when_judge_called():
    judge = _make_judge()
    judge.validate.return_value = True
    wrapper = ConfidenceAwareJudgeWrapper(judge)
    value = _make_value(LOW_CONFIDENCE)
    ctx = _make_ctx()
    initial_calls = ctx.llm_calls_so_far

    await wrapper.maybe_validate(value, ctx)

    assert ctx.llm_calls_so_far == initial_calls + 1


@pytest.mark.asyncio
async def test_does_not_increment_llm_calls_when_skipped():
    judge = _make_judge()
    wrapper = ConfidenceAwareJudgeWrapper(judge)
    value = _make_value(HIGH_CONFIDENCE)
    ctx = _make_ctx()
    initial_calls = ctx.llm_calls_so_far

    await wrapper.maybe_validate(value, ctx)

    assert ctx.llm_calls_so_far == initial_calls


@pytest.mark.asyncio
async def test_stats_tracks_all_outcomes():
    judge = _make_judge()
    # First call: accept, second call: reject
    judge.validate.side_effect = [True, False]
    wrapper = ConfidenceAwareJudgeWrapper(judge)
    ctx = _make_ctx()

    # Skip (high confidence)
    await wrapper.maybe_validate(_make_value(HIGH_CONFIDENCE), ctx)
    # Validated (low confidence, judge accepts)
    await wrapper.maybe_validate(_make_value(LOW_CONFIDENCE), ctx)
    # Invalidated (low confidence, judge rejects)
    await wrapper.maybe_validate(_make_value(LOW_CONFIDENCE), ctx)

    assert wrapper.stats == {"skipped": 1, "validated": 1, "invalidated": 1}


def test_source_property_delegates_to_judge():
    judge = _make_judge(source=Source.VISION)
    wrapper = ConfidenceAwareJudgeWrapper(judge)

    assert wrapper.source == Source.VISION
