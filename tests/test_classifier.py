"""Tests for LlmClassifier (Step F).

All tests use AsyncMock to avoid real LLM calls.
"""
import pytest
from unittest.mock import AsyncMock

from app.services.enrichment.base import (
    ExtractionContext, TargetAttribute, Source,
)
from app.services.enrichment.intelligence.classifier import (
    LlmClassifier, ClassifierDecision, _ClassifierResponse,
)
from app.services.providers.structured_adapter import StructuredLlmManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**kwargs) -> ExtractionContext:
    defaults = dict(product_id=1, product_name="iPhone 15 Pro", category_id=10)
    defaults.update(kwargs)
    return ExtractionContext(**defaults)


def _make_target(attr_id: int = 1, name: str = "Color", type_: str = "text", **kwargs) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=type_, **kwargs)


def _make_llm_mock(parsed=None, tokens=100) -> AsyncMock:
    mock = AsyncMock(spec=StructuredLlmManager)
    mock.structured_request.return_value = (parsed, tokens)
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_classify_empty_attrs_returns_empty():
    """Returns {} immediately when unfilled_attributes is empty."""
    classifier = LlmClassifier(llm_manager=_make_llm_mock())
    ctx = _make_context()
    result = await classifier.classify(ctx, [])
    assert result == {}


@pytest.mark.asyncio
async def test_classify_calls_llm_with_context_and_attrs():
    """LLM structured_request is called with product name and attr info in user_text."""
    decision = ClassifierDecision(
        attribute_id=1,
        suggested_sources=[Source.LLM_KNOWLEDGE],
        reasoning="brand product",
    )
    parsed = _ClassifierResponse(decisions=[decision])
    mock_llm = _make_llm_mock(parsed=parsed)
    classifier = LlmClassifier(llm_manager=mock_llm)

    ctx = _make_context(product_name="Samsung Galaxy S24", brand="Samsung")
    attr = _make_target(attr_id=1, name="Storage")
    await classifier.classify(ctx, [attr])

    mock_llm.structured_request.assert_called_once()
    call_args = mock_llm.structured_request.call_args
    user_text = call_args.kwargs.get("user_text") or call_args[0][1]
    assert "Samsung Galaxy S24" in user_text
    assert "Storage" in user_text


@pytest.mark.asyncio
async def test_classify_returns_decisions_dict_keyed_by_id():
    """classify() returns a dict keyed by attribute_id."""
    decisions = [
        ClassifierDecision(attribute_id=10, suggested_sources=[Source.VISION], reasoning="visual"),
        ClassifierDecision(attribute_id=20, suggested_sources=[Source.WEB_SEARCH, Source.LLM_KNOWLEDGE], reasoning="specs"),
    ]
    parsed = _ClassifierResponse(decisions=decisions)
    mock_llm = _make_llm_mock(parsed=parsed)
    classifier = LlmClassifier(llm_manager=mock_llm)

    ctx = _make_context()
    attrs = [_make_target(10, "Color"), _make_target(20, "Weight")]
    result = await classifier.classify(ctx, attrs)

    assert set(result.keys()) == {10, 20}
    assert result[10] == [Source.VISION]
    assert result[20] == [Source.WEB_SEARCH, Source.LLM_KNOWLEDGE]


@pytest.mark.asyncio
async def test_classify_includes_image_flag_in_prompt():
    """When context.image_urls is non-empty, 'Has photos: True' appears in user_text."""
    decision = ClassifierDecision(
        attribute_id=1,
        suggested_sources=[Source.VISION],
        reasoning="has images",
    )
    parsed = _ClassifierResponse(decisions=[decision])
    mock_llm = _make_llm_mock(parsed=parsed)
    classifier = LlmClassifier(llm_manager=mock_llm)

    ctx = _make_context(image_urls=["https://example.com/photo.jpg"])
    attr = _make_target(attr_id=1, name="Color")
    await classifier.classify(ctx, [attr])

    call_args = mock_llm.structured_request.call_args
    user_text = call_args.kwargs.get("user_text") or call_args[0][1]
    assert "Has photos: True" in user_text


@pytest.mark.asyncio
async def test_classify_fallback_on_llm_failure():
    """When LLM returns None, fallback returns {attr_id: [LLM_KNOWLEDGE]} for all attrs."""
    mock_llm = _make_llm_mock(parsed=None)
    classifier = LlmClassifier(llm_manager=mock_llm)

    ctx = _make_context()
    attrs = [_make_target(1, "Color"), _make_target(2, "Weight")]
    result = await classifier.classify(ctx, attrs)

    assert result == {1: [Source.LLM_KNOWLEDGE], 2: [Source.LLM_KNOWLEDGE]}


@pytest.mark.asyncio
async def test_classify_increments_llm_calls():
    """context.llm_calls_so_far is incremented by 1 after a successful LLM call."""
    decision = ClassifierDecision(
        attribute_id=1,
        suggested_sources=[Source.LLM_KNOWLEDGE],
        reasoning="ok",
    )
    parsed = _ClassifierResponse(decisions=[decision])
    mock_llm = _make_llm_mock(parsed=parsed)
    classifier = LlmClassifier(llm_manager=mock_llm)

    ctx = _make_context()
    assert ctx.llm_calls_so_far == 0
    await classifier.classify(ctx, [_make_target(1)])
    assert ctx.llm_calls_so_far == 1


def test_classifier_decision_model_validates_min_max_sources():
    """ClassifierDecision allows 0 sources (give-up signal) and caps at max_length=3."""
    # Valid: 1 source
    d = ClassifierDecision(
        attribute_id=1,
        suggested_sources=[Source.VISION],
        reasoning="ok",
    )
    assert len(d.suggested_sources) == 1

    # Valid: 3 sources (max)
    d3 = ClassifierDecision(
        attribute_id=2,
        suggested_sources=[Source.VISION, Source.LLM_KNOWLEDGE, Source.WEB_SEARCH],
        reasoning="try all",
    )
    assert len(d3.suggested_sources) == 3

    # Valid: 0 sources — намеренный give-up сигнал (min_length=0).
    # Downstream auto-fallback (classifier.py) подменяет пустой список на LLM_KNOWLEDGE.
    d0 = ClassifierDecision(
        attribute_id=3,
        suggested_sources=[],
        reasoning="nothing",
    )
    assert d0.suggested_sources == []

    # Invalid: 4 sources (max_length=3 violated)
    with pytest.raises(Exception):
        ClassifierDecision(
            attribute_id=4,
            suggested_sources=[Source.VISION, Source.LLM_KNOWLEDGE, Source.WEB_SEARCH, Source.DESCRIPTION],
            reasoning="too many",
        )
