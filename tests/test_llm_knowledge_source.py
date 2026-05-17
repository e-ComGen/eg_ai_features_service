"""Tests for LlmKnowledgeSource (Step C).

All tests use AsyncMock to avoid real LLM calls.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.enrichment.base import (
    ExtractionContext, TargetAttribute, Source, AttributeValue,
)
from app.services.enrichment.sources.llm_knowledge_source import (
    LlmKnowledgeSource, _KnowledgeResponse, _KnowledgeAttr,
)
from app.services.enrichment.judges.knowledge_judge import KnowledgeJudge
from app.services.providers.structured_adapter import StructuredLlmManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**kwargs) -> ExtractionContext:
    defaults = dict(product_id=1, product_name="iPhone 15 Pro", category_id=10)
    defaults.update(kwargs)
    return ExtractionContext(**defaults)


def _make_target(attr_id: int = 1, name: str = "Color", type_: str = "text") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=type_)


def _make_llm_mock(parsed=None, tokens=100) -> AsyncMock:
    mock = AsyncMock(spec=StructuredLlmManager)
    mock.structured_request.return_value = (parsed, tokens)
    return mock


# ---------------------------------------------------------------------------
# is_applicable tests
# ---------------------------------------------------------------------------

def test_is_applicable_with_brand():
    """Returns True when context.brand is set."""
    src = LlmKnowledgeSource(llm_manager=_make_llm_mock())
    ctx = _make_context(product_name="iPhone 15 Pro", brand="Apple")
    target = _make_target()
    assert src.is_applicable(ctx, target) is True


def test_is_applicable_short_name():
    """Returns False when product_name is shorter than 5 characters."""
    src = LlmKnowledgeSource(llm_manager=_make_llm_mock())
    ctx = _make_context(product_name="AB", brand=None)
    target = _make_target()
    assert src.is_applicable(ctx, target) is False


def test_is_applicable_no_brand_but_long_name():
    """Returns True for a long product name even without explicit brand (MVP heuristic)."""
    src = LlmKnowledgeSource(llm_manager=_make_llm_mock())
    ctx = _make_context(product_name="Samsung Galaxy S24 Ultra", brand=None)
    target = _make_target()
    assert src.is_applicable(ctx, target) is True


def test_is_applicable_empty_name():
    """Returns False when product_name is empty string."""
    src = LlmKnowledgeSource(llm_manager=_make_llm_mock())
    ctx = _make_context(product_name="")
    target = _make_target()
    assert src.is_applicable(ctx, target) is False


# ---------------------------------------------------------------------------
# extract tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_returns_empty_no_targets():
    """Returns [] immediately when targets list is empty."""
    src = LlmKnowledgeSource(llm_manager=_make_llm_mock())
    ctx = _make_context()
    result = await src.extract(ctx, [])
    assert result == []


@pytest.mark.asyncio
async def test_extract_calls_llm_with_brand_in_prompt():
    """LLM user_text must contain the brand string."""
    mock_llm = _make_llm_mock(parsed=_KnowledgeResponse(known_attributes=[]))
    src = LlmKnowledgeSource(llm_manager=mock_llm)
    ctx = _make_context(brand="Apple")
    await src.extract(ctx, [_make_target()])

    mock_llm.structured_request.assert_called_once()
    call_args = mock_llm.structured_request.call_args
    # structured_request is called with keyword args only
    user_text = call_args.kwargs.get("user_text") or call_args[0][1]
    assert "Apple" in user_text


@pytest.mark.asyncio
async def test_extract_returns_attribute_values_with_source_knowledge():
    """Returned AttributeValues must have source=LLM_KNOWLEDGE."""
    attr = _KnowledgeAttr(attribute_id=5, value="blue", confidence=0.95, reasoning="well known color")
    parsed = _KnowledgeResponse(known_attributes=[attr])
    src = LlmKnowledgeSource(llm_manager=_make_llm_mock(parsed=parsed))
    ctx = _make_context()

    result = await src.extract(ctx, [_make_target(5)])

    assert len(result) == 1
    assert result[0].source == Source.LLM_KNOWLEDGE
    assert result[0].attribute_id == 5
    assert result[0].value == "blue"
    assert result[0].confidence == 0.95
    assert result[0].evidence == "well known color"


@pytest.mark.asyncio
async def test_extract_increments_llm_calls():
    """context.llm_calls_so_far must be incremented by 1 after a successful call."""
    parsed = _KnowledgeResponse(known_attributes=[])
    src = LlmKnowledgeSource(llm_manager=_make_llm_mock(parsed=parsed))
    ctx = _make_context()
    assert ctx.llm_calls_so_far == 0

    await src.extract(ctx, [_make_target()])

    assert ctx.llm_calls_so_far == 1


@pytest.mark.asyncio
async def test_extract_returns_empty_on_llm_failure():
    """Returns [] and does NOT increment llm_calls when LLM returns None."""
    src = LlmKnowledgeSource(llm_manager=_make_llm_mock(parsed=None))
    ctx = _make_context()

    result = await src.extract(ctx, [_make_target()])

    assert result == []
    assert ctx.llm_calls_so_far == 0


# ---------------------------------------------------------------------------
# get_judge / source_type
# ---------------------------------------------------------------------------

def test_get_judge_returns_knowledge_judge():
    """get_judge() must return a KnowledgeJudge instance."""
    src = LlmKnowledgeSource(llm_manager=_make_llm_mock())
    judge = src.get_judge()
    assert isinstance(judge, KnowledgeJudge)


def test_source_type_is_llm_knowledge():
    """source_type property must return Source.LLM_KNOWLEDGE."""
    src = LlmKnowledgeSource(llm_manager=_make_llm_mock())
    assert src.source_type == Source.LLM_KNOWLEDGE
