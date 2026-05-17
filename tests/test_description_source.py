"""Unit tests for DescriptionSource and DescriptionJudge."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.enrichment.base import (
    AttributeValue, ExtractionContext, Source, TargetAttribute,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.enrichment.sources.description_source import (
    DescriptionSource, _ExtractionResponse, _ExtractedAttr,
)
from app.services.enrichment.judges.description_judge import DescriptionJudge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(description: str | None = "This product has red color and weighs 500g.") -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name="Test Product",
        product_description=description,
        category_id=42,
        category_path=["Electronics", "Gadgets"],
    )


def _make_target(attr_id: int = 10, name: str = "Color", attr_type: str = "enum") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=attr_type, allowed_values=["red", "blue", "green"])


def _make_mock_llm(extracted_attrs: list[_ExtractedAttr] | None = None) -> AsyncMock:
    mock_llm = AsyncMock(spec=StructuredLlmManager)
    if extracted_attrs is None:
        extracted_attrs = [
            _ExtractedAttr(attribute_id=10, value="red", confidence=0.9, evidence="red color"),
        ]
    mock_llm.structured_request.return_value = (
        _ExtractionResponse(extracted=extracted_attrs),
        100,
    )
    return mock_llm


# ---------------------------------------------------------------------------
# DescriptionSource tests
# ---------------------------------------------------------------------------

def test_is_applicable_with_description():
    """True if description has ≥10 chars."""
    source = DescriptionSource(llm_manager=MagicMock())
    ctx = _make_context("This product has red color and weighs 500g.")
    target = _make_target()
    assert source.is_applicable(ctx, target) is True


def test_is_applicable_empty_description():
    """False for empty, None, or short description."""
    source = DescriptionSource(llm_manager=MagicMock())
    target = _make_target()

    assert source.is_applicable(_make_context(None), target) is False
    assert source.is_applicable(_make_context(""), target) is False
    assert source.is_applicable(_make_context("short"), target) is False  # 5 chars < 10


@pytest.mark.asyncio
async def test_extract_returns_empty_when_no_description():
    """Empty description → empty result list."""
    source = DescriptionSource(llm_manager=MagicMock())
    ctx = _make_context(None)
    targets = [_make_target()]
    result = await source.extract(ctx, targets)
    assert result == []


@pytest.mark.asyncio
async def test_extract_returns_empty_when_no_targets():
    """Empty targets list → empty result list without calling LLM."""
    mock_llm = _make_mock_llm()
    source = DescriptionSource(llm_manager=mock_llm)
    ctx = _make_context()
    result = await source.extract(ctx, [])
    assert result == []
    mock_llm.structured_request.assert_not_called()


@pytest.mark.asyncio
async def test_extract_calls_llm_with_correct_prompt():
    """LLM prompt must contain product_name, description, and target attribute."""
    mock_llm = _make_mock_llm()
    source = DescriptionSource(llm_manager=mock_llm)
    ctx = _make_context("This product has red color and weighs 500g.")
    targets = [_make_target(attr_id=10, name="Color")]

    await source.extract(ctx, targets)

    mock_llm.structured_request.assert_called_once()
    call_kwargs = mock_llm.structured_request.call_args

    # Check system_prompt
    system_prompt = call_kwargs.kwargs.get("system_prompt") or call_kwargs.args[0]
    assert "product characteristics" in system_prompt.lower() or "extract" in system_prompt.lower()

    # Check user_text contains product name, description, and target name
    user_text = call_kwargs.kwargs.get("user_text") or call_kwargs.args[1]
    assert "Test Product" in user_text
    assert "red color" in user_text
    assert "Color" in user_text


@pytest.mark.asyncio
async def test_extract_returns_attribute_values_with_source_description():
    """All returned AttributeValue objects must have source=DESCRIPTION."""
    mock_llm = _make_mock_llm([
        _ExtractedAttr(attribute_id=10, value="red", confidence=0.9, evidence="red color"),
        _ExtractedAttr(attribute_id=11, value=500, confidence=0.85, evidence="500g"),
    ])
    source = DescriptionSource(llm_manager=mock_llm)
    ctx = _make_context()
    targets = [_make_target(10, "Color"), _make_target(11, "Weight", "numeric")]

    result = await source.extract(ctx, targets)

    assert len(result) == 2
    for av in result:
        assert av.source == Source.DESCRIPTION
        assert isinstance(av, AttributeValue)


@pytest.mark.asyncio
async def test_extract_increments_llm_calls_in_context():
    """context.llm_calls_so_far must be incremented by 1 after successful extraction."""
    mock_llm = _make_mock_llm()
    source = DescriptionSource(llm_manager=mock_llm)
    ctx = _make_context()
    assert ctx.llm_calls_so_far == 0

    await source.extract(ctx, [_make_target()])

    assert ctx.llm_calls_so_far == 1


@pytest.mark.asyncio
async def test_extract_returns_empty_on_llm_failure():
    """If LLM manager returns (None, 0) → extract returns []."""
    mock_llm = AsyncMock(spec=StructuredLlmManager)
    mock_llm.structured_request.return_value = (None, 0)
    source = DescriptionSource(llm_manager=mock_llm)
    ctx = _make_context()

    result = await source.extract(ctx, [_make_target()])

    assert result == []


def test_get_judge_returns_description_judge():
    """get_judge() must return a DescriptionJudge instance."""
    source = DescriptionSource(llm_manager=MagicMock())
    judge = source.get_judge()
    assert isinstance(judge, DescriptionJudge)
