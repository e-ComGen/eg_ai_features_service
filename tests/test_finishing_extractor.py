"""Tests for FinishingExtractor — focused re-extraction for empty required attributes.

All LLM calls are mocked; no live API calls made.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.enrichment.base import (
    Source,
    AttributeValue,
    TargetAttribute,
    ExtractionContext,
    AttributeSource,
    LlmJudge,
)
from app.services.enrichment.finishing import FinishingExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(
    id: int,
    name: str = "Тип",
    is_required: bool = False,
) -> TargetAttribute:
    return TargetAttribute(id=id, name=name, type="text", is_required=is_required)


def _make_ctx() -> ExtractionContext:
    return ExtractionContext(product_id=1, product_name="Test Product", category_id=1)


def _make_value(attr_id: int, source: Source = Source.DESCRIPTION) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value="Черный",
        confidence=0.95,
        source=source,
    )


def _mock_source(extract_return: list | None = None) -> MagicMock:
    """Return a mock AttributeSource that returns given values from extract()."""
    s = MagicMock(spec=AttributeSource)
    s.source_type = Source.DESCRIPTION
    s.is_applicable = MagicMock(return_value=True)
    s.extract = AsyncMock(return_value=extract_return or [])
    s._llm = None  # no _llm so _FocusedSourceProxy calls extract() directly
    judge = MagicMock(spec=LlmJudge)
    judge.source = Source.DESCRIPTION
    judge.validate = AsyncMock(return_value=True)
    s.get_judge = MagicMock(return_value=judge)
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_skipped_when_all_required_filled():
    """Finishing pass returns empty list when every required target already has a value."""
    source = _mock_source([])
    extractor = FinishingExtractor(sources=[source])

    targets = [_make_target(id=1, is_required=True)]
    already_filled = [_make_value(attr_id=1)]

    result = await extractor.extract_missing(_make_ctx(), targets, already_filled)

    assert result == []
    source.extract.assert_not_called()


@pytest.mark.asyncio
async def test_runs_on_optional_missing_targets():
    """Finishing pass also attempts optional (is_required=False) missing targets.

    By design (see finishing.py docstring) required attrs get priority under the
    per-pass limit, but optional missing targets are still re-extracted.
    """
    recovered = _make_value(attr_id=1)
    source = _mock_source([recovered])
    extractor = FinishingExtractor(sources=[source])

    targets = [_make_target(id=1, is_required=False)]
    already_filled: list[AttributeValue] = []

    result = await extractor.extract_missing(_make_ctx(), targets, already_filled)

    assert len(result) == 1
    assert result[0].attribute_id == 1
    source.extract.assert_called_once()
    called_targets = source.extract.call_args[0][1]
    assert [t.id for t in called_targets] == [1]


@pytest.mark.asyncio
async def test_runs_on_both_required_and_optional_required_first():
    """When both required and optional attributes are missing, both are passed to sources,
    with required ordered first (priority under the per-pass limit)."""
    recovered = _make_value(attr_id=10)
    source = _mock_source([recovered])
    extractor = FinishingExtractor(sources=[source])

    required_t = _make_target(id=10, is_required=True)
    optional_t = _make_target(id=20, is_required=False)
    targets = [required_t, optional_t]
    already_filled: list[AttributeValue] = []

    result = await extractor.extract_missing(_make_ctx(), targets, already_filled)

    assert len(result) == 1
    assert result[0].attribute_id == 10
    # Source was called with both targets, required first
    called_targets = source.extract.call_args[0][1]
    assert [t.id for t in called_targets] == [10, 20]


@pytest.mark.asyncio
async def test_adds_attribute_value_when_source_recovers():
    """FinishingExtractor returns new AttributeValues from source that recovers a value."""
    recovered_value = _make_value(attr_id=5)
    source = _mock_source([recovered_value])
    extractor = FinishingExtractor(sources=[source])

    targets = [_make_target(id=5, is_required=True)]
    already_filled: list[AttributeValue] = []

    result = await extractor.extract_missing(_make_ctx(), targets, already_filled)

    assert len(result) == 1
    assert result[0].attribute_id == 5
    assert result[0].value == "Черный"


@pytest.mark.asyncio
async def test_empty_result_when_source_recovers_nothing():
    """Graceful: source called but returns empty list — no crash, empty result."""
    source = _mock_source([])
    extractor = FinishingExtractor(sources=[source])

    targets = [_make_target(id=7, is_required=True)]
    already_filled: list[AttributeValue] = []

    result = await extractor.extract_missing(_make_ctx(), targets, already_filled)

    assert result == []
    source.extract.assert_called_once()


@pytest.mark.asyncio
async def test_source_exception_is_handled_gracefully():
    """If a source raises during focused pass, exception is caught and empty list returned."""
    source = _mock_source()
    source.extract = AsyncMock(side_effect=RuntimeError("LLM down"))
    extractor = FinishingExtractor(sources=[source])

    targets = [_make_target(id=3, is_required=True)]
    already_filled: list[AttributeValue] = []

    result = await extractor.extract_missing(_make_ctx(), targets, already_filled)

    assert result == []
