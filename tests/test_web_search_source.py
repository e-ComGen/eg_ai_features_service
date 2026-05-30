"""Unit tests for WebSearchSource.

All tests use mocked collaborators — no real API calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.base import (
    ExtractionContext, TargetAttribute, Source,
)
from app.services.enrichment.websearch_producer import WebSearchProducer
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.enrichment.sources.web_search_source import (
    WebSearchSource, _WebExtractionResponse, _WebExtractedAttr,
)
from app.services.enrichment.judges.websearch_judge import WebSearchJudge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(product_id: int = 1, product_name: str = "ACME Widget Pro") -> ExtractionContext:
    return ExtractionContext(
        product_id=product_id,
        product_name=product_name,
        category_id=10,
        brand="ACME",
        ean="1234567890123",
    )


def _make_target(attr_id: int = 42, name: str = "Weight", type_: str = "text",
                 semantic_type: str | None = None) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=type_, semantic_type=semantic_type)


def _make_extraction_response(
    attr_id: int = 42,
    value: str = "500g",
    confidence: float = 0.9,
    source_url: str | None = None,
    evidence: str | None = "500g product weight",
) -> _WebExtractionResponse:
    return _WebExtractionResponse(
        extracted=[
            _WebExtractedAttr(
                attribute_id=attr_id,
                value=value,
                confidence=confidence,
                source_url=source_url,
                evidence=evidence,
            )
        ]
    )


def _make_source(summary: str | None = "Weight is 500g.", extraction_response=None):
    """Return a WebSearchSource with fully mocked collaborators."""
    mock_producer = AsyncMock(spec=WebSearchProducer)
    mock_producer.produce_summary = AsyncMock(return_value=summary)

    mock_extractor = AsyncMock(spec=StructuredLlmManager)
    resp = extraction_response or _make_extraction_response()
    mock_extractor.structured_request = AsyncMock(return_value=(resp, 100))

    source = WebSearchSource(
        websearch_producer=mock_producer,
        extraction_manager=mock_extractor,
    )
    return source, mock_producer, mock_extractor


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# 1. is_applicable — True with a proper product_name
def test_is_applicable_with_product_name():
    source, _, _ = _make_source()
    ctx = _make_context(product_name="ACME Widget Pro 5000")
    target = _make_target()
    assert source.is_applicable(ctx, target) is True


# 2. is_applicable — False when product_name is too short
def test_is_applicable_short_name():
    source, _, _ = _make_source()
    ctx = _make_context(product_name="ABC")  # 3 chars < 5
    target = _make_target()
    assert source.is_applicable(ctx, target) is False


# 3. extract returns [] when targets list is empty
@pytest.mark.asyncio
async def test_extract_returns_empty_no_targets():
    source, mock_producer, _ = _make_source()
    ctx = _make_context()
    result = await source.extract(ctx, targets=[])
    assert result == []
    mock_producer.produce_summary.assert_not_called()


# 4. extract calls search then extraction LLM
@pytest.mark.asyncio
async def test_extract_calls_search_then_extraction():
    source, mock_producer, mock_extractor = _make_source(summary="Weight 500g.")
    ctx = _make_context()
    targets = [_make_target()]

    result = await source.extract(ctx, targets)

    mock_producer.produce_summary.assert_awaited_once_with(
        product_name=ctx.product_name,
        brand=ctx.brand,
        ean=ctx.ean,
        mpn=ctx.mpn,
    )
    mock_extractor.structured_request.assert_awaited_once()
    assert len(result) == 1


# 5. extract caches summary per product_id — two extracts → search called once
@pytest.mark.asyncio
async def test_extract_caches_summary_per_product():
    source, mock_producer, mock_extractor = _make_source(summary="Weight 500g.")
    ctx = _make_context(product_id=7)
    targets = [_make_target()]

    # First call
    await source.extract(ctx, targets)
    # Second call — same product_id
    await source.extract(ctx, targets)

    assert mock_producer.produce_summary.await_count == 1
    assert mock_extractor.structured_request.await_count == 2


# 6. extract returns AttributeValues with source=WEB_SEARCH
@pytest.mark.asyncio
async def test_extract_returns_attribute_values_with_source_web_search():
    source, _, _ = _make_source(summary="Weight is 500g.")
    ctx = _make_context()
    targets = [_make_target(attr_id=42)]

    result = await source.extract(ctx, targets)

    assert len(result) == 1
    av = result[0]
    assert av.attribute_id == 42
    assert av.source == Source.WEB_SEARCH
    assert av.value == "500g"
    assert av.confidence == pytest.approx(0.9)


# 7. extract includes [url] prefix in evidence when source_url returned by LLM
@pytest.mark.asyncio
async def test_extract_includes_source_url_in_evidence():
    resp = _make_extraction_response(
        source_url="https://manufacturer.com/specs",
        evidence="weight 500g per spec sheet",
    )
    source, _, _ = _make_source(summary="Weight 500g.", extraction_response=resp)
    ctx = _make_context()
    targets = [_make_target()]

    result = await source.extract(ctx, targets)

    assert len(result) == 1
    assert result[0].evidence.startswith("[https://manufacturer.com/specs]")


# 8. extract returns [] when producer summary is None
@pytest.mark.asyncio
async def test_extract_returns_empty_when_summary_none():
    source, mock_producer, mock_extractor = _make_source(summary=None)
    ctx = _make_context()
    targets = [_make_target()]

    result = await source.extract(ctx, targets)

    assert result == []
    mock_extractor.structured_request.assert_not_called()


# 9. get_judge returns a WebSearchJudge instance
def test_get_judge_returns_websearch_judge():
    source, _, _ = _make_source()
    judge = source.get_judge()
    assert isinstance(judge, WebSearchJudge)


# 10. extract copies semantic_type from target
@pytest.mark.asyncio
async def test_extract_copies_semantic_type_from_target():
    """value.semantic_type must match the target's semantic_type."""
    resp = _make_extraction_response(attr_id=42, value="0883412740906", confidence=0.9, evidence="EAN from specs")
    source, _, _ = _make_source(summary="EAN is 0883412740906.", extraction_response=resp)
    ctx = _make_context()
    target = _make_target(attr_id=42, name="EAN", semantic_type="ean")

    result = await source.extract(ctx, [target])

    assert len(result) == 1
    assert result[0].semantic_type == "ean"


@pytest.mark.asyncio
async def test_extract_semantic_type_none_when_target_has_no_semantic_type():
    """value.semantic_type is None when target has no semantic_type."""
    source, _, _ = _make_source(summary="Weight is 500g.")
    ctx = _make_context()
    target = _make_target(attr_id=42, name="Weight", semantic_type=None)

    result = await source.extract(ctx, [target])

    assert len(result) == 1
    assert result[0].semantic_type is None
