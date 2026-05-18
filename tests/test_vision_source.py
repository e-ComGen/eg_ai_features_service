"""Unit tests for VisionSource.

All tests use mocked LLM — no real API calls.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.base import (
    AttributeValue, ExtractionContext, Source, TargetAttribute,
)
from app.services.enrichment.sources.vision_source import VisionSource, VISUAL_SEMANTIC_TYPES
from app.services.enrichment.judges.vision_judge import VisionJudge
from app.services.enrichment.vision_producer import VisionProducer
from app.services.providers.structured_adapter import StructuredLlmManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()


def _make_context(
    product_id: int = 1,
    product_name: str = "Test Product",
    image_urls=_SENTINEL,
) -> ExtractionContext:
    if image_urls is _SENTINEL:
        image_urls = ["https://example.com/img1.jpg"]
    return ExtractionContext(
        product_id=product_id,
        product_name=product_name,
        category_id=10,
        image_urls=image_urls,
    )


def _make_target(
    attr_id: int = 101,
    name: str = "Color",
    semantic_type: str | None = "color",
) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type="enum", semantic_type=semantic_type)


def _make_vision_source(vision_text: str = "Red cylindrical product.", extracted=None):
    """Return a VisionSource with fully mocked dependencies."""
    mock_vision = AsyncMock(spec=VisionProducer)
    mock_vision.produce_description = AsyncMock(return_value=vision_text)

    mock_extractor = AsyncMock(spec=StructuredLlmManager)

    if extracted is not None:
        from app.services.enrichment.sources.vision_source import _VisionExtractionResponse, _VisionExtractedAttr
        resp = _VisionExtractionResponse(extracted=extracted)
        mock_extractor.structured_request = AsyncMock(return_value=(resp, 100))
    else:
        from app.services.enrichment.sources.vision_source import _VisionExtractionResponse
        resp = _VisionExtractionResponse(extracted=[])
        mock_extractor.structured_request = AsyncMock(return_value=(resp, 100))

    source = VisionSource(
        vision_producer=mock_vision,
        extraction_manager=mock_extractor,
    )
    return source, mock_vision, mock_extractor


# ---------------------------------------------------------------------------
# is_applicable tests
# ---------------------------------------------------------------------------

def test_is_applicable_with_image_and_visual_type():
    """Returns True when context has image_urls and target has a visual semantic_type."""
    source, _, _ = _make_vision_source()
    ctx = _make_context(image_urls=["https://example.com/img.jpg"])
    target = _make_target(semantic_type="color")
    assert source.is_applicable(ctx, target) is True


def test_is_applicable_no_images():
    """Returns False when context has no image_urls."""
    source, _, _ = _make_vision_source()
    ctx = _make_context(image_urls=[])
    target = _make_target(semantic_type="color")
    assert source.is_applicable(ctx, target) is False


def test_is_applicable_non_visual_semantic_type():
    """Returns False when target semantic_type is not in VISUAL_SEMANTIC_TYPES (e.g. weight)."""
    source, _, _ = _make_vision_source()
    ctx = _make_context(image_urls=["https://example.com/img.jpg"])
    target = _make_target(semantic_type="weight")
    assert source.is_applicable(ctx, target) is False


# ---------------------------------------------------------------------------
# extract tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_returns_empty_no_images():
    """Returns empty list when context.image_urls is empty."""
    source, mock_vision, mock_extractor = _make_vision_source()
    ctx = _make_context(image_urls=[])
    targets = [_make_target()]
    result = await source.extract(ctx, targets)
    assert result == []
    mock_vision.produce_description.assert_not_called()
    mock_extractor.structured_request.assert_not_called()


@pytest.mark.asyncio
async def test_extract_calls_vision_then_extraction():
    """Both vision and extraction LLM calls are made in order."""
    from app.services.enrichment.sources.vision_source import _VisionExtractedAttr
    extracted = [_VisionExtractedAttr(attribute_id=101, value="red", confidence=0.9, evidence="red surface")]
    source, mock_vision, mock_extractor = _make_vision_source(
        vision_text="Red surface, cylindrical.", extracted=extracted
    )
    ctx = _make_context()
    targets = [_make_target()]

    result = await source.extract(ctx, targets)

    mock_vision.produce_description.assert_called_once()
    mock_extractor.structured_request.assert_called_once()
    assert len(result) == 1


@pytest.mark.asyncio
async def test_extract_caches_vision_per_product():
    """Vision call is made only once when extract is called twice for the same product_id."""
    source, mock_vision, mock_extractor = _make_vision_source(vision_text="Blue surface.")
    ctx = _make_context(product_id=42)
    targets = [_make_target()]

    await source.extract(ctx, targets)
    await source.extract(ctx, targets)

    # Vision producer called only once despite two extract calls
    assert mock_vision.produce_description.call_count == 1
    # Extractor called twice (once per extract call)
    assert mock_extractor.structured_request.call_count == 2


@pytest.mark.asyncio
async def test_extract_returns_attribute_values_with_source_vision():
    """Returned AttributeValue objects have source=Source.VISION."""
    from app.services.enrichment.sources.vision_source import _VisionExtractedAttr
    extracted = [
        _VisionExtractedAttr(attribute_id=101, value="red", confidence=0.88, evidence="red surface"),
        _VisionExtractedAttr(attribute_id=102, value="round", confidence=0.75, evidence="circular shape"),
    ]
    source, _, _ = _make_vision_source(vision_text="Red round product.", extracted=extracted)
    ctx = _make_context()
    targets = [_make_target(101), _make_target(102, "Shape", "shape")]

    result = await source.extract(ctx, targets)

    assert len(result) == 2
    for av in result:
        assert av.source == Source.VISION


@pytest.mark.asyncio
async def test_extract_increments_llm_calls_by_two_on_first_call():
    """First extract: llm_calls_so_far increments by 2 (vision + extraction)."""
    from app.services.enrichment.sources.vision_source import _VisionExtractedAttr
    extracted = [_VisionExtractedAttr(attribute_id=101, value="red", confidence=0.9, evidence="red")]
    source, _, _ = _make_vision_source(vision_text="Red product.", extracted=extracted)
    ctx = _make_context()
    assert ctx.llm_calls_so_far == 0

    await source.extract(ctx, [_make_target()])

    assert ctx.llm_calls_so_far == 2


def test_get_judge_returns_vision_judge():
    """get_judge() returns an instance of VisionJudge."""
    source, _, _ = _make_vision_source()
    judge = source.get_judge()
    assert isinstance(judge, VisionJudge)


@pytest.mark.asyncio
async def test_extract_copies_semantic_type_from_target():
    """value.semantic_type must match the target's semantic_type."""
    from app.services.enrichment.sources.vision_source import _VisionExtractedAttr
    extracted = [_VisionExtractedAttr(attribute_id=101, value="red", confidence=0.88, evidence="red surface")]
    source, _, _ = _make_vision_source(vision_text="Red surface.", extracted=extracted)
    ctx = _make_context()
    target = _make_target(attr_id=101, name="Color", semantic_type="color")

    result = await source.extract(ctx, [target])

    assert len(result) == 1
    assert result[0].semantic_type == "color"


@pytest.mark.asyncio
async def test_extract_semantic_type_none_when_target_has_no_semantic_type():
    """value.semantic_type is None when target has no semantic_type."""
    from app.services.enrichment.sources.vision_source import _VisionExtractedAttr
    extracted = [_VisionExtractedAttr(attribute_id=101, value="red", confidence=0.88, evidence="red surface")]
    source, _, _ = _make_vision_source(vision_text="Red surface.", extracted=extracted)
    ctx = _make_context()
    target = _make_target(attr_id=101, name="Color", semantic_type=None)

    result = await source.extract(ctx, [target])

    assert len(result) == 1
    assert result[0].semantic_type is None
