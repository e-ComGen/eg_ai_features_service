"""Unit tests for the enrichment branch orchestration in JobProcessor.

Tests verify that vision / web-search branches are called (or skipped)
according to BatchOptions, and that branch failures are isolated.

All external I/O (LLM, DB, URL fetcher) is mocked.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.job_processor import JobProcessor
from app.services.enrichment import AttributeMerger, Source
from app.services.enrichment.attribute_merger import AttributeValue
from app.models import (
    ProductData,
    ProductContext,
    FeatureOption,
    ResearchMode,
    BatchOptions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_product(image_urls=None):
    return ProductData(
        id=1,
        category_id=42,
        name="Test Widget",
        description="A simple test widget.",
        context=ProductContext(existing_features={}, company_id=0),
        languages=["en"],
        source_urls=[],
        image_urls=image_urls or [],
    )


def _make_schema():
    return {
        "colour": FeatureOption(type="text", options=[]),
    }


def _make_pipeline():
    """Minimal AiFeaturePipeline mock that returns a value for 'colour'."""
    pipeline = MagicMock()
    pipeline.extract_feature = AsyncMock(return_value={
        "value": "red",
        "tokens": 10,
        "router_debug": {},
        "extraction_reasoning": "found in description",
        "deduced_context": None,
        "source": "description",
        "source_urls": None,
    })
    return pipeline


def _make_db_cache():
    cache = MagicMock()
    cache.get_cached_value = AsyncMock(return_value=None)
    cache.set_cached_value = AsyncMock(return_value=None)
    return cache


def _make_matcher():
    matcher = MagicMock()
    matcher.find_best_match = MagicMock(side_effect=lambda val, opts: val)
    return matcher


def _make_job_processor(vision_producer=None, websearch_producer=None):
    semaphore = asyncio.Semaphore(10)
    return JobProcessor(
        pipeline=_make_pipeline(),
        db_cache=_make_db_cache(),
        matcher=_make_matcher(),
        global_semaphore=semaphore,
        vision_producer=vision_producer,
        websearch_producer=websearch_producer,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_vision_not_called_when_disabled(monkeypatch):
    """enable_vision=False → VisionProducer.produce_description never called."""
    vision = MagicMock()
    vision.produce_description = AsyncMock(return_value="some text")

    processor = _make_job_processor(vision_producer=vision)

    with patch("app.services.job_processor.fetch_all", new=AsyncMock(return_value="")), \
         patch("app.services.job_processor.AsyncSessionLocal") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        await processor.process_product(
            product=_make_product(image_urls=["https://example.com/img.jpg"]),
            schema=_make_schema(),
            client_id=1,
            options=BatchOptions(enable_vision=False, enable_web_search=False),
        )

    vision.produce_description.assert_not_called()


@pytest.mark.asyncio
async def test_vision_not_called_when_no_images(monkeypatch):
    """enable_vision=True but no image_urls → VisionProducer not called."""
    vision = MagicMock()
    vision.produce_description = AsyncMock(return_value="some text")

    processor = _make_job_processor(vision_producer=vision)

    with patch("app.services.job_processor.fetch_all", new=AsyncMock(return_value="")), \
         patch("app.services.job_processor.AsyncSessionLocal") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        await processor.process_product(
            product=_make_product(image_urls=[]),  # no images
            schema=_make_schema(),
            client_id=1,
            options=BatchOptions(enable_vision=True, enable_web_search=False),
        )

    vision.produce_description.assert_not_called()


@pytest.mark.asyncio
async def test_vision_called_when_enabled_with_images():
    """enable_vision=True with images → VisionProducer.produce_description called."""
    vision = MagicMock()
    vision.produce_description = AsyncMock(return_value=None)  # returns None → no attrs

    processor = _make_job_processor(vision_producer=vision)

    with patch("app.services.job_processor.fetch_all", new=AsyncMock(return_value="")), \
         patch("app.services.job_processor.AsyncSessionLocal") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = await processor.process_product(
            product=_make_product(image_urls=["https://example.com/img.jpg"]),
            schema=_make_schema(),
            client_id=1,
            options=BatchOptions(enable_vision=True, enable_web_search=False),
        )

    vision.produce_description.assert_called_once()
    # Description branch still ran and returned "red".
    assert result["filled_features"].get("colour") == "red"


@pytest.mark.asyncio
async def test_vision_branch_exception_does_not_kill_description_branch():
    """If vision branch raises, description branch result is still returned."""
    vision = MagicMock()
    vision.produce_description = AsyncMock(side_effect=RuntimeError("vision API down"))

    processor = _make_job_processor(vision_producer=vision)

    with patch("app.services.job_processor.fetch_all", new=AsyncMock(return_value="")), \
         patch("app.services.job_processor.AsyncSessionLocal") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        result = await processor.process_product(
            product=_make_product(image_urls=["https://example.com/img.jpg"]),
            schema=_make_schema(),
            client_id=1,
            options=BatchOptions(enable_vision=True, enable_web_search=False),
        )

    # Description branch still returned "red" despite vision failure.
    assert result["filled_features"].get("colour") == "red"


@pytest.mark.asyncio
async def test_both_branches_disabled_by_default():
    """Default BatchOptions → neither vision nor web_search are called."""
    vision = MagicMock()
    vision.produce_description = AsyncMock(return_value="text")
    websearch = MagicMock()
    websearch.produce_summary = AsyncMock(return_value="text")

    processor = _make_job_processor(
        vision_producer=vision, websearch_producer=websearch
    )

    with patch("app.services.job_processor.fetch_all", new=AsyncMock(return_value="")), \
         patch("app.services.job_processor.AsyncSessionLocal") as mock_session_cls:
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = mock_session

        await processor.process_product(
            product=_make_product(image_urls=["https://example.com/img.jpg"]),
            schema=_make_schema(),
            client_id=1,
            # No options arg → defaults to BatchOptions()
        )

    vision.produce_description.assert_not_called()
    websearch.produce_summary.assert_not_called()
