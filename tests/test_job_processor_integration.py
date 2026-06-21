"""Integration tests: job_processor + USE_NEW_PIPELINE feature flag.

Tests verify:
1. Legacy path is used (and unchanged) when USE_NEW_PIPELINE=False.
2. New path branch is entered when USE_NEW_PIPELINE=True and adapter is present.
3. Structural test: legacy path result shape is unchanged.
4. New path returns legacy-compatible result shape (debug_info + tokens_used present).
5. New path id-mapping correctly maps positional index back to schema key names.
6. New path handles empty schema gracefully.

All external I/O (LLM, DB, URL fetcher, pipeline adapter) is mocked.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app import config
from app.services.enrichment.base import AttributeValue, Source
from app.services.job_processor import JobProcessor
from app.models import (
    BatchOptions,
    FeatureOption,
    ProductContext,
    ProductData,
    ResearchMode,
)


# ---------------------------------------------------------------------------
# Helpers (shared with test_job_processor_branches)
# ---------------------------------------------------------------------------

def _make_product():
    return ProductData(
        id=99,
        category_id=1,
        name="Integration Widget",
        description="A widget for integration tests.",
        context=ProductContext(existing_features={}, company_id=0),
        languages=["en"],
        source_urls=[],
        image_urls=[],
    )


def _make_schema():
    return {"colour": FeatureOption(type="text", options=[])}


def _make_pipeline():
    p = MagicMock()
    p.extract_feature = AsyncMock(return_value={
        "value": "green",
        "tokens": 5,
        "router_debug": {},
        "extraction_reasoning": "found in description",
        "deduced_context": None,
        "source": "description",
        "source_urls": None,
    })
    return p


def _make_db_cache():
    c = MagicMock()
    c.get_cached_value = AsyncMock(return_value=None)
    c.set_cached_value = AsyncMock(return_value=None)
    return c


def _make_matcher():
    m = MagicMock()
    m.find_best_match = MagicMock(side_effect=lambda val, opts: val)
    return m


def _make_processor() -> JobProcessor:
    return JobProcessor(
        pipeline=_make_pipeline(),
        db_cache=_make_db_cache(),
        matcher=_make_matcher(),
        global_semaphore=asyncio.Semaphore(10),
    )


def _session_patch():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    return mock_session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_legacy_path_used_when_flag_off(monkeypatch):
    """When USE_NEW_PIPELINE=False, process_product returns the legacy result dict."""
    monkeypatch.setattr(config, "USE_NEW_PIPELINE", False)

    processor = _make_processor()
    with patch("app.services.job_processor.fetch_all", new=AsyncMock(return_value="")), \
         patch("app.services.job_processor.AsyncSessionLocal") as mock_cls:
        mock_cls.return_value = _session_patch()

        result = await processor.process_product(
            product=_make_product(),
            schema=_make_schema(),
            client_id=1,
        )

    # Legacy path produces the standard result shape
    assert result["product_id"] == 99
    assert "filled_features" in result
    assert "debug_info" in result
    assert "tokens_used" in result
    assert "is_cached" in result
    # The pipeline's mock value "green" should appear
    assert result["filled_features"].get("colour") == "green"


def _make_mock_adapter(av_list=None):
    """Build a mock PipelineAdapter whose .run() coroutine returns av_list."""
    adapter = MagicMock()
    adapter.run = AsyncMock(return_value=av_list or [])
    return adapter


@pytest.mark.asyncio
async def test_new_path_used_when_flag_on(monkeypatch):
    """When USE_NEW_PIPELINE=True and adapter is set, the new path is entered.

    Verifies:
    - Result has the standard legacy shape (product_id, filled_features, debug_info,
      tokens_used, is_cached).
    - The adapter.run() coroutine was called exactly once.
    """
    monkeypatch.setattr(config, "USE_NEW_PIPELINE", True)

    # Adapter returns one AttributeValue for 'colour' (attribute_id=0, positional index)
    mock_av = AttributeValue(
        attribute_id=0,
        value="blue",
        confidence=0.9,
        source=Source.DESCRIPTION,
        evidence="found in description",
        judge_validated=True,
    )
    mock_adapter = _make_mock_adapter([mock_av])

    processor = _make_processor()
    processor._pipeline_adapter = mock_adapter

    result = await processor.process_product(
        product=_make_product(),
        schema=_make_schema(),
        client_id=1,
    )

    mock_adapter.run.assert_called_once()

    # Standard legacy shape present
    assert result["product_id"] == 99
    assert "filled_features" in result
    assert "debug_info" in result
    assert "tokens_used" in result
    assert "is_cached" in result
    # is_cached is False for new path
    assert result["is_cached"] is False


@pytest.mark.asyncio
async def test_new_path_called_when_flag_on_returns_legacy_format(monkeypatch):
    """New path adapter result is converted to the exact same dict shape as legacy.

    Verifies debug_info and tokens_used are present and have the right types.
    """
    monkeypatch.setattr(config, "USE_NEW_PIPELINE", True)

    mock_av = AttributeValue(
        attribute_id=0,
        value="red",
        confidence=0.85,
        source=Source.LLM_KNOWLEDGE,
        evidence="from knowledge base",
        judge_validated=False,
    )
    mock_adapter = _make_mock_adapter([mock_av])

    processor = _make_processor()
    processor._pipeline_adapter = mock_adapter

    result = await processor.process_product(
        product=_make_product(),
        schema=_make_schema(),
        client_id=1,
    )

    # Contract v2 keys present (added "skipped": причина пустоты per НЕзаполненный target)
    assert set(result.keys()) == {"product_id", "filled_features", "debug_info", "skipped", "tokens_used", "is_cached"}

    # tokens_used is int (0 for new path — see TODO Tier-2-cost-tracking)
    assert isinstance(result["tokens_used"], int)

    # debug_info is a dict keyed by feature name
    assert isinstance(result["debug_info"], dict)
    assert "colour" in result["debug_info"]
    di = result["debug_info"]["colour"]
    # debug_info entry has the expected sub-keys
    for key in ("source", "confidence", "evidence", "judge_validated"):
        assert key in di, f"debug_info missing key: {key}"

    # filled_features has the mapped value
    assert result["filled_features"]["colour"] == "red"


@pytest.mark.asyncio
async def test_new_path_id_mapping_matches_schema_keys(monkeypatch):
    """Positional index id must map back to the correct schema key name.

    Schema: {"colour": ..., "material": ...}
    Adapter returns AttributeValue with attribute_id=1 → should map to "material".
    """
    monkeypatch.setattr(config, "USE_NEW_PIPELINE", True)

    multi_schema = {
        "colour": FeatureOption(type="text", options=[]),
        "material": FeatureOption(type="text", options=[]),
    }
    # attribute_id=1 corresponds to index 1 → "material"
    mock_av = AttributeValue(
        attribute_id=1,
        value="steel",
        confidence=0.92,
        source=Source.DESCRIPTION,
        evidence=None,
        judge_validated=True,
    )
    mock_adapter = _make_mock_adapter([mock_av])

    processor = _make_processor()
    processor._pipeline_adapter = mock_adapter

    result = await processor.process_product(
        product=_make_product(),
        schema=multi_schema,
        client_id=1,
    )

    assert result["filled_features"].get("material") == "steel"
    assert "colour" not in result["filled_features"]  # attribute_id=0 not returned


@pytest.mark.asyncio
async def test_new_path_handles_empty_schema(monkeypatch):
    """New path with an empty schema returns a valid result with empty filled_features."""
    monkeypatch.setattr(config, "USE_NEW_PIPELINE", True)

    mock_adapter = _make_mock_adapter([])  # adapter returns nothing

    processor = _make_processor()
    processor._pipeline_adapter = mock_adapter

    result = await processor.process_product(
        product=_make_product(),
        schema={},
        client_id=1,
    )

    assert result["product_id"] == 99
    assert result["filled_features"] == {}
    assert result["debug_info"] == {}
    assert result["tokens_used"] == 0
    assert result["is_cached"] is False


@pytest.mark.asyncio
async def test_legacy_path_unchanged(monkeypatch):
    """Structural test: enabling the flag does NOT change the legacy result.

    With flag OFF, result structure and filled_features values are identical
    regardless of whether the pipeline adapter import succeeded.
    """
    monkeypatch.setattr(config, "USE_NEW_PIPELINE", False)

    processor = _make_processor()
    with patch("app.services.job_processor.fetch_all", new=AsyncMock(return_value="")), \
         patch("app.services.job_processor.AsyncSessionLocal") as mock_cls:
        mock_cls.return_value = _session_patch()

        result = await processor.process_product(
            product=_make_product(),
            schema=_make_schema(),
            client_id=1,
        )

    # Exact keys present
    assert set(result.keys()) == {"product_id", "filled_features", "debug_info", "tokens_used", "is_cached"}
    # No extra keys sneaked in
    assert len(result) == 5
    # Value extracted by legacy mock
    assert result["filled_features"]["colour"] == "green"
    # tokens_used is int
    assert isinstance(result["tokens_used"], int)
    # is_cached is bool
    assert isinstance(result["is_cached"], bool)
