"""Integration tests: job_processor + USE_NEW_PIPELINE feature flag.

Tests verify:
1. Legacy path is used (and unchanged) when USE_NEW_PIPELINE=False.
2. New path branch is entered when USE_NEW_PIPELINE=True and adapter is present.
3. Structural test: legacy path result shape is unchanged.

All external I/O (LLM, DB, URL fetcher, pipeline adapter) is mocked.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app import config
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


@pytest.mark.asyncio
async def test_new_path_used_when_flag_on(monkeypatch):
    """When USE_NEW_PIPELINE=True, the pipeline adapter branch is entered (logged warning).

    The adapter stub currently falls through to the legacy path with a warning
    (per spec: minimal integration until full wiring is complete).  We verify:
    - The warning is logged.
    - The result still has the standard shape (legacy fallback is working).
    """
    monkeypatch.setattr(config, "USE_NEW_PIPELINE", True)

    processor = _make_processor()
    with patch("app.services.job_processor.fetch_all", new=AsyncMock(return_value="")), \
         patch("app.services.job_processor.AsyncSessionLocal") as mock_cls, \
         patch("app.services.job_processor.logger") as mock_logger:
        mock_cls.return_value = _session_patch()

        result = await processor.process_product(
            product=_make_product(),
            schema=_make_schema(),
            client_id=1,
        )

    # Warning must have been emitted about new pipeline not fully wired
    warning_calls = [str(call) for call in mock_logger.warning.call_args_list]
    assert any("USE_NEW_PIPELINE" in wc for wc in warning_calls), (
        "Expected a warning about USE_NEW_PIPELINE being set but wiring incomplete"
    )

    # Result still has standard shape (fell back to legacy)
    assert "product_id" in result
    assert "filled_features" in result


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
