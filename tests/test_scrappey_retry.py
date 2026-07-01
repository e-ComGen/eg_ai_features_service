"""Unit tests for fail-fast retry logic in YandexMarketSource (scrape.do).

Covers:
  - YandexMarketSource._fetch_card_html: first attempt times out → retry fires → success.
  - YandexMarketSource._fetch_card_html: both attempts time out → None returned.

OzonCardSource's own retry wrapper was REMOVED (FIX-10, ozon_card_source.py switched to
scrape.do): scrapedo_fetch already retries transient failures internally, so there is no
separate Ozon-side retry logic left to test here — see test_ozon_card_scrapedo_fix10.py for
OzonCardSource._fetch_page coverage.

No real network. _scrapedo_fetch_page is patched with asyncio coroutines that simulate
immediate timeout or immediate success.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from app.services.enrichment.sources.yandex_market_source import (
    YandexMarketSource,
    _SCRAPPEY_MAX_ATTEMPTS as _YM_MAX_ATTEMPTS,
    _SCRAPPEY_PER_ATTEMPT_TIMEOUT as _YM_PER_ATTEMPT_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ym_source() -> YandexMarketSource:
    """Build a minimal YandexMarketSource for unit testing."""
    with patch(
        "app.services.enrichment.sources.yandex_market_source.get_web_search_client",
        return_value=MagicMock(),
    ):
        src = YandexMarketSource.__new__(YandexMarketSource)
        src._scrapedo_token = "fake-token"
        src._search_client = MagicMock()
        src._judge = MagicMock()
        src._cache = {}
        return src


# ---------------------------------------------------------------------------
# YandexMarketSource._fetch_card_html — timeout retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ym_fetch_card_html_returns_content_on_success(caplog):
    """Single Scrape.do call succeeds -> HTML content returned (no retry: max_attempts=1)."""
    import logging
    from app.services.providers.scrapfly_client import ScrapflyResult

    src = _make_ym_source()
    big_html = "y" * 50_000

    call_count = 0

    async def _fake_fetch_page(url, *, render=True, super_proxy=True, timeout=95.0):
        nonlocal call_count
        call_count += 1
        return ScrapflyResult(
            success=True, content=big_html, status_code=200, credits_used=1, error=None
        )

    with (
        caplog.at_level(logging.INFO, logger="app.services.enrichment.sources.yandex_market_source"),
        patch(
            "app.services.enrichment.sources.yandex_market_source._scrapedo_fetch_page",
            side_effect=_fake_fetch_page,
        ),
    ):
        result = await src._fetch_card_html("https://market.yandex.ru/product/123456")

    assert result == big_html
    assert call_count == 1, f"Expected 1 call (single attempt, no retry), got {call_count}"
    all_messages = " ".join(r.message for r in caplog.records)
    assert "scrapedo-retry" not in all_messages, (
        f"Single-attempt design must not retry, got: {all_messages!r}"
    )


@pytest.mark.asyncio
async def test_ym_fetch_card_html_returns_none_on_timeout(caplog):
    """The single Scrape.do attempt times out -> None returned, all-failed-empty logged."""
    import logging
    import app.services.enrichment.sources.yandex_market_source as ym_mod

    src = _make_ym_source()
    call_count = 0

    async def _always_hang(url, *, render=True, super_proxy=True, timeout=95.0):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(9999)

    with (
        patch.object(ym_mod, "_SCRAPPEY_PER_ATTEMPT_TIMEOUT", 0.05),
        caplog.at_level(logging.WARNING, logger="app.services.enrichment.sources.yandex_market_source"),
        patch(
            "app.services.enrichment.sources.yandex_market_source._scrapedo_fetch_page",
            side_effect=_always_hang,
        ),
    ):
        result = await src._fetch_card_html("https://market.yandex.ru/product/123456")

    assert result is None
    assert call_count == 1
    assert any("all-failed-empty" in r.message for r in caplog.records), (
        "Expected all-failed-empty log line"
    )


# ---------------------------------------------------------------------------
# YandexMarketSource constants sanity
# ---------------------------------------------------------------------------

def test_ym_retry_constants_are_sensible():
    """YM per-attempt timeout and max attempts must fit inside total cap."""
    assert _YM_PER_ATTEMPT_TIMEOUT == 95.0
    assert _YM_MAX_ATTEMPTS == 1
    worst_case = _YM_MAX_ATTEMPTS * _YM_PER_ATTEMPT_TIMEOUT + 2.0
    from app.services.enrichment.sources.yandex_market_source import _YM_TOTAL_TIMEOUT
    assert worst_case < _YM_TOTAL_TIMEOUT, (
        f"Worst-case ({worst_case}s) must fit inside total cap ({_YM_TOTAL_TIMEOUT}s)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
