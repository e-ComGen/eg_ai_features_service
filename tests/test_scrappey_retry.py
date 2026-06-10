"""Unit tests for fail-fast Scrappey retry logic in OzonCardSource and YandexMarketSource.

Covers:
  - OzonCardSource._scrappey_fetch: asyncio.TimeoutError on first attempt triggers retry;
    second attempt succeeds → HTML returned.
  - OzonCardSource._scrappey_fetch: both attempts time out → None returned, log emitted.
  - YandexMarketSource._fetch_card_html: first attempt times out → retry fires → success.
  - YandexMarketSource._fetch_card_html: both attempts time out → None returned.

No real network. _scrappey_fetch_once / _scrappey_fetch_page are patched with asyncio
coroutines that simulate immediate timeout or immediate success.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.services.enrichment.sources.ozon_card_source import (
    OzonCardSource,
    _SCRAPPEY_MAX_ATTEMPTS,
    _SCRAPPEY_PER_ATTEMPT_TIMEOUT,
)
from app.services.enrichment.sources.yandex_market_source import (
    YandexMarketSource,
    _SCRAPPEY_MAX_ATTEMPTS as _YM_MAX_ATTEMPTS,
    _SCRAPPEY_PER_ATTEMPT_TIMEOUT as _YM_PER_ATTEMPT_TIMEOUT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ozon_source() -> OzonCardSource:
    """Build a minimal OzonCardSource that won't touch disk/network at __init__."""
    with (
        patch(
            "app.services.enrichment.sources.ozon_card_source.get_ozon_characteristics_for_type",
            return_value={},
        ),
        patch(
            "app.services.enrichment.sources.ozon_card_source.OzonCardJudge",
            return_value=MagicMock(),
        ),
    ):
        src = OzonCardSource.__new__(OzonCardSource)
        src._scrappey_key = "fake-key"
        src._judge = MagicMock()
        src._cache = {}
        return src


def _make_ym_source() -> YandexMarketSource:
    """Build a minimal YandexMarketSource for unit testing."""
    with patch(
        "app.services.enrichment.sources.yandex_market_source.get_web_search_client",
        return_value=MagicMock(),
    ):
        src = YandexMarketSource.__new__(YandexMarketSource)
        src._scrappey_key = "fake-key"
        src._search_client = MagicMock()
        src._judge = MagicMock()
        src._cache = {}
        return src


# ---------------------------------------------------------------------------
# OzonCardSource._scrappey_fetch — timeout retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ozon_scrappey_fetch_retries_on_first_timeout(caplog):
    """First _scrappey_fetch_once call times out → retry fires → second succeeds.

    We patch _SCRAPPEY_PER_ATTEMPT_TIMEOUT to 0.05s so the test runs in milliseconds
    instead of 30s while preserving the real code path.
    """
    import logging
    import app.services.enrichment.sources.ozon_card_source as ozon_mod

    src = _make_ozon_source()
    big_html = "x" * 60_000  # > _MIN_VALID_HTML_LEN

    call_count = 0

    async def _fake_fetch_once(client, url):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate hung Scrappey — asyncio.wait_for with tiny timeout cancels us fast
            await asyncio.sleep(9999)
        return big_html

    fake_client = MagicMock(spec=httpx.AsyncClient)

    with (
        patch.object(ozon_mod, "_SCRAPPEY_PER_ATTEMPT_TIMEOUT", 0.05),
        patch.object(ozon_mod, "_SCRAPPEY_MAX_ATTEMPTS", 2),
        caplog.at_level(logging.INFO, logger="app.services.enrichment.sources.ozon_card_source"),
        patch.object(src, "_scrappey_fetch_once", side_effect=_fake_fetch_once),
    ):
        result = await src._scrappey_fetch(fake_client, "https://ozon.ru/fake")

    assert result == big_html, "Second attempt should succeed and return HTML"
    assert call_count == 2, "Should have made exactly 2 calls (1 timeout + 1 retry)"
    all_messages = " ".join(r.message for r in caplog.records)
    assert "scrappey-retry" in all_messages, (
        f"Expected a scrappey-retry log line, got: {all_messages!r}"
    )


@pytest.mark.asyncio
async def test_ozon_scrappey_fetch_gives_up_after_max_attempts(caplog):
    """Both attempts time out → _scrappey_fetch returns None and logs all-failed."""
    import logging
    import app.services.enrichment.sources.ozon_card_source as ozon_mod

    src = _make_ozon_source()
    call_count = 0

    async def _always_hang(client, url):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(9999)

    fake_client = MagicMock(spec=httpx.AsyncClient)

    with (
        patch.object(ozon_mod, "_SCRAPPEY_PER_ATTEMPT_TIMEOUT", 0.05),
        patch.object(ozon_mod, "_SCRAPPEY_MAX_ATTEMPTS", 2),
        caplog.at_level(logging.WARNING, logger="app.services.enrichment.sources.ozon_card_source"),
        patch.object(src, "_scrappey_fetch_once", side_effect=_always_hang),
    ):
        result = await src._scrappey_fetch(fake_client, "https://ozon.ru/fake")

    assert result is None
    assert call_count == 2, (
        f"Expected exactly 2 attempts, got {call_count}"
    )
    assert any("all-scrappey-attempts-timed-out" in r.message for r in caplog.records), (
        "Expected all-failed log line"
    )


# ---------------------------------------------------------------------------
# OzonCardSource constants sanity
# ---------------------------------------------------------------------------

def test_ozon_retry_constants_are_sensible():
    """Per-attempt timeout and max attempts must give a sane total budget.

    Timeouts were cut 80→45s total, 30→18s per-attempt (2026-06 fail-fast change).
    2 attempts × 18s + 2s backoff = 38s worst-case Scrappey path < 45s total cap.
    """
    assert _SCRAPPEY_PER_ATTEMPT_TIMEOUT == 18.0
    assert _SCRAPPEY_MAX_ATTEMPTS == 2
    worst_case = _SCRAPPEY_MAX_ATTEMPTS * _SCRAPPEY_PER_ATTEMPT_TIMEOUT + 2.0  # + 1 backoff
    from app.services.enrichment.sources.ozon_card_source import _OZON_CARD_TOTAL_TIMEOUT
    assert worst_case < _OZON_CARD_TOTAL_TIMEOUT, (
        f"Worst-case Scrappey path ({worst_case}s) must fit inside total cap "
        f"({_OZON_CARD_TOTAL_TIMEOUT}s)"
    )


# ---------------------------------------------------------------------------
# YandexMarketSource._fetch_card_html — timeout retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ym_fetch_card_html_retries_on_first_timeout(caplog):
    """First Scrappey call times out → retry fires → second succeeds."""
    import logging
    import app.services.enrichment.sources.yandex_market_source as ym_mod

    src = _make_ym_source()
    big_html = "y" * 50_000

    call_count = 0

    async def _fake_fetch_page(url, timeout=30.0):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await asyncio.sleep(9999)
        return big_html

    with (
        patch.object(ym_mod, "_SCRAPPEY_PER_ATTEMPT_TIMEOUT", 0.05),
        patch.object(ym_mod, "_SCRAPPEY_MAX_ATTEMPTS", 2),
        caplog.at_level(logging.INFO, logger="app.services.enrichment.sources.yandex_market_source"),
        patch(
            "app.services.enrichment.sources.yandex_market_source._scrappey_fetch_page",
            side_effect=_fake_fetch_page,
        ),
    ):
        result = await src._fetch_card_html("https://market.yandex.ru/product/123456")

    assert result == big_html
    assert call_count == 2, f"Expected 2 calls (1 timeout + 1 retry), got {call_count}"
    all_messages = " ".join(r.message for r in caplog.records)
    assert "scrappey-retry" in all_messages, (
        f"Expected scrappey-retry log, got: {all_messages!r}"
    )


@pytest.mark.asyncio
async def test_ym_fetch_card_html_returns_none_after_max_attempts(caplog):
    """Both attempts time out → None returned, all-failed-empty logged."""
    import logging
    import app.services.enrichment.sources.yandex_market_source as ym_mod

    src = _make_ym_source()
    call_count = 0

    async def _always_hang(url, timeout=30.0):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(9999)

    with (
        patch.object(ym_mod, "_SCRAPPEY_PER_ATTEMPT_TIMEOUT", 0.05),
        patch.object(ym_mod, "_SCRAPPEY_MAX_ATTEMPTS", 2),
        caplog.at_level(logging.WARNING, logger="app.services.enrichment.sources.yandex_market_source"),
        patch(
            "app.services.enrichment.sources.yandex_market_source._scrappey_fetch_page",
            side_effect=_always_hang,
        ),
    ):
        result = await src._fetch_card_html("https://market.yandex.ru/product/123456")

    assert result is None
    assert call_count == 2
    assert any("all-failed-empty" in r.message for r in caplog.records), (
        "Expected all-failed-empty log line"
    )


# ---------------------------------------------------------------------------
# YandexMarketSource constants sanity
# ---------------------------------------------------------------------------

def test_ym_retry_constants_are_sensible():
    """YM per-attempt timeout and max attempts must fit inside total cap."""
    assert _YM_PER_ATTEMPT_TIMEOUT == 30.0
    assert _YM_MAX_ATTEMPTS == 2
    worst_case = _YM_MAX_ATTEMPTS * _YM_PER_ATTEMPT_TIMEOUT + 2.0
    from app.services.enrichment.sources.yandex_market_source import _YM_TOTAL_TIMEOUT
    assert worst_case < _YM_TOTAL_TIMEOUT, (
        f"Worst-case ({worst_case}s) must fit inside total cap ({_YM_TOTAL_TIMEOUT}s)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
