"""
tests/test_scrappey_timeout.py

Tests for the fail-fast Scrappey timeout introduced in:
  perf(scrappey): tight fail-fast timeout on fallback + concurrent page fetch

Covers:
  1. Hanging Scrappey → returns None within ~cap, records TRANSIENT (not BLOCKED)
  2. Fast Scrappey success → returns content normally
  3. Concurrent multi-page fetch: one hanging page does NOT block others;
     total time ≈ cap, not 5×cap
  4. _scrappey_timeout_cap() respects URL_FETCHER_SCRAPPEY_TIMEOUT env var
  5. Timeout is TRANSIENT (block_strikes not incremented)

All tests are fully offline — no live calls.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import app.services.url_fetcher as uf
from app.services.url_fetcher import (
    _scrappey_timeout_cap,
    fetch_all_results,
    FetchResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_scrappey_counter():
    uf._scrappey_call_count = 0
    yield
    uf._scrappey_call_count = 0


@pytest.fixture()
def fast_cap(monkeypatch):
    """Set a very short Scrappey cap (0.15s) for speed in tests."""
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_TIMEOUT", "0.15")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_FALLBACK", "1")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_MAX", "200")


@pytest.fixture()
def fresh_domain_health(tmp_path, monkeypatch):
    """Isolated domain_health store in tmp_path."""
    store_file = str(tmp_path / "dh.json")
    monkeypatch.setenv("DOMAIN_HEALTH_STORE", store_file)
    import app.services.providers.domain_health as dh
    dh._store.clear()
    dh._store_loaded = False
    return dh


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_resp(status: int, body: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = body
    resp.headers = {}
    resp.raise_for_status = MagicMock()
    return resp


# ---------------------------------------------------------------------------
# 1. Hanging Scrappey → returns None within cap, records TRANSIENT
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hanging_scrappey_returns_none_within_cap(
    fast_cap, fresh_domain_health, tmp_path, monkeypatch
):
    """A Scrappey mock that sleeps longer than the cap must be killed within cap+slack.

    The test verifies:
      - _try_scrappey_fallback returns None
      - elapsed time < cap + 0.5s  (well under the old 120s)
      - domain_health records TRANSIENT, NOT BLOCKED (timeout ≠ confirmed dead)
    """
    monkeypatch.setattr(uf, "_CACHE_DIR", str(tmp_path / "cache"))
    cap = _scrappey_timeout_cap()
    slack = 0.5  # generous scheduling slack

    async def _slow_scrappey(url, timeout=25.0):
        await asyncio.sleep(10.0)   # much longer than cap
        return "X" * 1000

    with patch(
        "app.services.providers.scrappey_client.scrappey_fetch",
        new=AsyncMock(side_effect=_slow_scrappey),
    ):
        t0 = time.monotonic()
        result = await uf._try_scrappey_fallback(
            "https://wall-domain.ru/product/1",
            "HTTP 403",
            timeout=10,
        )
        elapsed = time.monotonic() - t0

    assert result is None, "Hanging Scrappey must return None"
    assert elapsed < cap + slack, (
        f"Expected completion in < {cap + slack:.2f}s, got {elapsed:.2f}s"
    )

    # Verify TRANSIENT was recorded (block_strikes must stay 0)
    dh = fresh_domain_health
    domain = dh._registrable_domain("wall-domain.ru")
    rec = dh._store.get(domain)
    assert rec is not None, "Domain record must exist after TRANSIENT"
    assert rec.block_strikes == 0, (
        f"Timeout must NOT increment block_strikes (got {rec.block_strikes})"
    )
    assert rec.transient_fails >= 1, "TRANSIENT must be counted"


# ---------------------------------------------------------------------------
# 2. Fast Scrappey success still returns content
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fast_scrappey_success_returns_content(
    fast_cap, fresh_domain_health, tmp_path, monkeypatch
):
    """A Scrappey mock that responds instantly must still return the response."""
    monkeypatch.setattr(uf, "_CACHE_DIR", str(tmp_path / "cache"))

    good_html = "Product description. " * 60  # > 500 chars, no block markers

    async def _fast_scrappey(url, timeout=25.0):
        return good_html

    with patch(
        "app.services.providers.scrappey_client.scrappey_fetch",
        new=AsyncMock(side_effect=_fast_scrappey),
    ):
        result = await uf._try_scrappey_fallback(
            "https://some-shop.ru/product/fast",
            "HTTP 403",
            timeout=10,
        )

    assert result is not None, "Fast Scrappey must return a response"
    assert result.status_code == 200
    assert good_html in result.text


# ---------------------------------------------------------------------------
# 3. Concurrent multi-page fetch: one slow page doesn't block the rest
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_fetch_one_slow_bounded_to_cap(tmp_path, monkeypatch):
    """fetch_all_results with 5 pages where 1 is slow (simulating Scrappey cap).

    Because fetch_all_results uses asyncio.gather (concurrent), total time
    must be approximately the slowest single task, NOT the sum of all tasks.

    Setup:
      - 4 pages return in ~0s (instant)
      - 1 page sleeps SLOW_DELAY seconds (simulating a Scrappey call hitting cap)
    Assert:
      - elapsed < SLOW_DELAY + slack  (concurrent, not SLOW_DELAY * 5)
      - 4 fast results returned, 1 slow dropped
    """
    SLOW_DELAY = 0.20    # 200ms — fast enough for unit tests
    TOTAL_SLACK = SLOW_DELAY + 0.40  # 400ms scheduling headroom

    fast_html = "Fast product content. " * 60
    monkeypatch.setattr(uf, "_CACHE_DIR", str(tmp_path / "cache"))

    async def _mock_fetch_one(url: str) -> FetchResult | None:
        """Slow URL sleeps SLOW_DELAY; all others return instantly."""
        if "slow" in url:
            await asyncio.sleep(SLOW_DELAY)
            return None
        return FetchResult(url=url, content=fast_html[:200], source_type="generic")

    with patch("app.services.url_fetcher._fetch_one_result", side_effect=_mock_fetch_one):
        urls = [
            "https://fast1.ru/p",
            "https://fast2.ru/p",
            "https://slow-wall.ru/p",   # this one has a delay
            "https://fast3.ru/p",
            "https://fast4.ru/p",
        ]
        t0 = time.monotonic()
        results = await fetch_all_results(urls)
        elapsed = time.monotonic() - t0

    # Total time must be ~SLOW_DELAY, not 5 × SLOW_DELAY
    assert elapsed < TOTAL_SLACK, (
        f"Concurrent fetch must complete in < {TOTAL_SLACK:.2f}s, got {elapsed:.3f}s "
        f"(if sequential, would be >{SLOW_DELAY * 5:.2f}s)"
    )
    # 4 fast pages returned; slow page was dropped (returned None)
    assert len(results) == 4, f"Expected 4 results (slow dropped), got {len(results)}"
    result_urls = {r.url for r in results}
    assert "https://slow-wall.ru/p" not in result_urls


# ---------------------------------------------------------------------------
# 4. _scrappey_timeout_cap respects env var
# ---------------------------------------------------------------------------

def test_scrappey_timeout_cap_reads_env(monkeypatch):
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_TIMEOUT", "42")
    assert _scrappey_timeout_cap() == 42.0


def test_scrappey_timeout_cap_defaults_to_25(monkeypatch):
    monkeypatch.delenv("URL_FETCHER_SCRAPPEY_TIMEOUT", raising=False)
    assert _scrappey_timeout_cap() == 25.0


def test_scrappey_timeout_cap_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_TIMEOUT", "not-a-number")
    assert _scrappey_timeout_cap() == 25.0


# ---------------------------------------------------------------------------
# 5. Timeout is TRANSIENT — does NOT count as a death strike (5 timeouts ≠ dead)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timeout_is_transient_not_death_strike(
    fast_cap, fresh_domain_health, tmp_path, monkeypatch
):
    """Even 5 timeout-based TRANSIENT outcomes must NOT mark the domain dead."""
    monkeypatch.setattr(uf, "_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")

    async def _slow_scrappey(url, timeout=25.0):
        await asyncio.sleep(10.0)
        return None

    host = "timeout-shop.ru"
    for i in range(5):
        with patch(
            "app.services.providers.scrappey_client.scrappey_fetch",
            new=AsyncMock(side_effect=_slow_scrappey),
        ):
            await uf._try_scrappey_fallback(
                f"https://{host}/product/{i}",
                "HTTP 403",
                timeout=10,
            )

    dh = fresh_domain_health
    # Must NOT be dead
    assert dh.should_skip_scrappey(host) is False, (
        "5 timeouts (TRANSIENT) must NOT mark domain dead"
    )
    domain = dh._registrable_domain(host)
    rec = dh._store.get(domain)
    assert rec is not None
    assert rec.block_strikes == 0, "Timeouts must not increment block_strikes"
    assert rec.transient_fails == 5
