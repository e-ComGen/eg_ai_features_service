"""
tests/test_domain_health.py

Unit tests for app/services/providers/domain_health.py.

All tests are offline — no real I/O to Scrappey.
Tests redirect the store path to a tmp_path to avoid polluting the real store.

Scenarios covered:
 1. 3 BLOCKED on distinct URLs → domain dead → should_skip_scrappey True
 2. USABLE resets strikes + clears dead_until (revives domain)
 3. 5× TRANSIENT does NOT mark domain dead
 4. ozon.ru never dead even after 10 BLOCKED
 5. wildberries.ru never dead even after 10 BLOCKED
 6. dead_until TTL expiry → should_skip_scrappey flips back to False
 7. Same-URL repeated BLOCKED does NOT over-count (distinct-URL guard)
 8. Persistence: write store, reload fresh module state, records survive
 9. url_fetcher integration: dead domain → _try_scrappey_fallback returns None
    without calling scrappey_fetch; healthy domain → calls it
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import httpx
import pytest

# ---------------------------------------------------------------------------
# Helpers to isolate module state per test
# ---------------------------------------------------------------------------


def _fresh_dh(tmp_path: Path, monkeypatch):
    """Return a freshly-loaded domain_health module with store in tmp_path."""
    store_file = str(tmp_path / "domain_health.json")
    monkeypatch.setenv("DOMAIN_HEALTH_STORE", store_file)
    import app.services.providers.domain_health as dh
    # Reset in-memory state
    dh._store.clear()
    dh._store_loaded = False
    return dh


# ---------------------------------------------------------------------------
# 1. Three BLOCKED on distinct URLs → domain dead
# ---------------------------------------------------------------------------

def test_three_blocked_distinct_urls_mark_dead(tmp_path, monkeypatch):
    dh = _fresh_dh(tmp_path, monkeypatch)
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")
    monkeypatch.setenv("DEAD_DOMAIN_TTL_DAYS", "14")

    host = "citilink.ru"
    assert dh.should_skip_scrappey(host) is False

    dh.record_scrappey_outcome(host, "https://citilink.ru/product/1", "BLOCKED")
    assert dh.should_skip_scrappey(host) is False  # 1 strike, not dead yet

    dh.record_scrappey_outcome(host, "https://citilink.ru/product/2", "BLOCKED")
    assert dh.should_skip_scrappey(host) is False  # 2 strikes

    dh.record_scrappey_outcome(host, "https://citilink.ru/product/3", "BLOCKED")
    assert dh.should_skip_scrappey(host) is True   # 3 strikes → DEAD


# ---------------------------------------------------------------------------
# 2. USABLE resets strikes and revives domain
# ---------------------------------------------------------------------------

def test_usable_resets_strikes_and_revives(tmp_path, monkeypatch):
    dh = _fresh_dh(tmp_path, monkeypatch)
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")

    host = "dns-shop.ru"
    dh.record_scrappey_outcome(host, "https://dns-shop.ru/a", "BLOCKED")
    dh.record_scrappey_outcome(host, "https://dns-shop.ru/b", "BLOCKED")
    dh.record_scrappey_outcome(host, "https://dns-shop.ru/c", "BLOCKED")
    assert dh.should_skip_scrappey(host) is True

    # A single USABLE wipes the death
    dh.record_scrappey_outcome(host, "https://dns-shop.ru/a", "USABLE")
    assert dh.should_skip_scrappey(host) is False
    rec = dh._store.get(dh._registrable_domain(host))
    assert rec is not None
    assert rec.block_strikes == 0
    assert rec.dead_until is None


# ---------------------------------------------------------------------------
# 3. TRANSIENT failures (even 5×) do NOT mark dead
# ---------------------------------------------------------------------------

def test_transient_failures_never_mark_dead(tmp_path, monkeypatch):
    dh = _fresh_dh(tmp_path, monkeypatch)
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")

    host = "some-shop.ru"
    for _ in range(5):
        dh.record_scrappey_outcome(host, "https://some-shop.ru/prod", "TRANSIENT")

    assert dh.should_skip_scrappey(host) is False
    rec = dh._store.get(dh._registrable_domain(host))
    assert rec is not None
    assert rec.block_strikes == 0
    assert rec.transient_fails == 5
    assert rec.dead_until is None


# ---------------------------------------------------------------------------
# 4 & 5. ozon.ru and wildberries.ru are NEVER dead
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("host,url_prefix", [
    ("ozon.ru", "https://ozon.ru/product/"),
    ("www.ozon.ru", "https://www.ozon.ru/product/"),
    ("wildberries.ru", "https://wildberries.ru/catalog/"),
    ("m.wildberries.ru", "https://m.wildberries.ru/catalog/"),
])
def test_never_dead_domains_always_scrappey_eligible(host, url_prefix, tmp_path, monkeypatch):
    dh = _fresh_dh(tmp_path, monkeypatch)
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")

    for i in range(10):
        dh.record_scrappey_outcome(host, f"{url_prefix}{i}", "BLOCKED")

    assert dh.should_skip_scrappey(host) is False


# ---------------------------------------------------------------------------
# 6. TTL expiry → should_skip_scrappey flips back to False
# ---------------------------------------------------------------------------

def test_dead_until_ttl_expiry_allows_reprobe(tmp_path, monkeypatch):
    dh = _fresh_dh(tmp_path, monkeypatch)
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")
    monkeypatch.setenv("DEAD_DOMAIN_TTL_DAYS", "14")

    host = "expired-shop.ru"
    dh.record_scrappey_outcome(host, "https://expired-shop.ru/a", "BLOCKED")
    dh.record_scrappey_outcome(host, "https://expired-shop.ru/b", "BLOCKED")
    dh.record_scrappey_outcome(host, "https://expired-shop.ru/c", "BLOCKED")
    assert dh.should_skip_scrappey(host) is True

    # Manually push dead_until into the past to simulate expiry
    domain = dh._registrable_domain(host)
    rec = dh._store[domain]
    rec.dead_until = time.time() - 1.0  # 1 second ago

    assert dh.should_skip_scrappey(host) is False  # expired → re-probe allowed


# ---------------------------------------------------------------------------
# 7. Same-URL repeated BLOCKED does NOT over-count
# ---------------------------------------------------------------------------

def test_same_url_repeated_blocked_does_not_overcount(tmp_path, monkeypatch):
    dh = _fresh_dh(tmp_path, monkeypatch)
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")

    host = "stubborn-shop.ru"
    url = "https://stubborn-shop.ru/product/42"

    # Same URL 5 times — should only count as 1 strike
    for _ in range(5):
        dh.record_scrappey_outcome(host, url, "BLOCKED")

    domain = dh._registrable_domain(host)
    rec = dh._store[domain]
    assert rec.block_strikes == 1
    assert dh.should_skip_scrappey(host) is False  # not dead yet, only 1 strike


def test_distinct_urls_each_count_as_one_strike(tmp_path, monkeypatch):
    dh = _fresh_dh(tmp_path, monkeypatch)
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")

    host = "multi-url-shop.ru"
    urls = [
        "https://multi-url-shop.ru/prod/1",
        "https://multi-url-shop.ru/prod/2",
        "https://multi-url-shop.ru/prod/3",
    ]
    for url in urls:
        dh.record_scrappey_outcome(host, url, "BLOCKED")

    domain = dh._registrable_domain(host)
    rec = dh._store[domain]
    assert rec.block_strikes == 3
    assert dh.should_skip_scrappey(host) is True


# ---------------------------------------------------------------------------
# 8. Persistence: write store, reload fresh state, records survive
# ---------------------------------------------------------------------------

def test_persistence_survives_reload(tmp_path, monkeypatch):
    store_file = str(tmp_path / "domain_health.json")
    monkeypatch.setenv("DOMAIN_HEALTH_STORE", store_file)
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")

    import app.services.providers.domain_health as dh
    dh._store.clear()
    dh._store_loaded = False

    host = "persist-test.ru"
    dh.record_scrappey_outcome(host, "https://persist-test.ru/a", "BLOCKED")
    dh.record_scrappey_outcome(host, "https://persist-test.ru/b", "BLOCKED")
    dh.record_scrappey_outcome(host, "https://persist-test.ru/c", "BLOCKED")

    assert os.path.exists(store_file)

    # Simulate fresh process: reset module state and reload
    dh._store.clear()
    dh._store_loaded = False

    # Check that reloading recovers the dead record
    assert dh.should_skip_scrappey(host) is True
    domain = dh._registrable_domain(host)
    rec = dh._store.get(domain)
    assert rec is not None
    assert rec.block_strikes == 3
    assert rec.dead_until is not None


def test_persistence_stores_valid_json(tmp_path, monkeypatch):
    store_file = str(tmp_path / "domain_health.json")
    monkeypatch.setenv("DOMAIN_HEALTH_STORE", store_file)

    import app.services.providers.domain_health as dh
    dh._store.clear()
    dh._store_loaded = False

    dh.record_scrappey_outcome("json-test.ru", "https://json-test.ru/p", "TRANSIENT")

    with open(store_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    assert "json-test.ru" in data
    assert "block_strikes" in data["json-test.ru"]
    assert "dead_until" in data["json-test.ru"]


# ---------------------------------------------------------------------------
# 9. url_fetcher integration: dead domain skips scrappey_fetch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetcher_skips_scrappey_on_dead_domain(tmp_path, monkeypatch):
    """A domain marked dead → _try_scrappey_fallback returns None without
    calling scrappey_fetch.  The free plain fetch still runs first."""
    store_file = str(tmp_path / "domain_health.json")
    monkeypatch.setenv("DOMAIN_HEALTH_STORE", store_file)
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_FALLBACK", "1")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_MAX", "200")

    import app.services.providers.domain_health as dh
    dh._store.clear()
    dh._store_loaded = False

    # Mark the domain dead manually
    host = "dead-domain-test.ru"
    dh.record_scrappey_outcome(host, "https://dead-domain-test.ru/a", "BLOCKED")
    dh.record_scrappey_outcome(host, "https://dead-domain-test.ru/b", "BLOCKED")
    dh.record_scrappey_outcome(host, "https://dead-domain-test.ru/c", "BLOCKED")
    assert dh.should_skip_scrappey(host) is True

    import app.services.url_fetcher as uf
    uf._scrappey_call_count = 0
    monkeypatch.setattr(uf, "_CACHE_DIR", str(tmp_path / "cache"))

    scrappey_called = {"n": 0}

    async def _fake_scrappey(url, timeout=120, browser=False):
        scrappey_called["n"] += 1
        return "X" * 1000

    with patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(side_effect=_fake_scrappey)), \
         patch("app.services.url_fetcher.asyncio.sleep", new=AsyncMock()):

        result = await uf._try_scrappey_fallback(
            "https://dead-domain-test.ru/product/1",
            "HTTP 403",
            10,
        )

    assert result is None
    assert scrappey_called["n"] == 0, "scrappey_fetch must NOT be called for dead domain"


@pytest.mark.asyncio
async def test_fetcher_calls_scrappey_on_healthy_domain(tmp_path, monkeypatch):
    """A healthy (non-dead) domain → _try_scrappey_fallback calls scrappey_fetch."""
    store_file = str(tmp_path / "domain_health.json")
    monkeypatch.setenv("DOMAIN_HEALTH_STORE", store_file)
    monkeypatch.setenv("DEAD_DOMAIN_STRIKES", "3")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_FALLBACK", "1")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_MAX", "200")

    import app.services.providers.domain_health as dh
    dh._store.clear()
    dh._store_loaded = False

    import app.services.url_fetcher as uf
    uf._scrappey_call_count = 0
    monkeypatch.setattr(uf, "_CACHE_DIR", str(tmp_path / "cache"))

    scrappey_called = {"n": 0}

    async def _fake_scrappey(url, timeout=120, browser=False):
        scrappey_called["n"] += 1
        return "Good product content. " * 50

    with patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(side_effect=_fake_scrappey)), \
         patch("app.services.url_fetcher.asyncio.sleep", new=AsyncMock()):

        result = await uf._try_scrappey_fallback(
            "https://healthy-domain-test.ru/product/1",
            "HTTP 403",
            10,
        )

    assert result is not None
    assert scrappey_called["n"] == 1, "scrappey_fetch MUST be called for healthy domain"


# ---------------------------------------------------------------------------
# Edge: _registrable_domain folding
# ---------------------------------------------------------------------------

def test_registrable_domain_strips_subdomains():
    import app.services.providers.domain_health as dh
    assert dh._registrable_domain("www.citilink.ru") == "citilink.ru"
    assert dh._registrable_domain("m.wildberries.ru") == "wildberries.ru"
    assert dh._registrable_domain("dns-shop.ru") == "dns-shop.ru"
    assert dh._registrable_domain("example.co.uk") == "example.co.uk"
    assert dh._registrable_domain("sub.example.co.uk") == "example.co.uk"


# ---------------------------------------------------------------------------
# Edge: graceful degradation on corrupt store
# ---------------------------------------------------------------------------

def test_corrupt_store_does_not_crash(tmp_path, monkeypatch):
    store_file = str(tmp_path / "domain_health.json")
    monkeypatch.setenv("DOMAIN_HEALTH_STORE", store_file)

    # Write garbage JSON
    with open(store_file, "w") as fh:
        fh.write("THIS IS NOT JSON !!!")

    import app.services.providers.domain_health as dh
    dh._store.clear()
    dh._store_loaded = False

    # Must not raise — degrade to "not dead"
    assert dh.should_skip_scrappey("some-shop.ru") is False
