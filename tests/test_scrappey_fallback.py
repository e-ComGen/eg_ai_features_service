"""
Tests for the Scrappey paid fallback in app/services/url_fetcher.py.

The fallback retries a 401/403 (anti-bot ban) on a configured domain through
Scrappey's browser-bypass. It is gated by URL_FETCHER_SCRAPPEY_FALLBACK and
URL_FETCHER_SCRAPPEY_DOMAINS. No network: both the HTTP client and
scrappey_fetch are mocked.
"""

import httpx
import pytest

from app.services import url_fetcher


def _make_get_returning(status_code: int):
    """Build a fake httpx.AsyncClient.get coroutine returning a fixed status."""

    async def _fake_get(self, url, *args, **kwargs):
        return httpx.Response(
            status_code=status_code,
            text="blocked" if status_code >= 400 else "ok",
            request=httpx.Request("GET", url),
        )

    return _fake_get


@pytest.fixture
def banned_url():
    return "https://www.dns-shop.ru/product/abc/"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("URL_FETCHER_SCRAPPEY_FALLBACK", raising=False)
    monkeypatch.delenv("URL_FETCHER_SCRAPPEY_DOMAINS", raising=False)
    yield


@pytest.mark.asyncio
async def test_flag_off_403_no_scrappey(monkeypatch, banned_url):
    """Flag OFF → 403 returns None and scrappey_fetch is never called."""
    monkeypatch.setattr(httpx.AsyncClient, "get", _make_get_returning(403))

    called = {"n": 0}

    async def _spy(url, timeout=120.0):
        called["n"] += 1
        return "<html>should not be used</html>"

    monkeypatch.setattr(
        "app.services.providers.scrappey_client.scrappey_fetch", _spy
    )

    resp = await url_fetcher._get_with_retry(banned_url)
    assert resp is None
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_flag_on_domain_in_set_403_uses_scrappey(monkeypatch, banned_url):
    """Flag ON + banned domain + 403 → scrappey_fetch called, its HTML returned."""
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_FALLBACK", "1")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_DOMAINS", "dns-shop.ru,ozon.ru")
    monkeypatch.setattr(httpx.AsyncClient, "get", _make_get_returning(403))

    called = {"n": 0}

    async def _spy(url, timeout=120.0):
        called["n"] += 1
        return "<html>bypassed content</html>"

    monkeypatch.setattr(
        "app.services.providers.scrappey_client.scrappey_fetch", _spy
    )

    resp = await url_fetcher._get_with_retry(banned_url)
    assert called["n"] == 1
    assert resp is not None
    assert resp.status_code == 200
    assert "bypassed content" in resp.text


@pytest.mark.asyncio
async def test_flag_on_domain_not_in_fast_set_403_still_uses_scrappey(monkeypatch):
    """Flag ON + domain NOT in fast set + 403 → scrappey IS called for any non-CDN host.

    The strict domain allowlist was replaced by a content-quality-based trigger:
    any host that is not in the non-text denylist (CDN/tracker/social) is now
    eligible for the Scrappey fallback.  URL_FETCHER_SCRAPPEY_DOMAINS remains
    supported as an optional 'fast-eligible' annotation but no longer gates
    which hosts can ever use the fallback.
    """
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_FALLBACK", "1")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_DOMAINS", "ozon.ru,dns-shop.ru")
    monkeypatch.setattr(httpx.AsyncClient, "get", _make_get_returning(403))

    called = {"n": 0}

    async def _spy(url, timeout=120.0):
        called["n"] += 1
        return "<html>real product content here with lots of text " + "x" * 600 + "</html>"

    monkeypatch.setattr(
        "app.services.providers.scrappey_client.scrappey_fetch", _spy
    )

    resp = await url_fetcher._get_with_retry("https://random-other-site.com/p/1")
    # Scrappey fires for any product domain (not just the fast-set)
    assert called["n"] == 1
    assert resp is not None
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_flag_on_404_no_scrappey(monkeypatch, banned_url):
    """Flag ON + banned domain but 404 → scrappey NOT called (404 is not a ban)."""
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_FALLBACK", "1")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_DOMAINS", "dns-shop.ru")
    monkeypatch.setattr(httpx.AsyncClient, "get", _make_get_returning(404))

    called = {"n": 0}

    async def _spy(url, timeout=120.0):
        called["n"] += 1
        return "<html>nope</html>"

    monkeypatch.setattr(
        "app.services.providers.scrappey_client.scrappey_fetch", _spy
    )

    resp = await url_fetcher._get_with_retry(banned_url)
    assert resp is None
    assert called["n"] == 0
