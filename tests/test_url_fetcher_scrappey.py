"""
tests/test_url_fetcher_scrappey.py

Unit tests for the content-aware Scrappey fallback in url_fetcher.py.

All tests are offline — httpx calls and scrappey_fetch are mocked.
The Scrappey fallback flag is enabled for all tests via monkeypatching
the env-checking function directly.

NOTE: Every test that calls fetch_url_content must redirect _CACHE_DIR to a
fresh tmp_path so a stale on-disk cache cannot short-circuit the mock.
"""
import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import app.services.url_fetcher as uf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_resp(status: int, body: str = "") -> MagicMock:
    """Construct a minimal httpx-like response mock."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.text = body
    resp.headers = {}
    if status < 400:
        resp.raise_for_status = MagicMock()
    else:
        resp.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                f"HTTP {status}",
                request=httpx.Request("GET", "https://example.com"),
                response=resp,
            )
        )
    return resp


def _make_async_client(responses):
    """
    Build a mock httpx.AsyncClient where .get() returns responses in order.
    responses: list of httpx.Response-like or Exception instances.
    """
    call_idx = {"i": 0}

    async def mock_get(url, **kwargs):
        idx = call_idx["i"]
        call_idx["i"] += 1
        item = responses[min(idx, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = mock_get
    return mock_client


GOOD_BODY = "X" * 1000   # clearly above the 500-char threshold
BLOCK_BODY = "<html>Just a moment...</html>"
SHORT_BODY = "hi"         # below 500-char threshold


# ---------------------------------------------------------------------------
# _looks_like_block unit tests
# ---------------------------------------------------------------------------

def test_looks_like_block_detects_captcha():
    assert uf._looks_like_block("Please solve the captcha to continue") is True


def test_looks_like_block_detects_cloudflare():
    assert uf._looks_like_block("cf-challenge detected, cloudflare is checking") is True


def test_looks_like_block_detects_datadome():
    assert uf._looks_like_block("datadome protection active") is True


def test_looks_like_block_detects_russian_markers():
    assert uf._looks_like_block("Подтвердите, что вы не робот") is True
    assert uf._looks_like_block("проверка безопасности") is True


def test_looks_like_block_passes_normal_content():
    assert uf._looks_like_block("Кроссовки Nike Air Max 270, размер 42, цвет белый") is False


def test_looks_like_block_empty_string_is_block():
    assert uf._looks_like_block("") is True


# ---------------------------------------------------------------------------
# _is_nontext_host unit tests
# ---------------------------------------------------------------------------

def test_is_nontext_host_cdn():
    assert uf._is_nontext_host("https://cdn.example.com/image.png") is True


def test_is_nontext_host_gtm():
    assert uf._is_nontext_host("https://www.googletagmanager.com/gtm.js") is True


def test_is_nontext_host_facebook():
    assert uf._is_nontext_host("https://www.facebook.com/share/p/123") is True


def test_is_nontext_host_doubleclick():
    assert uf._is_nontext_host("https://ad.doubleclick.net/click") is True


def test_is_nontext_host_product_domain_allowed():
    assert uf._is_nontext_host("https://sportmaster.ru/product/123") is False
    assert uf._is_nontext_host("https://lamoda.ru/p/abc/") is False
    assert uf._is_nontext_host("https://dns-shop.ru/product/xyz") is False


# ---------------------------------------------------------------------------
# Fixtures: reset global counter and patch env/sleep
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_scrappey_counter():
    """Reset the per-process Scrappey call counter before each test."""
    uf._scrappey_call_count = 0
    yield
    uf._scrappey_call_count = 0


@pytest.fixture()
def scrappey_env(monkeypatch, tmp_path):
    """Enable Scrappey fallback, patch sleep, redirect disk cache to tmp_path."""
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_FALLBACK", "1")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_MAX", "200")
    monkeypatch.setattr(uf, "_CACHE_DIR", str(tmp_path))
    with patch("app.services.url_fetcher.asyncio.sleep", new=AsyncMock()):
        yield


@pytest.fixture()
def scrappey_disabled(monkeypatch, tmp_path):
    """Disable Scrappey fallback, redirect disk cache to tmp_path."""
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_FALLBACK", "0")
    monkeypatch.setattr(uf, "_CACHE_DIR", str(tmp_path))
    with patch("app.services.url_fetcher.asyncio.sleep", new=AsyncMock()):
        yield


# ---------------------------------------------------------------------------
# a. Ban status 403 fires Scrappey
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrappey_fires_on_403(scrappey_env):
    scrappey_html = "X" * 1000

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(return_value=scrappey_html)) as mock_sf:

        mock_cls.return_value = _make_async_client([_make_resp(403)])
        result = await uf.fetch_url_content("https://sportmaster.ru/product/scrappey-test-403")

    mock_sf.assert_awaited_once()
    assert result is not None
    assert len(result.content) > 0
    assert uf._scrappey_call_count == 1


@pytest.mark.asyncio
async def test_scrappey_fires_on_418(scrappey_env):
    scrappey_html = "X" * 1000

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(return_value=scrappey_html)) as mock_sf:

        mock_cls.return_value = _make_async_client([_make_resp(418)])
        result = await uf.fetch_url_content("https://lamoda.ru/p/scrappey-test-418/")

    mock_sf.assert_awaited_once()
    assert result is not None
    assert uf._scrappey_call_count == 1


# ---------------------------------------------------------------------------
# d. 200 too-short body fires Scrappey
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrappey_fires_on_short_200(scrappey_env):
    scrappey_html = "Full product description. " * 40  # > 500 chars

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(return_value=scrappey_html)) as mock_sf:

        mock_cls.return_value = _make_async_client([_make_resp(200, SHORT_BODY)])
        result = await uf.fetch_url_content("https://some-shop.ru/scrappey-test-short")

    mock_sf.assert_awaited_once()
    assert uf._scrappey_call_count == 1


# ---------------------------------------------------------------------------
# d. 200 with captcha marker fires Scrappey
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrappey_fires_on_block_marker_200(scrappey_env):
    scrappey_html = "Real product page content. " * 40

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(return_value=scrappey_html)) as mock_sf:

        mock_cls.return_value = _make_async_client([_make_resp(200, BLOCK_BODY)])
        result = await uf.fetch_url_content("https://some-shop.ru/scrappey-test-block")

    mock_sf.assert_awaited_once()
    assert uf._scrappey_call_count == 1


# ---------------------------------------------------------------------------
# c. Redirect to block page (200 with block body after follow_redirects)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrappey_fires_on_redirect_to_block(scrappey_env):
    """A 200 whose body is a captcha wall (after redirect) triggers Scrappey."""
    block_body = "<html><body>Are you a robot? captcha required</body></html>"
    scrappey_html = "Product: Nike Air Max. " * 50

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(return_value=scrappey_html)) as mock_sf:

        mock_cls.return_value = _make_async_client([_make_resp(200, block_body)])
        result = await uf.fetch_url_content("https://market.yandex.ru/scrappey-test-redir")

    mock_sf.assert_awaited_once()
    assert uf._scrappey_call_count == 1


# ---------------------------------------------------------------------------
# No Scrappey on healthy 200 with good content
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_scrappey_on_healthy_200(scrappey_env):
    """A 200 with sufficient non-block content must NOT trigger Scrappey."""
    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock()) as mock_sf:

        mock_cls.return_value = _make_async_client([_make_resp(200, GOOD_BODY)])
        result = await uf.fetch_url_content("https://some-shop.ru/scrappey-test-good")

    mock_sf.assert_not_awaited()
    assert uf._scrappey_call_count == 0


# ---------------------------------------------------------------------------
# No Scrappey on denylisted (non-text) host
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_no_scrappey_on_denylisted_host(scrappey_env):
    """CDN/tracker domains must NEVER trigger Scrappey."""
    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock()) as mock_sf:

        mock_cls.return_value = _make_async_client([_make_resp(403)])
        result = await uf.fetch_url_content("https://cdn.example.com/asset.js")

    mock_sf.assert_not_awaited()
    assert uf._scrappey_call_count == 0


# ---------------------------------------------------------------------------
# b. Transient errors: Scrappey only AFTER retries exhausted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrappey_not_before_transient_retries_exhausted(scrappey_env):
    """On 429/503 Scrappey must NOT fire until all retry attempts are spent."""
    call_log = []

    async def _fake_scrappey(url, timeout=120):
        call_log.append("scrappey")
        return "Full product text. " * 60

    # Simulate: 2 transient 503s then success
    responses = [_make_resp(503), _make_resp(503), _make_resp(200, GOOD_BODY)]

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(side_effect=_fake_scrappey)):

        mock_cls.return_value = _make_async_client(responses)
        result = await uf.fetch_url_content("https://some-shop.ru/scrappey-test-transient")

    # Success on 3rd attempt — Scrappey must NOT have fired
    assert "scrappey" not in call_log
    assert uf._scrappey_call_count == 0


@pytest.mark.asyncio
async def test_scrappey_fires_after_all_transient_retries_exhausted(scrappey_env):
    """After all retries exhausted on transient status, Scrappey fires."""
    scrappey_html = "Product info page. " * 60

    # All 3 attempts return 503
    responses = [_make_resp(503), _make_resp(503), _make_resp(503)]

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(return_value=scrappey_html)) as mock_sf:

        mock_cls.return_value = _make_async_client(responses)
        result = await uf.fetch_url_content("https://some-shop.ru/scrappey-test-transient-exhaust")

    mock_sf.assert_awaited_once()
    assert uf._scrappey_call_count == 1


# ---------------------------------------------------------------------------
# Per-process cap stops further calls
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_process_cap_stops_scrappey(monkeypatch, tmp_path):
    """When the cap is reached, Scrappey must NOT fire even for a banned domain."""
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_FALLBACK", "1")
    monkeypatch.setenv("URL_FETCHER_SCRAPPEY_MAX", "2")
    monkeypatch.setattr(uf, "_CACHE_DIR", str(tmp_path))
    uf._scrappey_call_count = 2  # already at cap

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(return_value="html")) as mock_sf, \
         patch("app.services.url_fetcher.asyncio.sleep", new=AsyncMock()):

        mock_cls.return_value = _make_async_client([_make_resp(403)])
        result = await uf.fetch_url_content("https://sportmaster.ru/scrappey-test-cap")

    mock_sf.assert_not_awaited()
    assert uf._scrappey_call_count == 2


# ---------------------------------------------------------------------------
# Fail-closed: Scrappey block result is rejected (not surfaced as content)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrappey_block_result_is_rejected(scrappey_env):
    """If Scrappey itself returns a block/captcha page, return None (fail-closed)."""
    captcha_html = "<html>Just a moment... cloudflare checking</html>"

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(return_value=captcha_html)):

        mock_cls.return_value = _make_async_client([_make_resp(403)])
        result = await uf.fetch_url_content("https://some-shop.ru/scrappey-test-block-result")

    # Must not return the captcha page as content
    assert result is None


# ---------------------------------------------------------------------------
# Scrappey disabled by flag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scrappey_not_called_when_disabled(scrappey_disabled):
    """With URL_FETCHER_SCRAPPEY_FALLBACK=0, Scrappey must never be called."""
    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock()) as mock_sf:

        mock_cls.return_value = _make_async_client([_make_resp(403)])
        result = await uf.fetch_url_content("https://some-shop.ru/scrappey-test-disabled")

    mock_sf.assert_not_awaited()
    assert uf._scrappey_call_count == 0


# ---------------------------------------------------------------------------
# At-most-one Scrappey call per URL
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_at_most_one_scrappey_call_per_url(scrappey_env):
    """Scrappey must be called at most ONCE per URL (no retry on Scrappey)."""
    call_count = {"n": 0}

    async def _fake_scrappey(url, timeout=120):
        call_count["n"] += 1
        return None  # force failure each time

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_cls, \
         patch("app.services.providers.scrappey_client.scrappey_fetch",
               new=AsyncMock(side_effect=_fake_scrappey)):

        mock_cls.return_value = _make_async_client([_make_resp(403)])
        result = await uf.fetch_url_content("https://some-shop.ru/scrappey-test-once")

    assert call_count["n"] == 1
    assert result is None
