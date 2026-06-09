"""
tests/test_scrappey_client.py

Unit tests for app/services/providers/scrappey_client.scrappey_fetch.
All HTTP calls are mocked — no live network traffic.

Key regression covered:
  Scrappey returns solution.statusCode = None (not 200) for many Russian
  retail sites (lamoda.ru, dns-shop.ru, sportmaster.ru) even when it
  successfully fetched real HTML.  The old guard `upstream_status != 200`
  silently dropped real content because None != 200 is True.
  After the fix, None is treated as "status unknown / likely OK" and
  accepted so long as content is non-empty and not a DataDome block.
"""

import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scrappey_response(
    scrappey_http_status: int = 200,
    upstream_status: Optional[int] = 200,
    body_html: str = "<html><body>product page</body></html>",
    verified: bool = True,
) -> MagicMock:
    solution: dict = {"verified": verified, "response": body_html}
    if upstream_status is not None:
        solution["statusCode"] = upstream_status

    mock_resp = MagicMock()
    mock_resp.status_code = scrappey_http_status
    mock_resp.json.return_value = {"solution": solution}
    mock_resp.text = json.dumps({"solution": solution})
    return mock_resp


def _patch_httpx(mock_resp: MagicMock):
    """Context-manager factory: patches httpx.AsyncClient to return mock_resp."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_resp)

    class _ctx:
        def __enter__(self_):
            self_.patcher = patch("httpx.AsyncClient", return_value=mock_client)
            self_.patcher.start()
            return mock_client

        def __exit__(self_, *args):
            self_.patcher.stop()

    return _ctx()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_returns_html_when_upstream_status_200(monkeypatch):
    """Classic: Scrappey returns statusCode=200 — should pass through HTML."""
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    mock_resp = _make_scrappey_response(upstream_status=200, body_html="<html>ok</html>")

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://example.com")

    assert result == "<html>ok</html>"


@pytest.mark.asyncio
async def test_returns_html_when_upstream_status_none(monkeypatch):
    """
    Regression: statusCode absent (None) + verified=True + real HTML.
    Scrappey omits statusCode for lamoda.ru, dns-shop.ru, sportmaster.ru.
    Must return the HTML, not None.
    """
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    real_html = "<html><body>" + "product content " * 50 + "</body></html>"
    mock_resp = _make_scrappey_response(upstream_status=None, body_html=real_html)

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://www.lamoda.ru")

    assert result is not None, (
        "statusCode=None was incorrectly treated as non-200 and dropped real HTML"
    )
    assert len(result) > 100


@pytest.mark.asyncio
async def test_rejects_explicit_upstream_407(monkeypatch):
    """Explicit upstream 407 (proxy auth required) must still be rejected."""
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    mock_resp = _make_scrappey_response(upstream_status=407, body_html="proxy error body")

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://www.sportmaster.ru")

    assert result is None


@pytest.mark.asyncio
async def test_rejects_explicit_upstream_301(monkeypatch):
    """Explicit upstream 301 (redirect) must still be rejected."""
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    mock_resp = _make_scrappey_response(upstream_status=301, body_html="<html>redirect</html>")

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://market.yandex.ru")

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_without_key(monkeypatch):
    """No SCRAPPEY_KEY → return None immediately, no HTTP call."""
    monkeypatch.delenv("SCRAPPEY_KEY", raising=False)

    import importlib
    import app.services.providers.scrappey_client as mod
    importlib.reload(mod)
    result = await mod.scrappey_fetch("https://example.com")

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_on_scrappey_http_4xx(monkeypatch):
    """Scrappey API itself returns 4xx (bad key / no credits) → None."""
    monkeypatch.setenv("SCRAPPEY_KEY", "badkey")
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = '{"error": "Invalid API key"}'

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://example.com")

    assert result is None


@pytest.mark.asyncio
async def test_datadome_block_returns_none(monkeypatch):
    """DataDome challenge JSON in response body → None, even with statusCode=None."""
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    datadome_html = '{"incidentId":"abc123","blockURL":"https://geo.captcha-delivery.com/captcha/"}'
    mock_resp = _make_scrappey_response(upstream_status=None, body_html=datadome_html)

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://www.dns-shop.ru")

    assert result is None


@pytest.mark.asyncio
async def test_empty_response_returns_none(monkeypatch):
    """Empty solution.response → None regardless of statusCode."""
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    mock_resp = _make_scrappey_response(upstream_status=None, body_html="")

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://www.lamoda.ru")

    assert result is None
