"""
tests/test_scrappey_client.py

Unit tests for app/services/providers/scrappey_client.scrappey_fetch.
All HTTP calls are mocked — no live network traffic.

Key regressions / features covered:
  1. statusCode=None + real HTML → accepted (statusCode omitted by Scrappey
     for many Russian retail sites even when content is valid).
  2. browser=True → payload contains "requestType":"browser".
  3. Relaxed status-code guard: non-200 upstream code is accepted when the body
     is real content (not a block page).  We only reject empty or block bodies.
  4. Block / DataDome content → rejected regardless of statusCode.
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
# Bare mode — basic pass / fail
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


# ---------------------------------------------------------------------------
# Relaxed status-code guard: accept non-200 if body is real (not block)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_accepts_non200_status_when_body_is_real_content(monkeypatch):
    """
    Relaxed guard: upstream 407 with real HTML body → content is returned.
    The old strict guard rejected ANY non-200; the new guard only rejects
    when the body itself looks like a block page.
    """
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    real_html = "<html><body>Характеристики товара: цвет синий, размер 42</body></html>"
    mock_resp = _make_scrappey_response(upstream_status=407, body_html=real_html)

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://www.dns-shop.ru")

    assert result is not None, (
        "Non-200 upstream with real HTML body should be accepted under the relaxed guard"
    )
    assert "Характеристики" in result


@pytest.mark.asyncio
async def test_rejects_non200_status_when_body_is_block_page(monkeypatch):
    """Non-200 + block body → still rejected (block content wins)."""
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    block_html = "<html><body>Just a moment... cloudflare checking your browser</body></html>"
    mock_resp = _make_scrappey_response(upstream_status=407, body_html=block_html)

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://www.citilink.ru")

    assert result is None


# ---------------------------------------------------------------------------
# Browser mode — requestType injection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_browser_mode_injects_request_type(monkeypatch):
    """browser=True must add 'requestType':'browser' to the Scrappey payload."""
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    real_html = "<html><body>DNS-Shop product Характеристики ₽</body></html>"
    mock_resp = _make_scrappey_response(upstream_status=None, body_html=real_html)

    with _patch_httpx(mock_resp) as mock_client:
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://www.dns-shop.ru", browser=True)

    assert result is not None
    # Verify the payload sent to Scrappey contained requestType=browser
    call_args = mock_client.post.call_args
    sent_payload = call_args.kwargs.get("json") or call_args.args[1] if call_args.args else {}
    if not sent_payload and call_args.kwargs:
        sent_payload = call_args.kwargs.get("json", {})
    assert sent_payload.get("requestType") == "browser", (
        f"Expected requestType='browser' in payload, got: {sent_payload}"
    )


@pytest.mark.asyncio
async def test_bare_mode_does_not_inject_request_type(monkeypatch):
    """browser=False (default) must NOT add requestType to the payload."""
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    mock_resp = _make_scrappey_response(upstream_status=200, body_html="<html>ok</html>")

    with _patch_httpx(mock_resp) as mock_client:
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://example.com", browser=False)

    call_args = mock_client.post.call_args
    sent_payload = call_args.kwargs.get("json") or {}
    assert "requestType" not in sent_payload, (
        f"bare mode must not include requestType, got: {sent_payload}"
    )


@pytest.mark.asyncio
async def test_browser_mode_accepts_real_html_with_none_status(monkeypatch):
    """browser=True + statusCode=None + real HTML → returned (Qrator success scenario)."""
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    real_html = (
        "<html><body>Смартфон Samsung Galaxy S24 "
        "Характеристики: RAM 8 ГБ, ROM 256 ГБ, цена 89990 ₽"
        "</body></html>"
    )
    mock_resp = _make_scrappey_response(upstream_status=None, body_html=real_html)

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://www.dns-shop.ru/product/abc/", browser=True)

    assert result is not None
    assert "Характеристики" in result


# ---------------------------------------------------------------------------
# Common failure paths (both modes)
# ---------------------------------------------------------------------------

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
async def test_cloudflare_block_returns_none(monkeypatch):
    """Generic Cloudflare block page → None regardless of mode."""
    monkeypatch.setenv("SCRAPPEY_KEY", "testkey")
    cf_html = "<html><head><title>Just a moment...</title></head><body>cloudflare</body></html>"
    mock_resp = _make_scrappey_response(upstream_status=200, body_html=cf_html)

    with _patch_httpx(mock_resp):
        import importlib
        import app.services.providers.scrappey_client as mod
        importlib.reload(mod)
        result = await mod.scrappey_fetch("https://www.citilink.ru", browser=True)

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
