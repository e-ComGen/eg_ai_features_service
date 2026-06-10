"""app/services/providers/scrapfly_client.py

Async single-page fetch through Scrapfly (https://scrapfly.io).

Scrapfly bypasses Ozon's CDN bot-protection (DataDome, JS fingerprinting) via
ASP (Anti-Scraping Protection) + residential proxy + JS rendering. Unlike
Scrappey, it uses a REST GET endpoint with query params. Cost: 30 credits/call
for ASP+residential+render_js; typically 60 credits/product (search + features).

Usage:
    from app.services.providers.scrapfly_client import scrapfly_fetch

    result = await scrapfly_fetch("https://www.ozon.ru/product/...")
    if result.success:
        html = result.content
        print(f"credits used: {result.credits_used}")
"""
from __future__ import annotations

import logging
import os
import ssl
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_SCRAPFLY_ENDPOINT = "https://api.scrapfly.io/scrape"
_DEFAULT_TIMEOUT = 90.0  # seconds — Scrapfly JS rendering takes 10-30s


@dataclass(frozen=True)
class ScrapflyResult:
    """Result from a single Scrapfly fetch."""

    success: bool
    content: Optional[str]
    status_code: Optional[int]   # upstream HTTP status from the scraped page
    credits_used: int            # Scrapfly credits consumed
    error: Optional[str]         # human-readable error if success=False


def _is_block_content(content: str) -> bool:
    """Return True if the content looks like a WAF / captcha block page."""
    if not content:
        return True
    lower = content[:1000].lower()
    _BLOCK_MARKERS = [
        "incidentid",         # DataDome JSON block
        "captcha",
        "smartcaptcha",
        "access denied",
        "just a moment",      # Cloudflare waiting room
        "cf-challenge",
        "cloudflare",
        "проверка",
        "подтвердите, что вы не робот",
        "are you a robot",
    ]
    return any(m in lower for m in _BLOCK_MARKERS)


async def scrapfly_fetch(
    url: str,
    render_js: bool = True,
    wait_for_selector: Optional[str] = None,
    country: str = "ru",
    proxy_pool: str = "public_residential_pool",
    timeout: float = _DEFAULT_TIMEOUT,
) -> ScrapflyResult:
    """Fetch a URL via Scrapfly with ASP bypass.

    Parameters
    ----------
    url:
        Target URL to fetch.
    render_js:
        When True, Scrapfly spins a full browser for JS rendering.
        Required for Ozon which uses Vue/SSR hydration.
    wait_for_selector:
        Optional CSS selector to wait for before capturing snapshot.
        Useful for waiting for characteristics widget to fully hydrate.
        If None, Scrapfly uses its default page-ready heuristic.
    country:
        Two-letter country code for proxy exit node. "ru" for Ozon.
    proxy_pool:
        Scrapfly proxy pool name.
    timeout:
        httpx client-level timeout in seconds.

    Returns
    -------
    ScrapflyResult with success=True and content set, or success=False with
    error message and credits_used=0 on failure.
    """
    api_key = os.environ.get("SCRAPFLY_API_KEY") or os.environ.get("SCRAPFLY_KEY")
    if not api_key:
        logger.warning("[Scrapfly] SCRAPFLY_API_KEY not set — cannot fetch %s", url[:80])
        return ScrapflyResult(
            success=False,
            content=None,
            status_code=None,
            credits_used=0,
            error="SCRAPFLY_API_KEY not configured",
        )

    # Build the Scrapfly request URL manually to avoid httpx double-encoding the
    # target `url` param.  httpx's params= encodes values with quote_plus(), turning
    # spaces into `+`.  Scrapfly strictly validates the target URL and rejects any URL
    # that contains `+` (decoded as space) — returning HTTP 422 "invalid URL".
    # Using urllib.parse.urlencode with quote_via=quote produces %20 for spaces and
    # does NOT re-encode already-percent-encoded characters.
    params: dict = {
        "key": api_key,
        "url": url,
        "asp": "true",
        "render_js": "true" if render_js else "false",
        "country": country,
        "proxy_pool": proxy_pool,
        "retry": "false",   # we handle retries at call site
    }
    if wait_for_selector:
        params["wait_for_selector"] = wait_for_selector

    # Encode all params with %XX (not +) so the nested url value stays a valid URL.
    query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    request_url = f"{_SCRAPFLY_ENDPOINT}?{query_string}"

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(request_url)
    except (
        ssl.SSLError,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.TransportError,
        httpx.TimeoutException,
        httpx.HTTPError,
    ) as exc:
        logger.info("[Scrapfly] network/SSL err for %s: %s", url[:80], exc)
        return ScrapflyResult(
            success=False,
            content=None,
            status_code=None,
            credits_used=0,
            error=f"network error: {exc}",
        )

    if r.status_code >= 400:
        # Log the full body (up to 500 chars) so the exact Scrapfly error code
        # (e.g. ERR::SCRAPE::WAIT_FOR_SELECTOR_TIMEOUT) is visible in logs.
        logger.warning(
            "[Scrapfly] API HTTP %s for %s — body[:500]=%s",
            r.status_code, url[:80], r.text[:500],
        )
        return ScrapflyResult(
            success=False,
            content=None,
            status_code=r.status_code,
            credits_used=0,
            error=f"Scrapfly API HTTP {r.status_code}: {r.text[:200]}",
        )

    try:
        envelope = r.json()
    except Exception:
        logger.info("[Scrapfly] non-JSON envelope for %s", url[:80])
        return ScrapflyResult(
            success=False,
            content=None,
            status_code=r.status_code,
            credits_used=0,
            error="non-JSON envelope",
        )

    # Scrapfly envelope structure:
    # {
    #   "result": {
    #     "status": "DONE",
    #     "status_code": 200,       # upstream HTTP status
    #     "content": "<html>...",   # rendered page HTML
    #     "cost": 30,               # credits consumed
    #     "error": null,
    #   }
    # }
    result_obj = envelope.get("result") or {}
    content = result_obj.get("content") or ""
    upstream_status = result_obj.get("status_code")
    credits_used = int(result_obj.get("cost") or 0)
    scrapfly_status = result_obj.get("status") or ""
    scrapfly_error = result_obj.get("error")

    if scrapfly_status not in ("DONE", "") and scrapfly_error:
        logger.warning(
            "[Scrapfly] status=%s error=%s for %s (credits_used=%d)",
            scrapfly_status, scrapfly_error, url[:80], credits_used,
        )
        return ScrapflyResult(
            success=False,
            content=None,
            status_code=upstream_status,
            credits_used=credits_used,
            error=f"Scrapfly status={scrapfly_status}: {scrapfly_error}",
        )

    if not content:
        logger.info(
            "[Scrapfly] empty content (upstream=%s) for %s",
            upstream_status, url[:80],
        )
        return ScrapflyResult(
            success=False,
            content=None,
            status_code=upstream_status,
            credits_used=credits_used,
            error="empty content",
        )

    if _is_block_content(content):
        logger.info(
            "[Scrapfly] block/captcha page detected (upstream=%s) for %s",
            upstream_status, url[:80],
        )
        return ScrapflyResult(
            success=False,
            content=None,
            status_code=upstream_status,
            credits_used=credits_used,
            error="block/captcha page",
        )

    logger.info(
        "[Scrapfly] success (upstream=%s, len=%d, credits=%d) for %s",
        upstream_status, len(content), credits_used, url[:80],
    )
    return ScrapflyResult(
        success=True,
        content=content,
        status_code=upstream_status,
        credits_used=credits_used,
        error=None,
    )
