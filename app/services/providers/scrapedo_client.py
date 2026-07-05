"""app/services/providers/scrapedo_client.py

Async single-page fetch through Scrape.do (https://scrape.do).

PAY-AS-YOU-GO alternative to ZenRows. Like ZenRows, returns the rendered HTML
directly in the response body (no JSON envelope). Reuses ScrapflyResult so
callers need no type changes.

Recipe for RU anti-bot sites (DataDome / SmartCaptcha):
  GET https://api.scrape.do/
  Params: token, url, render=true (headless JS), super=true (residential/mobile
          premium proxy), geoCode=ru, customWait=<ms>
  Success: HTTP 200 AND len(body) >= threshold (~50 000 chars)

Usage:
    from app.services.providers.scrapedo_client import scrapedo_fetch
    result = await scrapedo_fetch("https://www.lamoda.ru/p/...")
    if result.success:
        html = result.content
"""
from __future__ import annotations

import asyncio
import logging
import os
import ssl
import urllib.parse
from typing import Optional

import httpx

from app.services.providers.scrapfly_client import ScrapflyResult

logger = logging.getLogger(__name__)

_SCRAPEDO_ENDPOINT = "https://api.scrape.do/"
_DEFAULT_TIMEOUT = 120.0
_MIN_BODY_LEN = 50_000
_MIN_JSON_BODY_LEN = 200

# Transient HTTP statuses worth retrying (Scrape.do proxy rotation hiccups:
# 502 ROTATION_FAILED "Occasional failures on premium" — explicitly retryable).
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds: 2s, 4s, 8s


async def scrapedo_fetch(
    url: str,
    *,
    render: bool = True,
    super_proxy: bool = True,
    geo: str = "ru",
    wait_ms: int = 5000,
    timeout: float = _DEFAULT_TIMEOUT,
) -> ScrapflyResult:
    """Fetch a URL via Scrape.do with residential proxy + JS rendering.

    Parameters
    ----------
    url: target URL.
    render: when True, Scrape.do uses a headless browser (JS rendering).
    super_proxy: when True, uses Residential & Mobile premium proxies
        (`super=true`) — needed to pierce DataDome / geo-locked RU sites.
    geo: two-letter country code for the proxy exit (`geoCode`). "ru".
    wait_ms: extra wait after page-ready, ms (`customWait`).
    timeout: httpx client timeout in seconds.

    Returns ScrapflyResult (success + content, or success=False + error).
    Cost (if exposed) is read from a Scrape.do response header.
    """
    token = os.environ.get("SCRAPEDO_TOKEN")
    if not token:
        logger.warning("[Scrape.do] SCRAPEDO_TOKEN not set — cannot fetch %s", url[:80])
        return ScrapflyResult(success=False, content=None, status_code=None,
                              credits_used=0, error="SCRAPEDO_TOKEN not configured")

    params: dict = {
        "token": token,
        "url": url,
        "render": "true" if render else "false",
        "super": "true" if super_proxy else "false",
        "geoCode": geo,
    }
    # customWait is rejected by some sites (e.g. dns-shop) under render mode with a
    # misleading "CustomWait can work with Render=True" 400 — send it only when >0 so
    # the on-400 retry below can drop it and still pierce the site.
    if wait_ms:
        params["customWait"] = str(wait_ms)
    query_string = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    request_url = f"{_SCRAPEDO_ENDPOINT}?{query_string}"

    r = None
    credits_used = 0
    last_err = "unknown"
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(request_url)
        except (
            ssl.SSLError, httpx.ConnectError, httpx.RemoteProtocolError,
            httpx.TransportError, httpx.TimeoutException, httpx.HTTPError,
        ) as exc:
            last_err = f"network error: {exc}"
            logger.info("[Scrape.do] %s for %s (attempt %d/%d)",
                        last_err, url[:80], attempt, _MAX_RETRIES)
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE * attempt)
                continue
            return ScrapflyResult(success=False, content=None, status_code=None,
                                  credits_used=0, error=last_err)

        # Scrape.do exposes remaining credits / cost in headers (best-effort).
        raw_cost = (
            r.headers.get("Scrape.do-Request-Cost")
            or r.headers.get("X-Request-Cost")
            or "0"
        )
        try:
            credits_used = round(float(raw_cost))
        except (ValueError, TypeError):
            credits_used = 0

        # Transient proxy/upstream failure (e.g. 502 ROTATION_FAILED) — retry.
        if r.status_code in _TRANSIENT_STATUS:
            last_err = f"Scrape.do HTTP {r.status_code}: {r.text[:200]}"
            logger.info("[Scrape.do] transient HTTP %s for %s (attempt %d/%d)",
                        r.status_code, url[:80], attempt, _MAX_RETRIES)
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE * attempt)
                continue
            return ScrapflyResult(success=False, content=None, status_code=r.status_code,
                                  credits_used=credits_used, error=last_err)

        # Non-transient 4xx — hard fail, no retry.
        if r.status_code >= 400:
            # Some sites reject customWait under render mode ("CustomWait can work
            # with Render=True" 400) — retry ONCE without customWait (render+super
            # alone pierces dns-shop: verified 200/73KB vs 400 with customWait).
            if r.status_code == 400 and wait_ms and "customwait" in r.text.lower():
                logger.info("[Scrape.do] customWait 400 for %s — retrying without customWait",
                            url[:80])
                return await scrapedo_fetch(url, render=render, super_proxy=super_proxy,
                                            geo=geo, wait_ms=0, timeout=timeout)
            logger.warning("[Scrape.do] HTTP %s for %s — body[:300]=%s",
                           r.status_code, url[:80], r.text[:300])
            return ScrapflyResult(success=False, content=None, status_code=r.status_code,
                                  credits_used=credits_used,
                                  error=f"Scrape.do HTTP {r.status_code}: {r.text[:200]}")

        break  # HTTP 200 — proceed to body checks

    content = r.text
    if not content:
        return ScrapflyResult(success=False, content=None, status_code=r.status_code,
                              credits_used=credits_used, error="empty body")

    min_body_len = _MIN_BODY_LEN if render else _MIN_JSON_BODY_LEN
    if len(content) < min_body_len:
        logger.info("[Scrape.do] body too short (%d chars, need %d) for %s",
                    len(content), min_body_len, url[:80])
        return ScrapflyResult(success=False, content=None, status_code=r.status_code,
                              credits_used=credits_used,
                              error=f"body too short ({len(content)} chars)")

    logger.info("[Scrape.do] success (HTTP %s, len=%d) for %s",
                r.status_code, len(content), url[:80])
    return ScrapflyResult(success=True, content=content, status_code=r.status_code,
                          credits_used=credits_used, error=None)
