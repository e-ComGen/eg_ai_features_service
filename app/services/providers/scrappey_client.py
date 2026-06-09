"""
app/services/providers/scrappey_client.py

Reusable single-page fetch through Scrappey (paid browser-bypass).

Scrappey runs a real browser to bypass anti-bot challenges (DataDome, Cloudflare,
401/403 WAF blocks) on banned domains such as Ozon, DNS-Shop, Wildberries,
Citilink. We already use it for live Ozon card scraping
(app/services/enrichment/sources/ozon_card_source.py); this module exposes the
single-page POST as a standalone helper so other fetchers (e.g. the generic
url_fetcher fallback) can reuse it without depending on OzonCardSource.

The logic mirrors OzonCardSource._scrappey_fetch_once: POST cmd=request.get,
read envelope.solution.response, treat empty / block-page content as failure
(return None).

Browser mode (browser=True)
----------------------------
Passes ``"requestType":"browser"`` to Scrappey, which spins up a full Chromium
instance with JS rendering.  This beats Qrator WAF and Cloudflare JS challenges
that the bare ``request.get`` mode cannot handle (dns-shop.ru, citilink.ru,
sportmaster.ru).  Browser mode is slower (~15–35 s vs ~3–8 s bare), so callers
should use a higher timeout (see URL_FETCHER_SCRAPPEY_BROWSER_TIMEOUT, default
40 s).

Status-code guard (relaxed vs bare mode)
-----------------------------------------
Scrappey frequently omits ``statusCode`` (returns None) even when it fetches a
real page via browser mode.  For both modes we now accept the response whenever
content is non-empty AND does NOT look like a block page — we only reject
explicit non-200 codes together with a block/empty body.  A real 410 or redirect
page that Scrappey rendered as actual HTML will pass through.
"""

import json
import logging
import os
import ssl
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_SCRAPPEY_ENDPOINT = "https://publisher.scrappey.com/api/v1"


def _is_datadome_block(content: str) -> bool:
    """Detect a DataDome challenge in the response body.

    DataDome returns JSON ``{"incidentId":"...","blockURL":"..."}`` (or
    ``supportURL``) instead of the real page.  A valid page never contains
    ``incidentId`` in its head, so checking the first 500 chars is enough.
    """
    if not content:
        return False
    return "incidentId" in content[:500]


def _is_block_content(content: str) -> bool:
    """Return True when content looks like a WAF / captcha block page.

    Combines DataDome detection with a small set of generic block markers so
    the caller does not need to import url_fetcher._looks_like_block (avoids a
    circular dependency).
    """
    if not content:
        return True
    if _is_datadome_block(content):
        return True
    lower = content.lower()
    _GENERIC_BLOCK = [
        "captcha",
        "smartcaptcha",
        "access denied",
        "just a moment",
        "cf-challenge",
        "cloudflare",
        "проверка",
        "подтвердите, что вы не робот",
        "are you a robot",
    ]
    return any(m in lower for m in _GENERIC_BLOCK)


async def scrappey_fetch(
    url: str,
    timeout: float = 120.0,
    browser: bool = False,
) -> Optional[str]:
    """Fetch a single page's HTML through Scrappey's browser bypass.

    Parameters
    ----------
    url:
        Target URL to fetch.
    timeout:
        httpx client timeout in seconds.  For browser mode callers should pass
        a higher value (40 s) because JS rendering takes longer.
    browser:
        When True, injects ``"requestType":"browser"`` into the Scrappey
        payload.  Chromium renders the page, executing JS and solving WAF
        JS challenges (Qrator, Cloudflare).  Use this when bare mode fails.

    Returns the upstream HTML (solution.response) on success, or None on any
    failure: missing SCRAPPEY_KEY, network/SSL error, Scrappey HTTP >= 400,
    non-json envelope, empty / block content.

    No retry here — callers decide whether to retry. This is a thin wrapper so
    it can be mocked easily in tests.
    """
    key = os.environ.get("SCRAPPEY_KEY")
    if not key:
        logger.warning("[Scrappey] SCRAPPEY_KEY not set — cannot fallback-fetch %s", url[:80])
        return None

    payload: dict = {"cmd": "request.get", "url": url}
    if browser:
        payload["requestType"] = "browser"
        logger.debug("[Scrappey] using browser mode for %s", url[:80])

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                _SCRAPPEY_ENDPOINT,
                params={"key": key},
                json=payload,
                headers={"Content-Type": "application/json"},
            )
    except (
        ssl.SSLError,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.TransportError,
        httpx.TimeoutException,
        httpx.HTTPError,
    ) as exc:
        logger.info("[Scrappey] network/SSL err (transient) for %s: %s", url[:80], exc)
        return None

    if r.status_code >= 400:
        logger.warning(
            "[Scrappey] HTTP %s for %s — skip (body[:200]=%s)",
            r.status_code, url[:80], r.text[:200],
        )
        return None

    try:
        envelope = r.json()
    except (ValueError, json.JSONDecodeError):
        logger.info("[Scrappey] non-json envelope for %s", url[:80])
        return None

    solution = envelope.get("solution") or {}
    upstream_status = solution.get("statusCode")
    content = solution.get("response") or ""

    if not content:
        logger.info("[Scrappey] empty content (upstream=%s) for %s", upstream_status, url[:80])
        return None

    # Relaxed status-code guard:
    # Accept the response whenever content is non-empty AND not a block page —
    # regardless of upstream_status.  Scrappey often omits statusCode (None)
    # even for real pages (verified=True), and browser mode can legitimately
    # render pages that return 4xx codes (e.g. a product page that also sets
    # HTTP 410 for SEO reasons).  We only reject when the body itself looks
    # like a block/WAF wall.
    if _is_block_content(content):
        logger.info(
            "[Scrappey] block/captcha content detected (upstream=%s) for %s",
            upstream_status, url[:80],
        )
        return None

    mode_tag = "browser" if browser else "bare"
    logger.info(
        "[Scrappey] %s mode success (upstream=%s, len=%d) for %s",
        mode_tag, upstream_status, len(content), url[:80],
    )
    return content
