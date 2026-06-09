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
read envelope.solution.response, treat empty / non-200 upstream / DataDome
challenge as failure (return None).
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

    DataDome returns JSON `{"incidentId":"...","blockURL":"..."}` (or
    `supportURL`) instead of the real page. A valid page never contains
    `incidentId` in its head, so checking the first 500 chars is enough.
    """
    if not content:
        return False
    return "incidentId" in content[:500]


async def scrappey_fetch(url: str, timeout: float = 120.0) -> Optional[str]:
    """Fetch a single page's HTML through Scrappey's browser bypass.

    Returns the upstream HTML (solution.response) on success, or None on any
    failure: missing SCRAPPEY_KEY, network/SSL error, Scrappey HTTP >= 400,
    non-json envelope, empty content, non-200 upstream status, or a DataDome
    challenge.

    No retry here — callers decide whether to retry. This is a thin, cheap
    wrapper so it can be mocked easily in tests.
    """
    key = os.environ.get("SCRAPPEY_KEY")
    if not key:
        logger.warning("[Scrappey] SCRAPPEY_KEY not set — cannot fallback-fetch %s", url[:80])
        return None

    payload = {"cmd": "request.get", "url": url}
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

    # Scrappey omits statusCode (returns None) for sites that don't echo it back
    # but still deliver real HTML (verified=True).  Treat None as "don't know /
    # likely 200" — only reject explicit non-200 codes (e.g. 301, 403, 407).
    if upstream_status is not None and upstream_status != 200:
        logger.info("[Scrappey] upstream HTTP %s for %s — likely block", upstream_status, url[:80])
        return None

    if _is_datadome_block(content):
        logger.info("[Scrappey] DataDome challenge in response for %s", url[:80])
        return None

    return content
