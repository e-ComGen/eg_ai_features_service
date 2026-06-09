"""Lamoda fabric-composition miner — prototype.

Pipeline
--------
1. Serper `site:lamoda.ru <brand> <product_name>` → top product URLs.
2. BrowserFetcher.fetch_with_warmup() — hits Lamoda homepage first to let
   DataDome cookie accumulate (mirrors the "lamoda.ru → then product" manual
   flow the owner confirmed works), then navigates to the product URL.
3. composition_extractor.extract_composition() → composition strings.
4. page_matches_brand() gate → fail-closed brand verification.
5. Return first hit: {composition, source_url, evidence}.

Integration seam
----------------
This module is self-contained; it does NOT touch the production source list yet.
To wire it in, add a call to `mine_lamoda_composition()` inside the existing
`WebSearchProducer.mine_composition()` fallback chain (or as a parallel source
in `PipelineOrchestrator`).

DataDome note
-------------
Real chrome.exe (--headless=new + real TLS fingerprint) defeats DataDome from
this VPS.  Bundled Playwright Chromium would be blocked.  BrowserFetcher is
responsible for ensuring the real binary is used.

Warmup strategy (owner-confirmed):
  The owner opened a Lamoda product page from THIS IP — it worked "со скрипом"
  (slowly) ONLY after hitting the homepage first.  Cold product-URL hits trigger
  the DataDome block stub ("Доступ ограничен").  We mirror that by:
  1. Navigating to https://www.lamoda.ru/ and waiting up to 60 s / 3 retries
     for the homepage to show real catalog content (DataDome cookie sets itself
     automatically via JS challenge execution in real headless Chrome).
  2. Once the cookie is set, navigating to the product URL in the SAME context.
  Session reuse means subsequent product fetches skip the homepage warmup.

Performance
-----------
Cold-start (first Lamoda fetch, with warmup): ~20-60 s depending on DataDome.
Warm (session reused, same BrowserFetcher instance): ~5-10 s per product page.
For production, keep ONE BrowserFetcher instance alive across requests (module-
level singleton pattern shown below via `_get_fetcher()`).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from app.services.enrichment.composition_extractor import (
    extract_composition,
    page_matches_brand,
)
from app.services.providers.browser_fetcher import BrowserFetcher
from app.services.providers.serper_client import SerperClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lamoda URL helpers
# ---------------------------------------------------------------------------

# Lamoda product URLs follow the pattern /p/<id>/<brand>-<slug>/
_LAMODA_PRODUCT_URL_RE = re.compile(
    r"https?://(?:www\.)?lamoda\.ru/p/[^/]+/[^/?#]+",
    re.IGNORECASE,
)

# CSS selector for the spec/characteristic block on Lamoda PDPs.
# The rendered DOM has a <div> with class containing "x-product-characteristics"
# or the generic product-description section.  We wait for whichever appears.
_SPEC_SELECTOR = "[class*='x-product-characteristics'], [class*='product-characteristics']"

# Lamoda homepage warmup URL — hit this BEFORE any product URL so DataDome can
# set its cookie via JS challenge.
_LAMODA_WARMUP_URL = "https://www.lamoda.ru/"

# CSS selector that confirms the Lamoda homepage is showing real catalog content
# (not a DataDome block).  A search/nav bar or catalog menu is always present.
_LAMODA_WARMUP_REAL_SELECTOR = (
    "[class*='header'], [class*='catalog'], [class*='navigation'], "
    "nav, header"
)

# Marker string that appears in DataDome block pages (original pattern)
_DATADOME_BLOCK_MARKER = "Доступ ограничен"

# Lamoda's own IP/rate-limit block page marker.
# When Lamoda's CDN blocks the IP it serves a custom ~4656-char page that:
#   - Has <title>Ошибка доступа</title>
#   - Has id="REQUEST-IP" / id="REQUEST-ID" debug metadata blocks
#   - Does NOT contain catalog/product content
# We detect it by the REQUEST-IP element which is unique to this error page.
_LAMODA_BLOCK_MARKER = 'id="REQUEST-IP"'

# Whether this session has already successfully cleared the DataDome warmup.
# We only need one homepage warmup per BrowserFetcher session — after that the
# datadome cookie persists for all subsequent product fetches.
_session_warmed: set[int] = set()  # keyed by id(fetcher)


def _is_lamoda_product_url(url: str) -> bool:
    """Return True when *url* looks like a Lamoda product-detail page."""
    return bool(_LAMODA_PRODUCT_URL_RE.match(url))


def _extract_lamoda_urls(search_results) -> list[str]:
    """Extract Lamoda product-page URLs from SerperResults."""
    urls: list[str] = []
    for result in search_results.organic_results:
        link = getattr(result, "link", "") or ""
        if _is_lamoda_product_url(link):
            urls.append(link)
    return urls


# ---------------------------------------------------------------------------
# Module-level singleton: keep one BrowserFetcher alive across calls so that
# the DataDome session cookie is reused (no re-challenge per call).
# Pool expansion point: replace this with a list[BrowserFetcher] + round-robin.
# ---------------------------------------------------------------------------

_fetcher_singleton: Optional[BrowserFetcher] = None


def _get_fetcher() -> BrowserFetcher:
    """Return (lazily create) the module-level BrowserFetcher singleton."""
    global _fetcher_singleton
    if _fetcher_singleton is None or _fetcher_singleton._closed:
        _fetcher_singleton = BrowserFetcher()
    return _fetcher_singleton


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def mine_lamoda_composition(
    product_name: str,
    brand: str,
    *,
    max_urls: int = 3,
    fetch_timeout: float = 45.0,
    serper_client: Optional[SerperClient] = None,
    fetcher: Optional[BrowserFetcher] = None,
) -> Optional[dict]:
    """Search Lamoda for *product_name* by *brand* and extract composition.

    Args:
        product_name: Full product name as sold (RU or EN), e.g.
                      "Толстовка худи Champion Reverse Weave".
        brand:        Brand token for Serper query + brand-gate, e.g. "Champion".
        max_urls:     Maximum Lamoda product URLs to try (stops on first hit).
        fetch_timeout: Per-page browser fetch timeout in seconds.
        serper_client: Optional injected SerperClient (for testing / DI).
        fetcher:      Optional injected BrowserFetcher (for testing / DI).

    Returns:
        dict with keys {composition, source_url, evidence} on success,
        or None when nothing was found / brand-gate rejected all pages.

    Never raises — returns None on any error (fail-closed).
    """
    if not product_name or not brand:
        logger.warning("mine_lamoda_composition: product_name and brand are required")
        return None

    # --- 1. Serper search ---------------------------------------------------
    _serper = serper_client or SerperClient()
    query = f"site:lamoda.ru {brand} {product_name}"
    logger.info("mine_lamoda_composition: searching %r", query)
    try:
        results = await _serper.search(query, num_results=10)
    except Exception as exc:
        logger.warning("mine_lamoda_composition: Serper error: %s", exc)
        return None

    lamoda_urls = _extract_lamoda_urls(results)
    if not lamoda_urls:
        logger.info(
            "mine_lamoda_composition: no Lamoda product URLs in Serper results for %r",
            query,
        )
        return None

    logger.info(
        "mine_lamoda_composition: %d Lamoda URLs found: %s",
        len(lamoda_urls),
        lamoda_urls[:max_urls],
    )

    # --- 2. Browser-fetch + extract -----------------------------------------
    _fetcher = fetcher or _get_fetcher()
    candidates = lamoda_urls[:max_urls]

    # Determine whether this fetcher instance already has a warm DataDome session
    # from a previous call.  On first use we run a homepage warmup; subsequent
    # product fetches in the same session reuse the stored datadome cookie.
    fetcher_id = id(_fetcher)
    already_warmed = fetcher_id in _session_warmed

    for url in candidates:
        logger.info("mine_lamoda_composition: fetching %s (warmed=%s)", url, already_warmed)
        try:
            if not already_warmed:
                # First product URL in a cold session — use homepage warmup path
                logger.info(
                    "mine_lamoda_composition: cold session — running warmup via %s",
                    _LAMODA_WARMUP_URL,
                )
                html = await _fetcher.fetch_with_warmup(
                    url,
                    warmup_url=_LAMODA_WARMUP_URL,
                    warmup_timeout=60.0,
                    product_timeout=fetch_timeout,
                    warmup_real_selector=_LAMODA_WARMUP_REAL_SELECTOR,
                    product_real_selector=_SPEC_SELECTOR,
                    warmup_block_marker=_LAMODA_BLOCK_MARKER,
                    product_block_marker=_LAMODA_BLOCK_MARKER,
                    warmup_retries=3,
                    warmup_retry_delay=6.0,
                )
                # Mark the session warm regardless of result — the cookie is set
                # after the homepage navigation even if the product URL failed.
                _session_warmed.add(fetcher_id)
                already_warmed = True
            else:
                # Subsequent URLs reuse the warm session — plain fetch
                html = await _fetcher.fetch(
                    url,
                    wait_selector=_SPEC_SELECTOR,
                    timeout=fetch_timeout,
                )
        except Exception as exc:
            logger.warning("mine_lamoda_composition: fetch error for %s: %s", url, exc)
            continue

        if not html:
            logger.info("mine_lamoda_composition: empty HTML for %s", url)
            continue

        html_len = len(html)

        # Detect Lamoda's IP-block / rate-limit error page (distinct from DataDome).
        # This custom ~4656-char page contains id="REQUEST-IP" metadata.
        if _LAMODA_BLOCK_MARKER in html:
            logger.warning(
                "mine_lamoda_composition: Lamoda IP-block page detected for %s "
                "(%d chars) — IP may be rate-limited; skipping URL",
                url, html_len,
            )
            continue

        logger.info("mine_lamoda_composition: got %d chars from %s", html_len, url)

        # --- 3. Brand-gate --------------------------------------------------
        if not page_matches_brand(html, url, brand, product_name=product_name):
            logger.info(
                "mine_lamoda_composition: brand-gate rejected %s (brand=%r)", url, brand
            )
            continue

        # --- 4. Composition extraction ---------------------------------------
        compositions = extract_composition(html)
        if not compositions:
            logger.info(
                "mine_lamoda_composition: no composition extracted from %s", url
            )
            continue

        composition_str = compositions[0]
        logger.info(
            "mine_lamoda_composition: FOUND composition=%r from %s",
            composition_str,
            url,
        )
        return {
            "composition": composition_str,
            "source_url": url,
            "evidence": composition_str,
            "all_compositions": compositions,
            "html_length": html_len,
        }

    logger.info(
        "mine_lamoda_composition: exhausted %d URLs without a composition hit "
        "(product=%r, brand=%r)",
        len(candidates),
        product_name,
        brand,
    )
    return None
