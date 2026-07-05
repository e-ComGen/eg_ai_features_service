"""Adaptive multi-site composition harvester for apparel products.

Strategy
--------
1. Two parallel Serper queries: generic brand+model+состав AND a site-targeted
   (site:kixbox.ru OR site:sportmaster.ru OR ...) query.
2. Candidate URLs are deduped and ordered by pool preference:
   open (httpx) sites first, then browser-walled sites.
3. For each candidate (up to max_sites):
   - Skip if domain_health marks it dead.
   - Route: open domain → plain url_fetcher; walled domain → BrowserFetcher.
4. Run extract_composition + page_matches_brand on the page.
5. Return first verified hit; move to next on failure.
6. Sequential + 3-5 s inter-request delay; one BrowserFetcher reused for all
   browser-sites; never crashes.

Routing table
-------------
open (plain httpx):
  kixbox.ru, sneakerhead.ru, basketshop.ru, brandshop.ru,
  street-beat.ru, blankstyle.com

browser (BrowserFetcher + warmup):
  sportmaster.ru, lamoda.ru
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

from app.services.enrichment.composition_extractor import (
    extract_composition,
    page_matches_brand,
)
from app.services.providers.serper_client import SerperClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Site pool definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteEntry:
    domain: str
    needs_browser: bool
    warmup_url: Optional[str] = None


APPAREL_SITE_POOL: list[SiteEntry] = [
    # --- Open sites (plain httpx) — cheap, fast, tried first ---
    # sneakerhead.ru leads: server-rendered structured label-value PDP pages with
    # full apparel surface (Состав, Пол, Страна, Цвет, Артикул) — best footwear hit.
    SiteEntry("sneakerhead.ru",  needs_browser=False),
    SiteEntry("kixbox.ru",       needs_browser=False),
    SiteEntry("basketshop.ru",   needs_browser=False),
    SiteEntry("brandshop.ru",    needs_browser=False),
    SiteEntry("street-beat.ru",  needs_browser=False),
    SiteEntry("blankstyle.com",  needs_browser=False),
    # --- Browser-walled sites — need real Chrome to bypass anti-bot ---
    SiteEntry("sportmaster.ru",  needs_browser=True,  warmup_url="https://www.sportmaster.ru/"),
    SiteEntry("lamoda.ru",       needs_browser=True,  warmup_url="https://www.lamoda.ru/"),
]

# Build lookup maps for quick routing
_POOL_DOMAINS: set[str] = {s.domain for s in APPAREL_SITE_POOL}
_BROWSER_DOMAINS: set[str] = {s.domain for s in APPAREL_SITE_POOL if s.needs_browser}
_OPEN_DOMAINS: set[str] = {s.domain for s in APPAREL_SITE_POOL if not s.needs_browser}
_SITE_BY_DOMAIN: dict[str, SiteEntry] = {s.domain: s for s in APPAREL_SITE_POOL}

# Site-targeted Serper query fragment: "site:kixbox.ru OR site:sneakerhead.ru OR ..."
_SITE_QUERY_FRAGMENT = " OR ".join(f"site:{s.domain}" for s in APPAREL_SITE_POOL)

# Inter-request politeness delay (seconds)
_INTER_REQUEST_DELAY = 3.5

# ---------------------------------------------------------------------------
# sneakerhead.ru structured label-value extractor
# ---------------------------------------------------------------------------
# sneakerhead PDP pages are server-rendered and expose a structured spec block:
#   Состав: Кожа, синтетика, текстиль, резина
#   Пол: Унисекс
#   Страна: Китай
#   Артикул: DH2987-102
#   Цвет: Белый/Чёрный
#
# We parse these as verbatim label→value pairs.  The label set is intentionally
# limited to fields that are product-invariant (safe to generalise across SKUs).

# Canonical sneakerhead label names → our normalised field names (lowercase key).
_SNEAKERHEAD_LABEL_MAP: dict[str, str] = {
    "состав":   "Состав",
    "материал": "Материал",
    "пол":      "Пол",
    "страна":   "Страна",
    "артикул":  "Артикул",
    "цвет":     "Цвет",
    "сезон":    "Сезон",
}

# Regex: matches "Label: Value" lines (colon separator, any leading whitespace).
# Captures the label and value verbatim; stops at end-of-line.
_SNEAKERHEAD_LABEL_RE = re.compile(
    r"^\s*(?P<label>" + "|".join(re.escape(k) for k in _SNEAKERHEAD_LABEL_MAP) + r")\s*:\s*"
    r"(?P<value>[^\n\r]{1,200})",
    re.IGNORECASE | re.MULTILINE,
)


def extract_sneakerhead_fields(html_or_text: str) -> dict[str, str]:
    """Parse sneakerhead.ru structured spec block into {field_name: verbatim_value}.

    Works on both raw HTML (strips tags first) and plain text.
    Returns only the fields present in _SNEAKERHEAD_LABEL_MAP; empty dict if none found.
    Values are returned verbatim (no normalisation) — callers do enum-matching.
    """
    if not html_or_text:
        return {}

    from app.services.enrichment.composition_extractor import _flatten_html
    flat = _flatten_html(html_or_text)

    result: dict[str, str] = {}
    for m in _SNEAKERHEAD_LABEL_RE.finditer(flat):
        label_low = m.group("label").strip().lower()
        field_name = _SNEAKERHEAD_LABEL_MAP.get(label_low)
        if field_name is None:
            continue
        value = m.group("value").strip().rstrip(".,;:")
        if value and field_name not in result:
            result[field_name] = value

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_host(url: str) -> str:
    """Return lowercase hostname from url, stripping www. prefix."""
    try:
        host = urlparse(url).hostname or ""
        return re.sub(r"^www\.", "", host.lower())
    except Exception:
        return ""


def _pool_rank(url: str) -> int:
    """Return pool index (lower = higher priority). Non-pool URLs → 9999."""
    host = _extract_host(url)
    for i, entry in enumerate(APPAREL_SITE_POOL):
        if host == entry.domain or host.endswith("." + entry.domain):
            return i
    return 9999


def _collect_urls(serper_results_list) -> list[str]:
    """Flatten and dedup URLs from a list of SerperResults objects."""
    seen: set[str] = set()
    urls: list[str] = []
    for results in serper_results_list:
        if results is None:
            continue
        for r in getattr(results, "organic_results", []):
            url = getattr(r, "link", "") or ""
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def _order_by_pool_preference(urls: list[str]) -> list[str]:
    """Sort: pool sites first (open before browser), unknown sites last."""
    return sorted(urls, key=_pool_rank)


def _is_dead(url: str) -> bool:
    """Dead-domain memory retired with Scrappey (scrape.do is the sole tier now);
    no per-domain skip list remains, so no domain is ever pre-skipped."""
    return False


# SPA app-root / hydration markers — presence with little visible text signals
# an unrendered client-side-rendered shell (httpx got HTML, but the spec block
# only appears after JS runs in a real browser).
_SPA_MARKERS: tuple[str, ...] = (
    'id="app"',
    "id='app'",
    "__next_data__",
    "data-reactroot",
    'id="root"',
    "id='root'",
    "data-server-rendered",
    "window.__nuxt__",
    "window.__initial_state__",
)

# Material words — if these appear ONLY inside <script>/JSON but not in visible
# text, the page is an unrendered SPA whose data lives in a JS payload.
_MATERIAL_HINT_WORDS: tuple[str, ...] = (
    "хлопок", "хлопка", "полиэстер", "эластан", "вискоз", "состав",
    "cotton", "polyester", "elastane", "material", "материал",
)

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)


def _visible_text_len(html: str) -> int:
    """Approximate length of human-visible text (scripts/styles/tags stripped)."""
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", without_scripts)
    return len(re.sub(r"\s+", " ", text).strip())


def _looks_like_spa_shell(html: Optional[str]) -> bool:
    """Heuristic: True if *html* looks like an unrendered SPA shell.

    Used ONLY after an open (httpx) fetch found NO composition, to decide
    whether a browser re-render is worth the cost.  General (no per-site
    hardcoding) — triggers when any of:
      1. A known app-root / hydration marker is present AND visible text is
         small relative to the raw HTML size (classic CSR shell).
      2. Visible text is tiny in absolute terms (page is essentially empty).
      3. Material/composition words exist in the raw HTML but NOT in the
         visible text — i.e. the data is locked inside a script/JSON payload
         that only a browser will hydrate into the DOM.
    """
    if not html:
        return False

    low = html.lower()
    visible_len = _visible_text_len(html)
    html_len = len(html)

    has_marker = any(m in low for m in _SPA_MARKERS)

    # 1 + 2: marker present with sparse visible text, or near-empty body.
    if has_marker and (visible_len < 800 or (html_len > 0 and visible_len / html_len < 0.05)):
        return True
    if visible_len < 200 and html_len > 2_000:
        return True

    # 3: material words live only in scripts/JSON, not in visible text.
    visible_low = _TAG_RE.sub(" ", _SCRIPT_STYLE_RE.sub(" ", html)).lower()
    in_raw = any(w in low for w in _MATERIAL_HINT_WORDS)
    in_visible = any(w in visible_low for w in _MATERIAL_HINT_WORDS)
    if in_raw and not in_visible:
        return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def harvest_composition(
    product_name: str,
    brand: str,
    *,
    max_sites: int = 4,
    serper_client: Optional[SerperClient] = None,
    browser_fetcher=None,  # BrowserFetcher instance for injection (testing/reuse)
) -> Optional[dict]:
    """Adaptively harvest fabric composition for an apparel product.

    Tries a pool of known retail sites, routing each through the correct
    fetcher (plain httpx for open sites, real Chrome for walled ones).

    Args:
        product_name: Full product name, e.g. "Толстовка худи Champion Reverse Weave".
        brand:        Brand token for search queries + brand verification gate.
        max_sites:    Maximum distinct candidate URLs to attempt before giving up.
        serper_client: Optional injected SerperClient (DI / testing).
        browser_fetcher: Optional injected BrowserFetcher (DI / reuse across calls).

    Returns:
        dict with keys {composition, material, source_url, site, evidence} on success,
        or None when no site yielded a verified composition.

    Never raises — all per-site failures are caught and logged.
    """
    if not product_name or not brand:
        logger.warning("harvest_composition: product_name and brand are required")
        return None

    _serper = serper_client or SerperClient()

    # --- 1. Dual Serper search -----------------------------------------------
    generic_query = f"{brand} {product_name} состав"
    targeted_query = f"({_SITE_QUERY_FRAGMENT}) {brand} {product_name}"

    logger.info(
        "harvest_composition: searching for %r / %r",
        generic_query,
        targeted_query,
    )

    async def _safe_search(q: str):
        try:
            return await _serper.search(q, num_results=10)
        except Exception as exc:
            logger.warning("harvest_composition: Serper error for %r: %s", q, exc)
            return None

    generic_res, targeted_res = await asyncio.gather(
        _safe_search(generic_query),
        _safe_search(targeted_query),
    )

    raw_urls = _collect_urls([generic_res, targeted_res])
    ordered_urls = _order_by_pool_preference(raw_urls)

    logger.info(
        "harvest_composition: %d candidate URLs (ordered, first %d to be tried): %s",
        len(ordered_urls),
        min(max_sites, len(ordered_urls)),
        [u for u in ordered_urls[:max_sites]],
    )

    # --- 2. Fetch + extract loop ---------------------------------------------
    # Lazy BrowserFetcher: create ONE instance, shared across all browser-sites
    # AND across SPA auto-escalations of open sites.
    _browser_fetcher_owned = False
    _browser_fetcher = browser_fetcher  # may be None initially

    def _get_browser_fetcher():
        """Return the shared BrowserFetcher, lazily creating it once."""
        nonlocal _browser_fetcher, _browser_fetcher_owned
        if _browser_fetcher is None:
            from app.services.providers.browser_fetcher import BrowserFetcher
            _browser_fetcher = BrowserFetcher()
            _browser_fetcher_owned = True
            logger.info("harvest_composition: created BrowserFetcher (shared)")
        return _browser_fetcher

    tried = 0
    try:
        for url in ordered_urls:
            if tried >= max_sites:
                logger.info("harvest_composition: max_sites=%d reached, stopping", max_sites)
                break

            host = _extract_host(url)

            # Skip dead/banned domains
            if _is_dead(url):
                logger.info("harvest_composition: SKIP dead domain %s (%s)", host, url)
                continue

            tried += 1

            # Determine routing
            needs_browser = any(
                host == entry.domain or host.endswith("." + entry.domain)
                for entry in APPAREL_SITE_POOL
                if entry.needs_browser
            )
            route = "browser" if needs_browser else "open"

            logger.info(
                "harvest_composition: trying %s [%s] (attempt %d/%d)",
                host, route, tried, max_sites,
            )

            html: Optional[str] = None

            try:
                if needs_browser:
                    # Lazy-init the shared BrowserFetcher on first browser-site hit
                    bf = _get_browser_fetcher()

                    # Find warmup URL for this domain
                    warmup_url: Optional[str] = None
                    for entry in APPAREL_SITE_POOL:
                        if host == entry.domain or host.endswith("." + entry.domain):
                            warmup_url = entry.warmup_url
                            break

                    if warmup_url:
                        html = await bf.fetch_with_warmup(
                            url,
                            warmup_url=warmup_url,
                            warmup_timeout=60.0,
                            product_timeout=60.0,
                        )
                    else:
                        html = await bf.fetch(url)

                else:
                    # Open site: httpx-direct → scrape.do (PRIMARY anti-bot tier) →
                    # BrowserFetcher.  Scrappey was retired 2026-06-14 (publisher.scrappey.com
                    # dead) — forcing it here only burned 25-45s per blocked link on a dead
                    # endpoint.  Dropped force_scrappey so this path respects the global
                    # Scrappey-off default and relies on scrape.do, which is already tried
                    # first inside fetch_url_content.
                    from app.services.url_fetcher import fetch_url_content
                    result = await fetch_url_content(url)
                    if result is not None:
                        # Use raw_html when available (preserves spec blocks)
                        html = result.raw_html or result.content

            except Exception as exc:
                logger.warning(
                    "harvest_composition: fetch error for %s [%s]: %s",
                    url, route, exc,
                )
                # Delay before next site even on failure (politeness)
                if tried < max_sites:
                    await asyncio.sleep(_INTER_REQUEST_DELAY)
                continue

            if not html:
                logger.info("harvest_composition: empty response from %s", url)
                if tried < max_sites:
                    await asyncio.sleep(_INTER_REQUEST_DELAY)
                continue

            # --- Brand gate ---
            if not page_matches_brand(html, url, brand, product_name=product_name):
                logger.info(
                    "harvest_composition: brand-gate rejected %s (brand=%r, product=%r)",
                    url, brand, product_name,
                )
                if tried < max_sites:
                    await asyncio.sleep(_INTER_REQUEST_DELAY)
                continue

            # --- Composition extraction (+ sneakerhead structured fields) ---
            # sneakerhead.ru: server-rendered label-value spec block — parse the full
            # structured surface (Состав, Пол, Страна, Цвет, Артикул, Сезон) verbatim.
            # The composition string is taken from the "Состав" or "Материал" field.
            # All other fields are collected into extra_fields for the caller to use.
            extra_fields: dict[str, str] = {}
            if host == "sneakerhead.ru" or host.endswith(".sneakerhead.ru"):
                sneaker_fields = extract_sneakerhead_fields(html)
                if sneaker_fields:
                    logger.info(
                        "harvest_composition: sneakerhead structured fields on %s: %s",
                        url, list(sneaker_fields.keys()),
                    )
                    # Surface composition from the structured block if present.
                    comp_from_struct = sneaker_fields.pop("Состав", None) \
                        or sneaker_fields.pop("Материал", None)
                    extra_fields = sneaker_fields  # remaining fields (Пол, Страна, etc.)
                    compositions = [comp_from_struct] if comp_from_struct else extract_composition(html)
                else:
                    compositions = extract_composition(html)
            else:
                compositions = extract_composition(html)

            # --- SPA auto-escalation (open sites only) ---
            # If httpx returned an unrendered SPA shell (spec block is in the JS
            # payload, not visible HTML), escalate to a real browser render ONCE.
            # No per-site hardcoding: the heuristic detects ANY SPA framework.
            if not compositions and not needs_browser and _looks_like_spa_shell(html):
                logger.info(
                    "harvest_composition: SPA shell detected on %s — escalating to browser render",
                    url,
                )
                try:
                    bf = _get_browser_fetcher()
                    spa_html = await bf.fetch(url)
                    if spa_html:
                        logger.info(
                            "harvest_composition: browser-rendered %s (%d chars)",
                            url, len(spa_html),
                        )
                        # Re-run composition extraction on the fully rendered DOM
                        compositions = extract_composition(spa_html)
                        if compositions:
                            # Update html so return block uses the rendered version
                            html = spa_html
                            route = "open+browser-spa"
                        else:
                            logger.info(
                                "harvest_composition: SPA escalation found no composition on %s", url
                            )
                    else:
                        logger.info(
                            "harvest_composition: SPA browser fetch returned empty for %s", url
                        )
                except Exception as exc:
                    logger.warning(
                        "harvest_composition: SPA escalation error for %s: %s", url, exc
                    )

            if not compositions:
                logger.info(
                    "harvest_composition: no composition extracted from %s", url
                )
                if tried < max_sites:
                    await asyncio.sleep(_INTER_REQUEST_DELAY)
                continue

            composition_str = compositions[0]
            logger.info(
                "harvest_composition: HIT — site=%s route=%s composition=%r url=%s",
                host, route, composition_str, url,
            )

            result: dict = {
                "composition": composition_str,
                "material": _primary_material_token(composition_str),
                "source_url": url,
                "site": host,
                "evidence": composition_str,
                "route": route,
                "all_compositions": compositions,
            }
            if extra_fields:
                result["extra_fields"] = extra_fields
            return result

            # Unreachable, but keeps the loop logic clear
            await asyncio.sleep(_INTER_REQUEST_DELAY)  # noqa: unreachable

    finally:
        # Close the BrowserFetcher only if WE created it (not if caller injected it)
        if _browser_fetcher_owned and _browser_fetcher is not None:
            try:
                await _browser_fetcher.close()
            except Exception:
                pass

    logger.info(
        "harvest_composition: exhausted %d candidates without a hit (product=%r, brand=%r)",
        tried, product_name, brand,
    )
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _primary_material_token(composition: str) -> Optional[str]:
    """Extract the dominant material name from a composition string.

    Delegates to composition_extractor.primary_material but imports locally
    to keep the module importable even if the extractor evolves.
    """
    try:
        from app.services.enrichment.composition_extractor import primary_material
        return primary_material([composition])
    except Exception:
        return None
