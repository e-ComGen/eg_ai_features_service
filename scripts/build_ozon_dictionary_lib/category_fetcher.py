"""Ozon category fetcher — Wayback Machine primary, Playwright fallback.

Uses archive.org Wayback Machine as the primary source to bypass DataDome
(archive.org fetches from its own IPs; DataDome has no effect).

Falls back to Playwright-based headless Chromium when no Wayback snapshot
is available or when the snapshot is too old.

TODO: Verify real Ozon API endpoint path and response schema.
      Ozon updates their internal API sporadically — the URL below may need
      updating if the endpoint changes.  Investigate via browser DevTools on
      https://www.ozon.ru/api/composer-api.bx/page/json/v2?url=/category/<slug>/
"""
import asyncio
import json
import re
import urllib.parse
import urllib.request
from typing import Any

try:
    from playwright.async_api import async_playwright, Page
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


# Known Ozon internal endpoint for category children
# TODO: Validate this URL against real Ozon traffic
_CATEGORY_API_URL = (
    "https://www.ozon.ru/api/entrypoint-api.bx/page/json/v2"
    "?url=/category/{slug}/"
)

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Wayback Machine helpers
# ---------------------------------------------------------------------------

def _wayback_snapshot_url(ozon_url: str) -> str | None:
    """Return the most recent Wayback Machine snapshot URL for *ozon_url*, or None."""
    # Try the simple availability API first (fast)
    api = f"https://archive.org/wayback/available?url={urllib.parse.quote(ozon_url, safe=':/')}"
    try:
        try:
            from curl_cffi import requests as _cffi
            resp = _cffi.get(api, timeout=12)
            data = resp.json()
        except ImportError:
            with urllib.request.urlopen(api, timeout=12) as r:
                data = json.loads(r.read())

        snapshot = data.get("archived_snapshots", {}).get("closest", {})
        if snapshot.get("available"):
            return snapshot["url"]
    except Exception:
        pass

    # Fall back to CDX API (slower but more complete)
    cdx = (
        "https://web.archive.org/cdx/search/cdx"
        f"?url={urllib.parse.quote(ozon_url, safe=':/')}"
        "&output=json&limit=1&fl=timestamp,original&filter=statuscode:200&fastLatest=true"
    )
    try:
        try:
            from curl_cffi import requests as _cffi
            resp = _cffi.get(cdx, timeout=15)
            rows = resp.json()
        except ImportError:
            with urllib.request.urlopen(cdx, timeout=15) as r:
                rows = json.loads(r.read())

        if len(rows) > 1:
            ts, orig = rows[1][0], rows[1][1]
            return f"https://web.archive.org/web/{ts}/{orig}"
    except Exception:
        pass

    return None


def _fetch_wayback_product_urls(ozon_url: str, limit: int = 20) -> list[str]:
    """Fetch *limit* product URLs from the latest Wayback Machine snapshot.

    Returns an empty list when no snapshot is found or on any network error.
    Category dictionaries on Ozon change slowly, so a snapshot from the past
    year is usually still valid for attribute/category work.
    """
    snapshot_url = _wayback_snapshot_url(ozon_url)
    if not snapshot_url:
        return []

    try:
        try:
            from curl_cffi import requests as _cffi
            resp = _cffi.get(snapshot_url, timeout=30, impersonate="chrome120")
            html = resp.text
        except ImportError:
            req = urllib.request.Request(
                snapshot_url,
                headers={"User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                html = r.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"[category_fetcher] Wayback fetch failed for {ozon_url}: {exc}")
        return []

    raw_links = re.findall(r'href=["\']([^"\']*?/product/[^"\']+)["\']', html)
    seen: set[str] = set()
    result: list[str] = []
    for lnk in raw_links:
        # Strip Wayback Machine wrapper prefix if present
        m = re.search(r"/(https?://[^/]*/product/[^?\"']+)", lnk)
        if m:
            clean = m.group(1).split("?")[0]
        elif "/product/" in lnk and "ozon.ru" in lnk:
            clean = lnk.split("?")[0]
        else:
            continue
        if clean not in seen:
            seen.add(clean)
            result.append(clean)
        if len(result) >= limit:
            break

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def fetch_sample_product_urls(
    category: dict,
    limit: int = 20,
    use_wayback: bool = True,
) -> list[str]:
    """Fetch up to *limit* sample product URLs from an Ozon category page.

    Accepts a category dict (as returned by seed_loader.load_seed_categories)
    with any of the following fields used to determine the category page URL,
    tried in priority order:

    1. ``url``   — full URL, e.g. "https://www.ozon.ru/category/smartfony-15502/"
    2. ``slug``  — slug only,  e.g. "smartfony-15502"
    3. ``id``    — numeric ID, e.g. 502

    Strategy (DataDome bypass):
    - Primary:  archive.org Wayback Machine (``use_wayback=True``, default).
                Wayback fetches from its own IPs; DataDome has no effect.
                Category structures on Ozon change slowly, so a cached snapshot
                from the past year is valid for attribute/dictionary work.
    - Fallback: headless Playwright (may be blocked by DataDome on datacenter IPs).

    Args:
        category:     dict with at least one of ``url``, ``slug``, or ``id``.
        limit:        Maximum number of product URLs to return.
        use_wayback:  Try archive.org first (default True).

    Returns:
        List of absolute Ozon product page URLs (https://www.ozon.ru/product/...),
        deduplicated, query-params stripped.  May be shorter than *limit* on error.
    """
    cat_url = _resolve_category_url(category)
    if not cat_url:
        return []

    # --- Primary: Wayback Machine ---
    if use_wayback:
        urls = _fetch_wayback_product_urls(cat_url, limit)
        if urls:
            print(f"[category_fetcher] Wayback: got {len(urls)} URLs for {cat_url}")
            return urls
        print(f"[category_fetcher] Wayback returned nothing for {cat_url}, falling back to Playwright")

    # --- Fallback: Playwright ---
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError(
            "playwright is not installed.  Run: pip install playwright && playwright install chromium"
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=_DEFAULT_USER_AGENT,
            viewport={"width": 1280, "height": 900},
            locale="ru-RU",
        )
        page = await context.new_page()
        try:
            urls = await _collect_product_urls(page, cat_url, limit)
        finally:
            await browser.close()

    return urls


def _resolve_category_url(category: dict) -> str:
    """Build the Ozon category page URL from a category dict.

    Priority: ``url`` > ``slug`` > ``id``.
    Returns an empty string if none of those fields are present.
    """
    url = category.get("url", "")
    if url:
        if not url.startswith("http"):
            url = f"https://www.ozon.ru{url}"
        return url

    slug = category.get("slug", "")
    if slug:
        return f"https://www.ozon.ru/category/{slug}/"

    cat_id = category.get("id")
    if cat_id is not None:
        return f"https://www.ozon.ru/category/{cat_id}/"

    return ""


async def _collect_product_urls(page: Any, category_url: str, n: int) -> list[str]:
    """Navigate to category page and scrape product hrefs.

    Uses domcontentloaded (not networkidle) to avoid hanging on antibot challenge
    pages that keep XHR activity alive indefinitely.  If the server returns a
    non-2xx status the page is skipped immediately.
    """
    product_urls: list[str] = []
    try:
        response = await page.goto(category_url, wait_until="domcontentloaded", timeout=30_000)
        status_code = getattr(response, "status", 200)
        if response is None or (isinstance(status_code, int) and status_code >= 400):
            print(
                f"[category_fetcher] Skipping {category_url}: "
                f"HTTP {status_code if response else 'no-response'}"
            )
            return product_urls

        # Give JS rendering a short fixed window (3 s) — enough for SSR/hydration
        # but doesn't block forever if the page is a challenge/captcha.
        await page.wait_for_timeout(3_000)

        # Ozon product links contain "/product/" in their href
        links = await page.eval_on_selector_all(
            "a[href*='/product/']",
            "els => els.map(e => e.href)",
        )
        seen: set[str] = set()
        for link in links:
            link = link.split("?")[0]  # strip query params
            if link not in seen:
                seen.add(link)
                product_urls.append(link)
            if len(product_urls) >= n:
                break
    except Exception as exc:
        print(f"[category_fetcher] Failed to collect URLs from {category_url}: {exc}")
    return product_urls
