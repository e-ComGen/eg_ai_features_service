"""Ozon category fetcher — Playwright-based fallback.

Uses Ozon's internal categoryChildV3 endpoint (accessed via headless Chromium)
to build/enrich the category tree and collect sample product URLs per category.

This module is a fallback when welel/ozon-scraper seed data is unavailable or
when sample_urls are missing from the seed.

TODO: Verify real Ozon API endpoint path and response schema.
      Ozon updates their internal API sporadically — the URL below may need
      updating if the endpoint changes.  Investigate via browser DevTools on
      https://www.ozon.ru/api/composer-api.bx/page/json/v2?url=/category/<slug>/
"""
import asyncio
import json
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


async def fetch_sample_product_urls(
    category: dict,
    limit: int = 20,
) -> list[str]:
    """Fetch up to *limit* sample product URLs from an Ozon category page.

    Accepts a category dict (as returned by seed_loader.load_seed_categories)
    with any of the following fields used to determine the category page URL,
    tried in priority order:

    1. ``url``   — full URL, e.g. "https://www.ozon.ru/category/smartfony-15502/"
    2. ``slug``  — slug only,  e.g. "smartfony-15502"
    3. ``id``    — numeric ID, e.g. 502

    Args:
        category: dict with at least one of ``url``, ``slug``, or ``id``.
        limit: Maximum number of product URLs to return.

    Returns:
        List of absolute Ozon product page URLs (https://www.ozon.ru/product/...),
        deduplicated, query-params stripped.  May be shorter than *limit* on error.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError(
            "playwright is not installed.  Run: pip install playwright && playwright install chromium"
        )

    cat_url = _resolve_category_url(category)
    if not cat_url:
        return []

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
    """Navigate to category page and scrape product hrefs."""
    product_urls: list[str] = []
    try:
        await page.goto(category_url, wait_until="networkidle", timeout=45_000)
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
