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
    category_id: int,
    category_slug: str = "",
    n: int = 20,
) -> list[str]:
    """Fetch N sample product URLs from an Ozon category page via Playwright.

    Args:
        category_id: Ozon category numeric ID.
        category_slug: URL slug of the category (e.g. "smartfony-15502").
        n: How many product URLs to collect.

    Returns:
        List of absolute Ozon product page URLs, possibly fewer than n on error.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError(
            "playwright is not installed.  Run: pip install playwright && playwright install chromium"
        )

    slug = category_slug or str(category_id)
    url = f"https://www.ozon.ru/category/{slug}/"

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            user_agent=_DEFAULT_USER_AGENT,
            viewport={"width": 1280, "height": 900},
            locale="ru-RU",
        )
        page = await context.new_page()
        try:
            urls = await _collect_product_urls(page, url, n)
        finally:
            await browser.close()

    return urls


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
