"""Tests for scripts/build_ozon_dictionary_lib/category_fetcher.py

All tests use mocks — no real Playwright launches, no live Ozon requests.
"""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make the scripts directory importable so we can import the lib directly.
_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from build_ozon_dictionary_lib.category_fetcher import (
    _resolve_category_url,
    fetch_sample_product_urls,
)


# ---------------------------------------------------------------------------
# _resolve_category_url — pure helper, no mocking needed
# ---------------------------------------------------------------------------


class TestResolveCategoryUrl:
    def test_uses_full_url_when_present(self):
        cat = {"id": 502, "url": "https://www.ozon.ru/category/smartfony-15502/"}
        assert _resolve_category_url(cat) == "https://www.ozon.ru/category/smartfony-15502/"

    def test_prepends_domain_for_relative_url(self):
        cat = {"url": "/category/smartfony-15502/"}
        assert _resolve_category_url(cat) == "https://www.ozon.ru/category/smartfony-15502/"

    def test_falls_back_to_slug(self):
        cat = {"id": 502, "slug": "smartfony-15502"}
        assert _resolve_category_url(cat) == "https://www.ozon.ru/category/smartfony-15502/"

    def test_falls_back_to_id(self):
        cat = {"id": 502}
        assert _resolve_category_url(cat) == "https://www.ozon.ru/category/502/"

    def test_returns_empty_string_when_no_url_fields(self):
        assert _resolve_category_url({"name": "Смартфоны"}) == ""

    def test_returns_empty_string_for_empty_dict(self):
        assert _resolve_category_url({}) == ""


# ---------------------------------------------------------------------------
# fetch_sample_product_urls — async, mocked Playwright
# ---------------------------------------------------------------------------


def _make_playwright_mock(href_list: list[str]) -> MagicMock:
    """Build a nested MagicMock that mimics async_playwright context manager."""
    page_mock = AsyncMock()
    page_mock.goto = AsyncMock()
    page_mock.eval_on_selector_all = AsyncMock(return_value=href_list)

    context_mock = AsyncMock()
    context_mock.new_page = AsyncMock(return_value=page_mock)

    browser_mock = AsyncMock()
    browser_mock.new_context = AsyncMock(return_value=context_mock)
    browser_mock.close = AsyncMock()

    chromium_mock = MagicMock()
    chromium_mock.launch = AsyncMock(return_value=browser_mock)

    pw_instance = MagicMock()
    pw_instance.chromium = chromium_mock
    # async context manager protocol
    pw_instance.__aenter__ = AsyncMock(return_value=pw_instance)
    pw_instance.__aexit__ = AsyncMock(return_value=False)

    return pw_instance


@pytest.mark.asyncio
async def test_fetch_returns_product_urls():
    """fetch_sample_product_urls returns deduplicated product URLs."""
    hrefs = [
        "https://www.ozon.ru/product/iphone-14-123456/?sku=1",
        "https://www.ozon.ru/product/samsung-s23-789012/",
        "https://www.ozon.ru/product/iphone-14-123456/?sku=2",  # duplicate
    ]
    pw_mock = _make_playwright_mock(hrefs)

    with patch(
        "build_ozon_dictionary_lib.category_fetcher.async_playwright",
        return_value=pw_mock,
    ):
        result = await fetch_sample_product_urls({"id": 502}, limit=20)

    # Duplicates removed, query params stripped
    assert "https://www.ozon.ru/product/iphone-14-123456/" in result
    assert "https://www.ozon.ru/product/samsung-s23-789012/" in result
    assert len(result) == 2


@pytest.mark.asyncio
async def test_fetch_respects_limit():
    """fetch_sample_product_urls returns at most *limit* URLs."""
    hrefs = [
        f"https://www.ozon.ru/product/item-{i}/" for i in range(30)
    ]
    pw_mock = _make_playwright_mock(hrefs)

    with patch(
        "build_ozon_dictionary_lib.category_fetcher.async_playwright",
        return_value=pw_mock,
    ):
        result = await fetch_sample_product_urls({"id": 502}, limit=5)

    assert len(result) == 5


@pytest.mark.asyncio
async def test_fetch_empty_when_no_id_or_url():
    """fetch_sample_product_urls returns [] when category has no URL/id fields."""
    # No Playwright should be launched at all since URL resolution fails early.
    pw_mock = _make_playwright_mock([])

    with patch(
        "build_ozon_dictionary_lib.category_fetcher.async_playwright",
        return_value=pw_mock,
    ):
        result = await fetch_sample_product_urls({"name": "NoURL"}, limit=10)

    assert result == []


@pytest.mark.asyncio
async def test_fetch_uses_category_url_field():
    """fetch_sample_product_urls uses category['url'] when present."""
    hrefs = ["https://www.ozon.ru/product/widget-42/"]
    pw_mock = _make_playwright_mock(hrefs)

    with patch(
        "build_ozon_dictionary_lib.category_fetcher.async_playwright",
        return_value=pw_mock,
    ):
        result = await fetch_sample_product_urls(
            {"url": "https://www.ozon.ru/category/custom-slug-999/"}, limit=10
        )

    assert result == ["https://www.ozon.ru/product/widget-42/"]


@pytest.mark.asyncio
async def test_fetch_playwright_not_available():
    """fetch_sample_product_urls raises RuntimeError when playwright is missing."""
    with patch(
        "build_ozon_dictionary_lib.category_fetcher.PLAYWRIGHT_AVAILABLE",
        False,
    ):
        with pytest.raises(RuntimeError, match="playwright is not installed"):
            await fetch_sample_product_urls({"id": 502}, limit=5)
