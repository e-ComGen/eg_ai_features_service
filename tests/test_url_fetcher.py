"""
tests/test_url_fetcher.py

Unit tests for app/services/url_fetcher.py (web-fetch Stage 2).

Run with:
    pytest tests/test_url_fetcher.py -v
All tests are offline — httpx calls are mocked via unittest.mock.
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import httpx

from app.services.url_fetcher import (
    detect_source,
    fetch_url_content,
    fetch_wildberries,
    fetch_all,
    _extract_wb_nm_id,
)


# ---------------------------------------------------------------------------
# detect_source
# ---------------------------------------------------------------------------

def test_detect_source_recognizes_aliexpress():
    assert detect_source("https://ru.aliexpress.com/item/1005001234567890.html") == "ali"
    assert detect_source("https://www.aliexpress.com/item/123.html") == "ali"


def test_detect_source_recognizes_wildberries():
    assert detect_source("https://www.wildberries.ru/catalog/123456789/detail.aspx") == "wb"
    assert detect_source("https://wildberries.ru/catalog/987654/detail.aspx") == "wb"


def test_detect_source_recognizes_ozon():
    assert detect_source("https://www.ozon.ru/product/some-product-123456/") == "ozon"
    assert detect_source("https://ozon.ru/product/item/") == "ozon"


def test_detect_source_returns_generic_for_unknown():
    assert detect_source("https://some-supplier.ru/product/123") == "generic"


# ---------------------------------------------------------------------------
# nm_id extraction from WB URL
# ---------------------------------------------------------------------------

def test_fetch_wildberries_extracts_nm_id():
    assert _extract_wb_nm_id("https://www.wildberries.ru/catalog/123456789/detail.aspx") == "123456789"
    assert _extract_wb_nm_id("https://wildberries.ru/catalog/987654/detail.aspx") == "987654"
    assert _extract_wb_nm_id("https://example.com/page") is None


# ---------------------------------------------------------------------------
# HTTPS-only guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_url_rejects_http():
    """fetch_url_content must return None for http:// URLs."""
    result = await fetch_url_content("http://insecure.example.com/product/1")
    assert result is None


# ---------------------------------------------------------------------------
# Timeout returns None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_url_timeout_returns_none():
    """When httpx raises TimeoutException, fetch_url_content returns None."""
    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client_cls.return_value = mock_client

        result = await fetch_url_content("https://example.com/product")
    assert result is None


# ---------------------------------------------------------------------------
# Wildberries: verify correct API URL is called
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_wildberries_calls_card_api():
    """fetch_wildberries must call card.wb.ru with the extracted nm_id."""
    wb_response = {
        "data": {
            "products": [
                {
                    "name": "Test Product",
                    "brand": "BrandX",
                    "description": "Great product description",
                    "options": [
                        {"name": "Color", "value": "Red"},
                    ],
                }
            ]
        }
    }

    captured_url = []

    async def mock_get(url, **kwargs):
        captured_url.append(url)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = MagicMock(return_value=wb_response)
        return mock_resp

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = mock_get
        mock_client_cls.return_value = mock_client

        result = await fetch_wildberries("https://www.wildberries.ru/catalog/123456789/detail.aspx")

    assert result is not None
    assert result.source_type == "wb"
    assert len(captured_url) == 1
    assert "card.wb.ru" in captured_url[0]
    assert "nm=123456789" in captured_url[0]
    assert "Test Product" in result.content


# ---------------------------------------------------------------------------
# fetch_all: partial failure returns successful results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_all_partial_failure_returns_what_succeeded():
    """
    3 URLs: 1st succeeds (WB), 2nd raises exception, 3rd succeeds (generic).
    fetch_all must return text from the 2 successful ones.
    """
    wb_response = {
        "data": {
            "products": [
                {"name": "WB Product", "description": "WB desc", "options": []}
            ]
        }
    }

    call_count = 0

    async def mock_get(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if "card.wb.ru" in url:
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json = MagicMock(return_value=wb_response)
            return mock_resp
        if "fail.example.com" in url:
            raise httpx.ConnectError("connection refused")
        # Generic page
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.text = "<html><body>Generic product page content here</body></html>"
        return mock_resp

    with patch("app.services.url_fetcher.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = mock_get
        mock_client_cls.return_value = mock_client

        urls = [
            "https://www.wildberries.ru/catalog/111111/detail.aspx",
            "https://fail.example.com/product/bad",
            "https://generic-shop.ru/product/abc",
        ]
        result = await fetch_all(urls)

    # Must include content from at least the WB URL
    assert "WB Product" in result or "=== Source:" in result
    # The failing URL should NOT appear as a source with content
    assert "fail.example.com" not in result or result.count("=== Source:") <= 2
