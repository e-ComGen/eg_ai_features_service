"""WB seller catalog fetch: httpx-first, Playwright antibot fallback."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.services.enrichment.sources.wb_card_cdn import _CHROME_UA

logger = logging.getLogger(__name__)

_CATALOG_URL = "https://www.wildberries.ru/__internal/u-catalog/sellers/v4/catalog"
_PAGE_SIZE_PARAMS = {
    "ab_testing": "false",
    "appType": "1",
    "curr": "rub",
    "dest": "-1257786",
    "hide_dtype": "15",
    "hide_vflags": "4294967296",
    "lang": "ru",
    "sort": "popular",
    "spp": "30",
}


class SellerCatalogProvider(ABC):
    @abstractmethod
    async def fetch_products(self, seller_id: str, max_pages: int = 20) -> tuple[list[dict], int]:
        """Return (products, catalog_total). products is a list of dicts with at least
        keys nm_id (int), subject_id (int|None), subject_name (str|None if present)."""
        raise NotImplementedError


class WbSellerCatalog(SellerCatalogProvider):
    async def fetch_products(self, seller_id: str, max_pages: int = 20) -> tuple[list[dict], int]:
        headers = {
            "User-Agent": _CHROME_UA,
            "Referer": f"https://www.wildberries.ru/seller/{seller_id}",
            "Accept": "application/json",
        }
        result = await self._fetch_via_httpx(seller_id, max_pages, headers)
        if result is not None:
            return result
        logger.info("[WbSellerCatalog] httpx path blocked for seller=%s, falling back to Playwright", seller_id)
        return await self._fetch_via_playwright(seller_id)

    async def _fetch_via_httpx(
        self, seller_id: str, max_pages: int, headers: dict
    ) -> Optional[tuple[list[dict], int]]:
        async with httpx.AsyncClient(headers=headers, timeout=15.0) as client:
            all_products: list[dict] = []
            catalog_total = 0
            seen_nm_ids: set[int] = set()

            for page in range(1, max_pages + 1):
                params = dict(_PAGE_SIZE_PARAMS)
                params["page"] = str(page)
                params["supplier"] = str(seller_id)

                async def _get_page() -> Optional[dict]:
                    try:
                        response = await client.get(_CATALOG_URL, params=params)
                        if response.status_code != 200:
                            return None
                        data = response.json()
                        if not isinstance(data.get("products"), list):
                            return None
                        return data
                    except (httpx.TimeoutException, httpx.HTTPError, ValueError):
                        return None

                data = await _get_page()
                if data is None:
                    await asyncio.sleep(2)
                    data = await _get_page()
                    if data is None:
                        if page == 1:
                            return None
                        break

                if page == 1:
                    catalog_total = int(data.get("total") or 0)

                products = data.get("products") or []
                if not products:
                    break

                for p in products:
                    nm_id = p.get("id")
                    if nm_id is not None and nm_id not in seen_nm_ids:
                        seen_nm_ids.add(nm_id)
                        all_products.append({
                            "nm_id": nm_id,
                            "subject_id": p.get("subjectId"),
                            "subject_name": p.get("subjectName"),
                        })

                await asyncio.sleep(0.8)

            return (all_products, catalog_total)

    async def _fetch_via_playwright(self, seller_id: str) -> tuple[list[dict], int]:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page(user_agent=_CHROME_UA)
            collected_responses: list[dict] = []

            async def _on_response(response):
                if "u-catalog/sellers/v4/catalog" in response.url:
                    try:
                        body = await response.json()
                        if isinstance(body, dict) and isinstance(body.get("products"), list):
                            collected_responses.append(body)
                    except Exception:
                        logger.debug("Failed to parse response from %s", response.url)

            page.on("response", _on_response)
            await page.goto(f"https://www.wildberries.ru/seller/{seller_id}", wait_until="networkidle")
            await asyncio.sleep(5)
            await page.mouse.wheel(0, 4000)
            await asyncio.sleep(5)
            await browser.close()

        catalog_total = 0
        all_products: list[dict] = []
        seen_nm_ids: set[int] = set()

        for resp in collected_responses:
            if catalog_total == 0:
                catalog_total = int(resp.get("total") or 0)
            products = resp.get("products") or []
            for p in products:
                nm_id = p.get("id")
                if nm_id is not None and nm_id not in seen_nm_ids:
                    seen_nm_ids.add(nm_id)
                    all_products.append({
                        "nm_id": nm_id,
                        "subject_id": p.get("subjectId"),
                        "subject_name": p.get("subjectName"),
                    })

        return (all_products, catalog_total)
