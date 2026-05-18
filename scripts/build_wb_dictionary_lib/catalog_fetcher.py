"""WB catalog fetcher.

Fetches sample nm_ids (product IDs) for a given category / subject from
the public WB catalog endpoint.  No auth required.
"""
import httpx
from typing import Optional

CATALOG_URL = "https://catalog.wb.ru/catalog/{shard}/v2/catalog"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


async def fetch_sample_nm_ids(
    subj_id: int,
    shard: Optional[str],
    limit: int = 30,
) -> list[int]:
    """Fetch sample nm_ids from category catalog.

    Returns up to `limit` nm_ids from the first page of the category.
    Falls back to _search_fallback when shard is unavailable.
    """
    if not shard:
        return await _search_fallback(subj_id, limit)

    url = CATALOG_URL.format(shard=shard)
    params = {
        "appType": 1,
        "cat": subj_id,
        "curr": "rub",
        "dest": -1257786,
        "sort": "popular",
        "spp": 30,
    }
    async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
        try:
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            products = data.get("data", {}).get("products", [])
            return [p["id"] for p in products[:limit]]
        except Exception:
            return []


async def _search_fallback(subj_id: int, limit: int) -> list[int]:
    """Fallback through search.wb.ru when catalog has no shard.

    Currently returns empty; can be implemented if needed.
    """
    # TODO: implement search.wb.ru fallback when shard is missing
    return []
