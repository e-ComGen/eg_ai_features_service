"""WB catalog fetcher.

Fetches sample nm_ids (product IDs) for a given category / subject.

catalog.wb.ru is blocked by Angie WAF (403) as of 2026.
Primary:  search.wb.ru/exactmatch/ru/common/v9/search  (200 with full browser
          headers; retries up to 3x on 429 with exponential back-off).
Fallback: returns empty list — the caller (build_wb_dictionary.py) skips the
          subject and continues with the rest.
"""
import asyncio
import httpx
from typing import Optional

SEARCH_URL = "https://search.wb.ru/exactmatch/ru/common/v9/search"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Origin": "https://www.wildberries.ru",
    "Referer": "https://www.wildberries.ru/",
    "sec-ch-ua": '"Chromium";v="120", "Not_A Brand";v="8"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
}

_RETRY_DELAYS = [5, 15, 30]  # seconds; 3 attempts before giving up


async def fetch_sample_nm_ids(
    subj_id: int,
    shard: Optional[str],
    limit: int = 30,
    query: Optional[str] = None,
) -> list[int]:
    """Fetch sample nm_ids via search.wb.ru for the given subject.

    Parameters
    ----------
    subj_id:
        WB subject/category integer ID (e.g. 8126 for bluzki).
    shard:
        Shard string from the menu (unused; kept for API compatibility).
    limit:
        Maximum number of nm_ids to return.
    query:
        Optional category query string from the menu node (e.g. ``"?cat=8126"``).
        When provided, its ``cat`` value is preferred over ``subj_id``.

    Returns
    -------
    List of nm_ids (may be empty on failure).
    """
    params = {
        "appType": "1",
        "curr": "rub",
        "dest": "-1257786",
        "resultset": "catalog",
        "sort": "popular",
        "spp": "30",
        "xsubject": str(subj_id),
    }

    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            await asyncio.sleep(delay)
        async with httpx.AsyncClient(timeout=20, headers=_HEADERS) as client:
            try:
                r = await client.get(SEARCH_URL, params=params)
                if r.status_code == 429:
                    continue  # retry after delay
                if r.status_code != 200:
                    return []
                data = r.json()
                products = data.get("data", {}).get("products", [])
                return [p["id"] for p in products[:limit] if "id" in p]
            except Exception:
                if attempt == len(_RETRY_DELAYS):
                    return []
                continue

    return []
