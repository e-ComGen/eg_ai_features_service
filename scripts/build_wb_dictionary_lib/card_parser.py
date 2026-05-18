"""WB card parser.

Fetches a single product card from the public WB card detail endpoint and
extracts the characteristics array.

Primary endpoint:  https://card.wb.ru/cards/v1/detail
Fallback endpoint: https://wbx-content-v2.wbstatic.net/ru/{nm_id}.json
"""
import httpx
from typing import Optional

CARD_URL = "https://card.wb.ru/cards/v1/detail"
CONTENT_URL = "https://wbx-content-v2.wbstatic.net/ru/{nm_id}.json"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


async def fetch_characteristics(nm_id: int) -> Optional[list[dict]]:
    """Fetch card detail and extract characteristics array.

    Tries card.wb.ru first; falls back to wbx-content-v2 static JSON if the
    primary endpoint returns no characteristics.

    Returns list of {"id": int, "name": str, "value": str | list} or None on failure.
    """
    result = await _fetch_from_card_api(nm_id)
    if result is not None:
        return result
    return await _fetch_from_content_static(nm_id)


async def _fetch_from_card_api(nm_id: int) -> Optional[list[dict]]:
    """Primary: card.wb.ru/cards/v1/detail"""
    params = {
        "appType": 1,
        "curr": "rub",
        "dest": -1257786,
        "spp": 30,
        "nm": nm_id,
    }
    async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
        try:
            r = await client.get(CARD_URL, params=params)
            r.raise_for_status()
            data = r.json()
            products = data.get("data", {}).get("products", [])
            if not products:
                return None
            product = products[0]
            charcs = product.get("characteristics") or product.get("colors") or []
            # Return None (to trigger fallback) when the list is empty
            return charcs if charcs else None
        except Exception:
            return None


async def _fetch_from_content_static(nm_id: int) -> Optional[list[dict]]:
    """Fallback: wbx-content-v2 static JSON which usually contains full specs."""
    url = CONTENT_URL.format(nm_id=nm_id)
    async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
        try:
            r = await client.get(url)
            r.raise_for_status()
            data = r.json()
            # Static JSON structure: top-level "compositions" or "options" arrays
            # Each entry typically has {"name": str, "value": str}; no numeric id
            charcs = data.get("compositions") or data.get("options") or []
            if charcs:
                # Normalise to same shape as primary API
                return [
                    {"id": 0, "name": c.get("name", ""), "value": c.get("value", "")}
                    for c in charcs
                    if c.get("name")
                ]
            return None
        except Exception:
            return None
