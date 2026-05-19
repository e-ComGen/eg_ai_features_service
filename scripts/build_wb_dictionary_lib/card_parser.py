"""WB card parser.

Fetches a single product card from the WB basket CDN and extracts the
``options`` (characteristics) array.

card.wb.ru is blocked by Angie WAF (403) as of 2026.
Primary:  basket-NN.wbbasket.ru/vol{vol}/part{part}/{nm}/info/ru/card.json
          Basket number is derived from the nm_id vol using the known WB CDN
          sharding table.
Fallback: wbx-content-v2.wbstatic.net/ru/{nm_id}.json  (static CDN).
"""
import httpx
from typing import Optional

_BASKET_THRESHOLDS = [
    (143,  "basket-01"),
    (287,  "basket-02"),
    (431,  "basket-03"),
    (719,  "basket-04"),
    (1007, "basket-05"),
    (1061, "basket-06"),
    (1115, "basket-07"),
    (1169, "basket-08"),
    (1313, "basket-09"),
    (1601, "basket-10"),
    (1655, "basket-11"),
    (1919, "basket-12"),
    (2045, "basket-13"),
    (2189, "basket-14"),
    (2405, "basket-15"),
    (2621, "basket-16"),
    (2837, "basket-17"),
]

CONTENT_URL = "https://wbx-content-v2.wbstatic.net/ru/{nm_id}.json"

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
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "cross-site",
}


def _basket_host(nm_id: int) -> str:
    """Return the wbbasket.ru hostname for the given nm_id."""
    vol = nm_id // 100000
    for threshold, name in _BASKET_THRESHOLDS:
        if vol <= threshold:
            return f"{name}.wbbasket.ru"
    return "basket-18.wbbasket.ru"


def _basket_card_url(nm_id: int) -> str:
    host = _basket_host(nm_id)
    vol = nm_id // 100000
    part = nm_id // 1000
    return f"https://{host}/vol{vol}/part{part}/{nm_id}/info/ru/card.json"


async def fetch_characteristics(nm_id: int) -> Optional[list[dict]]:
    """Fetch card detail and extract characteristics array.

    Tries the basket CDN first; falls back to wbx-content-v2 static JSON.

    Returns list of {"id": int, "name": str, "value": str | list} or None on
    failure.
    """
    result = await _fetch_from_basket_cdn(nm_id)
    if result is not None:
        return result
    return await _fetch_from_content_static(nm_id)


async def _fetch_from_basket_cdn(nm_id: int) -> Optional[list[dict]]:
    """Primary: basket-NN.wbbasket.ru card JSON.

    Response structure::

        {
          "nm_id": ...,
          "options": [{"name": "Цвет", "value": "Белый", "charc_type": 1}, ...],
          ...
        }
    """
    url = _basket_card_url(nm_id)
    async with httpx.AsyncClient(timeout=15, headers=_HEADERS) as client:
        try:
            r = await client.get(url)
            if r.status_code != 200:
                return None
            data = r.json()
            options = data.get("options") or []
            if not options:
                return None
            # Normalise to the same shape used by the rest of the pipeline:
            # {"id": int, "name": str, "value": str}
            return [
                {
                    "id": 0,
                    "name": o.get("name", ""),
                    "value": o.get("value", ""),
                }
                for o in options
                if o.get("name")
            ]
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
            charcs = data.get("compositions") or data.get("options") or []
            if charcs:
                return [
                    {"id": 0, "name": c.get("name", ""), "value": c.get("value", "")}
                    for c in charcs
                    if c.get("name")
                ]
            return None
        except Exception:
            return None
