"""Runtime value lookup for Ozon truncated-dictionary attributes.

When a dictionary attribute has ``values_truncated=True`` (>5000 possible values),
the cached list is incomplete. This module provides ``search_value`` — an async
function that queries the Ozon Seller API search endpoint to find the best matching
value id at runtime.

Endpoint used:
    POST https://api-seller.ozon.ru/v1/description-category/attribute/values/search
    Body: {
        "description_category_id": int, "type_id": int, "attribute_id": int,
        "value": str, "limit": 100
    }
    Response: {"result": [{"id": int, "value": str, "info": str}], "has_next": bool}

In-process cache:
    Results are cached in a plain dict keyed by (cat_id, type_id, attr_id, query).
    This avoids duplicate API calls within a single process/worker lifetime.
    The cache is intentionally simple (no TTL, no size limit) because:
    - Workers are typically short-lived (one job = one process lifetime).
    - Value vocabularies are stable between requests.
    Use ``clear_lookup_cache()`` in tests to reset state.
"""
import os
import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)

OZON_BASE_URL = "https://api-seller.ozon.ru"
SEARCH_ENDPOINT = "/v1/description-category/attribute/values/search"

# In-process cache: (cat_id, type_id, attr_id, query_lower) -> {"id": int, "value": str} | None
_lookup_cache: dict[tuple[int, int, int, str], Optional[dict]] = {}


def clear_lookup_cache() -> None:
    """Clear the in-process lookup cache. Useful in tests."""
    _lookup_cache.clear()


async def search_value(
    cat_id: int,
    type_id: int,
    attribute_id: int,
    query: str,
    *,
    client_id: Optional[str] = None,
    api_key: Optional[str] = None,
    limit: int = 100,
) -> Optional[dict]:
    """Search Ozon Seller API for a dictionary value matching *query*.

    Performs a partial-match search (e.g. "adid" finds "Adidas").
    Returns the first result as ``{"id": int, "value": str}``, or ``None`` if
    no match was found or the API call failed.

    Args:
        cat_id: Ozon description_category_id.
        type_id: Ozon type_id.
        attribute_id: Ozon attribute id.
        query: Partial string to search for (passed as ``"value"`` in request body).
        client_id: Ozon Client-Id header value. Falls back to ``OZON_CLIENT_ID`` env var.
        api_key: Ozon Api-Key header value. Falls back to ``OZON_API_KEY`` env var.
        limit: Number of results to request (1-100, default 100).

    Returns:
        ``{"id": int, "value": str}`` for the best (first) match, or ``None``.
    """
    cache_key = (cat_id, type_id, attribute_id, query.lower())
    if cache_key in _lookup_cache:
        return _lookup_cache[cache_key]

    resolved_client_id = client_id or os.getenv("OZON_CLIENT_ID", "")
    resolved_api_key = api_key or os.getenv("OZON_API_KEY", "")

    if not resolved_client_id or not resolved_api_key:
        log.warning(
            "ozon_runtime_lookup: OZON_CLIENT_ID / OZON_API_KEY not set; "
            "skipping runtime lookup for attr_id=%d query=%r",
            attribute_id, query,
        )
        _lookup_cache[cache_key] = None
        return None

    payload = {
        "description_category_id": cat_id,
        "type_id": type_id,
        "attribute_id": attribute_id,
        "value": query,
        "limit": limit,
    }
    headers = {
        "Client-Id": resolved_client_id,
        "Api-Key": resolved_api_key,
        "Content-Type": "application/json",
    }

    result: Optional[dict] = None
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                OZON_BASE_URL + SEARCH_ENDPOINT,
                headers=headers,
                json=payload,
            )
        if r.status_code == 200:
            body = r.json()
            items = body.get("result") or []
            if items:
                first = items[0]
                result = {"id": first["id"], "value": first["value"]}
                log.debug(
                    "ozon_runtime_lookup: attr_id=%d query=%r → id=%d value=%r",
                    attribute_id, query, result["id"], result["value"],
                )
            else:
                log.debug(
                    "ozon_runtime_lookup: attr_id=%d query=%r → no results",
                    attribute_id, query,
                )
        elif r.status_code == 404:
            log.debug(
                "ozon_runtime_lookup: attr_id=%d not dict-backed (404)", attribute_id
            )
        else:
            log.warning(
                "ozon_runtime_lookup: HTTP %d for attr_id=%d query=%r: %s",
                r.status_code, attribute_id, query, r.text[:200],
            )
    except httpx.HTTPError as exc:
        log.warning(
            "ozon_runtime_lookup: network error for attr_id=%d query=%r: %s",
            attribute_id, query, exc,
        )

    _lookup_cache[cache_key] = result
    return result
