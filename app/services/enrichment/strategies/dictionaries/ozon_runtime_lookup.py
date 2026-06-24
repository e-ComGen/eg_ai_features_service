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
import asyncio
import os
import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)

OZON_BASE_URL = "https://api-seller.ozon.ru"
SEARCH_ENDPOINT = "/v1/description-category/attribute/values/search"
TREE_ENDPOINT = "/v1/description-category/tree"

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


VALUES_ENDPOINT = "/v1/description-category/attribute/values"

# (cat_id, type_id, attr_id) -> list[{"id": int, "value": str}]
_values_cache: dict[tuple[int, int, int], list[dict]] = {}


def clear_values_cache() -> None:
    """Clear the in-process category-values cache. Useful in tests."""
    _values_cache.clear()


async def list_values(
    cat_id: int,
    type_id: int,
    attribute_id: int,
    *,
    client_id: Optional[str] = None,
    api_key: Optional[str] = None,
    page_limit: int = 100,
    max_values: int = 300,
) -> list[dict]:
    """List the authoritative per-category dictionary values for an attribute.

    Paginates POST ``/v1/description-category/attribute/values`` (``last_value_id``
    cursor) and returns up to ``max_values`` entries as ``{"id": int, "value":
    str}``. Used for ТН ВЭД / Тип, whose LOCALLY-cached values are stale/generic
    while the live API holds the real category-specific list. Returns ``[]`` on
    missing creds, a non-dict-backed (404) attribute, or any error. Cached per
    (cat, type, attr) for the process lifetime.
    """
    cache_key = (cat_id, type_id, attribute_id)
    if cache_key in _values_cache:
        return _values_cache[cache_key]

    resolved_client_id = client_id or os.getenv("OZON_CLIENT_ID", "")
    resolved_api_key = api_key or os.getenv("OZON_API_KEY", "")
    if not resolved_client_id or not resolved_api_key:
        _values_cache[cache_key] = []
        return []

    headers = {
        "Client-Id": resolved_client_id,
        "Api-Key": resolved_api_key,
        "Content-Type": "application/json",
    }
    out: list[dict] = []
    last_value_id = 0
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            while len(out) < max_values:
                payload = {
                    "description_category_id": cat_id,
                    "type_id": type_id,
                    "attribute_id": attribute_id,
                    "language": "DEFAULT",
                    "limit": page_limit,
                    "last_value_id": last_value_id,
                }
                r = await client.post(OZON_BASE_URL + VALUES_ENDPOINT, headers=headers, json=payload)
                if r.status_code != 200:
                    if r.status_code != 404:
                        log.warning(
                            "ozon list_values: HTTP %d attr_id=%d: %s",
                            r.status_code, attribute_id, r.text[:200],
                        )
                    break
                body = r.json()
                items = body.get("result") or []
                if not items:
                    break
                for it in items:
                    out.append({"id": it["id"], "value": it["value"]})
                    last_value_id = it["id"]
                if not body.get("has_next"):
                    break
    except httpx.HTTPError as exc:
        log.warning("ozon list_values: network error attr_id=%d: %s", attribute_id, exc)

    _values_cache[cache_key] = out
    return out


# ---------------------------------------------------------------------------
# Live description_category_id resolution from the category tree
# ---------------------------------------------------------------------------
# Зачем: category_id из шаблона caller'а (eg_importer) может быть УСТАРЕВШИМ —
# values-API отвечает «category with level_3_id=... and type=... is not found».
# Правильный description_category_id — это родитель type_id в ЖИВОМ дереве Ozon
# (тостер: type 96031 живёт под dcid 17039630, а не под шаблонным 47156221).
# Грузим дерево один раз на процесс, строим {type_id: parent_description_category_id}.
_type_to_dcid: dict[int, int] = {}
_tree_loaded = False
_tree_lock = asyncio.Lock()


def clear_tree_cache() -> None:
    """Сбросить кэш дерева категорий. Для тестов."""
    global _tree_loaded
    _type_to_dcid.clear()
    _tree_loaded = False


def _walk_tree(nodes, parent_dcid: Optional[int]) -> None:
    for n in nodes or []:
        dcid = n.get("description_category_id")
        tid = n.get("type_id")
        if tid is not None and parent_dcid is not None:
            _type_to_dcid[int(tid)] = int(parent_dcid)
        child_parent = dcid if dcid is not None else parent_dcid
        _walk_tree(n.get("children"), child_parent)


async def _load_tree(client_id: Optional[str], api_key: Optional[str]) -> None:
    global _tree_loaded
    cid = client_id or os.getenv("OZON_CLIENT_ID", "")
    key = api_key or os.getenv("OZON_API_KEY", "")
    if not cid or not key:
        _tree_loaded = True  # нечем грузить — не долбим повторно
        return
    headers = {"Client-Id": cid, "Api-Key": key, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                OZON_BASE_URL + TREE_ENDPOINT, headers=headers, json={"language": "DEFAULT"}
            )
        if r.status_code == 200:
            _walk_tree(r.json().get("result"), None)
            log.info("ozon tree: загружено %d type→category маппингов", len(_type_to_dcid))
        else:
            log.warning("ozon tree: HTTP %d: %s", r.status_code, r.text[:200])
    except httpx.HTTPError as exc:  # pragma: no cover — сеть не должна ронять стейдж
        log.warning("ozon tree: network error: %s", exc)
    _tree_loaded = True


async def resolve_description_category_id(
    type_id: Optional[int],
    *,
    client_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[int]:
    """Вернуть живой description_category_id (родитель type_id) или None.

    None если: type_id не задан, дерево недоступно (нет кред/сеть) или type_id в
    дереве не найден. Вызывающая сторона тогда фоллбэчит на свой category_id.
    """
    if type_id is None:
        return None
    if not _tree_loaded:
        async with _tree_lock:
            if not _tree_loaded:
                await _load_tree(client_id, api_key)
    return _type_to_dcid.get(int(type_id))
