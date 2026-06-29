"""Runtime value lookup for WildBerries Content API dictionary attributes.

WB does not use numeric value IDs for most fields — a characteristic accepts a
canonical string from WB's own vocabulary. This module resolves an extracted
string to its canonical form via the WB Content API dictionaries.

Endpoints used (base: https://content-api.wildberries.ru):
    GET /content/v2/object/charcs/{subjectId}?locale=ru
        → {"data":[{"charcID":int,"name":str,"required":bool,...}], "error":bool}
    GET /content/v2/directory/countries?locale=ru
        → {"data":[{"id":int,"name":str,"fullName":str}]}
    GET /content/v2/directory/colors?locale=ru
        → {"data":[{"name":str,"parentName":str}]}  (937 items)
    GET /content/v2/directory/seasons?locale=ru
        → {"data":["круглогодичный","лето",...]}  (plain strings)
    GET /content/v2/directory/tnved?subjectID={id}&locale=ru
        → {"data":[...]}

In-process cache:
    Plain dicts keyed by (endpoint_key). No TTL — vocabularies are stable
    within a worker lifetime. Use ``clear_lookup_cache()`` in tests.
"""
import os
import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)

WB_BASE_URL = "https://content-api.wildberries.ru"

# Charc names that map to specific WB directories (lowercased for matching)
_COLOR_NAMES = frozenset({"цвет", "цвета", "основной цвет", "цвет товара"})
_COUNTRY_NAMES = frozenset({"страна производства", "страна изготовитель", "страна производитель"})
_SEASON_NAMES = frozenset({"сезон"})
_TNVED_NAMES = frozenset({"тн вэд", "tnved", "тнвэд", "код тн вэд"})
_GENDER_NAMES = frozenset({"пол", "gender"})

# In-process caches
_charcs_cache: dict[int, list[dict]] = {}          # subject_id → list of charc dicts
_directory_cache: dict[str, list[dict]] = {}        # dir_name → list of value dicts
_tnved_cache: dict[int, list[dict]] = {}            # subject_id → list of tnved dicts
# resolve cache: (subject_id, charc_name_lower, value_lower) → {"value": str, "id": int|None} | None
_resolve_cache: dict[tuple, Optional[dict]] = {}


def clear_lookup_cache() -> None:
    """Clear all in-process lookup caches. Useful in tests."""
    _charcs_cache.clear()
    _directory_cache.clear()
    _tnved_cache.clear()
    _resolve_cache.clear()


def _get_headers() -> Optional[dict]:
    """Return auth headers if WB_API_KEY is set, else None."""
    api_key = os.getenv("WB_API_KEY", "")
    if not api_key:
        return None
    return {"Authorization": api_key}


async def get_subject_charcs(subject_id: int) -> list[dict]:
    """Fetch characteristics for a WB subject (category).

    Returns list of charc dicts, empty on auth failure or network error.
    Caches by subject_id.
    """
    if subject_id in _charcs_cache:
        return _charcs_cache[subject_id]

    headers = _get_headers()
    if headers is None:
        log.warning("wb_runtime_lookup: WB_API_KEY not set; skipping charcs fetch for subject_id=%d", subject_id)
        _charcs_cache[subject_id] = []
        return []

    url = f"{WB_BASE_URL}/content/v2/object/charcs/{subject_id}"
    result: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers=headers, params={"locale": "ru"})
        if r.status_code == 200:
            body = r.json()
            result = body.get("data") or []
            log.debug("wb_runtime_lookup: subject_id=%d → %d charcs", subject_id, len(result))
        else:
            log.warning(
                "wb_runtime_lookup: HTTP %d for charcs subject_id=%d: %s",
                r.status_code, subject_id, r.text[:200],
            )
    except httpx.HTTPError as exc:
        log.warning("wb_runtime_lookup: network error for charcs subject_id=%d: %s", subject_id, exc)

    _charcs_cache[subject_id] = result
    return result


async def get_directory(name: str) -> list[dict]:
    """Fetch a WB dictionary by name: 'countries', 'colors', or 'seasons'.

    For 'seasons' the API returns plain strings — they are normalised to
    ``[{"name": str}]`` for uniform handling.

    Returns empty list on auth failure or network error. Caches by name.
    """
    name_lower = name.lower()
    if name_lower in _directory_cache:
        return _directory_cache[name_lower]

    headers = _get_headers()
    if headers is None:
        log.warning("wb_runtime_lookup: WB_API_KEY not set; skipping directory=%r", name)
        _directory_cache[name_lower] = []
        return []

    url = f"{WB_BASE_URL}/content/v2/directory/{name_lower}"
    result: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers=headers, params={"locale": "ru"})
        if r.status_code == 200:
            body = r.json()
            raw = body.get("data") or []
            # seasons returns plain strings → normalise
            if raw and isinstance(raw[0], str):
                result = [{"name": s} for s in raw]
            else:
                result = raw
            log.debug("wb_runtime_lookup: directory=%r → %d items", name, len(result))
        else:
            log.warning(
                "wb_runtime_lookup: HTTP %d for directory=%r: %s",
                r.status_code, name, r.text[:200],
            )
    except httpx.HTTPError as exc:
        log.warning("wb_runtime_lookup: network error for directory=%r: %s", name, exc)

    _directory_cache[name_lower] = result
    return result


async def get_tnved(subject_id: int) -> list[dict]:
    """Fetch ТН ВЭД codes for a WB subject.

    Returns empty list on auth failure or network error. Caches by subject_id.
    """
    if subject_id in _tnved_cache:
        return _tnved_cache[subject_id]

    headers = _get_headers()
    if headers is None:
        log.warning("wb_runtime_lookup: WB_API_KEY not set; skipping tnved for subject_id=%d", subject_id)
        _tnved_cache[subject_id] = []
        return []

    url = f"{WB_BASE_URL}/content/v2/directory/tnved"
    result: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, headers=headers, params={"locale": "ru", "subjectID": subject_id})
        if r.status_code == 200:
            body = r.json()
            result = body.get("data") or []
            log.debug("wb_runtime_lookup: tnved subject_id=%d → %d items", subject_id, len(result))
        else:
            log.warning(
                "wb_runtime_lookup: HTTP %d for tnved subject_id=%d: %s",
                r.status_code, subject_id, r.text[:200],
            )
    except httpx.HTTPError as exc:
        log.warning("wb_runtime_lookup: network error for tnved subject_id=%d: %s", subject_id, exc)

    _tnved_cache[subject_id] = result
    return result


def _fuzzy_match(value: str, candidates: list[str], threshold: int = 88) -> Optional[str]:
    """Return the best-matching candidate using rapidfuzz WRatio, or None.

    Falls back gracefully if rapidfuzz is not installed.
    """
    try:
        from rapidfuzz import process as rf_process, fuzz
        match = rf_process.extractOne(
            value, candidates, scorer=fuzz.WRatio, score_cutoff=threshold
        )
        if match:
            return match[0]
        return None
    except ImportError:
        log.debug("wb_runtime_lookup: rapidfuzz not installed; skipping fuzzy match")
        return None


async def resolve_value(
    subject_id: int,
    charc_name: str,
    value: str,
) -> Optional[dict]:
    """Resolve *value* to a canonical WB dictionary entry for *charc_name*.

    Resolution order:
        1. Exact match (case-insensitive).
        2. Fuzzy match via rapidfuzz WRatio >= 88.

    Returns:
        ``{"value": <canonical string>, "id": <int if available, else None>}``
        or ``None`` if no match found or no API key present.
    """
    cache_key = (subject_id, charc_name.lower(), value.lower())
    if cache_key in _resolve_cache:
        return _resolve_cache[cache_key]

    result = await _resolve_for_charc(subject_id, charc_name, value)
    _resolve_cache[cache_key] = result
    return result


async def _resolve_for_charc(
    subject_id: int,
    charc_name: str,
    value: str,
) -> Optional[dict]:
    """Internal: determine dictionary type and perform matching."""
    name_lower = charc_name.lower().strip()
    value_lower = value.lower().strip()

    if name_lower in _COLOR_NAMES:
        items = await get_directory("colors")
        # colors: [{"name": str, "parentName": str}]
        candidates = [it["name"] for it in items if it.get("name")]
        matched = _exact_then_fuzzy(value_lower, candidates)
        if matched:
            return {"value": matched, "id": None}
        return None

    if name_lower in _COUNTRY_NAMES:
        items = await get_directory("countries")
        # countries: [{"id": int, "name": str, "fullName": str}]
        names = [it["name"] for it in items if it.get("name")]
        matched = _exact_then_fuzzy(value_lower, names)
        if matched:
            matched_id = next(
                (it["id"] for it in items if it.get("name", "").lower() == matched.lower()),
                None,
            )
            return {"value": matched, "id": matched_id}
        return None

    if name_lower in _SEASON_NAMES:
        items = await get_directory("seasons")
        # seasons normalised to [{"name": str}]
        candidates = [it["name"] for it in items if it.get("name")]
        matched = _exact_then_fuzzy(value_lower, candidates)
        if matched:
            return {"value": matched, "id": None}
        return None

    if name_lower in _TNVED_NAMES:
        items = await get_tnved(subject_id)
        # tnved structure varies; try common key names
        candidates: list[str] = []
        for it in items:
            for key in ("name", "tnvedName", "code", "tnved"):
                if it.get(key):
                    candidates.append(str(it[key]))
                    break
        matched = _exact_then_fuzzy(value_lower, candidates)
        if matched:
            matched_id = next(
                (it.get("id") for it in items
                 if str(it.get("name", it.get("tnvedName", ""))).lower() == matched.lower()),
                None,
            )
            return {"value": matched, "id": matched_id}
        return None

    if name_lower in _GENDER_NAMES:
        # WB has no gender directory; canonicalise via the shared synonym map
        # (мужские→Мужской). No numeric value_id for WB gender → id=None.
        from app.services.enrichment.strategies.dictionaries.ozon_loader import (
            normalize_gender_value,
        )
        canon = normalize_gender_value(value)
        if canon:
            return {"value": canon, "id": None}
        return None

    # No known dictionary for this charc_name
    log.debug(
        "wb_runtime_lookup: charc_name=%r not mapped to any WB directory; skipping resolve",
        charc_name,
    )
    return None


def _exact_then_fuzzy(value_lower: str, candidates: list[str]) -> Optional[str]:
    """Exact case-insensitive match first, then rapidfuzz fallback."""
    for c in candidates:
        if c.lower() == value_lower:
            return c
    return _fuzzy_match(value_lower, candidates)
