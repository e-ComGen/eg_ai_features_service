"""BooksSource — verbatim book metadata from Open Library + Google Books APIs.

Books carry an ISBN-13 (EAN with prefix 978/979). ``context.ean`` (already
extracted by BarcodeSource) IS the ISBN for books. This source fires when
context.ean starts with 978 or 979 AND there are still-empty book-relevant
targets.

Two free APIs — no scraping, no Serper, no LLM:
  1. Open Library (primary, no key needed)
     ``https://openlibrary.org/isbn/{isbn}.json``
     Works → title, publishers[], number_of_pages, publish_date,
     languages[], works[]. The work key is followed to get subjects[].
     Authors: follow author keys → /authors/{key}.json → name.

  2. Google Books (enrichment/fallback, key optional)
     ``https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}``
     API key from env ``GOOGLE_BOOKS_API_KEY``; absent → skipped.
     Items[0].volumeInfo → title, authors[], publisher, publishedDate,
     pageCount, categories[], language, description.

Merge: prefer whichever has the field; map to target attributes by name
(fuzzy enum-matcher WRatio≥88 via rapidfuzz):
  - Free-text targets (Автор, Издательство): verbatim.
  - Enum targets (Язык, Жанр/Тематика/Раздел): MUST pass resolve_value_id;
    drop on no-match (verbatim-safe, no guessing).
  - Numeric (Количество страниц, Год издания): verbatim str.

Cache by ISBN (LRU 256). Error-graceful: network/404 → [].
Source.WB_CARD semantics (verbatim, no LLM).

Pipeline position: Stage 0.56 — after IceCat (0.55), before Regard (0.57).
Fires ONLY when context.ean is a book ISBN AND remaining > 0.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import OrderedDict
from typing import Any, Optional, Union

import httpx

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.wb_card_judge import WbCardJudge
from app.services.enrichment.prompt_router import filter_already_filled_targets
from app.services.enrichment.sources.ozon_card_source import (
    _norm_char_name,
    _split_multivalue,
)
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    resolve_value_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_OL_ISBN_URL = "https://openlibrary.org/isbn/{isbn}.json"
_OL_WORK_URL = "https://openlibrary.org{key}.json"
_OL_AUTHOR_URL = "https://openlibrary.org{key}.json"
_GB_URL = "https://www.googleapis.com/books/v1/volumes"

_HTTP_TIMEOUT = 15.0
_CACHE_MAX = 256

# Confidence: ISBN is an exact product identifier — no brand-gate uncertainty.
_CONF = 0.93

# Skip-guard: if ≥80% targets already filled, skip the API calls.
_SKIP_FILL_RATIO = 0.80

# Fuzzy name-match threshold for attribute name → target mapping.
_FUZZY_THRESHOLD = 88

# User-Agent for Open Library (polite tier: 3 req/sec allowed).
_OL_USER_AGENT = "e-comgen/1.0 (davisaraceli591@gmail.com)"

# Env var for Google Books API key.
_GB_KEY_ENV = "GOOGLE_BOOKS_API_KEY"

# ISBN-13 prefix for books: 978 or 979.
_BOOK_ISBN_PREFIXES = ("978", "979")

# ---------------------------------------------------------------------------
# Language normalisation: ISO 639-1 / Open Library codes → Russian label.
# Used to map API language codes to human-readable RU values for enum matching.
# ---------------------------------------------------------------------------
_LANG_CODE_TO_RU: dict[str, str] = {
    "ru": "Русский",
    "rus": "Русский",
    "en": "Английский",
    "eng": "Английский",
    "de": "Немецкий",
    "ger": "Немецкий",
    "deu": "Немецкий",
    "fr": "Французский",
    "fre": "Французский",
    "fra": "Французский",
    "es": "Испанский",
    "spa": "Испанский",
    "it": "Итальянский",
    "ita": "Итальянский",
    "zh": "Китайский",
    "chi": "Китайский",
    "zho": "Китайский",
    "ja": "Японский",
    "jpn": "Японский",
    "pt": "Португальский",
    "por": "Португальский",
    "ar": "Арабский",
    "ara": "Арабский",
    "ko": "Корейский",
    "kor": "Корейский",
    "pl": "Польский",
    "pol": "Польский",
    "uk": "Украинский",
    "ukr": "Украинский",
    "nl": "Нидерландский",
    "nld": "Нидерландский",
    "sv": "Шведский",
    "swe": "Шведский",
    "fi": "Финский",
    "fin": "Финский",
    "cs": "Чешский",
    "cze": "Чешский",
    "tr": "Турецкий",
    "tur": "Турецкий",
    "he": "Иврит",
    "heb": "Иврит",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Canonical book attribute name mappings to API field names.
# Used for direct matching before fuzzy fallback.
# Keys are normalised lower-case substrings of typical RU book attribute names.
_ATTR_NAME_TO_FIELD: dict[str, str] = {
    # Author(s)
    "автор": "authors",
    "авторы": "authors",
    "author": "authors",
    # Publisher
    "издательство": "publisher",
    "publisher": "publisher",
    "издатель": "publisher",
    # Year
    "год издания": "year",
    "год выпуска": "year",
    "год": "year",
    # Pages
    "количество страниц": "pages",
    "страниц": "pages",
    "объем": "pages",
    "объём": "pages",
    "pages": "pages",
    "число страниц": "pages",
    # Language
    "язык": "language",
    "язык издания": "language",
    "language": "language",
    # Genre / Subject
    "жанр": "genres",
    "тематика": "genres",
    "раздел": "genres",
    "genre": "genres",
    "тема": "genres",
    "subject": "genres",
    # ISBN
    "isbn": "isbn",
    "штрихкод": "isbn",
    "ean": "isbn",
}


def _is_book_isbn(ean: Optional[str]) -> bool:
    """True when the EAN looks like an ISBN-13 (978/979 prefix, 13 digits)."""
    if not ean:
        return False
    s = ean.strip().replace("-", "").replace(" ", "")
    return len(s) == 13 and s.startswith(_BOOK_ISBN_PREFIXES) and s.isdigit()


def _parse_year(date_str: Optional[str]) -> Optional[str]:
    """Extract 4-digit year from a date string like '2021', '2021-05-01', 'May 2021'."""
    if not date_str:
        return None
    m = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", date_str)
    return m.group(1) if m else None


def _normalise_lang_code(code: str) -> str:
    """Convert ISO 639-1/2/Open Library language key to RU label."""
    # Open Library languages look like "/languages/eng".
    key = code.strip().lower()
    if "/" in key:
        key = key.rsplit("/", 1)[-1]
    return _LANG_CODE_TO_RU.get(key, code)


def _clean_subject(subject: str) -> str:
    """Strip OL subject noise (parenthetical, trailing punctuation)."""
    s = re.sub(r"\s*\([^)]*\)", "", subject).strip().rstrip(".,;:")
    return s


async def _empty_coro() -> dict[str, Any]:
    """Async no-op returning an empty dict — used when Google Books key is absent."""
    return {}


# ---------------------------------------------------------------------------
# Open Library fetcher
# ---------------------------------------------------------------------------

async def _fetch_open_library(isbn: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch Open Library book metadata for the given ISBN.

    Returns a flat dict with keys: title, authors (list[str]), publisher (str),
    year (str), pages (str), languages (list[str]), genres (list[str]), isbn (str).
    Missing fields are absent from the dict.
    """
    data: dict[str, Any] = {"isbn": isbn}

    # --- Edition ---
    try:
        url = _OL_ISBN_URL.format(isbn=isbn)
        resp = await client.get(url)
        if resp.status_code == 404:
            logger.debug("[Books] Open Library 404 for ISBN %s", isbn)
            return {}
        if resp.status_code != 200:
            logger.debug("[Books] Open Library HTTP %s for ISBN %s", resp.status_code, isbn)
            return {}
        edition = resp.json()
    except Exception as exc:
        logger.debug("[Books] Open Library fetch error for %s: %s", isbn, exc)
        return {}

    if not isinstance(edition, dict):
        return {}

    # Title
    title = edition.get("title") or edition.get("full_title") or ""
    if title:
        data["title"] = str(title).strip()

    # Publisher
    publishers = edition.get("publishers") or []
    if publishers:
        pub = publishers[0] if isinstance(publishers[0], str) else str(publishers[0])
        data["publisher"] = pub.strip()

    # Year from publish_date
    year = _parse_year(edition.get("publish_date"))
    if year:
        data["year"] = year

    # Pages
    pages = edition.get("number_of_pages")
    if pages is not None:
        try:
            data["pages"] = str(int(pages))
        except (ValueError, TypeError):
            pass

    # Languages: [{key: "/languages/eng"}, ...]
    lang_entries = edition.get("languages") or []
    langs: list[str] = []
    for le in lang_entries:
        key = le.get("key", "") if isinstance(le, dict) else str(le)
        label = _normalise_lang_code(key)
        if label and label not in langs:
            langs.append(label)
    if langs:
        data["languages"] = langs

    # --- Work (for subjects/genres) ---
    work_keys = edition.get("works") or []
    if work_keys:
        try:
            wk = work_keys[0].get("key", "") if isinstance(work_keys[0], dict) else ""
            if wk:
                work_url = _OL_WORK_URL.format(key=wk)
                wresp = await client.get(work_url)
                if wresp.status_code == 200:
                    work = wresp.json()
                    raw_subjects = work.get("subjects") or []
                    # OL subjects are plain strings.
                    genres = [_clean_subject(str(s)) for s in raw_subjects if s]
                    genres = [g for g in genres if g]
                    if genres:
                        data["genres"] = genres[:20]  # cap to avoid noise
        except Exception as exc:
            logger.debug("[Books] OL work fetch error: %s", exc)

    # --- Authors ---
    author_keys = edition.get("authors") or []
    authors: list[str] = []
    for ak in author_keys[:3]:  # max 3 author requests
        try:
            akey = ak.get("key", "") if isinstance(ak, dict) else ""
            if not akey:
                continue
            aurl = _OL_AUTHOR_URL.format(key=akey)
            aresp = await client.get(aurl)
            if aresp.status_code == 200:
                adata = aresp.json()
                aname = adata.get("name") or adata.get("personal_name") or ""
                if aname and str(aname).strip():
                    authors.append(str(aname).strip())
        except Exception as exc:
            logger.debug("[Books] OL author fetch error: %s", exc)
    if authors:
        data["authors"] = authors

    return data


# ---------------------------------------------------------------------------
# Google Books fetcher
# ---------------------------------------------------------------------------

async def _fetch_google_books(isbn: str, api_key: str, client: httpx.AsyncClient) -> dict[str, Any]:
    """Fetch Google Books metadata for the given ISBN.

    Returns the same flat dict structure as _fetch_open_library.
    """
    data: dict[str, Any] = {}
    try:
        params: dict[str, str] = {"q": f"isbn:{isbn}"}
        if api_key:
            params["key"] = api_key
        resp = await client.get(_GB_URL, params=params)
        if resp.status_code != 200:
            logger.debug("[Books] Google Books HTTP %s for ISBN %s", resp.status_code, isbn)
            return {}
        raw = resp.json()
    except Exception as exc:
        logger.debug("[Books] Google Books fetch error for %s: %s", isbn, exc)
        return {}

    items = raw.get("items") or []
    if not items:
        return {}

    vi = items[0].get("volumeInfo") or {}
    if not isinstance(vi, dict):
        return {}

    title = vi.get("title") or ""
    if title:
        data["title"] = str(title).strip()

    authors = vi.get("authors") or []
    if authors:
        data["authors"] = [str(a).strip() for a in authors if a]

    publisher = vi.get("publisher") or ""
    if publisher:
        data["publisher"] = str(publisher).strip()

    year = _parse_year(vi.get("publishedDate"))
    if year:
        data["year"] = year

    pages = vi.get("pageCount")
    if pages is not None:
        try:
            data["pages"] = str(int(pages))
        except (ValueError, TypeError):
            pass

    categories = vi.get("categories") or []
    if categories:
        data["genres"] = [str(c).strip() for c in categories if c]

    lang_code = vi.get("language") or ""
    if lang_code:
        label = _normalise_lang_code(lang_code)
        data["languages"] = [label]

    description = vi.get("description") or ""
    if description:
        data["description"] = str(description)[:800]

    return data


# ---------------------------------------------------------------------------
# Merge two metadata dicts: prefer first (OL), fill from second (GB).
# ---------------------------------------------------------------------------

def _merge_metadata(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Merge two metadata dicts: primary (Open Library) wins per key."""
    result = dict(primary)
    for k, v in fallback.items():
        if k not in result or not result[k]:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Mapping: book field → target attributes
# ---------------------------------------------------------------------------

def _field_value_for_attr(
    field: str,
    meta: dict[str, Any],
) -> Optional[Union[str, list[str]]]:
    """Extract the value for a book metadata field.

    Returns a scalar for single-value fields (publisher, year, pages, language,
    isbn) or a list for multi-value fields (authors, genres).
    """
    if field == "authors":
        return meta.get("authors") or None
    if field == "publisher":
        return meta.get("publisher") or None
    if field == "year":
        return meta.get("year") or None
    if field == "pages":
        return meta.get("pages") or None
    if field == "language":
        langs = meta.get("languages") or []
        return langs[0] if langs else None
    if field == "genres":
        return meta.get("genres") or None
    if field == "isbn":
        return meta.get("isbn") or None
    return None


# ---------------------------------------------------------------------------
# BooksSource
# ---------------------------------------------------------------------------

class BooksSource(AttributeSource):
    """Verbatim book metadata from Open Library + Google Books APIs.

    Fires ONLY when context.ean is a book ISBN (978/979 prefix, 13 digits).
    ISBN is an exact product identifier — no brand-gate uncertainty, no LLM.

    Enum attributes: resolve_value_id; None → drop (verbatim-safe).
    Free-text and numeric attrs: verbatim.
    Emits Source.WB_CARD (verbatim, no LLM — same as RegardSource/OnlinerSource).
    """

    def __init__(self, **kwargs: Any) -> None:
        _ = kwargs
        self._judge = WbCardJudge()
        self._cache: OrderedDict[str, list[AttributeValue]] = OrderedDict()
        self._gb_api_key: str = os.environ.get(_GB_KEY_ENV, "")

    @property
    def source_type(self) -> Source:
        return Source.WB_CARD

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Applicable only when the EAN is a book ISBN (978/979 prefix)."""
        return _is_book_isbn(context.ean)

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        if not targets:
            return []

        # ISBN gate
        isbn = (context.ean or "").strip()
        if not _is_book_isbn(isbn):
            return []

        already_filled = already_filled or []

        # Skip-guard: ≥80% targets already filled → don't spend the API calls.
        filled_ids = {av.attribute_id for av in already_filled if av.confidence >= 0.85}
        if targets and len(filled_ids & {t.id for t in targets}) / len(targets) >= _SKIP_FILL_RATIO:
            logger.debug("[Books] skip — ≥%.0f%% targets filled", _SKIP_FILL_RATIO * 100)
            return []

        effective = filter_already_filled_targets(targets, already_filled)
        if not effective:
            return []

        # Cache by ISBN
        if isbn in self._cache:
            self._cache.move_to_end(isbn)
            return self._filter_for_targets(self._cache[isbn], effective)

        try:
            all_values = await asyncio.wait_for(
                self._do_extract(isbn, context, targets),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[Books] timeout (30s) for ISBN %s — skipping", isbn)
            return []
        except Exception as exc:
            logger.warning("[Books] unexpected error for ISBN %s: %s", isbn, exc)
            return []

        if all_values:
            self._cache_put(isbn, all_values)
        return self._filter_for_targets(all_values, effective)

    def get_judge(self) -> LlmJudge:
        return self._judge

    # ------------------------------------------------------------------
    # Core flow
    # ------------------------------------------------------------------

    async def _do_extract(
        self,
        isbn: str,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Fetch from Open Library and (optionally) Google Books; merge; map."""
        headers = {"User-Agent": _OL_USER_AGENT}
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
            headers=headers,
        ) as client:
            ol_task = _fetch_open_library(isbn, client)
            gb_task = (
                _fetch_google_books(isbn, self._gb_api_key, client)
                if self._gb_api_key
                else _empty_coro()
            )
            ol_meta, gb_meta = await asyncio.gather(ol_task, gb_task)

        meta = _merge_metadata(ol_meta, gb_meta)
        if not meta:
            logger.info("[Books] no metadata found for ISBN %s", isbn)
            return []

        # Always add ISBN itself so a target named «ISBN»/«Штрихкод» gets filled.
        meta["isbn"] = isbn

        logger.info(
            "[Books] ISBN %s → OL fields=%s GB fields=%s",
            isbn,
            list(ol_meta.keys()),
            list(gb_meta.keys()),
        )
        return self._map_metadata(meta, targets, context, isbn)

    # ------------------------------------------------------------------
    # Mapping: metadata → AttributeValue list
    # ------------------------------------------------------------------

    def _map_metadata(
        self,
        meta: dict[str, Any],
        targets: list[TargetAttribute],
        context: ExtractionContext,
        isbn: str,
    ) -> list[AttributeValue]:
        """Map book metadata fields to target attributes.

        Steps:
          1. Load Ozon dict names for attr_id → canonical display name.
          2. Build name lookup: target.name raw/norm + Ozon dict name.
          3. For each target: lookup field via _ATTR_NAME_TO_FIELD
             (exact substring → fuzzy WRatio≥88 fallback).
          4. Free-text/numeric: verbatim.
          5. Enum: resolve_value_id → None → DROP (verbatim-safe).
          6. Collection attrs: per-element for lists.
        """
        # Ozon dict names for richer name matching.
        ozon_chars: list[dict] = []
        cat_id: Optional[int] = None
        type_id: Optional[int] = None
        try:
            cat_id = int(context.category_id) if context.category_id else None
            type_id = context.ozon_type_id
            if cat_id and type_id:
                ozon_chars = get_ozon_characteristics_for_type(cat_id, type_id)
        except (ValueError, TypeError):
            cat_id = None
            type_id = None

        attr_id_to_dict_name: dict[int, str] = {}
        for oc in ozon_chars:
            if isinstance(oc, dict) and "id" in oc and "name" in oc:
                attr_id_to_dict_name[int(oc["id"])] = str(oc["name"])

        # Build target name → field mapping.
        target_by_id: dict[int, TargetAttribute] = {t.id: t for t in targets}

        # For each target, determine which book field it maps to.
        # Priority: exact substring match in _ATTR_NAME_TO_FIELD → fuzzy.
        def _target_to_field(target: TargetAttribute) -> Optional[str]:
            names_to_check = [target.name.lower()]
            dn = attr_id_to_dict_name.get(target.id)
            if dn:
                names_to_check.append(dn.lower())
            names_to_check.append(_norm_char_name(target.name))

            for name_low in names_to_check:
                # Exact key match
                if name_low in _ATTR_NAME_TO_FIELD:
                    return _ATTR_NAME_TO_FIELD[name_low]
                # Substring match: any known key is a substring of the target name
                for key, field in _ATTR_NAME_TO_FIELD.items():
                    if key in name_low or name_low in key:
                        return field

            # Fuzzy fallback via rapidfuzz
            try:
                from rapidfuzz import process, fuzz
                field_keys = list(_ATTR_NAME_TO_FIELD.keys())
                q = _norm_char_name(target.name)
                best = process.extractOne(q, field_keys, scorer=fuzz.WRatio)
                if best is not None and best[1] >= _FUZZY_THRESHOLD:
                    return _ATTR_NAME_TO_FIELD[best[0]]
            except ImportError:
                pass

            return None

        results: list[AttributeValue] = []
        used_fields: set[str] = set()  # prevent double-filling same field

        for target in targets:
            field = _target_to_field(target)
            if field is None:
                continue

            raw_value = _field_value_for_attr(field, meta)
            if raw_value is None:
                continue

            evidence = f"books:isbn={isbn}|ol+gb"

            value_id: Optional[int] = None
            value_ids: Optional[list[int]] = None
            value_out: Union[str, list[str]]

            if target.is_collection or isinstance(raw_value, list):
                # Multi-value: e.g. authors, genres
                parts: list[str] = (
                    raw_value if isinstance(raw_value, list)
                    else _split_multivalue(str(raw_value))
                )
                value_out = parts

                if target.type == "enum" and cat_id and type_id:
                    # Enum collection: resolve each element; keep resolved subset.
                    resolved_ids: list[Optional[int]] = []
                    resolved_parts: list[str] = []
                    for p in parts:
                        try:
                            vid = resolve_value_id(cat_id, type_id, target.id, p)
                        except Exception:
                            vid = None
                        if vid is not None:
                            resolved_ids.append(vid)
                            resolved_parts.append(p)
                    if not resolved_parts:
                        logger.debug(
                            "[Books] enum collection attr %s: no elements resolved — drop",
                            target.id,
                        )
                        continue
                    value_out = resolved_parts
                    value_ids = resolved_ids
                elif cat_id and type_id:
                    # Non-enum: try to resolve but don't drop on miss.
                    try:
                        resolved = [
                            resolve_value_id(cat_id, type_id, target.id, p)
                            for p in parts
                        ]
                        if any(r is not None for r in resolved):
                            value_ids = resolved
                    except Exception as exc:
                        logger.debug("[Books] resolve_value_id (list) failed: %s", exc)

            else:
                # Scalar value
                value_out = str(raw_value).strip()

                if target.type == "enum" and cat_id and type_id:
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, value_out)
                    except Exception as exc:
                        logger.debug("[Books] resolve_value_id failed: %s", exc)
                    if value_id is None:
                        logger.debug(
                            "[Books] enum attr %s value '%s' not in dict — drop",
                            target.id, value_out[:40],
                        )
                        continue
                elif cat_id and type_id:
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, value_out)
                    except Exception as exc:
                        logger.debug("[Books] resolve_value_id (non-enum) failed: %s", exc)

            # Mark field as used (but allow multiple targets to map the same field
            # only if they differ by is_collection vs scalar).
            field_key = f"{field}:{'collection' if target.is_collection else 'scalar'}"
            if field_key in used_fields:
                continue
            used_fields.add(field_key)

            results.append(AttributeValue(
                attribute_id=target.id,
                value=value_out,
                confidence=_CONF,
                source=Source.WB_CARD,
                evidence=evidence,
                semantic_type=target.semantic_type,
                is_collection=target.is_collection,
                value_id=value_id,
                value_ids=value_ids,
            ))

        logger.info(
            "[Books] ISBN %s → %d attrs filled (from %d targets)",
            isbn, len(results), len(targets),
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_for_targets(
        values: list[AttributeValue],
        effective: list[TargetAttribute],
    ) -> list[AttributeValue]:
        eff_ids = {t.id for t in effective}
        return [v for v in values if v.attribute_id in eff_ids]

    def _cache_put(self, key: str, value: list[AttributeValue]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)
