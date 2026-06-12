"""Unit tests for BooksSource — no live network, no LLM.

Covers:
  • _is_book_isbn: 978/979 detect; non-book EAN; edge cases.
  • _parse_year: various date formats.
  • _normalise_lang_code: ISO 639-1/2 and OL path form.
  • _merge_metadata: OL wins per-key; fallback from GB.
  • _fetch_open_library: fixture JSON → flat dict (mocked httpx).
  • _fetch_google_books: fixture JSON → flat dict (mocked httpx).
  • extract(): book ISBN → fills attrs; non-book EAN → []; 404 → [].
  • extract(): enum target dropped when resolve_value_id returns None.
  • extract(): skip-guard ≥80% filled → source doesn't call API.
  • extract(): authors as collection target.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.sources.books_source import (
    BooksSource,
    _is_book_isbn,
    _parse_year,
    _normalise_lang_code,
    _merge_metadata,
    _fetch_open_library,
    _fetch_google_books,
)
from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)


# ---------------------------------------------------------------------------
# Fixtures: sample API responses
# ---------------------------------------------------------------------------

OL_EDITION_JSON: dict = {
    "title": "Clean Code",
    "publishers": ["Prentice Hall"],
    "publish_date": "2008-08-01",
    "number_of_pages": 431,
    "languages": [{"key": "/languages/eng"}],
    "works": [{"key": "/works/OL1234W"}],
    "authors": [{"key": "/authors/OL1A"}],
}

OL_WORK_JSON: dict = {
    "subjects": [
        "Computer programming",
        "Software engineering (Object-oriented methods)",
        "Clean code",
    ],
}

OL_AUTHOR_JSON: dict = {
    "name": "Robert C. Martin",
    "personal_name": "Robert C. Martin",
}

GB_RESPONSE_JSON: dict = {
    "items": [
        {
            "volumeInfo": {
                "title": "Clean Code: A Handbook of Agile Software Craftsmanship",
                "authors": ["Robert C. Martin"],
                "publisher": "Prentice Hall",
                "publishedDate": "2008-08-01",
                "pageCount": 431,
                "categories": ["Computers / Programming / General"],
                "language": "en",
                "description": "A landmark programming book about writing clean, maintainable code.",
            }
        }
    ]
}

RU_OL_EDITION_JSON: dict = {
    "title": "Мастер и Маргарита",
    "publishers": ["Азбука"],
    "publish_date": "2020",
    "number_of_pages": 480,
    "languages": [{"key": "/languages/rus"}],
    "works": [{"key": "/works/OL5678W"}],
    "authors": [{"key": "/authors/OL2A"}],
}

RU_OL_WORK_JSON: dict = {
    "subjects": [
        "Русская литература",
        "Советская литература",
        "Роман",
    ],
}

RU_OL_AUTHOR_JSON: dict = {
    "name": "Михаил Булгаков",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(
    attr_id: int,
    name: str,
    attr_type: str = "text",
    is_collection: bool = False,
) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=attr_type, is_collection=is_collection)


def _make_context(ean: str = "9780132350884") -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name="Clean Code Robert Martin",
        category_id=90635,
        ean=ean,
    )


# ---------------------------------------------------------------------------
# _is_book_isbn
# ---------------------------------------------------------------------------

def test_is_book_isbn_valid_978() -> None:
    assert _is_book_isbn("9780132350884")


def test_is_book_isbn_valid_979() -> None:
    assert _is_book_isbn("9791032309049")


def test_is_book_isbn_hyphenated() -> None:
    assert _is_book_isbn("978-0-13-235088-4")


def test_is_book_isbn_non_book_ean() -> None:
    # Non-book EAN — starts with 4
    assert not _is_book_isbn("4607075010313")


def test_is_book_isbn_short_barcode() -> None:
    assert not _is_book_isbn("12345")


def test_is_book_isbn_none() -> None:
    assert not _is_book_isbn(None)


def test_is_book_isbn_empty() -> None:
    assert not _is_book_isbn("")


def test_is_book_isbn_wrong_length() -> None:
    # 12 digits starting with 978 → not valid ISBN-13
    assert not _is_book_isbn("978013235088")


# ---------------------------------------------------------------------------
# _parse_year
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("date_str,expected", [
    ("2021", "2021"),
    ("2021-05-01", "2021"),
    ("May 2021", "2021"),
    ("August 1, 2008", "2008"),
    (None, None),
    ("", None),
    ("No date here", None),
    ("1999-12-31", "1999"),
])
def test_parse_year(date_str, expected) -> None:
    from app.services.enrichment.sources.books_source import _parse_year
    assert _parse_year(date_str) == expected


# ---------------------------------------------------------------------------
# _normalise_lang_code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected", [
    ("en", "Английский"),
    ("eng", "Английский"),
    ("/languages/eng", "Английский"),
    ("/languages/rus", "Русский"),
    ("ru", "Русский"),
    ("de", "Немецкий"),
    ("fr", "Французский"),
    ("ja", "Японский"),
    ("unknown_xyz", "unknown_xyz"),  # passthrough for unknown codes
])
def test_normalise_lang_code(code, expected) -> None:
    assert _normalise_lang_code(code) == expected


# ---------------------------------------------------------------------------
# _merge_metadata
# ---------------------------------------------------------------------------

def test_merge_metadata_primary_wins() -> None:
    primary = {"title": "Clean Code", "publisher": "Prentice Hall", "year": "2008"}
    fallback = {"title": "Different Title", "publisher": "Other Publisher", "pages": "431"}
    result = _merge_metadata(primary, fallback)
    assert result["title"] == "Clean Code"
    assert result["publisher"] == "Prentice Hall"
    assert result["pages"] == "431"  # from fallback


def test_merge_metadata_fallback_fills_missing() -> None:
    primary = {"title": "Book A"}
    fallback = {"year": "2020", "languages": ["Русский"]}
    result = _merge_metadata(primary, fallback)
    assert result["year"] == "2020"
    assert result["languages"] == ["Русский"]


def test_merge_metadata_empty_primary_uses_fallback() -> None:
    primary: dict = {}
    fallback = {"title": "Fallback Title", "pages": "100"}
    result = _merge_metadata(primary, fallback)
    assert result["title"] == "Fallback Title"
    assert result["pages"] == "100"


# ---------------------------------------------------------------------------
# _fetch_open_library (mocked httpx)
# ---------------------------------------------------------------------------

def _build_mock_client(responses: dict[str, dict]) -> MagicMock:
    """Build an async httpx client mock mapping URL → JSON response."""
    async def _get(url, **_kwargs):
        for key, data in responses.items():
            if key in url:
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = data
                return resp
        # 404 for everything else
        resp = MagicMock()
        resp.status_code = 404
        return resp

    client = MagicMock()
    client.get = _get
    return client


@pytest.mark.asyncio
async def test_fetch_open_library_happy_path() -> None:
    client = _build_mock_client({
        "/isbn/9780132350884": OL_EDITION_JSON,
        "/works/OL1234W": OL_WORK_JSON,
        "/authors/OL1A": OL_AUTHOR_JSON,
    })
    result = await _fetch_open_library("9780132350884", client)
    assert result["title"] == "Clean Code"
    assert result["publisher"] == "Prentice Hall"
    assert result["year"] == "2008"
    assert result["pages"] == "431"
    assert "Английский" in result["languages"]
    assert "Robert C. Martin" in result["authors"]
    assert any("Computer programming" in g for g in result["genres"])


@pytest.mark.asyncio
async def test_fetch_open_library_404_returns_empty() -> None:
    async def _get(url, **_kwargs):
        resp = MagicMock()
        resp.status_code = 404
        return resp

    client = MagicMock()
    client.get = _get
    result = await _fetch_open_library("9780000000000", client)
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_open_library_network_error_returns_empty() -> None:
    async def _get(url, **_kwargs):
        raise ConnectionError("network error")

    client = MagicMock()
    client.get = _get
    result = await _fetch_open_library("9780132350884", client)
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_open_library_ru_book() -> None:
    client = _build_mock_client({
        "/isbn/9785389163430": RU_OL_EDITION_JSON,
        "/works/OL5678W": RU_OL_WORK_JSON,
        "/authors/OL2A": RU_OL_AUTHOR_JSON,
    })
    result = await _fetch_open_library("9785389163430", client)
    assert "Русский" in result["languages"]
    assert "Михаил Булгаков" in result["authors"]
    assert result["year"] == "2020"


# ---------------------------------------------------------------------------
# _fetch_google_books (mocked httpx)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_google_books_happy_path() -> None:
    async def _get(url, params=None, **_kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = GB_RESPONSE_JSON
        return resp

    client = MagicMock()
    client.get = _get
    result = await _fetch_google_books("9780132350884", "fake-key", client)
    assert result["title"] == "Clean Code: A Handbook of Agile Software Craftsmanship"
    assert "Robert C. Martin" in result["authors"]
    assert result["publisher"] == "Prentice Hall"
    assert result["year"] == "2008"
    assert result["pages"] == "431"
    assert result["languages"] == ["Английский"]
    assert "description" in result


@pytest.mark.asyncio
async def test_fetch_google_books_empty_items_returns_empty() -> None:
    async def _get(url, params=None, **_kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"items": []}
        return resp

    client = MagicMock()
    client.get = _get
    result = await _fetch_google_books("9780132350884", "fake-key", client)
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_google_books_http_error_returns_empty() -> None:
    async def _get(url, params=None, **_kwargs):
        resp = MagicMock()
        resp.status_code = 403
        return resp

    client = MagicMock()
    client.get = _get
    result = await _fetch_google_books("9780132350884", "fake-key", client)
    assert result == {}


# ---------------------------------------------------------------------------
# BooksSource.extract() — integration-style (mocked _do_extract)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_non_book_ean_returns_empty() -> None:
    """Non-book EAN (not 978/979) → source skips entirely."""
    source = BooksSource()
    targets = [_make_target(1, "Автор")]
    context = _make_context(ean="4607075010313")
    result = await source.extract(context, targets)
    assert result == []


@pytest.mark.asyncio
async def test_extract_no_ean_returns_empty() -> None:
    """Missing EAN → source skips."""
    source = BooksSource()
    targets = [_make_target(1, "Автор")]
    context = ExtractionContext(
        product_id=1,
        product_name="Some Book",
        category_id=90635,
        ean=None,
    )
    result = await source.extract(context, targets)
    assert result == []


@pytest.mark.asyncio
async def test_extract_fills_free_text_attrs() -> None:
    """Happy path: ISBN → OL + GB return metadata → attrs filled verbatim."""
    source = BooksSource()

    fake_meta = {
        "isbn": "9780132350884",
        "title": "Clean Code",
        "authors": ["Robert C. Martin"],
        "publisher": "Prentice Hall",
        "year": "2008",
        "pages": "431",
        "languages": ["Английский"],
        "genres": ["Computer programming", "Software engineering"],
    }

    targets = [
        _make_target(1, "Автор"),
        _make_target(2, "Издательство"),
        _make_target(3, "Год издания"),
        _make_target(4, "Количество страниц"),
    ]
    context = _make_context(ean="9780132350884")

    with patch.object(source, "_do_extract", new=AsyncMock(
        return_value=source._map_metadata(fake_meta, targets, context, "9780132350884")
    )):
        results = await source.extract(context, targets)

    attr_ids = {r.attribute_id for r in results}
    assert 2 in attr_ids  # Издательство
    assert 3 in attr_ids  # Год издания
    assert 4 in attr_ids  # Количество страниц
    for r in results:
        assert r.source == Source.WB_CARD
        assert r.confidence >= 0.90
        assert "books:isbn=" in (r.evidence or "")


@pytest.mark.asyncio
async def test_extract_isbn_target_always_filled() -> None:
    """A target named 'ISBN' gets the verbatim ISBN string."""
    source = BooksSource()

    fake_meta = {
        "isbn": "9780132350884",
        "title": "Clean Code",
    }

    targets = [_make_target(99, "ISBN")]
    context = _make_context(ean="9780132350884")

    with patch.object(source, "_do_extract", new=AsyncMock(
        return_value=source._map_metadata(fake_meta, targets, context, "9780132350884")
    )):
        results = await source.extract(context, targets)

    assert any(r.attribute_id == 99 and r.value == "9780132350884" for r in results)


@pytest.mark.asyncio
async def test_extract_enum_attr_dropped_when_no_match() -> None:
    """Enum target: value not found in Ozon dict → dropped (verbatim-safe)."""
    source = BooksSource()

    fake_meta = {
        "isbn": "9780132350884",
        "languages": ["Английский"],
    }

    targets = [_make_target(5, "Язык", attr_type="enum")]
    context = ExtractionContext(
        product_id=1,
        product_name="Clean Code",
        category_id=90635,
        ean="9780132350884",
        ozon_type_id=999,
    )

    mapped = source._map_metadata(fake_meta, targets, context, "9780132350884")

    with (
        patch.object(source, "_do_extract", new=AsyncMock(return_value=mapped)),
        patch(
            "app.services.enrichment.sources.books_source.resolve_value_id",
            return_value=None,
        ),
    ):
        results = await source.extract(context, targets)

    assert results == []


@pytest.mark.asyncio
async def test_extract_skip_guard_all_filled() -> None:
    """≥80% targets already filled → source skips without calling APIs."""
    source = BooksSource()

    targets = [_make_target(i, f"Attr {i}") for i in range(10)]
    context = _make_context(ean="9780132350884")

    already_filled = [
        AttributeValue(
            attribute_id=t.id,
            value="val",
            confidence=0.95,
            source=Source.WB_CARD,
        )
        for t in targets
    ]

    do_extract_mock = AsyncMock(return_value=[])
    with patch.object(source, "_do_extract", new=do_extract_mock):
        results = await source.extract(context, targets, already_filled=already_filled)

    assert results == []
    do_extract_mock.assert_not_called()


@pytest.mark.asyncio
async def test_extract_404_returns_empty() -> None:
    """Open Library 404 + Google Books empty → extract returns []."""
    source = BooksSource()
    source._gb_api_key = ""  # disable GB

    targets = [_make_target(1, "Автор")]
    context = _make_context(ean="9780000000000")

    async def _get(url, **_kwargs):
        resp = MagicMock()
        resp.status_code = 404
        return resp

    with patch("httpx.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=MagicMock(get=_get))
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client
        results = await source.extract(context, targets)

    assert results == []


@pytest.mark.asyncio
async def test_extract_authors_collection_target() -> None:
    """Authors mapped to a is_collection target produce a list value."""
    source = BooksSource()

    fake_meta = {
        "isbn": "9780132350884",
        "authors": ["Robert C. Martin", "Co Author"],
    }

    targets = [_make_target(7, "Авторы", attr_type="text", is_collection=True)]
    context = _make_context(ean="9780132350884")

    mapped = source._map_metadata(fake_meta, targets, context, "9780132350884")
    assert any(r.attribute_id == 7 and isinstance(r.value, list) for r in mapped)
    for r in mapped:
        if r.attribute_id == 7:
            assert "Robert C. Martin" in r.value


# ---------------------------------------------------------------------------
# BooksSource.is_applicable
# ---------------------------------------------------------------------------

def test_is_applicable_book_isbn() -> None:
    source = BooksSource()
    target = _make_target(1, "Автор")
    context = _make_context(ean="9780132350884")
    assert source.is_applicable(context, target)


def test_is_applicable_non_book_ean() -> None:
    source = BooksSource()
    target = _make_target(1, "Автор")
    context = _make_context(ean="4607075010313")
    assert not source.is_applicable(context, target)


def test_is_applicable_no_ean() -> None:
    source = BooksSource()
    target = _make_target(1, "Автор")
    context = ExtractionContext(
        product_id=1,
        product_name="Some Book",
        category_id=90635,
        ean=None,
    )
    assert not source.is_applicable(context, target)
