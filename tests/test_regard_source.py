"""Unit tests for RegardSource — no live network, no LLM.

Covers:
  • _parse_regard_specs: spec table → (title, specs) pairs.
  • _brand_gate: pass on correct brand+model tokens; fail on mismatch.
  • _score_title / _classify_match: scoring thresholds.
  • extract(): correct fills, brand-mismatch page → [], no-match enum → dropped.
  • _regard_product_re: URL recognition.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.sources.regard_source import (
    RegardSource,
    _parse_regard_specs,
    _REGARD_PRODUCT_RE,
)
from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)


# ---------------------------------------------------------------------------
# Fixture HTML — minimal regard.ru product page
# ---------------------------------------------------------------------------

REGARD_SAMPLE_HTML = """<!DOCTYPE html>
<html><head>
<title>Intel Core i5-12400 OEM — regard.ru</title>
</head><body>
<h1>Intel Core i5-12400 OEM</h1>
<div class="product-description">
  <h2>Характеристики</h2>
  <table>
    <tr><td>Процессорный разъём</td><td>LGA1700</td></tr>
    <tr><td>Количество ядер</td><td>6</td></tr>
    <tr><td>Базовая частота</td><td>2500 МГц</td></tr>
    <tr><td>Кэш L3</td><td>18 МБ</td></tr>
    <tr><td>Артикул</td><td>BX8071512400</td></tr>
    <tr><td>Бренд</td><td>Intel</td></tr>
  </table>
</div>
</body></html>
"""

# Brand-mismatch page: AMD CPU page served for Intel query.
REGARD_WRONG_BRAND_HTML = """<!DOCTYPE html>
<html><head>
<title>AMD Ryzen 5 5600 OEM — regard.ru</title>
</head><body>
<h2>Характеристики</h2>
<table>
  <tr><td>Процессорный разъём</td><td>AM4</td></tr>
  <tr><td>Количество ядер</td><td>6</td></tr>
</table>
</body></html>
"""

# Page with no spec table.
REGARD_NO_SPECS_HTML = """<!DOCTYPE html>
<html><head><title>Intel Core i5-12400 — regard.ru</title></head>
<body><p>Нет характеристик</p></body></html>
"""


# ---------------------------------------------------------------------------
# URL recognition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://www.regard.ru/catalog/product/123456", True),
    ("https://regard.ru/catalog/product/789", True),
    ("https://regard.ru/goods/4567890", True),
    ("https://regard.ru/catalog/", False),
    ("https://citilink.ru/catalog/product/123456", False),
])
def test_regard_product_re(url: str, expected: bool) -> None:
    assert bool(_REGARD_PRODUCT_RE.search(url)) == expected


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def test_parse_specs_extracts_title_and_pairs() -> None:
    title, specs = _parse_regard_specs(REGARD_SAMPLE_HTML)
    assert "i5-12400" in title.lower() or "intel" in title.lower()
    names = {s["name"] for s in specs}
    assert "Процессорный разъём" in names
    assert "Количество ядер" in names
    assert "Базовая частота" in names


def test_parse_specs_deduplicates_rows() -> None:
    # Duplicate spec rows should yield only one entry.
    html = REGARD_SAMPLE_HTML.replace(
        "<tr><td>Кэш L3</td><td>18 МБ</td></tr>",
        "<tr><td>Кэш L3</td><td>18 МБ</td></tr><tr><td>Кэш L3</td><td>18 МБ</td></tr>",
    )
    _, specs = _parse_regard_specs(html)
    cache_entries = [s for s in specs if s["name"] == "Кэш L3"]
    assert len(cache_entries) == 1


def test_parse_specs_no_spec_section_returns_empty() -> None:
    title, specs = _parse_regard_specs(REGARD_NO_SPECS_HTML)
    assert specs == []


def test_parse_specs_values_correct() -> None:
    _, specs = _parse_regard_specs(REGARD_SAMPLE_HTML)
    val_map = {s["name"]: s["value"] for s in specs}
    assert val_map["Процессорный разъём"] == "LGA1700"
    assert val_map["Количество ядер"] == "6"


# ---------------------------------------------------------------------------
# Brand-gate
# ---------------------------------------------------------------------------

def test_brand_gate_pass_correct_brand_and_model() -> None:
    title = "Intel Core i5-12400 OEM — regard.ru"
    assert RegardSource._brand_gate(title, "Intel", "Процессор Intel Core i5-12400")


def test_brand_gate_fail_wrong_brand() -> None:
    title = "AMD Ryzen 5 5600 OEM — regard.ru"
    # brand=Intel, model token i5-12400 not in AMD title
    assert not RegardSource._brand_gate(title, "Intel", "Процессор Intel Core i5-12400")


def test_brand_gate_fail_model_mismatch() -> None:
    # Title has Intel but wrong model number.
    title = "Intel Core i9-13900K OEM — regard.ru"
    # product_name has model token "i5" and "12400" — neither in i9-13900k title
    result = RegardSource._brand_gate(title, "Intel", "Процессор Intel Core i5-12400")
    # i5 / 12400 not in "i9-13900k" — gate should fail
    assert not result


def test_brand_gate_no_brand_uses_model_only() -> None:
    title = "i5-12400 — regard.ru"
    # brand empty → only model tokens matter
    assert RegardSource._brand_gate(title, "", "Intel i5-12400")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_classify_match_exact() -> None:
    assert RegardSource._classify_match(80.0) == "exact"


def test_classify_match_brand_line() -> None:
    assert RegardSource._classify_match(65.0) == "brand_line"


def test_classify_match_skip() -> None:
    assert RegardSource._classify_match(30.0) == "skip"


# ---------------------------------------------------------------------------
# extract() — unit (no live network)
# ---------------------------------------------------------------------------

def _make_target(attr_id: int, name: str, attr_type: str = "text") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=attr_type)


def _make_context(product_name: str, brand: str = "Intel") -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name=product_name,
        category_id=7500,
        brand=brand,
    )


def _make_serper_result(url: str):
    item = MagicMock()
    item.link = url
    result = MagicMock()
    result.organic_results = [item]
    return result


@pytest.mark.asyncio
async def test_extract_fills_from_correct_page() -> None:
    """Happy path: Serper finds regard.ru page, brand-gate passes, attrs filled."""
    search_client = MagicMock()
    search_client.search = AsyncMock(return_value=_make_serper_result(
        "https://www.regard.ru/catalog/product/123456"
    ))

    source = RegardSource(web_search_client=search_client)

    targets = [
        _make_target(1, "Процессорный разъём"),
        _make_target(2, "Количество ядер"),
    ]
    context = _make_context("Процессор Intel Core i5-12400")

    with patch.object(source, "_fetch_page", new=AsyncMock(return_value=REGARD_SAMPLE_HTML)):
        results = await source.extract(context, targets)

    assert len(results) == 2
    names_filled = {r.attribute_id for r in results}
    assert 1 in names_filled
    assert 2 in names_filled
    for r in results:
        assert r.source == Source.WB_CARD
        assert r.confidence >= 0.82


@pytest.mark.asyncio
async def test_extract_brand_mismatch_returns_empty() -> None:
    """Brand-mismatch page: gate fails → no fills."""
    search_client = MagicMock()
    search_client.search = AsyncMock(return_value=_make_serper_result(
        "https://www.regard.ru/catalog/product/999"
    ))

    source = RegardSource(web_search_client=search_client)

    targets = [_make_target(1, "Процессорный разъём")]
    # Querying Intel i5-12400 but page is AMD Ryzen
    context = _make_context("Intel Core i5-12400", brand="Intel")

    with patch.object(source, "_fetch_page", new=AsyncMock(return_value=REGARD_WRONG_BRAND_HTML)):
        results = await source.extract(context, targets)

    assert results == []


@pytest.mark.asyncio
async def test_extract_enum_attr_dropped_when_no_match() -> None:
    """Enum target: value not in Ozon dict → dropped (verbatim-safe)."""
    search_client = MagicMock()
    search_client.search = AsyncMock(return_value=_make_serper_result(
        "https://www.regard.ru/catalog/product/123456"
    ))

    source = RegardSource(web_search_client=search_client)

    # Enum target — resolve_value_id will return None for arbitrary value
    targets = [_make_target(99, "Процессорный разъём", attr_type="enum")]
    context = ExtractionContext(
        product_id=1,
        product_name="Intel Core i5-12400",
        category_id=7500,
        brand="Intel",
        ozon_type_id=1234,
    )

    with patch.object(source, "_fetch_page", new=AsyncMock(return_value=REGARD_SAMPLE_HTML)):
        # Patch resolve_value_id to always return None (value not in dict)
        with patch(
            "app.services.enrichment.sources.regard_source.resolve_value_id",
            return_value=None,
        ):
            results = await source.extract(context, targets)

    # Enum value not resolved → dropped, result is empty
    assert results == []


@pytest.mark.asyncio
async def test_extract_no_serper_results_returns_empty() -> None:
    """No Serper results → empty extract."""
    search_client = MagicMock()
    search_client.search = AsyncMock(return_value=MagicMock(organic_results=[]))

    source = RegardSource(web_search_client=search_client)
    targets = [_make_target(1, "Процессорный разъём")]
    context = _make_context("Intel Core i5-12400")

    results = await source.extract(context, targets)
    assert results == []


@pytest.mark.asyncio
async def test_extract_skip_guard_all_filled() -> None:
    """If ≥80% targets already filled with high confidence → skip."""
    search_client = MagicMock()
    search_client.search = AsyncMock()  # should NOT be called

    source = RegardSource(web_search_client=search_client)
    targets = [_make_target(i, f"Attr {i}") for i in range(10)]
    context = _make_context("Intel Core i5-12400")

    # All 10 targets already filled at high confidence
    already_filled = [
        AttributeValue(
            attribute_id=t.id,
            value="val",
            confidence=0.95,
            source=Source.WB_CARD,
        )
        for t in targets
    ]

    results = await source.extract(context, targets, already_filled=already_filled)

    assert results == []
    search_client.search.assert_not_called()


@pytest.mark.asyncio
async def test_extract_no_search_client_returns_empty() -> None:
    """No search client → extract returns []."""
    source = RegardSource(web_search_client=None)
    # Manually clear to ensure None (factory might succeed in test env)
    source._search_client = None

    targets = [_make_target(1, "Attr")]
    context = _make_context("Intel i5")

    results = await source.extract(context, targets)
    assert results == []
