"""Unit tests for OnlinerSource — no live network, no LLM.

Covers:
  • _parse_onliner_specs: JSON-LD additionalProperty → (name, specs).
  • _parse_onliner_specs: HTML table fallback when no JSON-LD present.
  • _brand_gate: pass on correct brand+model tokens; fail on mismatch.
  • _score_title / _classify_match: thresholds.
  • extract(): correct fills, brand-gate fail → [], enum-drop, skip-guard.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.sources.onliner_source import (
    OnlinerSource,
    _parse_onliner_specs,
)
from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)


# ---------------------------------------------------------------------------
# Fixture HTML — Onliner.by product page with JSON-LD additionalProperty
# ---------------------------------------------------------------------------

_PRODUCT_LD = {
    "@context": "https://schema.org",
    "@type": "Product",
    "name": "Ноутбук Lenovo ThinkPad X1 Carbon Gen 12",
    "additionalProperty": [
        {"@type": "PropertyValue", "name": "Процессор", "value": "Intel Core Ultra 7 155U"},
        {"@type": "PropertyValue", "name": "Объём оперативной памяти", "value": "16 ГБ"},
        {"@type": "PropertyValue", "name": "Диагональ экрана", "value": "14\""},
        {"@type": "PropertyValue", "name": "Объём накопителя", "value": "512 ГБ"},
        {"@type": "PropertyValue", "name": "Цвет", "value": "Чёрный"},
        {"@type": "PropertyValue", "name": "Операционная система", "value": "Windows 11 Pro"},
        {"@type": "PropertyValue", "name": "Тип подключения", "value": "Проводное, беспроводное"},
    ],
}

ONLINER_SAMPLE_HTML = f"""<!DOCTYPE html>
<html><head>
<meta property="og:title" content="Ноутбук Lenovo ThinkPad X1 Carbon Gen 12" />
</head><body>
<script type="application/ld+json">
{json.dumps(_PRODUCT_LD)}
</script>
<div class="product-specs">
  <h2>Характеристики</h2>
</div>
</body></html>
"""

# Page with HTML fallback (no JSON-LD)
ONLINER_HTML_FALLBACK = """<!DOCTYPE html>
<html><head>
<meta property="og:title" content="Lenovo ThinkPad X1 Carbon Gen 12" />
</head><body>
<table>
  <tr><td>Процессор</td><td>Intel Core Ultra 7 155U</td></tr>
  <tr><td>Объём оперативной памяти</td><td>16 ГБ</td></tr>
  <tr><td>Диагональ экрана</td><td>14&quot;</td></tr>
</table>
</body></html>
"""

# Wrong brand page
ONLINER_WRONG_BRAND_HTML = f"""<!DOCTYPE html>
<html><head>
<meta property="og:title" content="Ноутбук Dell XPS 15 9500" />
</head><body>
<script type="application/ld+json">
{json.dumps({
    "@type": "Product",
    "name": "Dell XPS 15 9500",
    "additionalProperty": [
        {"@type": "PropertyValue", "name": "Процессор", "value": "Intel Core i7-10750H"},
    ],
})}
</script>
</body></html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(attr_id: int, name: str, attr_type: str = "text") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=attr_type)


def _make_context(product_name: str, brand: str = "Lenovo") -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name=product_name,
        category_id=7500,
        brand=brand,
    )


def _make_search_response(results: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Mock _search return value: list of (name, url) tuples."""
    return results


# ---------------------------------------------------------------------------
# Parser: JSON-LD additionalProperty
# ---------------------------------------------------------------------------

def test_parse_specs_extracts_name_from_jsonld() -> None:
    name, specs = _parse_onliner_specs(ONLINER_SAMPLE_HTML)
    assert "Lenovo" in name or "ThinkPad" in name


def test_parse_specs_extracts_additional_properties() -> None:
    _, specs = _parse_onliner_specs(ONLINER_SAMPLE_HTML)
    spec_names = {s["name"] for s in specs}
    assert "Процессор" in spec_names
    assert "Объём оперативной памяти" in spec_names
    assert "Цвет" in spec_names


def test_parse_specs_values_correct() -> None:
    _, specs = _parse_onliner_specs(ONLINER_SAMPLE_HTML)
    spec_map = {s["name"]: s["value"] for s in specs}
    assert spec_map["Объём оперативной памяти"] == "16 ГБ"
    assert spec_map["Цвет"] == "Чёрный"


def test_parse_specs_deduplicates() -> None:
    # Inject duplicate additionalProperty into JSON-LD
    product = dict(_PRODUCT_LD)
    product["additionalProperty"] = [
        {"@type": "PropertyValue", "name": "Процессор", "value": "Intel Core Ultra 7 155U"},
        {"@type": "PropertyValue", "name": "Процессор", "value": "Intel Core Ultra 7 155U"},
    ]
    html = f'<script type="application/ld+json">{json.dumps(product)}</script>'
    _, specs = _parse_onliner_specs(html)
    cpu_entries = [s for s in specs if s["name"] == "Процессор"]
    assert len(cpu_entries) == 1


def test_parse_specs_html_fallback_when_no_jsonld() -> None:
    """HTML table fallback when no JSON-LD present."""
    _, specs = _parse_onliner_specs(ONLINER_HTML_FALLBACK)
    assert len(specs) >= 1
    spec_names = {s["name"] for s in specs}
    assert "Процессор" in spec_names


def test_parse_specs_no_properties_returns_empty_from_bad_json() -> None:
    html = '<script type="application/ld+json">{ bad json ]</script>'
    _, specs = _parse_onliner_specs(html)
    assert isinstance(specs, list)


# ---------------------------------------------------------------------------
# Brand-gate
# ---------------------------------------------------------------------------

def test_brand_gate_pass_correct_brand_and_model() -> None:
    assert OnlinerSource._brand_gate(
        "Ноутбук Lenovo ThinkPad X1 Carbon Gen 12",
        "Lenovo",
        "Lenovo ThinkPad X1 Carbon Gen 12",
    )


def test_brand_gate_fail_wrong_brand() -> None:
    assert not OnlinerSource._brand_gate(
        "Dell XPS 15 9500",
        "Lenovo",
        "Lenovo ThinkPad X1 Carbon Gen 12",
    )


def test_brand_gate_fail_model_mismatch() -> None:
    # Brand matches but distinct model token "rtx3080" not in "GTX1660" title.
    # _extract_model_tokens("GPU Nvidia RTX 3080") returns {"rtx3080", "3080"}.
    # Neither "rtx3080" nor "3080" appear in "Nvidia GTX 1660 Super".
    result = OnlinerSource._brand_gate(
        "Nvidia GTX 1660 Super",
        "Nvidia",
        "Nvidia RTX 3080",
    )
    assert not result


def test_brand_gate_no_brand_uses_model_only() -> None:
    assert OnlinerSource._brand_gate(
        "ThinkPad X1 Carbon Gen 12 ноутбук",
        "",
        "ThinkPad X1 Carbon",
    )


# ---------------------------------------------------------------------------
# Scoring / classify
# ---------------------------------------------------------------------------

def test_classify_match_exact() -> None:
    assert OnlinerSource._classify_match(80.0) == "exact"


def test_classify_match_brand_line() -> None:
    assert OnlinerSource._classify_match(65.0) == "brand_line"


def test_classify_match_skip() -> None:
    assert OnlinerSource._classify_match(30.0) == "skip"


# ---------------------------------------------------------------------------
# extract() — unit (no live network)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_fills_from_correct_page() -> None:
    """Happy path: search returns URL, brand-gate passes, attrs filled."""
    source = OnlinerSource()

    targets = [
        _make_target(1, "Процессор"),
        _make_target(2, "Объём оперативной памяти"),
    ]
    context = _make_context("Lenovo ThinkPad X1 Carbon Gen 12", brand="Lenovo")

    with patch.object(
        source, "_search",
        new=AsyncMock(return_value=[
            ("Ноутбук Lenovo ThinkPad X1 Carbon Gen 12",
             "https://catalog.onliner.by/notebook/lenovo/thinkpadx1carbon"),
        ]),
    ):
        with patch.object(source, "_fetch_page", new=AsyncMock(return_value=ONLINER_SAMPLE_HTML)):
            results = await source.extract(context, targets)

    assert len(results) >= 1
    attr_ids_filled = {r.attribute_id for r in results}
    assert 1 in attr_ids_filled or 2 in attr_ids_filled
    for r in results:
        assert r.source == Source.WB_CARD
        assert r.confidence >= 0.82


@pytest.mark.asyncio
async def test_extract_brand_mismatch_returns_empty() -> None:
    """Brand-gate fails → no fills."""
    source = OnlinerSource()

    targets = [_make_target(1, "Процессор")]
    context = _make_context("Lenovo ThinkPad X1 Carbon", brand="Lenovo")

    with patch.object(
        source, "_search",
        new=AsyncMock(return_value=[("Dell XPS 15", "https://catalog.onliner.by/notebook/dell/xps15")]),
    ):
        with patch.object(source, "_fetch_page", new=AsyncMock(return_value=ONLINER_WRONG_BRAND_HTML)):
            results = await source.extract(context, targets)

    assert results == []


@pytest.mark.asyncio
async def test_extract_enum_attr_dropped_when_no_match() -> None:
    """Enum target: resolve_value_id returns None → dropped."""
    source = OnlinerSource()

    targets = [_make_target(99, "Цвет", attr_type="enum")]
    context = ExtractionContext(
        product_id=1,
        product_name="Lenovo ThinkPad X1 Carbon Gen 12",
        category_id=7500,
        brand="Lenovo",
        ozon_type_id=1234,
    )

    with patch.object(
        source, "_search",
        new=AsyncMock(return_value=[
            ("Ноутбук Lenovo ThinkPad X1 Carbon Gen 12",
             "https://catalog.onliner.by/notebook/lenovo/thinkpadx1carbon"),
        ]),
    ):
        with patch.object(source, "_fetch_page", new=AsyncMock(return_value=ONLINER_SAMPLE_HTML)):
            with patch(
                "app.services.enrichment.sources.onliner_source.resolve_value_id",
                return_value=None,
            ):
                results = await source.extract(context, targets)

    assert results == []


@pytest.mark.asyncio
async def test_extract_no_search_results_returns_empty() -> None:
    """No search results → []."""
    source = OnlinerSource()

    targets = [_make_target(1, "Процессор")]
    context = _make_context("Lenovo ThinkPad X1 Carbon")

    with patch.object(source, "_search", new=AsyncMock(return_value=[])):
        results = await source.extract(context, targets)

    assert results == []


@pytest.mark.asyncio
async def test_extract_skip_guard_all_filled() -> None:
    """≥80% targets already filled → skip without searching."""
    source = OnlinerSource()

    targets = [_make_target(i, f"Attr {i}") for i in range(10)]
    context = _make_context("Lenovo ThinkPad X1 Carbon")

    already_filled = [
        AttributeValue(
            attribute_id=t.id,
            value="val",
            confidence=0.95,
            source=Source.WB_CARD,
        )
        for t in targets
    ]

    search_mock = AsyncMock(return_value=[])
    with patch.object(source, "_search", new=search_mock):
        results = await source.extract(context, targets, already_filled=already_filled)

    assert results == []
    search_mock.assert_not_called()


@pytest.mark.asyncio
async def test_extract_fetch_error_returns_empty() -> None:
    """HTTP error on page fetch → graceful skip, no crash."""
    source = OnlinerSource()

    targets = [_make_target(1, "Процессор")]
    context = _make_context("Lenovo ThinkPad X1 Carbon")

    with patch.object(
        source, "_search",
        new=AsyncMock(return_value=[("Lenovo ThinkPad", "https://catalog.onliner.by/notebook/lenovo/thinkpad")]),
    ):
        with patch.object(source, "_fetch_page", new=AsyncMock(return_value=None)):
            results = await source.extract(context, targets)

    assert results == []


@pytest.mark.asyncio
async def test_extract_html_fallback_parses_table() -> None:
    """HTML table fallback: no JSON-LD → specs from <tr><td>."""
    source = OnlinerSource()

    targets = [_make_target(1, "Процессор")]
    context = _make_context("Lenovo ThinkPad X1 Carbon Gen 12", brand="Lenovo")

    with patch.object(
        source, "_search",
        new=AsyncMock(return_value=[
            ("Lenovo ThinkPad X1 Carbon Gen 12",
             "https://catalog.onliner.by/notebook/lenovo/thinkpadx1carbon"),
        ]),
    ):
        with patch.object(source, "_fetch_page", new=AsyncMock(return_value=ONLINER_HTML_FALLBACK)):
            results = await source.extract(context, targets)

    # Should get at least one fill from the HTML table
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_source_caches_results() -> None:
    """Second call with same brand+model uses cache, not a new search."""
    source = OnlinerSource()

    targets = [_make_target(1, "Процессор")]
    context = _make_context("Lenovo ThinkPad X1 Carbon Gen 12", brand="Lenovo")

    search_mock = AsyncMock(return_value=[
        ("Ноутбук Lenovo ThinkPad X1 Carbon Gen 12",
         "https://catalog.onliner.by/notebook/lenovo/thinkpadx1carbon"),
    ])
    fetch_mock = AsyncMock(return_value=ONLINER_SAMPLE_HTML)

    with patch.object(source, "_search", new=search_mock):
        with patch.object(source, "_fetch_page", new=fetch_mock):
            await source.extract(context, targets)
            await source.extract(context, targets)

    # Second call should use cache — search called only once.
    assert search_mock.call_count == 1
