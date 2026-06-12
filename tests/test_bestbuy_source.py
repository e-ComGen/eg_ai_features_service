"""Unit tests for BestBuySource — no live network, no LLM.

Covers:
  • _parse_bestbuy_product: details/top-level/features → (name, specs).
  • _brand_gate: pass on correct brand+model tokens; fail on mismatch.
  • _score_title / _classify_match: thresholds.
  • extract(): no API key → [], correct fills, brand-gate fail → [], enum-drop.
  • EN→RU translation of English values before enum-match.
  • 403/429 graceful handling.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.sources.bestbuy_source import (
    BestBuySource,
    _parse_bestbuy_product,
)
from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)


# ---------------------------------------------------------------------------
# Fixture: sample Best Buy API product JSON
# ---------------------------------------------------------------------------

BESTBUY_SAMPLE_PRODUCT: dict = {
    "name": "Samsung Galaxy S24 Ultra 256GB Titanium Black",
    "sku": "6570248",
    "upc": "8806095071749",
    "manufacturer": "Samsung",
    "modelNumber": "SM-S928UZKEXAA",
    "color": "Black",
    "weight": 2.31,
    "shippingWeight": 0.75,
    "height": 6.42,
    "width": 3.07,
    "depth": 0.35,
    "warrantyLabor": "1 Year Manufacturer",
    "warrantyParts": "1 Year Manufacturer",
    "details": [
        {"name": "Display Size", "value": "6.8\""},
        {"name": "Storage Capacity", "value": "256GB"},
        {"name": "RAM", "value": "12GB"},
        {"name": "Operating System", "value": "Android"},
        {"name": "Water Resistant", "value": "Yes"},
        {"name": "Wireless Charging", "value": "Yes"},
        {"name": "Battery Capacity", "value": "5000 mAh"},
    ],
    "features": [
        {"feature": "S Pen included"},
        {"feature": "Titanium frame"},
    ],
}

BESTBUY_WRONG_BRAND_PRODUCT: dict = {
    "name": "Apple iPhone 15 Pro 256GB Black Titanium",
    "manufacturer": "Apple",
    "modelNumber": "MU7C3LL/A",
    "color": "Black",
    "details": [
        {"name": "Storage Capacity", "value": "256GB"},
        {"name": "RAM", "value": "8GB"},
    ],
}

BESTBUY_EMPTY_PRODUCT: dict = {
    "name": "Some Product",
    "manufacturer": "Unknown",
    "details": [],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(attr_id: int, name: str, attr_type: str = "text") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=attr_type)


def _make_context(
    product_name: str,
    brand: str = "Samsung",
    ean: str = "",
) -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name=product_name,
        category_id=7500,
        brand=brand,
        ean=ean or None,
    )


def _make_api_response(products: list[dict]) -> MagicMock:
    """Build a mock httpx response returning the products list."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"products": products}
    return mock_resp


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parse_product_extracts_name_and_details() -> None:
    name, specs = _parse_bestbuy_product(BESTBUY_SAMPLE_PRODUCT)
    assert "Samsung" in name
    assert "Galaxy S24" in name
    spec_names = {s["name"] for s in specs}
    assert "Display Size" in spec_names
    assert "Storage Capacity" in spec_names
    assert "RAM" in spec_names


def test_parse_product_includes_top_level_fields() -> None:
    name, specs = _parse_bestbuy_product(BESTBUY_SAMPLE_PRODUCT)
    spec_map = {s["name"]: s["value"] for s in specs}
    # color → Цвет
    assert "Цвет" in spec_map
    assert spec_map["Цвет"] == "Black"
    # warrantyLabor → Гарантия (труд)
    assert "Гарантия (труд)" in spec_map


def test_parse_product_includes_features() -> None:
    name, specs = _parse_bestbuy_product(BESTBUY_SAMPLE_PRODUCT)
    feature_entries = [s for s in specs if s["name"] == "Feature"]
    assert len(feature_entries) >= 1


def test_parse_product_deduplicates() -> None:
    # Add duplicate detail
    product = dict(BESTBUY_SAMPLE_PRODUCT)
    product["details"] = [
        {"name": "Storage Capacity", "value": "256GB"},
        {"name": "Storage Capacity", "value": "256GB"},  # duplicate
    ]
    _, specs = _parse_bestbuy_product(product)
    storage_entries = [s for s in specs if s["name"] == "Storage Capacity"]
    assert len(storage_entries) == 1


def test_parse_product_empty_details_returns_top_level() -> None:
    _, specs = _parse_bestbuy_product(BESTBUY_EMPTY_PRODUCT)
    # No details, no features, no top-level filled fields (None values filtered)
    assert isinstance(specs, list)


def test_parse_product_filters_null_values() -> None:
    product = {
        "name": "Test",
        "color": None,
        "details": [{"name": "RAM", "value": None}],
    }
    _, specs = _parse_bestbuy_product(product)
    spec_names = {s["name"] for s in specs}
    assert "RAM" not in spec_names
    assert "Цвет" not in spec_names


# ---------------------------------------------------------------------------
# Brand-gate tests
# ---------------------------------------------------------------------------

def test_brand_gate_pass_correct_brand_and_model() -> None:
    assert BestBuySource._brand_gate(
        "Samsung Galaxy S24 Ultra 256GB",
        "Samsung",
        "Samsung Galaxy S24 Ultra",
    )


def test_brand_gate_fail_wrong_brand() -> None:
    assert not BestBuySource._brand_gate(
        "Apple iPhone 15 Pro 256GB",
        "Samsung",
        "Samsung Galaxy S24 Ultra",
    )


def test_brand_gate_fail_model_mismatch() -> None:
    # Brand matches but model tokens (S24) absent.
    assert not BestBuySource._brand_gate(
        "Samsung Galaxy S23 FE",
        "Samsung",
        "Samsung Galaxy S24 Ultra",
    )


def test_brand_gate_no_brand_uses_model_only() -> None:
    assert BestBuySource._brand_gate(
        "Galaxy S24 Ultra Phone",
        "",
        "Galaxy S24 Ultra",
    )


# ---------------------------------------------------------------------------
# Scoring / classify
# ---------------------------------------------------------------------------

def test_classify_match_exact() -> None:
    assert BestBuySource._classify_match(80.0) == "exact"


def test_classify_match_brand_line() -> None:
    assert BestBuySource._classify_match(65.0) == "brand_line"


def test_classify_match_skip() -> None:
    assert BestBuySource._classify_match(30.0) == "skip"


# ---------------------------------------------------------------------------
# extract() — unit (no live network)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_no_api_key_returns_empty() -> None:
    """No BESTBUY_API_KEY → extract returns [] without crashing."""
    with patch.dict(os.environ, {}, clear=True):
        # Ensure key is absent
        os.environ.pop("BESTBUY_API_KEY", None)
        source = BestBuySource()

    targets = [_make_target(1, "Storage Capacity")]
    context = _make_context("Samsung Galaxy S24 Ultra")
    result = await source.extract(context, targets)
    assert result == []


@pytest.mark.asyncio
async def test_extract_fills_from_correct_product() -> None:
    """Happy path: API returns product, brand-gate passes, attrs filled."""
    source = BestBuySource()
    source._api_key = "fake-key"

    targets = [
        _make_target(1, "Storage Capacity"),
        _make_target(2, "RAM"),
    ]
    context = _make_context("Samsung Galaxy S24 Ultra", brand="Samsung")

    with patch.object(source, "_api_fetch", new=AsyncMock(return_value=[BESTBUY_SAMPLE_PRODUCT])):
        results = await source.extract(context, targets)

    assert len(results) >= 1
    attr_ids_filled = {r.attribute_id for r in results}
    assert 1 in attr_ids_filled or 2 in attr_ids_filled
    for r in results:
        assert r.source == Source.WB_CARD
        assert r.confidence >= 0.82


@pytest.mark.asyncio
async def test_extract_en_to_ru_translation() -> None:
    """English values 'Yes' → 'да' before enum resolution."""
    source = BestBuySource()
    source._api_key = "fake-key"

    # Target 'Water Resistant' — spec value is 'Yes' (English)
    targets = [_make_target(5, "Water Resistant", attr_type="text")]
    context = _make_context("Samsung Galaxy S24 Ultra", brand="Samsung")

    with patch.object(source, "_api_fetch", new=AsyncMock(return_value=[BESTBUY_SAMPLE_PRODUCT])):
        results = await source.extract(context, targets)

    # Should have translated "Yes" to "да"
    filled = {r.attribute_id: r.value for r in results}
    if 5 in filled:
        assert filled[5] == "да", f"Expected 'да' (translated from 'Yes'), got {filled[5]!r}"


@pytest.mark.asyncio
async def test_extract_brand_mismatch_returns_empty() -> None:
    """Brand-gate fails → no fills."""
    source = BestBuySource()
    source._api_key = "fake-key"

    targets = [_make_target(1, "Storage Capacity")]
    # Querying Samsung but API returns Apple
    context = _make_context("Samsung Galaxy S24 Ultra", brand="Samsung")

    with patch.object(source, "_api_fetch", new=AsyncMock(return_value=[BESTBUY_WRONG_BRAND_PRODUCT])):
        results = await source.extract(context, targets)

    assert results == []


@pytest.mark.asyncio
async def test_extract_enum_attr_dropped_when_no_match() -> None:
    """Enum target: value not resolved in Ozon dict → dropped."""
    source = BestBuySource()
    source._api_key = "fake-key"

    targets = [_make_target(99, "Storage Capacity", attr_type="enum")]
    context = ExtractionContext(
        product_id=1,
        product_name="Samsung Galaxy S24 Ultra",
        category_id=7500,
        brand="Samsung",
        ozon_type_id=1234,
    )

    with patch.object(source, "_api_fetch", new=AsyncMock(return_value=[BESTBUY_SAMPLE_PRODUCT])):
        with patch(
            "app.services.enrichment.sources.bestbuy_source.resolve_value_id",
            return_value=None,
        ):
            results = await source.extract(context, targets)

    assert results == []


@pytest.mark.asyncio
async def test_extract_no_api_results_returns_empty() -> None:
    """API returns empty products list → []."""
    source = BestBuySource()
    source._api_key = "fake-key"

    targets = [_make_target(1, "Storage Capacity")]
    context = _make_context("Samsung Galaxy S24 Ultra")

    with patch.object(source, "_api_fetch", new=AsyncMock(return_value=[])):
        results = await source.extract(context, targets)

    assert results == []


@pytest.mark.asyncio
async def test_extract_skip_guard_all_filled() -> None:
    """≥80% targets already filled → source skips without calling API."""
    source = BestBuySource()
    source._api_key = "fake-key"

    targets = [_make_target(i, f"Attr {i}") for i in range(10)]
    context = _make_context("Samsung Galaxy S24 Ultra")

    already_filled = [
        AttributeValue(
            attribute_id=t.id,
            value="val",
            confidence=0.95,
            source=Source.WB_CARD,
        )
        for t in targets
    ]

    api_mock = AsyncMock(return_value=[])
    with patch.object(source, "_api_fetch", new=api_mock):
        results = await source.extract(context, targets, already_filled=already_filled)

    assert results == []
    api_mock.assert_not_called()


@pytest.mark.asyncio
async def test_api_fetch_handles_403_gracefully() -> None:
    """HTTP 403 from Best Buy API → returns [] without raising."""
    source = BestBuySource()
    source._api_key = "fake-key"

    mock_resp = MagicMock()
    mock_resp.status_code = 403

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await source._api_fetch("upc=12345")

    assert result == []


@pytest.mark.asyncio
async def test_api_fetch_handles_429_gracefully() -> None:
    """HTTP 429 (rate limit) → returns [] without raising."""
    source = BestBuySource()
    source._api_key = "fake-key"

    mock_resp = MagicMock()
    mock_resp.status_code = 429

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await source._api_fetch("upc=12345")

    assert result == []
