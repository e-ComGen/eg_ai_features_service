"""Unit tests for ozon_payload.py — structural validation, no live API calls."""
import json
import pytest
from pydantic import ValidationError

from app.services.enrichment.base import AttributeValue, Source
from app.services.enrichment.marketplaces.ozon_payload import (
    OzonProductInput,
    OzonDimensions,
    OzonImportPayload,
    build_ozon_import_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dims(**kw) -> OzonDimensions:
    defaults = dict(depth=200, width=100, height=50, dimension_unit="mm")
    defaults.update(kw)
    return OzonDimensions(**defaults)


def _product(**kw) -> OzonProductInput:
    defaults = dict(
        offer_id="SKU-001",
        name="Кроссовки Adidas",
        price="1500",
        vat="0.20",
        currency_code="RUB",
        images=["https://cdn.example.com/1.jpg"],
        weight=500,
        weight_unit="g",
        dimensions=_dims(),
    )
    defaults.update(kw)
    return OzonProductInput(**defaults)


def _av(attr_id: int, value, *, is_collection=False, value_id=None, value_ids=None,
        source=Source.DESCRIPTION) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=0.9,
        source=source,
        is_collection=is_collection,
        value_id=value_id,
        value_ids=value_ids,
    )


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

def test_round_trip_matches_expected_structure():
    """build_ozon_import_payload → структура совпадает с ожидаемым JSON."""
    attrs = [
        _av(85, "Adidas", value_id=971042156),
        _av(9024, ["Синий", "Тёмно-синий"], is_collection=True, value_ids=[61576, 61577]),
    ]
    product = _product(
        offer_id="MERCHANT_SKU_123",
        name="Кроссовки",
        price="1500",
        old_price="2000",
        barcode="1234567890123",
        images=["https://cdn.example.com/1.jpg", "https://cdn.example.com/2.jpg"],
    )

    payload = build_ozon_import_payload(product, 17027949, 95001, attrs)
    d = payload.to_api_dict()

    assert d["items"][0]["offer_id"] == "MERCHANT_SKU_123"
    assert d["items"][0]["description_category_id"] == 17027949
    assert d["items"][0]["type_id"] == 95001

    attrs_out = d["items"][0]["attributes"]
    brand_attr = next(a for a in attrs_out if a["id"] == 85)
    assert brand_attr["complex_id"] == 0
    assert brand_attr["values"] == [{"dictionary_value_id": 971042156, "value": "Adidas"}]

    color_attr = next(a for a in attrs_out if a["id"] == 9024)
    assert len(color_attr["values"]) == 2
    assert color_attr["values"][0] == {"dictionary_value_id": 61576, "value": "Синий"}
    assert color_attr["values"][1] == {"dictionary_value_id": 61577, "value": "Тёмно-синий"}


# ---------------------------------------------------------------------------
# Attribute value cases
# ---------------------------------------------------------------------------

def test_is_collection_with_value_ids_emits_multiple_dict_entries():
    av = _av(9024, ["Синий", "Красный"], is_collection=True, value_ids=[61576, 61577])
    payload = build_ozon_import_payload(_product(), 100, 200, [av])
    values = payload.to_api_dict()["items"][0]["attributes"][0]["values"]
    assert len(values) == 2
    for v in values:
        assert "dictionary_value_id" in v
        assert "value" in v


def test_is_collection_free_text_emits_multiple_value_only_entries():
    av = _av(123, ["Хлопок", "Полиэстер"], is_collection=True)
    payload = build_ozon_import_payload(_product(), 100, 200, [av])
    values = payload.to_api_dict()["items"][0]["attributes"][0]["values"]
    assert len(values) == 2
    for v in values:
        assert "dictionary_value_id" not in v
        assert "value" in v


def test_scalar_with_value_id_emits_single_entry_with_both_fields():
    av = _av(85, "Adidas", value_id=971042156)
    payload = build_ozon_import_payload(_product(), 100, 200, [av])
    values = payload.to_api_dict()["items"][0]["attributes"][0]["values"]
    assert len(values) == 1
    assert values[0]["dictionary_value_id"] == 971042156
    assert values[0]["value"] == "Adidas"


def test_scalar_free_text_emits_single_value_only_entry():
    av = _av(85, "SomeBrand")
    payload = build_ozon_import_payload(_product(), 100, 200, [av])
    values = payload.to_api_dict()["items"][0]["attributes"][0]["values"]
    assert len(values) == 1
    assert "dictionary_value_id" not in values[0]
    assert values[0]["value"] == "SomeBrand"


def test_empty_attributes_list_produces_empty_attributes_array():
    payload = build_ozon_import_payload(_product(), 100, 200, [])
    assert payload.to_api_dict()["items"][0]["attributes"] == []


# ---------------------------------------------------------------------------
# Required-field validation
# ---------------------------------------------------------------------------

def test_empty_images_raises():
    with pytest.raises(ValidationError, match="images list must not be empty"):
        _product(images=[])


def test_too_many_images_raises():
    with pytest.raises(ValidationError, match="at most 15"):
        _product(images=[f"https://img/{i}.jpg" for i in range(16)])


def test_price_zero_raises():
    with pytest.raises(ValidationError, match="price must be > 0"):
        _product(price="0")


def test_price_negative_raises():
    with pytest.raises(ValidationError, match="price must be > 0"):
        _product(price="-100")


def test_price_non_numeric_raises():
    with pytest.raises(ValidationError, match="price must be numeric"):
        _product(price="бесплатно")


def test_old_price_zero_raises():
    with pytest.raises(ValidationError, match="old_price must be > 0"):
        _product(old_price="0")


def test_invalid_vat_raises():
    with pytest.raises(ValidationError):
        _product(vat="0.15")


def test_invalid_weight_unit_raises():
    with pytest.raises(ValidationError):
        _product(weight_unit="lb")


def test_invalid_dimension_unit_raises():
    with pytest.raises(ValidationError):
        _dims(dimension_unit="ft")


# ---------------------------------------------------------------------------
# primary_image behaviour
# ---------------------------------------------------------------------------

def test_primary_image_defaults_to_first_image():
    imgs = ["https://cdn.example.com/1.jpg", "https://cdn.example.com/2.jpg"]
    product = _product(images=imgs)
    payload = build_ozon_import_payload(product, 100, 200, [])
    assert payload.to_api_dict()["items"][0]["primary_image"] == imgs[0]


def test_primary_image_override():
    imgs = ["https://cdn.example.com/1.jpg", "https://cdn.example.com/2.jpg"]
    product = _product(images=imgs)
    payload = build_ozon_import_payload(product, 100, 200, [], primary_image=imgs[1])
    assert payload.to_api_dict()["items"][0]["primary_image"] == imgs[1]


# ---------------------------------------------------------------------------
# JSON-serializable
# ---------------------------------------------------------------------------

def test_to_api_dict_is_json_serializable():
    attrs = [
        _av(85, "Adidas", value_id=971042156),
        _av(9024, ["Синий", "Красный"], is_collection=True, value_ids=[61576, 61577]),
    ]
    payload = build_ozon_import_payload(_product(), 17027949, 95001, attrs)
    result = json.dumps(payload.to_api_dict())
    parsed = json.loads(result)
    assert "items" in parsed


# ---------------------------------------------------------------------------
# optional fields excluded when None
# ---------------------------------------------------------------------------

def test_none_optional_fields_excluded_from_dict():
    """old_price, barcode — None → не должны попасть в to_api_dict."""
    product = _product()  # no old_price, no barcode
    payload = build_ozon_import_payload(product, 100, 200, [])
    d = payload.to_api_dict()
    item = d["items"][0]
    assert "old_price" not in item
    assert "barcode" not in item


def test_old_price_included_when_set():
    product = _product(old_price="2000")
    payload = build_ozon_import_payload(product, 100, 200, [])
    item = payload.to_api_dict()["items"][0]
    assert item["old_price"] == "2000"


# ---------------------------------------------------------------------------
# complex_id always 0
# ---------------------------------------------------------------------------

def test_all_attributes_have_complex_id_zero():
    attrs = [_av(10, "X", value_id=1), _av(20, "Y")]
    payload = build_ozon_import_payload(_product(), 100, 200, attrs)
    for a in payload.to_api_dict()["items"][0]["attributes"]:
        assert a["complex_id"] == 0


# ---------------------------------------------------------------------------
# value_ids shorter than values list — partial fill, no crash
# ---------------------------------------------------------------------------

def test_value_ids_shorter_than_values_list_no_crash():
    """value_ids=[1] but value=["A","B","C"] → first gets id, rest free-text."""
    av = _av(99, ["A", "B", "C"], is_collection=True, value_ids=[42])
    payload = build_ozon_import_payload(_product(), 100, 200, [av])
    values = payload.to_api_dict()["items"][0]["attributes"][0]["values"]
    assert len(values) == 3
    assert values[0]["dictionary_value_id"] == 42
    assert "dictionary_value_id" not in values[1]
    assert "dictionary_value_id" not in values[2]
