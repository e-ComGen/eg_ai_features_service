# -*- coding: utf-8 -*-
"""Unit tests for unit_normalizer: lossless conversion of card scalar values
whose explicit unit differs from the field's explicit unit (same dimension).

The 16 "real" cases are the actual unit-mismatch fills observed in the
guards-ON diverse-40 run (multi_cat_20260617_010654.json) — Ray-Ban lens
dimensions, lawnmower cut height, package weights. The guard cases assert the
normalizer never touches ambiguous input."""
import pytest

from app.services.enrichment.strategies.dictionaries.unit_normalizer import (
    normalize_value,
)


# (field_name, value) -> expected converted string
REAL_CANDIDATES = [
    ("Вес товара, г", "9 кг", "9000"),
    ("Вес с упаковкой, г", "0.5 кг", "500"),
    ("Размеры, мм", "5.3 см", "53"),
    ("Вес с упаковкой, г", "1 кг", "1000"),
    ("Высота линзы, мм", "4.6 см", "46"),
    ("Ширина линзы, мм", "5.4 см", "54"),
    ("Размер заушника, мм", "15 см", "150"),
    ("Общая ширина, мм", "14.7 см", "147"),
    ("Мин. высота среза травы, см", "40 мм", "4"),
    ("Макс. высота среза травы, см", "70 мм", "7"),
    ("Вес с упаковкой, г", "13.39 кг", "13390"),
    ("Вес с упаковкой, г", "6.6 кг", "6600"),
    ("Размеры, мм", "94.5 см", "945"),
    ("Вес с упаковкой, г", "0.08 кг", "80"),
    ("Размеры, мм", "29.44 см", "294.4"),
    ("Вес товара, г", "9.5 кг", "9500"),
    # extra dimensions (sanity, not in the 16)
    ("Мощность, Вт", "1.4 кВт", "1400"),
    ("Объём, л", "500 мл", "0.5"),
]

# inputs that must pass through UNCHANGED
GUARD_CASES = [
    ("Ширина, мм", "54"),                 # bare number — unit unknown
    ("Ширина, мм", "5.4 мм"),             # already field's unit
    ("Высота, см", "5 - 7 см"),           # range
    ("Вес, г", ["9 кг", "10 кг"]),        # list / non-scalar
    ("Длина, м", "5 кг"),                 # different dimension in value
    ("Вес, г", "5.4 см"),                 # value dim != field dim
    ("Бренд", "Гарант"),                  # field carries no unit (no comma)
    ("Размеры, мм", "5 x 3 см"),          # multiple numbers → not a scalar
    ("Вес, кг", ""),                      # empty
]


@pytest.mark.parametrize("field,value,expected", REAL_CANDIDATES)
def test_real_candidates_convert(field, value, expected):
    r = normalize_value(field, value)
    assert r.changed is True
    assert r.value == expected


@pytest.mark.parametrize("field,value", GUARD_CASES)
def test_guards_leave_value_untouched(field, value):
    r = normalize_value(field, value)
    assert r.changed is False
    # unchanged value is returned as the stringified original
    assert r.value == str(value).strip()


def test_token_boundary_no_substring_false_match():
    """'кг' must not be detected as the 'г' token; 'см'/'мм' must not match 'м'."""
    # "9 кг" in a "г" field converts via кг (×1000), not misread as г.
    assert normalize_value("Вес, г", "9 кг").value == "9000"
    # "5 см" in an "м" field converts via см, not misread as bare м.
    assert normalize_value("Длина, м", "5 см").value == "0.05"
