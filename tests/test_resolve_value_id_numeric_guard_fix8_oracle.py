"""Functional oracle for FIX-8 (unit-strip retry + numeric semantic guard, INV-9).

Spec: docs/MANIFEST_resolve_value_id_numeric_guard.md.
Encodes the manifest's worked-example oracle O1..O9 verbatim -- does NOT
invent expected values, only asserts the ones the manifest specifies.

The dict fixture reproduces the real Ozon probe (cat_id=15621050,
type_id=95139, attribute_id=5186 "Разрешение экрана") with the three enum
values quoted in the manifest, using plain ASCII "x" as the stored form (the
manifest's grounded probe confirms "2712×1220"/"2712x1220"/"2712 x 1220" and
"1220x2712" all already resolve correctly pre-fix via the existing exact/fuzzy
strategies -- this fixture reproduces that behavior self-consistently). The
semantic matcher is mocked (no sentence_transformers model loaded) so only the
wiring of FIX-8a/FIX-8b is exercised -- the O9 no-FP case documents that a
real semantic match on a non-numeric attribute is unaffected by the guard.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

CAT_ID = 15621050
TYPE_ID = 95139
ATTR_ID = 5186  # "Разрешение экрана"

ID_2712X1220 = 972270457
ID_1220X2712 = 971969162
ID_2640X1200 = 970715447  # wrong neighbor a naive semantic fallback would pick

_DICT_RESOLUTION = {
    "schema_version": 2,
    "source": "ozon_seller_api",
    "categories": {
        f"{CAT_ID}:{TYPE_ID}": {
            "description_category_id": CAT_ID,
            "type_id": TYPE_ID,
            "name": "Test",
            "path": ["Test"],
            "characteristics": [
                {
                    "id": ATTR_ID,
                    "name": "Разрешение экрана",
                    "type": "Option",
                    "is_required": False,
                    "is_collection": False,
                    "description": "",
                    "values": [
                        {"id": ID_2712X1220, "value": "2712x1220"},
                        {"id": ID_1220X2712, "value": "1220x2712"},
                        {"id": ID_2640X1200, "value": "2640x1200"},
                    ],
                }
            ],
        }
    },
}

COLOR_ATTR_ID = 4180
_DICT_COLOR = {
    "schema_version": 2,
    "source": "ozon_seller_api",
    "categories": {
        "500:100": {
            "description_category_id": 500,
            "type_id": 100,
            "name": "Test",
            "path": ["Test"],
            "characteristics": [
                {
                    "id": COLOR_ATTR_ID,
                    "name": "Цвет",
                    "type": "Option",
                    "is_required": False,
                    "is_collection": False,
                    "description": "",
                    "values": [
                        {"id": 1001, "value": "Синий"},
                        {"id": 1002, "value": "Красный"},
                    ],
                }
            ],
        }
    },
}


def _reset_cache():
    from app.services.enrichment.strategies.dictionaries.ozon_loader import (
        load_ozon_dictionary,
    )
    load_ozon_dictionary.cache_clear()


def _resolve_with_dict(tmp_path, dict_payload, cat_id, type_id, attr_id, value,
                        matcher_return=None):
    """Run resolve_value_id against a synthetic dict, with the semantic
    matcher mocked to always return matcher_return (None disables it, like
    matcher unavailable)."""
    import app.services.enrichment.strategies.dictionaries.ozon_loader as mod
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(dict_payload, ensure_ascii=False), encoding="utf-8"
    )
    mock_matcher = MagicMock()
    mock_matcher.find_best_match.return_value = matcher_return
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        _reset_cache()
        mod._matcher_instance = mock_matcher if matcher_return is not None else None
        mod._matcher_attempted = True
        from app.services.enrichment.strategies.dictionaries.ozon_loader import (
            resolve_value_id,
        )
        return resolve_value_id(cat_id, type_id, attr_id, value)


# ---------------------------------------------------------------------------
# O1-O4: regression -- unchanged behavior, no unit suffix involved.
# ---------------------------------------------------------------------------

def test_o1_exact_multiplication_sign(tmp_path):
    result = _resolve_with_dict(tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
                                 "2712×1220")
    assert result == ID_2712X1220


def test_o2_latin_x(tmp_path):
    result = _resolve_with_dict(tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
                                 "2712x1220")
    assert result == ID_2712X1220


def test_o3_spaced_x(tmp_path):
    result = _resolve_with_dict(tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
                                 "2712 x 1220")
    assert result == ID_2712X1220


def test_o4_other_existing_id(tmp_path):
    result = _resolve_with_dict(tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
                                 "1220x2712")
    assert result == ID_1220X2712


# ---------------------------------------------------------------------------
# O5-O7: FIX-8a -- unit-strip retry resolves the clean value exactly.
# ---------------------------------------------------------------------------

def test_o5_pixels_suffix_stripped(tmp_path):
    result = _resolve_with_dict(tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
                                 "2712×1220 pixels")
    assert result == ID_2712X1220


def test_o6_px_suffix_stripped(tmp_path):
    result = _resolve_with_dict(tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
                                 "2712×1220 px")
    assert result == ID_2712X1220


def test_o7_cyrillic_tochek_suffix_stripped(tmp_path):
    result = _resolve_with_dict(tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
                                 "2712 x 1220 точек")
    assert result == ID_2712X1220


# ---------------------------------------------------------------------------
# O8: FIX-8b -- no exact match exists; a naive semantic fallback would pick
# the wrong neighbor (2640x1200) -- digit guard must abstain (None).
# ---------------------------------------------------------------------------

def test_o8_no_exact_match_semantic_abstains_on_digit_mismatch(tmp_path):
    # Mock matcher simulates a real semantic model picking the nearest (but
    # numerically WRONG) neighbor for the unit-suffixed, non-dict value.
    result = _resolve_with_dict(
        tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
        "2699x1200 pixels", matcher_return="2640x1200",
    )
    assert result is None


# ---------------------------------------------------------------------------
# O9 (non-numeric, no-FP): color query has no digits -- guard must not block
# a legitimate semantic match.
# ---------------------------------------------------------------------------

def test_o9_no_fp_non_numeric_semantic_match_untouched(tmp_path):
    result = _resolve_with_dict(
        tmp_path, _DICT_COLOR, 500, 100, COLOR_ATTR_ID,
        "тёмно-синий", matcher_return="Синий",
    )
    assert result == 1001
