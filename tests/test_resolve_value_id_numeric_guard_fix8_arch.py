"""Arch/property tests for FIX-8 (unit-strip retry + numeric semantic guard).

Spec: docs/MANIFEST_resolve_value_id_numeric_guard.md.
Covers INV-9a..e. The O1..O9 worked-example oracle lives in
test_resolve_value_id_numeric_guard_fix8_oracle.py (separate file, same
no-self-bias split style as the FIX-7 variant_token_guard tests).
"""
import json
from unittest.mock import MagicMock, patch

from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    _digits_compatible,
    _strip_unit_suffix,
)

CAT_ID = 15621050
TYPE_ID = 95139
ATTR_ID = 5186

ID_2712X1220 = 972270457
ID_1220X2712 = 971969162
ID_2640X1200 = 970715447

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


def _reset_cache():
    from app.services.enrichment.strategies.dictionaries.ozon_loader import (
        load_ozon_dictionary,
    )
    load_ozon_dictionary.cache_clear()


def _resolve_with_dict(tmp_path, dict_payload, cat_id, type_id, attr_id, value,
                        matcher_return=None):
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
# INV-9a -- _strip_unit_suffix strips ONLY whitelist suffixes; legitimate
# dictionary units (мм/В/Ah/...) are left untouched (returns None).
# ---------------------------------------------------------------------------

def test_inv9a_whitelist_suffix_stripped_pixels():
    assert _strip_unit_suffix("2712×1220 pixels") == "2712×1220"


def test_inv9a_whitelist_suffix_stripped_px():
    assert _strip_unit_suffix("2712×1220 px") == "2712×1220"


def test_inv9a_whitelist_suffix_stripped_tochek():
    assert _strip_unit_suffix("2712 x 1220 точек") == "2712 x 1220"


def test_inv9a_legit_unit_mm_not_stripped():
    assert _strip_unit_suffix("5 мм") is None


def test_inv9a_legit_unit_volts_not_stripped():
    assert _strip_unit_suffix("18 В") is None


def test_inv9a_legit_unit_ah_not_stripped():
    assert _strip_unit_suffix("6 Ah") is None


def test_inv9a_no_suffix_at_all_is_inert():
    assert _strip_unit_suffix("2712x1220") is None


# ---------------------------------------------------------------------------
# INV-9b -- _digits_compatible: True for non-numeric query (no \d), True on
# multiset match, False on mismatch.
# ---------------------------------------------------------------------------

def test_inv9b_non_numeric_query_always_true():
    assert _digits_compatible("тёмно-синий", "Синий") is True


def test_inv9b_digits_match_true():
    assert _digits_compatible("2712x1220", "2712x1220") is True


def test_inv9b_digits_match_regardless_of_separator():
    assert _digits_compatible("2712×1220 pixels", "2712x1220") is True


def test_inv9b_digits_mismatch_false():
    assert _digits_compatible("2699x1200 pixels", "2640x1200") is False


# ---------------------------------------------------------------------------
# INV-9c -- idempotency: repeated resolve_value_id calls return the same
# id/None on the same input.
# ---------------------------------------------------------------------------

def test_inv9c_idempotent_on_resolved_value(tmp_path):
    first = _resolve_with_dict(tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
                                "2712×1220 pixels")
    second = _resolve_with_dict(tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
                                 "2712×1220 pixels")
    assert first == second == ID_2712X1220


def test_inv9c_idempotent_on_abstained_value(tmp_path):
    first = _resolve_with_dict(
        tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
        "2699x1200 pixels", matcher_return="2640x1200",
    )
    second = _resolve_with_dict(
        tmp_path, _DICT_RESOLUTION, CAT_ID, TYPE_ID, ATTR_ID,
        "2699x1200 pixels", matcher_return="2640x1200",
    )
    assert first is None
    assert second is None


# ---------------------------------------------------------------------------
# INV-9e -- resolve_value_id signature unchanged (4 positional params, same
# order/names) -- a regression guard against accidental signature drift.
# ---------------------------------------------------------------------------

def test_inv9e_signature_unchanged():
    import inspect
    from app.services.enrichment.strategies.dictionaries.ozon_loader import (
        resolve_value_id,
    )
    sig = inspect.signature(resolve_value_id)
    assert list(sig.parameters.keys()) == ["cat_id", "type_id", "attribute_id", "value"]
