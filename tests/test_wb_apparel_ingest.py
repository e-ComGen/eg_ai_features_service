"""Unit tests for the pure payload-builder of build_wb_apparel_rag_index.

No network / no Qdrant / no embedding model — only the mapping logic that turns
a WB dataset row into the Qdrant payload {variantid, name, description, categories,
characteristics}. The `characteristics` dict shape must match what
CompetitorRagSource._find_attr_value reads back.
"""
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_wb_apparel_rag_index.py"
_spec = importlib.util.spec_from_file_location("build_wb_apparel_rag_index", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _sample_row():
    """A dict mimicking expected WB fields (options + compositions)."""
    return {
        "nm_id": 123456,
        "imt_name": "Платье женское летнее",
        "description": "Лёгкое платье из хлопка",
        "subj_root_name": "Одежда",
        "subj_name": "Платья",
        "options": [
            {"name": "Цвет", "value": "красный"},
            {"name": "Пол", "value": "женский"},
            {"name": "Сезон", "value": ["лето", "весна"]},
        ],
        "compositions": [
            {"name": "хлопок", "percentage": 80},
            {"name": "эластан", "percentage": 20},
        ],
    }


def test_payload_shape_matches_ozon():
    payload = mod.build_payload(_sample_row(), category_field="subj_root_name")
    assert set(payload.keys()) == {
        "variantid", "name", "description", "categories", "characteristics"
    }
    assert payload["variantid"] == 123456
    assert payload["name"] == "Платье женское летнее"
    assert payload["description"] == "Лёгкое платье из хлопка"
    assert payload["categories"] == ["Одежда"]


def test_characteristics_is_flat_dict():
    """characteristics must be {attr_name: stringified_value} for _find_attr_value."""
    chars = mod.extract_characteristics(_sample_row())
    assert isinstance(chars, dict)
    assert chars["Цвет"] == "красный"
    assert chars["Пол"] == "женский"
    # list value joined readably
    assert chars["Сезон"] == "лето, весна"
    # composition list joined into "Состав"
    assert chars["Состав"] == "хлопок 80%, эластан 20%"
    # all values are strings
    assert all(isinstance(v, str) for v in chars.values())


def test_characteristics_dict_form():
    """WB may expose characteristics as a {name: value} dict instead of a list."""
    row = {
        "nm_id": 1,
        "name": "Куртка",
        "subj_name": "Верхняя одежда",
        "characteristics": {"Цвет": "чёрный", "Материал": "полиэстер"},
    }
    chars = mod.extract_characteristics(row)
    assert chars["Цвет"] == "чёрный"
    assert chars["Материал"] == "полиэстер"


def test_find_attr_value_reads_our_payload():
    """End-to-end: the source's _find_attr_value must read our characteristics dict."""
    from app.services.enrichment.sources.competitor_rag_source import CompetitorRagSource
    src = CompetitorRagSource.__new__(CompetitorRagSource)  # no __init__ side effects
    chars = mod.extract_characteristics(_sample_row())
    assert src._find_attr_value(chars, "Цвет") == "красный"
    assert src._find_attr_value(chars, "Пол") == "женский"


def test_is_apparel_filter():
    allow = ["одежда", "обувь", "аксессуары"]
    assert mod.is_apparel(_sample_row(), allow, "subj_root_name") is True
    electronics = {"subj_root_name": "Электроника", "subj_name": "Наушники"}
    assert mod.is_apparel(electronics, allow, "subj_root_name") is False
    # substring against concatenated fields when category_field unknown
    shoes = {"subj_name": "Кроссовки", "subj_root_name": "Обувь"}
    assert mod.is_apparel(shoes, allow, None) is True


def test_pick_id_and_name_fallbacks():
    assert mod.pick_id({"variantid": 7}) == 7
    assert mod.pick_id({"id": 9}) == 9
    assert mod.pick_name({"title": "X"}) == "X"
    assert mod.pick_name({"name": "Y"}) == "Y"


def test_malformed_row_does_not_crash():
    # missing everything → empty-ish payload, no exception
    payload = mod.build_payload({}, category_field=None)
    assert payload["name"] == ""
    assert payload["characteristics"] == {}
