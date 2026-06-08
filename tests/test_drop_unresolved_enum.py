"""Unit tests for the GENERAL conservative finalize-stage cleanup
_drop_unresolved_optional_enums / _drop_unresolved_required_enums /
_is_placeholder_value in app/services/enrichment/pipeline.py.

After FULL value_id resolution (deterministic resolve_value_ids + llm_resolve_tail),
any enum-target value (optional OR required) that resolved to NO allowed dictionary
option (value_id is None) is a fake-fill that Ozon would reject on upload. We drop it.

Strict scope:
- ONLY enum targets (allowed_values present). Free-text never touched.
- Optional enums: _drop_unresolved_optional_enums (unchanged behavior).
- Required enums: _drop_unresolved_required_enums (NEW — primary safety guard).
- Drop only when value_id is None AFTER resolution.
- is_collection: drop only unresolved elements; if all unresolved → drop field.
- Correctly-resolved values (value_id set) NEVER dropped, regardless of is_required.
- Placeholder values ('нет','-','n/a',...) filtered by _is_placeholder_value.
"""
from app.services.enrichment.base import (
    AttributeValue,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    _drop_unresolved_optional_enums,
    _drop_unresolved_required_enums,
    _is_placeholder_value,
)


def _target(
    attr_id, *, name="Attr", allowed=None, is_required=False, is_collection=False
) -> TargetAttribute:
    return TargetAttribute(
        id=attr_id,
        name=name,
        type="enum" if allowed else "text",
        allowed_values=allowed,
        is_required=is_required,
        is_collection=is_collection,
    )


def _value(
    attr_id, value, *, value_id=None, value_ids=None, is_collection=False
) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=0.8,
        source=Source.WEB_SEARCH,
        value_id=value_id,
        value_ids=value_ids,
        is_collection=is_collection,
    )


def _ids(values):
    return {v.attribute_id: v for v in values}


# ── (a) optional enum, no value_id → dropped ───────────────────────────────────

def test_optional_enum_no_value_id_dropped():
    targets = [_target(10, allowed=["A", "B"], is_required=False)]
    merged = [_value(10, "Да", value_id=None)]
    out = _drop_unresolved_optional_enums(merged, targets)
    assert out == []


# ── (b) optional enum WITH value_id → kept ─────────────────────────────────────

def test_optional_enum_with_value_id_kept():
    targets = [_target(10, allowed=["A", "B"], is_required=False)]
    merged = [_value(10, "A", value_id=123)]
    out = _drop_unresolved_optional_enums(merged, targets)
    assert _ids(out)[10].value_id == 123


# ── (c) free-text optional (no allowed_values), no value_id → KEPT ─────────────

def test_free_text_optional_no_value_id_kept():
    targets = [_target(20, allowed=None, is_required=False)]
    merged = [_value(20, "any free text", value_id=None)]
    out = _drop_unresolved_optional_enums(merged, targets)
    assert len(out) == 1
    assert out[0].value == "any free text"


# ── (d) REQUIRED enum, no value_id → KEPT (out of scope) ───────────────────────

def test_required_enum_no_value_id_kept():
    targets = [_target(30, allowed=["A", "B"], is_required=True)]
    merged = [_value(30, "wrong", value_id=None)]
    out = _drop_unresolved_optional_enums(merged, targets)
    assert len(out) == 1
    assert out[0].value == "wrong"


# ── (e) is_collection enum: [resolved, unresolved] → keeps only resolved ───────

def test_collection_partial_keeps_resolved():
    # Production semantics: ozon resolve_value_ids COMPACTS value_ids to the list
    # of resolved ids (drops None), order-preserved, WITHOUT removing the unresolved
    # element from `value`. So value=["X","bogus"], value_ids=[111] (one resolved).
    targets = [
        _target(40, allowed=["X", "Y", "Z"], is_required=False, is_collection=True)
    ]
    merged = [
        _value(40, ["X", "bogus"], value_ids=[111], is_collection=True)
    ]
    out = _drop_unresolved_optional_enums(merged, targets)
    assert len(out) == 1
    assert out[0].value == ["X"]
    assert out[0].value_ids == [111]


def test_collection_all_unresolved_drops_field():
    # All unresolved → resolver leaves value_ids None/empty → drop whole field.
    targets = [
        _target(40, allowed=["X", "Y"], is_required=False, is_collection=True)
    ]
    merged = [
        _value(40, ["bogus1", "bogus2"], value_ids=None, is_collection=True)
    ]
    out = _drop_unresolved_optional_enums(merged, targets)
    assert out == []


def test_collection_all_resolved_kept():
    targets = [
        _target(40, allowed=["X", "Y"], is_required=False, is_collection=True)
    ]
    merged = [
        _value(40, ["X", "Y"], value_ids=[111, 222], is_collection=True)
    ]
    out = _drop_unresolved_optional_enums(merged, targets)
    assert len(out) == 1
    assert out[0].value == ["X", "Y"]


# ── (f) realistic Monitor-style fake-fill → dropped ────────────────────────────

def test_monitor_coating_fake_fill_dropped():
    """«Покрытие экрана» enum=[matte/glossy], LLM put «1,07 миллиардов цветов»
    (a wrong-attribute value) → no value_id, optional → dropped."""
    targets = [
        _target(
            50,
            name="Покрытие экрана",
            allowed=["Матовое", "Глянцевое"],
            is_required=False,
        )
    ]
    merged = [_value(50, "1,07 миллиардов цветов", value_id=None)]
    out = _drop_unresolved_optional_enums(merged, targets)
    assert out == []


# ── NEW: required enum guard ────────────────────────────────────────────────────

def test_required_enum_no_value_id_dropped_by_required_guard():
    """REQUIRED enum with value_id=None IS dropped by _drop_unresolved_required_enums.

    Core safety property: 'Тип'='без карманов'/'открытые' (feature values that
    the Ozon enum matcher rejects → value_id=None) must be purged, not left as
    garbage fills. Empty is better than wrong on a required field.
    """
    targets = [_target(60, name="Тип", allowed=["Футболка", "Джинсы", "Куртка"], is_required=True)]
    # Simulate 'без карманов' surviving to finalize with value_id=None
    merged = [_value(60, "без карманов", value_id=None)]
    out = _drop_unresolved_required_enums(merged, targets)
    assert out == [], f"Expected empty, got {[v.value for v in out]}"


def test_required_enum_with_value_id_survives():
    """REQUIRED enum with a resolved value_id MUST NOT be dropped.

    This is the primary safety contract: correctly-resolved required fields
    (brand-from-name, gender-fill, real Тип='Футболка' with value_id=999) survive.
    """
    targets = [_target(60, name="Тип", allowed=["Футболка", "Джинсы", "Куртка"], is_required=True)]
    merged = [_value(60, "Футболка", value_id=999)]
    out = _drop_unresolved_required_enums(merged, targets)
    assert len(out) == 1
    assert out[0].value == "Футболка"
    assert out[0].value_id == 999


def test_required_enum_placeholder_нет_dropped():
    """Placeholder 'нет' must never fill a required enum.

    In practice this comes from the pipeline placeholder-filter before resolve,
    but _drop_unresolved_required_enums acts as the final safety net: 'нет' will
    not resolve to a value_id, so it gets dropped here too.
    """
    targets = [_target(61, name="Тип", allowed=["Круглый", "V-образный"], is_required=True)]
    # 'нет' would have value_id=None after resolution
    merged = [_value(61, "нет", value_id=None)]
    out = _drop_unresolved_required_enums(merged, targets)
    assert out == []


def test_optional_enum_guard_does_not_touch_required():
    """_drop_unresolved_optional_enums must NOT drop required enums (even unresolved).

    The required-enum guard is separate; the optional guard must leave required
    fields alone so they are handled by _drop_unresolved_required_enums in order.
    """
    targets = [_target(62, name="Тип", allowed=["A", "B"], is_required=True)]
    merged = [_value(62, "garbage", value_id=None)]
    out = _drop_unresolved_optional_enums(merged, targets)
    # optional guard leaves required alone (required guard handles it separately)
    assert len(out) == 1
    assert out[0].value == "garbage"


# ── NEW: placeholder filter ─────────────────────────────────────────────────────

def test_is_placeholder_value_cases():
    """_is_placeholder_value detects canonical placeholder strings."""
    from app.services.enrichment.pipeline import _is_placeholder_value
    positives = ["нет", "НЕТ", " нет ", "-", "—", "none", "None", "n/a", "N/A",
                 "без", "не указано", "нет данных"]
    negatives = ["Футболка", "Мужской", "100", "Хлопок", "без рукавов", "нет в наличии"]
    for p in positives:
        assert _is_placeholder_value(p), f"Expected placeholder: {p!r}"
    for n in negatives:
        assert not _is_placeholder_value(n), f"Expected NOT placeholder: {n!r}"
