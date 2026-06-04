"""Unit tests for the GENERAL conservative finalize-stage cleanup
_drop_unresolved_optional_enums in app/services/enrichment/pipeline.py.

After FULL value_id resolution (deterministic resolve_value_ids + llm_resolve_tail),
an OPTIONAL enum-target value that resolved to NO allowed dictionary option
(value_id is None) is a fake-fill that Ozon would reject on upload. We drop it.

Strict scope:
- ONLY enum targets (allowed_values present). Free-text never touched.
- ONLY optional (is_required == False). Required-enum left intact.
- Drop only when value_id is None AFTER resolution.
- is_collection: drop only unresolved elements; if all unresolved → drop field.
"""
from app.services.enrichment.base import (
    AttributeValue,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import _drop_unresolved_optional_enums


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
