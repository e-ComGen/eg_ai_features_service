"""Unit tests for the GENERAL post-merge type-from-category guard.

Covers _apply_type_from_category in app/services/enrichment/pipeline.py — the
category leaf IS the garment type (leaf "Футболки" → enum option "Футболка").
For each REQUIRED enum target that is currently EMPTY or has value_id=None, the
normalised (lemmatised) leaf is matched EXACTLY (no fuzzy) against the target's
allowed options; on a unique match the field is filled with that option + its
dict value_id. No per-category / per-field hardcode — non-«Тип» enums whose
options do not match the leaf are a harmless no-op. Already-resolved values
(value_id set) are NEVER overridden.
"""
import pytest

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    _apply_type_from_category,
    _type_lemma,
)

_TYPE_ID = 8229  # arbitrary «Тип» attr id
_COLOR_ID = 10


def _ctx(category_path) -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name="some product",
        category_id=1,
        category_path=category_path,
    )


def _enum_target(allowed, *, id=_TYPE_ID, name="Тип", required=True) -> TargetAttribute:
    return TargetAttribute(
        id=id, name=name, type="enum", allowed_values=allowed, is_required=required
    )


def _val(attr_id, value, value_id=None, source=Source.LLM_KNOWLEDGE) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id, value=value, value_id=value_id,
        confidence=0.8, source=source,
    )


def _val_for(values, attr_id=_TYPE_ID):
    return next((v for v in values if v.attribute_id == attr_id), None)


# ── helper sanity: lemmatizer reused from wb_card_source ──────────────────────

def test_lemma_singularizes_plural_leaf():
    # «Футболки» (plural) → «футболка» (lemma) so it matches the enum option.
    assert _type_lemma("Футболки") == _type_lemma("Футболка")


# ── (a) empty required «Тип» → filled from leaf ───────────────────────────────

def test_empty_type_filled_from_leaf():
    """leaf 'Футболки' + options ['Поло','Футболка'] → fills 'Футболка'."""
    ctx = _ctx(["Одежда", "Футболки"])
    target = _enum_target(["Поло", "Футболка"])
    out = _apply_type_from_category([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Футболка"
    assert v.source == Source.DESCRIPTION
    assert v.evidence == "type_from_category"
    assert v.value_id is None  # no value_id_fn given → stays None


def test_empty_type_filled_with_value_id():
    """value_id_fn supplies the dict id → filled type carries it (survives drop)."""
    ctx = _ctx(["Одежда", "Футболки"])
    target = _enum_target(["Поло", "Футболка"])
    out = _apply_type_from_category(
        [], [target], ctx,
        value_id_fn=lambda attr_id, val: 555 if val == "Футболка" else None,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Футболка"
    assert v.value_id == 555


def test_unresolved_existing_type_overwritten():
    """Existing garbage with value_id=None ('без карманов') → overwritten to leaf type."""
    ctx = _ctx(["Одежда", "Футболки"])
    target = _enum_target(["Поло", "Футболка"])
    garbage = _val(_TYPE_ID, "без карманов", value_id=None)
    out = _apply_type_from_category(
        [garbage], [target], ctx,
        value_id_fn=lambda attr_id, val: 7,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Футболка"
    assert v.value_id == 7
    assert v.evidence == "type_from_category"
    assert len(out) == 1  # overwritten in place, not duplicated


# ── (b) no matching option → stays empty ──────────────────────────────────────

def test_no_matching_option_stays_empty():
    """leaf with NO matching enum option → field NOT filled (drop-guard empties)."""
    ctx = _ctx(["Электроника", "Наушники"])
    target = _enum_target(["Поло", "Футболка"])  # neither matches 'наушники'
    out = _apply_type_from_category([], [target], ctx)
    assert _val_for(out) is None


def test_no_category_path_noop():
    """No category_path → nothing to derive, no-op."""
    ctx = _ctx([])
    target = _enum_target(["Поло", "Футболка"])
    out = _apply_type_from_category([], [target], ctx)
    assert _val_for(out) is None


# ── (c) already-resolved value → untouched ────────────────────────────────────

def test_already_resolved_type_untouched():
    """Existing «Тип» with a value_id (resolved) is NEVER overridden."""
    ctx = _ctx(["Одежда", "Футболки"])
    target = _enum_target(["Поло", "Футболка"])
    resolved = _val(_TYPE_ID, "Поло", value_id=42, source=Source.OZON_CARD)
    out = _apply_type_from_category(
        [resolved], [target], ctx,
        value_id_fn=lambda attr_id, val: 999,
    )
    v = _val_for(out)
    assert v is not None and v.value == "Поло"  # untouched
    assert v.value_id == 42
    assert v.source == Source.OZON_CARD


# ── (d) non-«Тип» enum whose options don't match the leaf → no-op ─────────────

def test_non_type_enum_noop():
    """A Цвет enum (options unrelated to leaf) is a harmless no-op."""
    ctx = _ctx(["Одежда", "Футболки"])
    color = _enum_target(["Чёрный", "Белый"], id=_COLOR_ID, name="Цвет")
    out = _apply_type_from_category([], [color], ctx)
    assert _val_for(out, attr_id=_COLOR_ID) is None  # leaf doesn't match colors


def test_optional_enum_not_filled():
    """Scope is REQUIRED only — an optional enum matching the leaf is left alone."""
    ctx = _ctx(["Одежда", "Футболки"])
    target = _enum_target(["Поло", "Футболка"], required=False)
    out = _apply_type_from_category([], [target], ctx)
    assert _val_for(out) is None


# ── (e) ambiguity / safety ────────────────────────────────────────────────────

def test_two_options_match_leaf_skip():
    """If >1 option normalises to the leaf lemma → ambiguous, do NOT fill."""
    ctx = _ctx(["Одежда", "Футболки"])
    # both 'Футболка' and 'Футболки' lemmatise to the same lemma → ambiguous
    target = _enum_target(["Футболка", "Футболки"])
    out = _apply_type_from_category([], [target], ctx)
    assert _val_for(out) is None


def test_value_id_fn_failure_does_not_crash():
    """value_id_fn raising → type still filled, value_id stays None."""
    ctx = _ctx(["Одежда", "Футболки"])
    target = _enum_target(["Поло", "Футболка"])

    def _boom(_attr_id, _val):
        raise RuntimeError("dict unavailable")

    out = _apply_type_from_category([], [target], ctx, value_id_fn=_boom)
    v = _val_for(out)
    assert v is not None and v.value == "Футболка"
    assert v.value_id is None


def test_free_text_target_skipped():
    """Required target WITHOUT allowed_values (free-text) → skipped (nothing to match)."""
    ctx = _ctx(["Одежда", "Футболки"])
    target = TargetAttribute(id=_TYPE_ID, name="Тип", type="text", is_required=True)
    out = _apply_type_from_category([], [target], ctx)
    assert _val_for(out) is None


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
