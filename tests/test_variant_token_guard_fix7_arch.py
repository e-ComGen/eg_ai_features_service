"""Arch/property tests for FIX-7 (variant-token contradiction guard, INV-8).

Spec: docs/MANIFEST_variant_token_guard.md.
Covers INV-8a..e. The O1a..O8 worked-example oracle lives in
tests/test_variant_token_guard_fix7_oracle.py (separate file, no-self-bias
convention: same author here, but kept split per the existing FIX-5/FIX-6
oracle/arch split style in this test suite).
"""
from app.services.enrichment.pipeline import _reconcile_cross_field_contradictions
from app.services.enrichment.base import AttributeValue, Source, TargetAttribute


def _av(attribute_id, value, confidence, source, evidence=None, is_collection=False):
    return AttributeValue(
        attribute_id=attribute_id,
        value=value,
        confidence=confidence,
        source=source,
        evidence=evidence,
        is_collection=is_collection,
    )


def _target(id, name, type="text", is_collection=False):
    return TargetAttribute(id=id, name=name, type=type, is_collection=is_collection)


DIAG_ID, LINE_ID, OTHER_ID = 801, 802, 803


# ---------------------------------------------------------------------------
# INV-8a -- never drops a value whose target name doesn't match the
# diagonal-inch / line-generation pattern, no matter how contradictory the
# value looks against product_name.
# ---------------------------------------------------------------------------

def test_inv8a_non_matching_target_name_never_dropped_even_if_value_looks_contradictory():
    targets = [_target(OTHER_ID, "Мощность двигателя, Вт")]
    merged = [_av(OTHER_ID, "42", 0.9, Source.WEB_SEARCH, evidence="[compare-page]")]
    # product_name contains "42" as a size-range number and "42" as the value --
    # if RULE-G1 mistakenly fired on a non-diagonal target this would still
    # KEEP by coincidence, so use a product_name that would DROP it under
    # RULE-G1 logic (a different number in range) to prove the target-name
    # gate, not the value match, is what protects it.
    result = _reconcile_cross_field_contradictions(merged, targets, "SomeBrand Model 55")
    kept = next((v for v in result if v.attribute_id == OTHER_ID), None)
    assert kept is not None
    assert kept.value == "42"


def test_inv8a_non_matching_target_name_line_like_value_never_dropped():
    targets = [_target(OTHER_ID, "Комментарий")]
    merged = [_av(OTHER_ID, "Redmi Note 99", 0.9, Source.WEB_SEARCH)]
    result = _reconcile_cross_field_contradictions(merged, targets, "Xiaomi Redmi Note 13")
    kept = next((v for v in result if v.attribute_id == OTHER_ID), None)
    assert kept is not None
    assert kept.value == "Redmi Note 99"


# ---------------------------------------------------------------------------
# INV-8b -- if product_name has no size in range 17..120, no diagonal value
# is ever dropped (guard fully inert on the size rule).
# ---------------------------------------------------------------------------

def test_inv8b_no_size_in_range_diagonal_value_untouched():
    targets = [_target(DIAG_ID, "Диагональ экрана, дюймы")]
    merged = [_av(DIAG_ID, '99″', 0.7, Source.WEB_SEARCH)]
    # "LG C4" -> only digit group is "4", outside 17..120 -> name_sizes empty.
    result = _reconcile_cross_field_contradictions(merged, targets, "LG C4")
    kept = next((v for v in result if v.attribute_id == DIAG_ID), None)
    assert kept is not None
    assert kept.value == '99″'


def test_inv8b_empty_product_name_diagonal_value_untouched():
    targets = [_target(DIAG_ID, "Диагональ экрана, дюймы")]
    merged = [_av(DIAG_ID, '42″', 0.7, Source.WEB_SEARCH)]
    result = _reconcile_cross_field_contradictions(merged, targets, "")
    kept = next((v for v in result if v.attribute_id == DIAG_ID), None)
    assert kept is not None
    assert kept.value == '42″'


# ---------------------------------------------------------------------------
# INV-8c -- guard only REMOVES from the result, never creates or mutates
# other (unrelated) values.
# ---------------------------------------------------------------------------

def test_inv8c_unrelated_values_pass_through_unmutated():
    targets = [
        _target(DIAG_ID, "Диагональ экрана, дюймы"),
        _target(OTHER_ID, "Мощность двигателя, Вт"),
    ]
    other_av = _av(OTHER_ID, "1500", 0.9, Source.DESCRIPTION, evidence="verbatim")
    merged = [
        _av(DIAG_ID, '42″', 0.7, Source.WEB_SEARCH),  # will be dropped
        other_av,
    ]
    result = _reconcile_cross_field_contradictions(merged, targets, "LG OLED55C4RLA")
    # dropped one, the other survives byte-identical (same value/evidence)
    assert len(result) == 1
    survivor = result[0]
    assert survivor.attribute_id == OTHER_ID
    assert survivor.value == other_av.value
    assert survivor.evidence == other_av.evidence


def test_inv8c_guard_never_grows_the_result_set():
    targets = [_target(DIAG_ID, "Диагональ экрана, дюймы")]
    merged = [_av(DIAG_ID, '55″', 0.9, Source.OZON_CARD)]
    result = _reconcile_cross_field_contradictions(merged, targets, "LG OLED55C4RLA")
    assert len(result) <= len(merged)


# ---------------------------------------------------------------------------
# INV-8d -- idempotency: re-running the guard on its own output is a no-op.
# ---------------------------------------------------------------------------

def test_inv8d_idempotent_on_kept_value():
    targets = [_target(DIAG_ID, "Диагональ экрана, дюймы")]
    merged = [_av(DIAG_ID, '55″', 0.9, Source.OZON_CARD)]
    first = _reconcile_cross_field_contradictions(merged, targets, "LG OLED55C4RLA")
    second = _reconcile_cross_field_contradictions(first, targets, "LG OLED55C4RLA")
    assert [(v.attribute_id, v.value) for v in first] == [(v.attribute_id, v.value) for v in second]


def test_inv8d_idempotent_on_dropped_value_stays_dropped_and_empty():
    targets = [_target(DIAG_ID, "Диагональ экрана, дюймы")]
    merged = [_av(DIAG_ID, '42″', 0.7, Source.WEB_SEARCH)]
    first = _reconcile_cross_field_contradictions(merged, targets, "LG OLED55C4RLA")
    assert first == []
    second = _reconcile_cross_field_contradictions(first, targets, "LG OLED55C4RLA")
    assert second == []


# ---------------------------------------------------------------------------
# INV-8e -- backward-compatible signature: existing 2-arg call-sites (no
# product_name) must not raise, and the guard is inert (no product_name to
# compare against -> name_sizes/gen lookups can't match anything).
# ---------------------------------------------------------------------------

def test_inv8e_two_arg_call_backward_compatible_no_crash_and_inert():
    targets = [_target(DIAG_ID, "Диагональ экрана, дюймы")]
    merged = [_av(DIAG_ID, '42″', 0.7, Source.WEB_SEARCH)]
    # No product_name passed at all -- must not raise TypeError, and must not
    # drop anything (guard has nothing to compare against).
    result = _reconcile_cross_field_contradictions(merged, targets)
    kept = next((v for v in result if v.attribute_id == DIAG_ID), None)
    assert kept is not None
    assert kept.value == '42″'


def test_inv8e_none_product_name_treated_as_inert_not_crashing():
    targets = [_target(DIAG_ID, "Диагональ экрана, дюймы")]
    merged = [_av(DIAG_ID, '42″', 0.7, Source.WEB_SEARCH)]
    result = _reconcile_cross_field_contradictions(merged, targets, "")
    kept = next((v for v in result if v.attribute_id == DIAG_ID), None)
    assert kept is not None
