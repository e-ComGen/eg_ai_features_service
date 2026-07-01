"""Arch/property tests for grounding-arbitration Round 3 (FIX-5 + FIX-6).

Spec: docs/MANIFEST_grounding_arbitration.md §7.
Covers INV-5-chuck (Комплектация chuck-key evidence + drill-type/SDS
consistency: no surviving SDS-Plus chuck on a confirmed plain "Дрель") and
INV-7 (every number+unit in a generated annotation corresponds to a filled
field, unit-normalized; else stripped).

These are PROPERTY tests exercising the unit-level functions directly
(_reconcile_cross_field_contradictions, _check_annotation_numeric_grounding,
_extract_field_numeric_grounding, _normalize_annotation_unit) with synthetic
inputs -- no live enrichment, no live LLM call.
The O4/O4b/O5/O5b/O5c worked-example oracle lives in
tests/test_grounding_arbitration_fix5_fix6_oracle.py (functional, authored
by a different model family per the no-self-bias rule).
"""
import pytest

from app.services.enrichment.base import AttributeValue, Source, TargetAttribute
from app.services.enrichment.pipeline import (
    _reconcile_cross_field_contradictions,
    _check_annotation_numeric_grounding,
    _extract_field_numeric_grounding,
    _normalize_annotation_unit,
)


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


# ---------------------------------------------------------------------------
# INV-5-chuck -- Комплектация chuck-key evidence + drill-type/SDS consistency
# ---------------------------------------------------------------------------

TYPE_ID, KOMP_ID, CHUCK_ID = 920, 912, 913


def test_inv5_chuck_patronnyi_klyuch_forces_keyed():
    """Комплектация naming 'патронный ключ' (chuck-KEY tool) -- new pattern
    coverage -- must force Тип патрона to Ключевой, even with no drill-type
    context involved at all (pure RULE B extension)."""
    targets = [_target(KOMP_ID, "Комплектация"), _target(CHUCK_ID, "Тип патрона")]
    merged = [
        _av(KOMP_ID, "дрель, патронный ключ, кейс для переноски", 0.9,
            Source.DESCRIPTION, evidence="verbatim product description"),
        _av(CHUCK_ID, "Быстрозажимной", 0.6, Source.SAFE_ENUM_FILL, evidence="enum default"),
    ]
    result = _reconcile_cross_field_contradictions(merged, targets)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert "ключев" in chuck.value.lower()
    assert "быстрозажимн" not in chuck.value.lower()


def test_inv5_chuck_sds_invalid_for_plain_drill_corrects_via_komplektatsiya():
    """O4: SDS-Plus on a confirmed plain 'Дрель' with Комплектация naming a
    chuck-key -> corrected to Ключевой, never left as SDS-Plus."""
    targets = [
        _target(TYPE_ID, "Тип"),
        _target(KOMP_ID, "Комплектация"),
        _target(CHUCK_ID, "Тип патрона"),
    ]
    merged = [
        _av(TYPE_ID, "Дрель", 0.9, Source.DESCRIPTION),
        _av(KOMP_ID, "аккумулятор, зарядное устройство, патронный ключ, кейс",
            0.9, Source.DESCRIPTION, evidence="verbatim"),
        _av(CHUCK_ID, "SDS-Plus", 0.7, Source.WEB_SEARCH, evidence="guess"),
    ]
    result = _reconcile_cross_field_contradictions(merged, targets)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert "sds" not in chuck.value.lower()
    assert "ключев" in chuck.value.lower()


def test_inv5_chuck_sds_invalid_for_plain_drill_abstains_without_evidence():
    """Plain 'Дрель' + SDS-Plus chuck, but NO Комплектация evidence to
    correct from -- must ABSTAIN (drop), never silently keep SDS-Plus."""
    targets = [
        _target(TYPE_ID, "Тип"),
        _target(CHUCK_ID, "Тип патрона"),
    ]
    merged = [
        _av(TYPE_ID, "Дрель", 0.9, Source.DESCRIPTION),
        _av(CHUCK_ID, "SDS-Plus", 0.7, Source.WEB_SEARCH, evidence="guess"),
    ]
    result = _reconcile_cross_field_contradictions(merged, targets)
    assert not any(v.attribute_id == CHUCK_ID for v in result)


def test_inv5_chuck_sds_invalid_even_when_source_priority_high():
    """SDS-Plus-for-plain-Дрель is a domain impossibility -- correction is
    UNCONDITIONAL regardless of SOURCE_PRIORITY (unlike RULE B's generic
    Комплектация-vs-Тип-патрона dispute, which DOES respect priority).
    Even a high-priority OZON_CARD SDS-Plus fill must not survive."""
    targets = [
        _target(TYPE_ID, "Тип"),
        _target(KOMP_ID, "Комплектация"),
        _target(CHUCK_ID, "Тип патрона"),
    ]
    merged = [
        _av(TYPE_ID, "Дрель", 0.9, Source.DESCRIPTION),
        _av(KOMP_ID, "патронный ключ, кейс", 0.5, Source.LLM_KNOWLEDGE, evidence="guess"),
        _av(CHUCK_ID, "SDS-Plus", 0.95, Source.OZON_CARD, evidence="real card data"),
    ]
    result = _reconcile_cross_field_contradictions(merged, targets)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert "sds" not in chuck.value.lower()


@pytest.mark.parametrize("rotary_type", ["Перфоратор", "Перфоратор SDS-Max", "Отбойный молоток"])
def test_inv5_chuck_sds_valid_for_rotary_hammer_unchanged(rotary_type):
    """Regression: SDS-Plus IS valid on an actual rotary hammer/perforator --
    RULE C must not fire (and must not touch) when Тип names a rotary tool."""
    targets = [
        _target(TYPE_ID, "Тип"),
        _target(CHUCK_ID, "Тип патрона"),
    ]
    merged = [
        _av(TYPE_ID, rotary_type, 0.9, Source.DESCRIPTION),
        _av(CHUCK_ID, "SDS-Plus", 0.8, Source.OZON_CARD, evidence="real card data"),
    ]
    result = _reconcile_cross_field_contradictions(merged, targets)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert chuck.value == "SDS-Plus"  # untouched


def test_inv5_chuck_sds_no_type_target_leaves_chuck_unchanged():
    """No 'Тип' target present at all -> RULE C cannot confirm plain-drill,
    must ABSTAIN from touching Тип патрона entirely (false positives here
    would be worse than missing the fix)."""
    targets = [_target(CHUCK_ID, "Тип патрона")]
    merged = [_av(CHUCK_ID, "SDS-Plus", 0.8, Source.OZON_CARD, evidence="real card data")]
    result = _reconcile_cross_field_contradictions(merged, targets)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert chuck.value == "SDS-Plus"  # untouched, no target to confirm drill-type


def test_inv5_chuck_o4b_agreeing_keyed_unchanged():
    """O4b regression: a genuine Ключевой that AGREES with Комплектация ->
    unchanged (RULE B/idempotence, still holds after the pattern extension)."""
    targets = [_target(KOMP_ID, "Комплектация"), _target(CHUCK_ID, "Тип патрона")]
    merged = [
        _av(KOMP_ID, "патронный ключ, кейс", 0.9, Source.DESCRIPTION, evidence="x"),
        _av(CHUCK_ID, "Ключевой", 0.6, Source.SAFE_ENUM_FILL, evidence="x"),
    ]
    result = _reconcile_cross_field_contradictions(merged, targets)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert chuck.value == "Ключевой"  # untouched, no spurious re-write


def test_inv5_chuck_o4b_agreeing_quickrelease_unchanged():
    """O4b regression (other direction): genuine Быстрозажимной with no chuck-
    key mention in Комплектация -> unchanged."""
    targets = [_target(KOMP_ID, "Комплектация"), _target(CHUCK_ID, "Тип патрона")]
    merged = [
        _av(KOMP_ID, "аккумулятор, зарядное устройство, кейс", 0.9, Source.DESCRIPTION, evidence="x"),
        _av(CHUCK_ID, "Быстрозажимной", 0.6, Source.SAFE_ENUM_FILL, evidence="x"),
    ]
    result = _reconcile_cross_field_contradictions(merged, targets)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert chuck.value == "Быстрозажимной"  # untouched


# ---------------------------------------------------------------------------
# INV-7 -- annotation numeric grounding (deterministic, non-LLM)
# ---------------------------------------------------------------------------

WEIGHT_ID, DIAM_ID, IMPACT_ID = 930, 931, 932


def test_inv7_ungrounded_number_unit_is_flagged_and_stripped():
    """O5: annotation invents a torque figure from a totally different
    field (impacts/min) under the WRONG unit -- no torque field exists at
    all -> the claim is flagged and its sentence stripped."""
    filled_by_id = {
        IMPACT_ID: _av(IMPACT_ID, "48000", 0.9, Source.OZON_CARD),
    }
    target_names = {IMPACT_ID: "Частота ударов, уд/мин"}
    text = "Мощный инструмент для дома и стройки. Крутящий момент 48 Нм впечатляет. Удобная рукоятка."
    cleaned, violations = _check_annotation_numeric_grounding(text, filled_by_id, target_names)
    assert violations, "expected a violation for the ungrounded '48 Нм' claim"
    assert "Нм" not in cleaned
    assert "48" not in cleaned
    # unrelated sentences survive
    assert "Мощный инструмент" in cleaned
    assert "Удобная рукоятка" in cleaned


def test_inv7_no_false_strip_unit_reformat_kg_vs_g():
    """O5b (no false-strip): annotation says '1,8 кг', field stores '1800'
    with unit 'г' in its name suffix -- legitimate unit-equivalent reformat,
    must be KEPT unchanged."""
    filled_by_id = {WEIGHT_ID: _av(WEIGHT_ID, "1800", 0.9, Source.OZON_CARD)}
    target_names = {WEIGHT_ID: "Вес, г"}
    text = "Компактный и лёгкий инструмент весом всего 1,8 кг."
    cleaned, violations = _check_annotation_numeric_grounding(text, filled_by_id, target_names)
    assert violations == []
    assert cleaned == text


def test_inv7_no_false_strip_verbatim_all_match():
    """O5c: every number+unit in the annotation matches a filled field
    verbatim -> unchanged, zero violations."""
    filled_by_id = {
        DIAM_ID: _av(DIAM_ID, "13", 0.9, Source.OZON_CARD),
        WEIGHT_ID: _av(WEIGHT_ID, "1800", 0.9, Source.OZON_CARD),
    }
    target_names = {DIAM_ID: "Диаметр патрона, мм", WEIGHT_ID: "Вес, г"}
    text = "Патрон диаметром 13 мм и вес всего 1800 г делают инструмент удобным."
    cleaned, violations = _check_annotation_numeric_grounding(text, filled_by_id, target_names)
    assert violations == []
    assert cleaned == text


def test_inv7_sentence_split_does_not_garble_decimal_point_english_style():
    """Gate-2 (deepseek) round-2 finding: the sentence-boundary splitter must
    not treat a dot-decimal number (e.g. '1.8', English-style, no space) as a
    sentence end. When a dot-decimal grounded claim shares ONE run-on
    sentence with an ungrounded claim (no period between them), a splitter
    that mis-fires on the decimal point tears the sentence apart AT THE
    NUMBER, dropping the half containing the violation but leaving a
    GARBLED remainder (e.g. 'Вес всего 1.' -- truncated mid-number, worse
    than dropping the whole sentence). The fixed splitter treats the whole
    run-on sentence as one unit, so it is dropped cleanly (empty result),
    never left as a mangled fragment ending in a bare digit-dot."""
    filled_by_id = {WEIGHT_ID: _av(WEIGHT_ID, "1800", 0.9, Source.OZON_CARD)}
    target_names = {WEIGHT_ID: "Вес, г"}
    text = "Вес всего 1.8 кг при впечатляющем крутящем моменте 48 Нм."
    cleaned, violations = _check_annotation_numeric_grounding(text, filled_by_id, target_names)
    assert violations, "expected the ungrounded '48 Нм' claim to be flagged"
    # must NOT be torn mid-decimal into a garbled truncated fragment
    assert cleaned != "Вес всего 1.", "sentence was garbled at the decimal point instead of being dropped cleanly"
    assert not cleaned.endswith("1."), "must not leave a dangling truncated decimal"
    assert cleaned == ""  # the whole run-on sentence (grounded+ungrounded, no period between) drops as one unit


def test_inv7_field_value_with_embedded_unit_grounds_annotation():
    """Field VALUE itself embeds the unit (e.g. '48000 уд/мин' as a single
    text value, not split name-suffix/bare-number) -- annotation citing the
    exact same figure+unit must be recognized as grounded."""
    filled_by_id = {IMPACT_ID: _av(IMPACT_ID, "48000 уд/мин", 0.9, Source.OZON_CARD)}
    target_names = {IMPACT_ID: "Частота ударов"}
    text = "Частота ударов достигает 48000 уд/мин для быстрого сверления."
    cleaned, violations = _check_annotation_numeric_grounding(text, filled_by_id, target_names)
    assert violations == []
    assert cleaned == text


def test_inv7_unknown_unit_token_is_ignored_not_flagged():
    """A number followed by a token NOT in the unit alias table (e.g. a
    year) must never be extracted/flagged at all -- it's not our concern."""
    filled_by_id = {}
    target_names = {}
    text = "Модель выпускается с 2024 года и уже завоевала популярность."
    cleaned, violations = _check_annotation_numeric_grounding(text, filled_by_id, target_names)
    assert violations == []
    assert cleaned == text


def test_inv7_normalize_annotation_unit_kg_to_g():
    """Unit test of the pure normalizer: кг -> canonical grams x1000."""
    result = _normalize_annotation_unit("1,8", "кг")
    assert result is not None
    unit, value = result
    assert unit == "g"
    assert value == pytest.approx(1800.0)


def test_inv7_normalize_annotation_unit_unknown_returns_none():
    assert _normalize_annotation_unit("48", "Нм_unknown_alias") is None


def test_inv7_extract_field_numeric_grounding_skips_list_values():
    """Collection-valued fields (lists) are never numeric-grounding
    candidates -- must not raise, must simply be skipped."""
    filled_by_id = {
        1: _av(1, ["Сверление", "Сверление с ударом"], 0.8, Source.WEB_SEARCH, is_collection=True),
        WEIGHT_ID: _av(WEIGHT_ID, "1800", 0.9, Source.OZON_CARD),
    }
    target_names = {1: "Режимы работы", WEIGHT_ID: "Вес, г"}
    grounded = _extract_field_numeric_grounding(filled_by_id, target_names)
    assert ("g", 1800.0) in grounded
    assert len(grounded) == 1


@pytest.mark.parametrize("unit_word,mult", [("кг", 1000.0), ("г", 1.0), ("мм", 1.0), ("см", 10.0)])
def test_inv7_property_unit_normalization_multiplier(unit_word, mult):
    """Property: normalizing N <unit> always yields N*multiplier in the
    canonical unit, for every alias in the table exercised here."""
    unit, value = _normalize_annotation_unit("2", unit_word)
    assert value == pytest.approx(2.0 * mult)
