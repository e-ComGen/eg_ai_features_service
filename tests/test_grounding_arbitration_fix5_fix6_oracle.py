import pytest
from app.services.enrichment.pipeline import (
    _reconcile_cross_field_contradictions,
    _check_annotation_numeric_grounding,
)
from app.services.enrichment.base import AttributeValue, Source, TargetAttribute


def _av(attribute_id, value, confidence, source, evidence=None, is_collection=False):
    """Helper to construct AttributeValue objects."""
    return AttributeValue(
        attribute_id=attribute_id,
        value=value,
        confidence=confidence,
        source=source,
        evidence=evidence,
        is_collection=is_collection,
    )


def _target(id, name, type="text", is_collection=False):
    """Helper to construct TargetAttribute objects."""
    return TargetAttribute(
        id=id,
        name=name,
        type=type,
        is_collection=is_collection,
    )


def test_o4_sds_chuck_corrected_for_plain_drill_dewalt_dwd024():
    """
    O4 -- SDS chuck corrected for plain drill DeWalt DWD024.
    
    Simulate the live DeWalt DWD024 defect where a plain drill (Тип=Дрель)
    had an incorrectly guessed SDS-Plus chuck type. The Комплектация mentions
    "патронный ключ", which indicates a keyed (Ключевой) chuck, not SDS-Plus.
    The reconciliation should correct the chuck type.
    """
    # Create the three AttributeValue objects
    tip_attr = _av(
        attribute_id=920,
        value="Дрель",
        confidence=0.95,
        source=Source.DESCRIPTION,
    )
    
    komplektatsiya_attr = _av(
        attribute_id=912,
        value="аккумулятор, зарядное устройство, патронный ключ, кейс для переноски",
        confidence=0.9,
        source=Source.DESCRIPTION,
        evidence="verbatim product description",
    )
    
    tip_patrona_attr = _av(
        attribute_id=913,
        value="SDS-Plus",
        confidence=0.5,
        source=Source.WEB_SEARCH,
        evidence="guess, wrong",
    )
    
    merged = [tip_attr, komplektatsiya_attr, tip_patrona_attr]
    
    targets = [
        _target(id=920, name="Тип"),
        _target(id=912, name="Комплектация"),
        _target(id=913, name="Тип патрона"),
    ]
    
    result = _reconcile_cross_field_contradictions(merged, targets)
    
    # Find the Тип патрона result
    tip_patrona_result = None
    for av in result:
        if av.attribute_id == 913:
            tip_patrona_result = av
            break
    
    assert tip_patrona_result is not None, "Тип патрона should be in results"
    
    # Assert the value no longer contains "sds" (case-insensitive)
    assert "sds" not in tip_patrona_result.value.lower(), (
        f"Тип патрона should not contain 'sds', got: {tip_patrona_result.value}"
    )
    
    # Assert the value now contains "ключев" (case-insensitive)
    assert "ключев" in tip_patrona_result.value.lower(), (
        f"Тип патрона should contain 'ключев', got: {tip_patrona_result.value}"
    )


def test_o4b_agreeing_chuck_type_unchanged_regression():
    """
    O4b -- Agreeing chuck type unchanged (regression test).
    
    When Комплектация mentions "ключевой патрон" and Тип патрона is already
    correctly set to "Ключевой", they agree -- no correction should be made.
    This is a regression test to ensure we don't spuriously rewrite agreeing data.
    """
    komplektatsiya_attr = _av(
        attribute_id=912,
        value="ключевой патрон, кейс",
        confidence=0.9,
        source=Source.DESCRIPTION,
    )
    
    tip_patrona_attr = _av(
        attribute_id=913,
        value="Ключевой",
        confidence=0.95,
        source=Source.SAFE_ENUM_FILL,
    )
    
    merged = [komplektatsiya_attr, tip_patrona_attr]
    
    targets = [
        _target(id=912, name="Комплектация"),
        _target(id=913, name="Тип патрона"),
    ]
    
    result = _reconcile_cross_field_contradictions(merged, targets)
    
    # Find the Тип патрона result
    tip_patrona_result = None
    for av in result:
        if av.attribute_id == 913:
            tip_patrona_result = av
            break
    
    assert tip_patrona_result is not None, "Тип патрона should be in results"
    
    # Assert the value is unchanged, exactly "Ключевой"
    assert tip_patrona_result.value == "Ключевой", (
        f"Тип патрона should be unchanged as 'Ключевой', got: {tip_patrona_result.value}"
    )


def test_o5_annotation_strips_ungrounded_torque_claim_makita():
    """
    O5 -- Annotation strips ungrounded torque claim (Makita HP1630 defect).
    
    Simulate the live Makita HP1630 defect where the annotation mentions
    "Крутящий момент 48 Нм" but the only numeric field present is impacts-per-minute
    (48000 уд/мин). There is no torque field, so the "48 Нм" claim is ungrounded
    and should be caught as a violation and stripped from the cleaned text.
    """
    some_id = 12345
    
    filled_by_id = {
        some_id: _av(
            attribute_id=some_id,
            value="48000",
            confidence=0.9,
            source=Source.OZON_CARD,
        )
    }
    
    target_names = {
        some_id: "Частота ударов, уд/мин"
    }
    
    annotation_text = (
        "Этот мощный инструмент отлично справляется с задачами. "
        "Крутящий момент 48 Нм делает его незаменимым помощником. "
        "Эргономичная рукоятка снижает усталость руки."
    )
    
    cleaned_text, violations = _check_annotation_numeric_grounding(
        annotation_text, filled_by_id, target_names
    )
    
    # Assert violations list is non-empty (the ungrounded "48 Нм" claim was caught)
    assert len(violations) > 0, (
        f"Violations list should be non-empty for ungrounded torque claim, got: {violations}"
    )
    
    # Assert the cleaned text does NOT contain "Нм"
    assert "Нм" not in cleaned_text, (
        f"Cleaned text should not contain 'Нм', got: {cleaned_text}"
    )
    
    # Assert the other two unrelated sentences are still present
    assert "мощный инструмент" in cleaned_text, (
        f"First sentence should still be present, got: {cleaned_text}"
    )
    assert "Эргономичная рукоятка" in cleaned_text, (
        f"Third sentence should still be present, got: {cleaned_text}"
    )


def test_o5b_annotation_weight_unit_reformat_kept_regression():
    """
    O5b -- Annotation weight unit reformat kept (regression test, no false-strip).
    
    When the field value is "1800" (bare number) and the target name is "Вес, г"
    (unit in the name suffix), and the annotation says "1,8 кг" (comma-decimal,
    different but equivalent unit), this is a legitimate unit reformat and
    should NOT be stripped.
    """
    attr_id = 54321
    
    filled_by_id = {
        attr_id: _av(
            attribute_id=attr_id,
            value="1800",
            confidence=0.95,
            source=Source.DESCRIPTION,
        )
    }
    
    target_names = {
        attr_id: "Вес, г"
    }
    
    annotation_text = (
        "Этот инструмент удивительно лёгкий -- всего 1,8 кг веса, "
        "что удобно для длительной работы."
    )
    
    cleaned_text, violations = _check_annotation_numeric_grounding(
        annotation_text, filled_by_id, target_names
    )
    
    # Assert violations list is empty
    assert violations == [], (
        f"Violations list should be empty for legitimate unit reformat, got: {violations}"
    )
    
    # Assert the cleaned text is exactly equal to the original annotation_text
    assert cleaned_text == annotation_text, (
        f"Cleaned text should be unchanged, got: {cleaned_text}"
    )


def test_o5c_annotation_all_numbers_verbatim_match_unchanged():
    """
    O5c -- Annotation all numbers verbatim match unchanged (regression test).
    
    When filled_by_id has multiple numeric fields with matching unit suffixes
    in target_names, and the annotation quotes both numbers with the SAME units
    verbatim, nothing should be stripped or altered.
    """
    diameter_id = 11111
    weight_id = 22222
    
    filled_by_id = {
        diameter_id: _av(
            attribute_id=diameter_id,
            value="13",
            confidence=0.95,
            source=Source.DESCRIPTION,
        ),
        weight_id: _av(
            attribute_id=weight_id,
            value="1800",
            confidence=0.9,
            source=Source.DESCRIPTION,
        ),
    }
    
    target_names = {
        diameter_id: "Диаметр патрона, мм",
        weight_id: "Вес, г",
    }
    
    annotation_text = "Патрон диаметром 13 мм и вес 1800 г делают инструмент удобным."
    
    cleaned_text, violations = _check_annotation_numeric_grounding(
        annotation_text, filled_by_id, target_names
    )
    
    # Assert violations list is empty
    assert violations == [], (
        f"Violations list should be empty for verbatim match, got: {violations}"
    )
    
    # Assert the cleaned text equals the original annotation_text unchanged
    assert cleaned_text == annotation_text, (
        f"Cleaned text should be unchanged, got: {cleaned_text}"
    )
