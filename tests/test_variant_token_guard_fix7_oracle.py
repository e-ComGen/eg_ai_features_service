"""Functional oracle for FIX-7 (variant-token contradiction guard, INV-8).

Spec: docs/MANIFEST_variant_token_guard.md.
Encodes the manifest's worked-example oracle O1a..O8 verbatim -- does NOT
invent expected values, only asserts the ones the manifest specifies.
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


DIAG_ID = 701
LINE_ID = 702
IMPACT_ID = 703
CHUCK_MIN_DIAM_ID = 704
MODEL_NAME_ID = 705
RESOLUTION_ID = 706


def _diag_target():
    return _target(DIAG_ID, "Диагональ экрана, дюймы")


def _line_target():
    return _target(LINE_ID, "Линейка мобильных устройств")


def test_o1a_lg_oled55c4rla_diagonal_42_dropped():
    """O1a: LG OLED55C4RLA / Диагональ = 42" -> DROP (42 not in {55})."""
    merged = [_av(DIAG_ID, '42″', 0.7, Source.WEB_SEARCH, evidence="[compare-page]")]
    targets = [_diag_target()]
    result = _reconcile_cross_field_contradictions(merged, targets, "LG OLED55C4RLA")
    assert not any(v.attribute_id == DIAG_ID for v in result)


def test_o1b_lg_oled55c4rla_diagonal_55_kept():
    """O1b: LG OLED55C4RLA / Диагональ = 55" -> KEEP."""
    merged = [_av(DIAG_ID, '55″', 0.9, Source.OZON_CARD)]
    targets = [_diag_target()]
    result = _reconcile_cross_field_contradictions(merged, targets, "LG OLED55C4RLA")
    kept = next((v for v in result if v.attribute_id == DIAG_ID), None)
    assert kept is not None
    assert kept.value == '55″'


def test_o2_samsung_qe65qn95d_diagonal_65_kept():
    """O2: Samsung QE65QN95D / Диагональ = 65" -> KEEP (65 in {65, 95})."""
    merged = [_av(DIAG_ID, '65″', 0.9, Source.OZON_CARD)]
    targets = [_diag_target()]
    result = _reconcile_cross_field_contradictions(merged, targets, "Samsung QE65QN95D")
    kept = next((v for v in result if v.attribute_id == DIAG_ID), None)
    assert kept is not None
    assert kept.value == '65″'


def test_o3_lg_oled42c4rla_diagonal_42_kept():
    """O3: LG OLED42C4RLA / Диагональ = 42" -> KEEP."""
    merged = [_av(DIAG_ID, '42″', 0.9, Source.OZON_CARD)]
    targets = [_diag_target()]
    result = _reconcile_cross_field_contradictions(merged, targets, "LG OLED42C4RLA")
    kept = next((v for v in result if v.attribute_id == DIAG_ID), None)
    assert kept is not None
    assert kept.value == '42″'


def test_o4a_xiaomi_redmi_note_13_line_gen_14_dropped():
    """O4a: Xiaomi Redmi Note 13 4G / Линейка = Redmi Note 14 -> DROP (14 != 13)."""
    merged = [_av(LINE_ID, "Redmi Note 14", 0.6, Source.WEB_SEARCH, evidence="[compare-page]")]
    targets = [_line_target()]
    result = _reconcile_cross_field_contradictions(merged, targets, "Xiaomi Redmi Note 13 4G")
    assert not any(v.attribute_id == LINE_ID for v in result)


def test_o4b_xiaomi_redmi_note_13_line_gen_13_kept():
    """O4b: Xiaomi Redmi Note 13 4G / Линейка = Redmi Note 13 -> KEEP (13 == 13)."""
    merged = [_av(LINE_ID, "Redmi Note 13", 0.9, Source.DESCRIPTION)]
    targets = [_line_target()]
    result = _reconcile_cross_field_contradictions(merged, targets, "Xiaomi Redmi Note 13 4G")
    kept = next((v for v in result if v.attribute_id == LINE_ID), None)
    assert kept is not None
    assert kept.value == "Redmi Note 13"


def test_o5_no_fp_bosch_impact_rate_untouched():
    """O5 (no-FP): Bosch GSB 16 RE / Количество ударов, уд./мин = 47600 -> KEEP
    (target is neither diagonal nor line-generation pattern)."""
    merged = [_av(IMPACT_ID, "47600", 0.9, Source.OZON_CARD)]
    targets = [_target(IMPACT_ID, "Количество ударов, уд./мин")]
    result = _reconcile_cross_field_contradictions(merged, targets, "Bosch GSB 16 RE")
    kept = next((v for v in result if v.attribute_id == IMPACT_ID), None)
    assert kept is not None
    assert kept.value == "47600"


def test_o6_no_fp_makita_min_chuck_diameter_untouched():
    """O6 (no-FP): Makita HP1631 / Мин. диаметр патрона, мм = 1.5 -> KEEP."""
    merged = [_av(CHUCK_MIN_DIAM_ID, "1.5", 0.9, Source.OZON_CARD)]
    targets = [_target(CHUCK_MIN_DIAM_ID, "Мин. диаметр патрона, мм")]
    result = _reconcile_cross_field_contradictions(merged, targets, "Makita HP1631")
    kept = next((v for v in result if v.attribute_id == CHUCK_MIN_DIAM_ID), None)
    assert kept is not None
    assert kept.value == "1.5"


def test_o7_no_fp_lg_merge_echo_model_name_untouched():
    """O7 (no-FP): LG OLED55C4RLA / 'Название модели (для объединения в одну
    карточку)' = OLED55C4RLA -> KEEP (merge-echo target, not a линейка target;
    must not be caught by RULE-G2)."""
    merged = [_av(MODEL_NAME_ID, "OLED55C4RLA", 0.95, Source.DESCRIPTION)]
    targets = [_target(MODEL_NAME_ID, "Название модели (для объединения в одну карточку)")]
    result = _reconcile_cross_field_contradictions(merged, targets, "LG OLED55C4RLA")
    kept = next((v for v in result if v.attribute_id == MODEL_NAME_ID), None)
    assert kept is not None
    assert kept.value == "OLED55C4RLA"


def test_o8_no_fp_poco_resolution_untouched():
    """O8 (no-FP): POCO X6 5G / Разрешение экрана = 2560x1440 -> KEEP (value has
    no variant token relevant to RULE-G1/G2 -- out of FIX-7 scope)."""
    merged = [_av(RESOLUTION_ID, "2560x1440", 0.9, Source.OZON_CARD)]
    targets = [_target(RESOLUTION_ID, "Разрешение экрана")]
    result = _reconcile_cross_field_contradictions(merged, targets, "POCO X6 5G")
    kept = next((v for v in result if v.attribute_id == RESOLUTION_ID), None)
    assert kept is not None
    assert kept.value == "2560x1440"
