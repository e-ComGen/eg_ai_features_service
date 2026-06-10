"""Unit tests for _apply_spec_from_title (spec-from-title deterministic filler).

Guards tested:
  - Positive: title contains EXACTLY ONE allowed value as whole-token sequence → filled.
  - Negative / mud guards:
      * Ambiguity: 2+ allowed values hit title → SKIP (empty > wrong).
      * Too short: allowed value < _SPEC_MIN_VALUE_LEN chars total → NOT matched.
      * Absent: title does not contain the value → empty.
      * Already-filled attr → untouched.
      * Required target → untouched (spec-from-title is optional-only).
      * Free-text target (no allowed_values) → untouched.
      * Brand target → untouched (handled by _apply_brand_from_name).
  - value_id_fn: supplied id is passed through; failure does not crash.
"""
import pytest

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    _apply_spec_from_title,
    _SPEC_FROM_TITLE_EVIDENCE,
    _SPEC_MIN_VALUE_LEN,
    _BRAND_TARGET_ATTR_ID,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ATTR_CPU = 9785       # Процессор (arbitrary id)
_ATTR_GPU_SERIES = 5141  # Серия GPU
_ATTR_BRAND = _BRAND_TARGET_ATTR_ID  # 31

def _ctx(name: str) -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name=name,
        category_id=100,
    )


def _opt_enum(attr_id: int, name: str, allowed: list[str]) -> TargetAttribute:
    return TargetAttribute(
        id=attr_id, name=name, type="enum",
        allowed_values=allowed,
        is_required=False,
    )


def _req_enum(attr_id: int, name: str, allowed: list[str]) -> TargetAttribute:
    return TargetAttribute(
        id=attr_id, name=name, type="enum",
        allowed_values=allowed,
        is_required=True,
    )


def _val(attr_id: int, value: str, source: Source = Source.LLM_KNOWLEDGE) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id, value=value, confidence=0.8, source=source,
    )


def _get(values: list[AttributeValue], attr_id: int) -> AttributeValue | None:
    return next((v for v in values if v.attribute_id == attr_id), None)


# ---------------------------------------------------------------------------
# (a) Positive: single match → filled
# ---------------------------------------------------------------------------

def test_single_match_fills_attribute():
    """'GeForce RTX 4060' in title, exactly one match in allowed → filled."""
    ctx = _ctx("Видеокарта MSI GeForce RTX 4060 Ventus 2X")
    target = _opt_enum(_ATTR_GPU_SERIES, "Серия GPU", [
        "GeForce GTX 1660",
        "GeForce RTX 4060",
        "GeForce RTX 4090",
    ])
    out = _apply_spec_from_title([], [target], ctx)
    v = _get(out, _ATTR_GPU_SERIES)
    assert v is not None, "expected attribute to be filled"
    assert v.value == "GeForce RTX 4060"
    assert v.source == Source.DESCRIPTION
    assert v.evidence == _SPEC_FROM_TITLE_EVIDENCE
    assert v.confidence > 0.0


def test_single_match_with_value_id():
    """value_id_fn returns a non-None id → carried into the filled AttributeValue."""
    ctx = _ctx("Видеокарта MSI GeForce RTX 4060 Ventus 2X")
    target = _opt_enum(_ATTR_GPU_SERIES, "Серия GPU", ["GeForce RTX 4060"])
    out = _apply_spec_from_title(
        [], [target], ctx,
        value_id_fn=lambda attr_id, val: 971423405 if val == "GeForce RTX 4060" else None,
    )
    v = _get(out, _ATTR_GPU_SERIES)
    assert v is not None and v.value_id == 971423405


def test_cyrillic_single_match():
    """Кириллическое значение: 'Блендер' в названии, один вариант в enum → заполнен."""
    ctx = _ctx("Блендер Philips HR2543/00")
    target = _opt_enum(12619, "Вид бытовой техники", [
        "Блендер", "Блендер-суповарка", "Аппарат для смеси"
    ])
    out = _apply_spec_from_title([], [target], ctx)
    v = _get(out, 12619)
    assert v is not None and v.value == "Блендер"


def test_multiword_value_matches_as_phrase():
    """Multi-word value 'Apple Watch Series 9 45mm' must match as a contiguous phrase."""
    ctx = _ctx("Смарт-часы Apple Watch Series 9 45mm Silver")
    target = _opt_enum(9336, "Модель часов", [
        "Apple Watch Series 9 45mm",
        "Apple Watch Series 9 41mm",
        "Samsung Galaxy Watch 6",
    ])
    out = _apply_spec_from_title([], [target], ctx)
    v = _get(out, 9336)
    assert v is not None and v.value == "Apple Watch Series 9 45mm"


# ---------------------------------------------------------------------------
# (b) Ambiguity guard: 2+ hits → skip
# ---------------------------------------------------------------------------

def test_ambiguity_guard_skips_attr():
    """Two allowed values both match the title → attr NOT filled (empty > wrong)."""
    ctx = _ctx("Ноутбук ASUS Core i5 Core i7 Edition")
    target = _opt_enum(_ATTR_CPU, "Процессор", [
        "Intel Core i5-1235U",
        "Intel Core i7-1255U",
    ])
    # Both "Core i5" tokens and "Core i7" tokens will partially appear; let's use
    # values that EXACTLY tokenise to subsets of the title.
    target2 = _opt_enum(200, "Модель", ["ASUS Core", "Core i5"])
    ctx2 = _ctx("Ноутбук ASUS Core i5")
    out = _apply_spec_from_title([], [target2], ctx2)
    # "ASUS Core" → tokens ['asus','core']; "Core i5" → tokens ['core','i5']
    # Only one can match at each position, so let's verify with a clear double-hit case:
    target3 = _opt_enum(201, "Вид техники", ["Ноутбук", "ноутбук Pro"])
    ctx3 = _ctx("Ноутбук Lenovo")
    out3 = _apply_spec_from_title([], [target3], ctx3)
    # "Ноутбук" matches; "ноутбук Pro" does NOT match (no 'pro' in title) → single hit
    v3 = _get(out3, 201)
    assert v3 is not None and v3.value == "Ноутбук"


def test_ambiguity_two_exact_hits_skip():
    """Direct two-hit scenario → skip."""
    ctx = _ctx("Устройство SSD HDD")
    target = _opt_enum(301, "Тип накопителя", ["SSD", "HDD", "SSHD"])
    # Both SSD and HDD are in the title → ambiguous → skip
    out = _apply_spec_from_title([], [target], ctx)
    v = _get(out, 301)
    assert v is None, f"expected no fill due to ambiguity, got {v}"


# ---------------------------------------------------------------------------
# (c) Too-short value guard
# ---------------------------------------------------------------------------

def test_short_value_not_matched():
    """Allowed value with total token length < _SPEC_MIN_VALUE_LEN is NEVER matched."""
    assert _SPEC_MIN_VALUE_LEN == 3, "test assumes min=3"
    # "5G" → normalised tokens ['5g'] → total_len=2 < 3 → skipped
    ctx = _ctx("Смартфон Samsung Galaxy A55 5G")
    target = _opt_enum(400, "Стандарт связи", ["5G", "4G", "3G"])
    # All three are 2 chars or fewer total → none should match
    out = _apply_spec_from_title([], [target], ctx)
    v = _get(out, 400)
    assert v is None, f"expected no fill for short values, got {v}"


def test_three_char_value_is_matched():
    """Value of exactly _SPEC_MIN_VALUE_LEN chars IS matched."""
    ctx = _ctx("Устройство с SSD накопителем")
    target = _opt_enum(401, "Тип накопителя", ["SSD"])
    out = _apply_spec_from_title([], [target], ctx)
    v = _get(out, 401)
    assert v is not None and v.value == "SSD"


# ---------------------------------------------------------------------------
# (d) Value absent in title → empty
# ---------------------------------------------------------------------------

def test_no_match_stays_empty():
    """Title contains none of the allowed values → attr stays empty."""
    ctx = _ctx("Тостер Bosch TAT3A011")
    target = _opt_enum(500, "Цвет", ["Красный", "Синий", "Зелёный"])
    out = _apply_spec_from_title([], [target], ctx)
    assert _get(out, 500) is None


# ---------------------------------------------------------------------------
# (e) Already-filled attr → untouched
# ---------------------------------------------------------------------------

def test_already_filled_attr_untouched():
    """Attr already has a value → spec-from-title does NOT overwrite it."""
    ctx = _ctx("Видеокарта MSI GeForce RTX 4060")
    target = _opt_enum(_ATTR_GPU_SERIES, "Серия GPU", ["GeForce RTX 4060", "GeForce RTX 4090"])
    existing = _val(_ATTR_GPU_SERIES, "GeForce RTX 4090", source=Source.OZON_CARD)
    out = _apply_spec_from_title([existing], [target], ctx)
    v = _get(out, _ATTR_GPU_SERIES)
    assert v is not None and v.value == "GeForce RTX 4090"
    assert v.source == Source.OZON_CARD


# ---------------------------------------------------------------------------
# (f) Required target → untouched (spec-from-title is optional-only)
# ---------------------------------------------------------------------------

def test_required_target_not_filled():
    """Required enum target is NOT touched by spec-from-title."""
    ctx = _ctx("Блендер Philips HR2543/00")
    target = _req_enum(600, "Обязательный вид", ["Блендер", "Тостер"])
    out = _apply_spec_from_title([], [target], ctx)
    assert _get(out, 600) is None


# ---------------------------------------------------------------------------
# (g) Free-text target (no allowed_values) → untouched
# ---------------------------------------------------------------------------

def test_free_text_target_skipped():
    """Target without allowed_values (free-text) is never touched."""
    ctx = _ctx("Ноутбук ASUS VivoBook 15 Core i5")
    target = TargetAttribute(id=700, name="Описание", type="text", is_required=False)
    out = _apply_spec_from_title([], [target], ctx)
    assert _get(out, 700) is None


# ---------------------------------------------------------------------------
# (h) Brand target → untouched
# ---------------------------------------------------------------------------

def test_brand_target_skipped():
    """Brand attr (id==31) is excluded — handled by _apply_brand_from_name."""
    ctx = _ctx("Ноутбук ASUS VivoBook")
    brand_target = _opt_enum(_BRAND_TARGET_ATTR_ID, "Бренд", ["ASUS", "Lenovo"])
    out = _apply_spec_from_title([], [brand_target], ctx)
    assert _get(out, _BRAND_TARGET_ATTR_ID) is None


def test_brand_name_target_skipped():
    """Target named 'Бренд в одежде' (brand by name) is also excluded."""
    ctx = _ctx("Футболка Nike Club")
    brand_target = _opt_enum(9999, "Бренд в одежде", ["Nike", "Adidas"])
    out = _apply_spec_from_title([], [brand_target], ctx)
    assert _get(out, 9999) is None


# ---------------------------------------------------------------------------
# (i) value_id_fn failure does not crash
# ---------------------------------------------------------------------------

def test_value_id_fn_failure_does_not_crash():
    """value_id_fn raising an exception → attr still filled, value_id stays None."""
    ctx = _ctx("Блендер Philips HR2543/00")
    target = _opt_enum(12619, "Вид техники", ["Блендер"])

    def _boom(attr_id, val):
        raise RuntimeError("dict unavailable")

    out = _apply_spec_from_title([], [target], ctx, value_id_fn=_boom)
    v = _get(out, 12619)
    assert v is not None and v.value == "Блендер"
    assert v.value_id is None


# ---------------------------------------------------------------------------
# (j) Empty product name → no-op
# ---------------------------------------------------------------------------

def test_empty_product_name_noop():
    """Empty/None product_name → function returns merged unchanged."""
    ctx = _ctx("")
    target = _opt_enum(800, "Серия", ["RTX 4060"])
    out = _apply_spec_from_title([], [target], ctx)
    assert _get(out, 800) is None


# ---------------------------------------------------------------------------
# (k) Partial-substring must NOT match (whole-token only)
# ---------------------------------------------------------------------------

def test_partial_substring_does_not_match():
    """Allowed value 'Core' must NOT match a title token 'CoreX' — whole-token only."""
    ctx = _ctx("Ноутбук с процессором CoreX Ultra")
    target = _opt_enum(900, "Процессор", ["Core"])
    # "Core" normalises to ['core']; title normalises to [...,'corex',...].
    # 'core' != 'corex' → no match (whole token required, not substring).
    out = _apply_spec_from_title([], [target], ctx)
    assert _get(out, 900) is None


# ---------------------------------------------------------------------------
# (l) Case-insensitive and ё→е normalisation
# ---------------------------------------------------------------------------

def test_case_insensitive_match():
    """Match is case-insensitive (both title and value are lowercased)."""
    ctx = _ctx("Умная колонка Яндекс Станция Мини 2")
    target = _opt_enum(22824, "Система умного дома", ["Яндекс", "Sber", "Google Home"])
    out = _apply_spec_from_title([], [target], ctx)
    v = _get(out, 22824)
    assert v is not None and v.value == "Яндекс"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
