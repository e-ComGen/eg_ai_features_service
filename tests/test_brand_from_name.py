"""Unit tests for the GENERAL post-merge brand-from-name resolver.

Covers _apply_brand_from_name in app/services/enrichment/pipeline.py — the product
NAME is authoritative for brand. If EXACTLY ONE allowed-enum brand appears in the
name as a contiguous token-run (>=3 chars), an empty brand field is filled and a
garbage (wrong allowed-enum) brand is OVERRIDDEN. Zero or ambiguous matches leave
the field untouched (anti-garbage guard). No per-brand hardcode.
"""
import pytest

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    _apply_brand_from_name,
    _is_brand_target_name,
    _BRAND_TARGET_ATTR_ID,
)

_BRAND_ID = _BRAND_TARGET_ATTR_ID  # 31


def _ctx(name: str) -> ExtractionContext:
    return ExtractionContext(product_id=1, product_name=name, category_id=1)


def _brand_target(allowed, *, id=_BRAND_ID, name="Бренд") -> TargetAttribute:
    return TargetAttribute(id=id, name=name, type="enum", allowed_values=allowed)


def _brand_value(value, source=Source.WEB_SEARCH) -> AttributeValue:
    return AttributeValue(
        attribute_id=_BRAND_ID, value=value, confidence=0.85, source=source
    )


def _val_for(values, attr_id=_BRAND_ID):
    return next((v for v in values if v.attribute_id == attr_id), None)


# ── helper-level ──────────────────────────────────────────────────────────────

def test_is_brand_target_name():
    assert _is_brand_target_name("Бренд")
    assert _is_brand_target_name("Бренд в одежде")
    assert _is_brand_target_name("Brand")
    assert _is_brand_target_name("Торговая марка")
    assert not _is_brand_target_name("Цвет")
    assert not _is_brand_target_name("Пол")


# ── (a) unfilled → filled ─────────────────────────────────────────────────────

def test_unfilled_filled_from_name():
    """(a) name contains 'Champion', brand UNFILLED → filled 'Champion'."""
    ctx = _ctx("Толстовка худи Champion Reverse Weave")
    target = _brand_target(["Nike", "Champion", "Adidas", "Reebok"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Champion"
    assert v.source == Source.DESCRIPTION
    assert v.evidence == "brand_from_name"


# ── (b) garbage → overridden ──────────────────────────────────────────────────

def test_garbage_overridden_from_name():
    """(b) name 'Levi's', brand wrongly 'HUGO' → OVERRIDDEN to "Levi's"."""
    ctx = _ctx("Джинсы мужские Levi's 501")
    target = _brand_target(["HUGO", "Levi's", "Wrangler", "Lee"])
    out = _apply_brand_from_name([_brand_value("HUGO")], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Levi's"
    assert v.evidence == "brand_from_name"
    assert v.value_id is None  # re-resolved by strategy downstream


# ── (c) no allowed brand in name → untouched ──────────────────────────────────

def test_no_brand_in_name_untouched_unfilled():
    """(c) no allowed brand present → unfilled stays unfilled (no guess)."""
    ctx = _ctx("Толстовка худи мужская оверсайз чёрная")
    target = _brand_target(["Nike", "Champion", "Adidas"])
    out = _apply_brand_from_name([], [target], ctx)
    assert _val_for(out) is None


def test_no_brand_in_name_keeps_existing():
    """(c') no allowed brand in name → existing value NOT touched."""
    ctx = _ctx("Толстовка худи мужская оверсайз чёрная")
    target = _brand_target(["Nike", "Champion", "HUGO"])
    out = _apply_brand_from_name([_brand_value("HUGO")], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "HUGO"  # untouched, no name evidence


# ── (d) two brands in name → untouched (ambiguous) ────────────────────────────

def test_two_brands_ambiguous_untouched():
    """(d) TWO allowed brands appear in name → left untouched."""
    ctx = _ctx("Кроссовки Nike x Adidas коллаборация")
    target = _brand_target(["Nike", "Adidas", "Puma"])
    out = _apply_brand_from_name([], [target], ctx)
    assert _val_for(out) is None  # ambiguous: not filled


def test_two_brands_ambiguous_existing_untouched():
    ctx = _ctx("Кроссовки Nike x Adidas коллаборация")
    target = _brand_target(["Nike", "Adidas", "HUGO"])
    out = _apply_brand_from_name([_brand_value("HUGO")], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "HUGO"  # ambiguous → not overridden


# ── (e) multi-word brand ──────────────────────────────────────────────────────

def test_multiword_brand_matched():
    """(e) multi-word brand 'The North Face' matched as contiguous tokens."""
    ctx = _ctx("Куртка The North Face Resolve мужская")
    target = _brand_target(["Columbia", "The North Face", "Patagonia"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "The North Face"


def test_multiword_brand_with_punctuation():
    """Punctuation-bearing brand 'be quiet!' matched via tokenization."""
    ctx = _ctx("Блок питания be quiet! Pure Power 12")
    target = _brand_target(["Corsair", "be quiet!", "Seasonic"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "be quiet!"


def test_apostrophe_brand_matched():
    """Apostrophe brand 'Levi's' matched even though name has it as 'Levi's'."""
    ctx = _ctx("Джинсы Levi's классические")
    target = _brand_target(["Levi's", "Lee", "Wrangler"])
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Levi's"


# ── (f) short brand not false-matched ─────────────────────────────────────────

def test_short_two_char_brand_not_false_matched():
    """(f) 2-char brand 'GG' must not false-match on incidental letters."""
    ctx = _ctx("Сумка GG женская кожаная")  # 'GG' present but <3 chars → ignored
    target = _brand_target(["GG", "Gucci", "Prada"])
    out = _apply_brand_from_name([], [target], ctx)
    # 'GG' is too short to be trusted; 'Gucci'/'Prada' absent → nothing filled.
    assert _val_for(out) is None


# ── invariants / scope ────────────────────────────────────────────────────────

def test_never_writes_brand_absent_from_allowed():
    """B is always chosen from allowed_values AND present in name."""
    ctx = _ctx("Джинсы Levi's 501")
    target = _brand_target(["HUGO", "Lee"])  # Levi's NOT in allowed
    out = _apply_brand_from_name([], [target], ctx)
    # Levi's in name but not allowed; HUGO/Lee not in name → nothing.
    assert _val_for(out) is None


def test_free_text_brand_target_skipped():
    """Brand target WITHOUT allowed_values (free-text) is out of scope."""
    ctx = _ctx("Джинсы Levi's 501")
    target = TargetAttribute(id=_BRAND_ID, name="Бренд", type="text")
    out = _apply_brand_from_name([], [target], ctx)
    assert _val_for(out) is None  # free-text not handled


def test_detect_by_id_when_name_differs():
    """Target with id==31 but odd name still detected as brand target."""
    ctx = _ctx("Толстовка Champion Reverse Weave")
    target = _brand_target(["Champion", "Nike"], name="Производитель")
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Champion"


def test_already_correct_brand_left_intact():
    """Filled with the right brand already → kept (no needless override)."""
    ctx = _ctx("Толстовка Champion Reverse Weave")
    target = _brand_target(["Champion", "Nike"])
    existing = _brand_value("Champion", source=Source.OZON_CARD)
    out = _apply_brand_from_name([existing], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Champion"
    assert v.source == Source.OZON_CARD  # untouched, not rewritten


def test_non_brand_attribute_untouched():
    ctx = _ctx("Толстовка Champion Reverse Weave")
    color = TargetAttribute(id=10, name="Цвет", type="enum", allowed_values=["Чёрный"])
    cand = [AttributeValue(attribute_id=10, value="Чёрный", confidence=0.9, source=Source.WEB_SEARCH)]
    out = _apply_brand_from_name(cand, [color], ctx)
    assert len(out) == 1 and out[0].value == "Чёрный"


def test_yo_normalization_in_brand():
    """ё/е equivalence in brand vs name."""
    ctx = _ctx("Конфеты Алёнка молочные")
    target = _brand_target(["Аленка", "Рот Фронт"])  # 'Аленка' with е
    out = _apply_brand_from_name([], [target], ctx)
    v = _val_for(out)
    assert v is not None and v.value == "Аленка"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
