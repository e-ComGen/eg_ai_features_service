"""Unit tests for Lever 2 (service-life verbatim extractor) and Lever 3 (warranty alias cross-fill).

Lever 2 guards tested:
  - Verbatim hit in description fills the attr.
  - Verbatim hit in evidence string (web_search snippet) fills the attr.
  - No 'срок службы/эксплуатации' phrase → no fill.
  - Number WITHOUT 'срок службы' context → no fill.
  - Already-filled target → NOT overwritten.
  - Multiple targets with same name → all filled from the single hit.

Lever 3 guards tested:
  - Compatible alias (Гарантийный срок filled → Гарантия cross-filled).
  - Compatible alias (Гарантия filled → Гарантийный срок cross-filled).
  - Incompatible type ('Гарантия на товар, мес.' Integer + 'Гарантийный срок' String)
    → NO cross-fill (qualifier guard blocks bare 'гарантия' match).
  - Both filled → no cross-fill (ambiguity).
  - Neither filled → no cross-fill.
  - Enum target in alias bucket → skipped (free-text guard).
"""
import pytest

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    _apply_service_life_from_text,
    _apply_warranty_alias_cross_fill,
    _extract_service_life_number,
    _is_service_life_target,
    _warranty_alias_key,
    _SERVICE_LIFE_EVIDENCE_PREFIX,
    _WARRANTY_ALIAS_EVIDENCE,
    _SERVICE_LIFE_CONF,
    _WARRANTY_ALIAS_CONF,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(description: str = "") -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name="Тестовый товар",
        product_description=description or None,
        category_id=100,
    )


def _free_text(attr_id: int, name: str) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type="String", is_required=False)


def _integer_target(attr_id: int, name: str) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type="Integer", is_required=False)


def _enum_target(attr_id: int, name: str, allowed: list[str]) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type="enum", allowed_values=allowed, is_required=False)


def _av(attr_id: int, value: str, source: Source = Source.WEB_SEARCH, evidence: str = "") -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=0.9,
        source=source,
        evidence=evidence or None,
    )


def _get(values: list[AttributeValue], attr_id: int) -> AttributeValue | None:
    return next((v for v in values if v.attribute_id == attr_id), None)


# ---------------------------------------------------------------------------
# Lever 2 unit: _extract_service_life_number
# ---------------------------------------------------------------------------

class TestExtractServiceLifeNumber:

    def test_pattern_a_integer(self):
        """'срок службы 5 лет' → '5'"""
        assert _extract_service_life_number("срок службы 5 лет") == "5"

    def test_pattern_a_with_colon(self):
        """'срок службы: 7 лет' → '7'"""
        assert _extract_service_life_number("срок службы: 7 лет") == "7"

    def test_pattern_a_ekspl(self):
        """'срок эксплуатации 10 лет' → '10'"""
        assert _extract_service_life_number("срок эксплуатации 10 лет") == "10"

    def test_pattern_a_goda(self):
        """'срок службы 3 года' → '3'"""
        assert _extract_service_life_number("срок службы 3 года") == "3"

    def test_pattern_a_case_insensitive(self):
        """Uppercase variant"""
        assert _extract_service_life_number("Срок Службы 5 Лет") == "5"

    def test_pattern_a_decimal(self):
        """'срок службы 1,5 лет' → '1.5'"""
        result = _extract_service_life_number("срок службы 1,5 лет")
        assert result == "1.5"

    def test_pattern_a_in_longer_text(self):
        """Phrase embedded in a sentence"""
        text = "Данный прибор имеет срок службы 5 лет согласно документации."
        assert _extract_service_life_number(text) == "5"

    def test_no_phrase_returns_none(self):
        """Number without срок службы context → None"""
        assert _extract_service_life_number("Гарантия 12 месяцев") is None

    def test_number_without_year_unit_returns_none(self):
        """'срок службы 5' without лет/год/года → None"""
        assert _extract_service_life_number("срок службы 5") is None

    def test_empty_text_returns_none(self):
        assert _extract_service_life_number("") is None

    def test_none_returns_none(self):
        assert _extract_service_life_number(None) is None

    def test_number_only_in_different_context(self):
        """Random number next to non-service-life phrase → None"""
        assert _extract_service_life_number("Мощность 1500 Вт, гарантия 2 года") is None


# ---------------------------------------------------------------------------
# Lever 2 unit: _is_service_life_target
# ---------------------------------------------------------------------------

class TestIsServiceLifeTarget:

    def test_srok_sluzhby(self):
        assert _is_service_life_target("Срок службы, лет") is True

    def test_srok_ekspluatatsii(self):
        assert _is_service_life_target("Срок эксплуатации") is True

    def test_garantiya_not_service_life(self):
        assert _is_service_life_target("Гарантия") is False

    def test_garantiynyy_srok_not_service_life(self):
        assert _is_service_life_target("Гарантийный срок") is False

    def test_case_insensitive(self):
        assert _is_service_life_target("СРОК СЛУЖБЫ") is True


# ---------------------------------------------------------------------------
# Lever 2 integration: _apply_service_life_from_text
# ---------------------------------------------------------------------------

class TestApplyServiceLifeFromText:

    _SL_ATTR_ID = 6036

    def test_fills_from_description(self):
        """Phrase in product_description → fills the attr."""
        ctx = _ctx("Срок службы 5 лет при правильной эксплуатации.")
        target = _free_text(self._SL_ATTR_ID, "Срок службы, лет")
        out = _apply_service_life_from_text([], [target], ctx)
        v = _get(out, self._SL_ATTR_ID)
        assert v is not None
        assert v.value == "5"
        assert v.source == Source.DESCRIPTION
        assert v.evidence is not None
        assert v.evidence.startswith(_SERVICE_LIFE_EVIDENCE_PREFIX)
        assert v.confidence == _SERVICE_LIFE_CONF

    def test_fills_from_evidence_string(self):
        """Phrase in an existing AV's evidence string → fills the attr."""
        ctx = _ctx()  # no description
        existing_av = _av(
            attr_id=999,
            value="Хороший товар",
            source=Source.WEB_SEARCH,
            evidence="Производитель заявляет срок службы 7 лет при правильном хранении",
        )
        target = _free_text(self._SL_ATTR_ID, "Срок службы, лет")
        out = _apply_service_life_from_text([existing_av], [target], ctx)
        v = _get(out, self._SL_ATTR_ID)
        assert v is not None
        assert v.value == "7"
        assert v.source == Source.DESCRIPTION

    def test_no_phrase_no_fill(self):
        """No 'срок службы' phrase anywhere → no fill."""
        ctx = _ctx("Отличный товар, гарантия 1 год.")
        target = _free_text(self._SL_ATTR_ID, "Срок службы, лет")
        out = _apply_service_life_from_text([], [target], ctx)
        assert _get(out, self._SL_ATTR_ID) is None

    def test_number_without_phrase_no_fill(self):
        """Number '5' in description but without срок службы phrase → no fill."""
        ctx = _ctx("5 скоростей, гарантия 2 года.")
        target = _free_text(self._SL_ATTR_ID, "Срок службы, лет")
        out = _apply_service_life_from_text([], [target], ctx)
        assert _get(out, self._SL_ATTR_ID) is None

    def test_already_filled_not_overwritten(self):
        """Already-filled attr → NOT overwritten even if phrase found in description."""
        ctx = _ctx("Срок службы 5 лет.")
        existing = _av(self._SL_ATTR_ID, "3", source=Source.OZON_CARD)
        target = _free_text(self._SL_ATTR_ID, "Срок службы, лет")
        out = _apply_service_life_from_text([existing], [target], ctx)
        v = _get(out, self._SL_ATTR_ID)
        # existing value survives (no overwrite), new fill not added
        assert v is not None and v.value == "3"
        # Check no duplicate
        assert sum(1 for x in out if x.attribute_id == self._SL_ATTR_ID) == 1

    def test_description_priority_over_evidence(self):
        """Description hit takes priority over evidence string hit."""
        ctx = _ctx("Срок службы 5 лет.")
        existing_av = _av(
            999, "other",
            evidence="срок эксплуатации 10 лет",
        )
        target = _free_text(self._SL_ATTR_ID, "Срок службы, лет")
        out = _apply_service_life_from_text([existing_av], [target], ctx)
        v = _get(out, self._SL_ATTR_ID)
        assert v is not None and v.value == "5"  # description wins

    def test_multiple_sl_targets_all_filled(self):
        """Two service-life targets both empty → both filled from the same hit."""
        ctx = _ctx("Срок службы 5 лет.")
        t1 = _free_text(6036, "Срок службы, лет")
        t2 = _free_text(8345, "Срок службы, лет")
        out = _apply_service_life_from_text([], [t1, t2], ctx)
        assert _get(out, 6036) is not None and _get(out, 6036).value == "5"
        assert _get(out, 8345) is not None and _get(out, 8345).value == "5"

    def test_non_service_life_target_not_filled(self):
        """Target whose name doesn't match → not filled."""
        ctx = _ctx("Срок службы 5 лет.")
        target = _free_text(10400, "Гарантия")
        out = _apply_service_life_from_text([], [target], ctx)
        assert _get(out, 10400) is None


# ---------------------------------------------------------------------------
# Lever 3 unit: _warranty_alias_key
# ---------------------------------------------------------------------------

class TestWarrantyAliasKey:

    def test_garantiynyy_srok(self):
        group = _warranty_alias_key("Гарантийный срок")
        assert group is not None
        assert "гарантийный срок" in group

    def test_garantiya_bare(self):
        group = _warranty_alias_key("Гарантия")
        assert group is not None

    def test_garantiya_with_qualifier_blocked(self):
        """'Гарантия на товар, мес.' has qualifier 'товар' → NOT matched."""
        group = _warranty_alias_key("Гарантия на товар, мес.")
        assert group is None

    def test_garantiya_na_vnutrenniy_bak(self):
        """'Гарантия на внутренний бак, мес.' → NOT matched."""
        group = _warranty_alias_key("Гарантия на внутренний бак, мес.")
        assert group is None

    def test_unrelated_name(self):
        assert _warranty_alias_key("Срок службы, лет") is None

    def test_brand(self):
        assert _warranty_alias_key("Бренд") is None


# ---------------------------------------------------------------------------
# Lever 3 integration: _apply_warranty_alias_cross_fill
# ---------------------------------------------------------------------------

class TestApplyWarrantyAliasCrossFill:

    _GARANTIA_ID = 10400       # "Гарантия" — free-text String
    _GARANTIYNYY_ID = 4385     # "Гарантийный срок" — free-text String
    _GARANTIYA_MES_ID = 4164   # "Гарантия на товар, мес." — Integer (NOT an alias)

    def _targets(self, garantia=True, garantiynyy=True, mes=False):
        out = []
        if garantia:
            out.append(_free_text(self._GARANTIA_ID, "Гарантия"))
        if garantiynyy:
            out.append(_free_text(self._GARANTIYNYY_ID, "Гарантийный срок"))
        if mes:
            out.append(_integer_target(self._GARANTIYA_MES_ID, "Гарантия на товар, мес."))
        return out

    def test_garantiynyy_filled_garantia_empty(self):
        """'Гарантийный срок'=filled → 'Гарантия' cross-filled."""
        targets = self._targets()
        filled = _av(self._GARANTIYNYY_ID, "12 месяцев", source=Source.DESCRIPTION)
        out = _apply_warranty_alias_cross_fill([filled], targets)
        v = _get(out, self._GARANTIA_ID)
        assert v is not None
        assert v.value == "12 месяцев"
        assert v.source == Source.DESCRIPTION
        assert v.evidence == _WARRANTY_ALIAS_EVIDENCE
        assert v.confidence == _WARRANTY_ALIAS_CONF

    def test_garantia_filled_garantiynyy_empty(self):
        """'Гарантия'=filled → 'Гарантийный срок' cross-filled."""
        targets = self._targets()
        filled = _av(self._GARANTIA_ID, "1 год", source=Source.WEB_SEARCH)
        out = _apply_warranty_alias_cross_fill([filled], targets)
        v = _get(out, self._GARANTIYNYY_ID)
        assert v is not None
        assert v.value == "1 год"

    def test_both_filled_no_cross_fill(self):
        """Both aliases filled → no new fills (ambiguity guard)."""
        targets = self._targets()
        av1 = _av(self._GARANTIA_ID, "12 месяцев")
        av2 = _av(self._GARANTIYNYY_ID, "24 месяца")
        out = _apply_warranty_alias_cross_fill([av1, av2], targets)
        # No new entries added (already 2 items)
        assert len(out) == 2

    def test_neither_filled_no_cross_fill(self):
        """Neither alias filled → no action."""
        targets = self._targets()
        out = _apply_warranty_alias_cross_fill([], targets)
        assert len(out) == 0

    def test_incompatible_type_not_cross_filled(self):
        """'Гарантия на товар, мес.' (Integer, with qualifier) NOT cross-filled with 'Гарантийный срок'."""
        # Both targets: Гарантийный срок (free-text) + Гарантия на товар, мес. (Integer)
        targets = [
            _free_text(self._GARANTIYNYY_ID, "Гарантийный срок"),
            _integer_target(self._GARANTIYA_MES_ID, "Гарантия на товар, мес."),
        ]
        filled = _av(self._GARANTIYNYY_ID, "12 месяцев")
        out = _apply_warranty_alias_cross_fill([filled], targets)
        # "Гарантия на товар, мес." should NOT be filled (qualifier guard blocks alias key)
        v_mes = _get(out, self._GARANTIYA_MES_ID)
        assert v_mes is None

    def test_enum_alias_target_skipped(self):
        """If one alias target is an enum (has allowed_values) → free-text guard skips it."""
        enum_garantia = _enum_target(self._GARANTIA_ID, "Гарантия", ["Да", "Нет"])
        free_garantiynyy = _free_text(self._GARANTIYNYY_ID, "Гарантийный срок")
        targets = [enum_garantia, free_garantiynyy]
        filled = _av(self._GARANTIYNYY_ID, "12 месяцев")
        out = _apply_warranty_alias_cross_fill([filled], targets)
        # enum Гарантия should NOT be filled (free-text guard)
        assert _get(out, self._GARANTIA_ID) is None

    def test_original_fills_preserved(self):
        """Existing AVs in merged are preserved unchanged."""
        targets = self._targets()
        filled = _av(self._GARANTIYNYY_ID, "12 месяцев")
        out = _apply_warranty_alias_cross_fill([filled], targets)
        # Original filled av still present
        orig = _get(out, self._GARANTIYNYY_ID)
        assert orig is not None and orig.value == "12 месяцев"

    def test_no_targets_no_op(self):
        """No alias targets → function is a no-op."""
        targets = [_free_text(999, "Срок службы, лет")]
        filled = _av(999, "5")
        out = _apply_warranty_alias_cross_fill([filled], targets)
        assert out == [filled]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
