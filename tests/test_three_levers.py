"""Unit tests for the three new pipeline levers.

Lever 1: "Название" — fill product-title attr verbatim from input product name.
Lever 2: General numeric-spec extractor (generalization of service-life extractor).
Lever 3a: Boolean Да/Нет — stated-only verbatim fill.
Lever 3b: "Комплектация" — verbatim list from product description.

Each lever has POSITIVE tests (stated → filled) and NEGATIVE mud-guards
(no mention → NOT filled; wrong-unit number → NOT filled; cross-attribute
number bleed → NOT filled; default-Да never happens).
"""
import pytest

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    # Lever 1
    _apply_title_from_input,
    _is_title_target,
    _TITLE_FROM_INPUT_CONF,
    _TITLE_FROM_INPUT_EVIDENCE,
    # Lever 2
    _apply_numeric_spec_from_text,
    _apply_service_life_from_text,   # backward-compat wrapper
    _attr_keyword,
    _build_numspec_regex,
    _extract_numspec_number,
    _is_service_life_target,
    _extract_service_life_number,
    _SERVICE_LIFE_EVIDENCE_PREFIX,
    _NUMSPEC_CONF,
    # Lever 3
    _apply_boolean_stated_from_text,
    _apply_komplektatsiya_from_text,
    _is_bool_target,
    _is_komplektatsiya_target,
    _bool_stated_keywords,
    _BOOL_STATED_CONF,
    _BOOL_STATED_EVIDENCE_PREFIX,
    _KOMPLEKTATSIYA_CONF,
    _KOMPLEKTATSIYA_EVIDENCE,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ctx(description: str = "", product_name: str = "Тестовый товар") -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name=product_name,
        product_description=description or None,
        category_id=100,
    )


def _free_text(attr_id: int, name: str) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type="String", is_required=False)


def _numeric(attr_id: int, name: str) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type="Integer", is_required=False)


def _bool_attr(attr_id: int, name: str) -> TargetAttribute:
    return TargetAttribute(
        id=attr_id, name=name, type="enum",
        allowed_values=["Да", "Нет"], is_required=False,
    )


def _enum_attr(attr_id: int, name: str, values: list[str]) -> TargetAttribute:
    return TargetAttribute(
        id=attr_id, name=name, type="enum",
        allowed_values=values, is_required=False,
    )


def _av(attr_id: int, value: str, source: Source = Source.WEB_SEARCH,
        evidence: str = "") -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id, value=value, confidence=0.9,
        source=source, evidence=evidence or None,
    )


def _get(values: list[AttributeValue], attr_id: int) -> AttributeValue | None:
    return next((v for v in values if v.attribute_id == attr_id), None)


# ===========================================================================
# LEVER 1: "Название" — fill from input product name
# ===========================================================================

class TestIsTitleTarget:
    """Unit tests for _is_title_target predicate."""

    def test_nazvanie_exact(self):
        assert _is_title_target("Название") is True

    def test_nazvanie_lower(self):
        assert _is_title_target("название") is True

    def test_naimenovanie(self):
        assert _is_title_target("Наименование") is True

    def test_naimenovanie_tovara(self):
        assert _is_title_target("Наименование товара") is True

    def test_nazvanie_tovara(self):
        assert _is_title_target("Название товара") is True

    def test_polnoe_nazvanie(self):
        assert _is_title_target("Полное название") is True

    def test_polnoe_naimenovanie(self):
        assert _is_title_target("Полное наименование") is True

    # Negative: compound names that should NOT match
    def test_nazvanie_tsveta_rejected(self):
        """'Название цвета' must NOT match — different attr."""
        assert _is_title_target("Название цвета") is False

    def test_nazvanie_brenda_rejected(self):
        assert _is_title_target("Название бренда") is False

    def test_nazvanie_modeli_rejected(self):
        assert _is_title_target("Название модели") is False

    def test_nazvanie_serii_rejected(self):
        assert _is_title_target("Название серии") is False

    def test_unrelated_attr_rejected(self):
        assert _is_title_target("Цвет") is False

    def test_brand_attr_rejected(self):
        assert _is_title_target("Бренд") is False


class TestApplyTitleFromInput:
    """Integration tests for _apply_title_from_input."""

    _TITLE_ATTR_ID = 9000

    def test_fills_from_product_name(self):
        """Empty 'Название' attr → filled verbatim with product_name."""
        ctx = _ctx(product_name="Смартфон Samsung Galaxy S24 Ultra 256GB")
        target = _free_text(self._TITLE_ATTR_ID, "Название")
        out = _apply_title_from_input([], [target], ctx)
        v = _get(out, self._TITLE_ATTR_ID)
        assert v is not None
        assert v.value == "Смартфон Samsung Galaxy S24 Ultra 256GB"
        assert v.source == Source.DESCRIPTION
        assert v.evidence == _TITLE_FROM_INPUT_EVIDENCE
        assert v.confidence == _TITLE_FROM_INPUT_CONF

    def test_fills_naimenovanie_tovara(self):
        ctx = _ctx(product_name="Кроссовки Nike Air Max")
        target = _free_text(self._TITLE_ATTR_ID, "Наименование товара")
        out = _apply_title_from_input([], [target], ctx)
        v = _get(out, self._TITLE_ATTR_ID)
        assert v is not None
        assert v.value == "Кроссовки Nike Air Max"

    def test_never_overwrites_already_filled(self):
        """Already-filled 'Название' → NOT overwritten."""
        ctx = _ctx(product_name="Новое название")
        existing = _av(self._TITLE_ATTR_ID, "Старое название", source=Source.OZON_CARD)
        target = _free_text(self._TITLE_ATTR_ID, "Название")
        out = _apply_title_from_input([existing], [target], ctx)
        v = _get(out, self._TITLE_ATTR_ID)
        assert v is not None and v.value == "Старое название"
        assert sum(1 for x in out if x.attribute_id == self._TITLE_ATTR_ID) == 1

    def test_empty_product_name_no_fill(self):
        """Empty product_name → no fill (guard)."""
        ctx = _ctx(product_name="")
        target = _free_text(self._TITLE_ATTR_ID, "Название")
        out = _apply_title_from_input([], [target], ctx)
        assert _get(out, self._TITLE_ATTR_ID) is None

    def test_nazvanie_tsveta_not_filled(self):
        """'Название цвета' is NOT the title target — must not be filled."""
        ctx = _ctx(product_name="Синий")
        target = _free_text(9001, "Название цвета")
        out = _apply_title_from_input([], [target], ctx)
        assert _get(out, 9001) is None

    def test_no_title_target_noop(self):
        """No title targets in list → no fills."""
        ctx = _ctx(product_name="Товар 123")
        target = _free_text(9002, "Бренд")
        out = _apply_title_from_input([], [target], ctx)
        assert _get(out, 9002) is None

    def test_multiple_title_targets_all_filled(self):
        """Multiple title targets → all filled from the same product_name."""
        ctx = _ctx(product_name="Планшет Lenovo Tab 12")
        t1 = _free_text(9010, "Название")
        t2 = _free_text(9011, "Наименование товара")
        out = _apply_title_from_input([], [t1, t2], ctx)
        assert _get(out, 9010).value == "Планшет Lenovo Tab 12"
        assert _get(out, 9011).value == "Планшет Lenovo Tab 12"


# ===========================================================================
# LEVER 2: General numeric-spec extractor
# ===========================================================================

class TestAttrKeyword:
    """Unit tests for keyword derivation."""

    def test_strips_unit_suffix_chas(self):
        assert _attr_keyword("Время автономной работы, ч") == "время автономной работы"

    def test_strips_unit_suffix_min(self):
        assert _attr_keyword("Время зарядки, мин") == "время зарядки"

    def test_strips_unit_suffix_let(self):
        assert _attr_keyword("Срок службы, лет") == "срок службы"

    def test_strips_unit_suffix_vt(self):
        assert _attr_keyword("Мощность, Вт") == "мощность"

    def test_no_suffix(self):
        assert _attr_keyword("Частота обновления") == "частота обновления"

    def test_empty(self):
        assert _attr_keyword("") == ""


class TestBuildNumspecRegex:
    """Unit tests for regex builder."""

    def test_builds_for_min_unit(self):
        pat = _build_numspec_regex("время зарядки", "min")
        assert pat is not None

    def test_builds_for_w_unit(self):
        pat = _build_numspec_regex("мощность", "w")
        assert pat is not None

    def test_returns_none_for_unknown_unit(self):
        pat = _build_numspec_regex("нечто неизвестное", "xyz_unknown")
        assert pat is None

    def test_returns_none_for_empty_keyword(self):
        pat = _build_numspec_regex("", "min")
        assert pat is None


class TestExtractNumspecNumber:
    """Unit tests for the general number extractor."""

    def test_forward_match(self):
        """keyword ... N unit → found."""
        import re
        pat = _build_numspec_regex("время зарядки", "min")
        result = _extract_numspec_number("Время зарядки: 90 мин при стандартной зарядке.", pat)
        assert result is not None
        num, _ = result
        assert num == "90"

    def test_no_keyword_no_match(self):
        """Number present but keyword absent → None."""
        import re
        pat = _build_numspec_regex("время зарядки", "min")
        result = _extract_numspec_number("Время разговора: 45 мин", pat)
        assert result is None

    def test_wrong_unit_no_match(self):
        """Keyword present but unit is wrong (hours instead of minutes) → None."""
        pat = _build_numspec_regex("время зарядки", "min")
        # "2 часа" — no "мин" → should not match min pattern
        result = _extract_numspec_number("Время зарядки: 2 часа", pat)
        assert result is None


class TestCrossAttributeBleedGuard:
    """Critical: numbers for one attr must NOT bleed into another attr."""

    _ZARYAD_ID = 4001   # Время зарядки, мин
    _RAZG_ID = 4002     # Время разговора, мин
    _AVTONOM_ID = 4003  # Время автономной работы, ч

    def test_charge_time_does_not_fill_talk_time(self):
        """'зарядка 2 часа' MUST NOT fill 'время разговора'."""
        # Use units that would match but keyword won't
        ctx = _ctx("Время зарядки: 90 мин. Телефон удобен в использовании.")
        target_talk = _numeric(self._RAZG_ID, "Время разговора, мин")
        out = _apply_numeric_spec_from_text([], [target_talk], ctx)
        # "Время разговора" keyword NOT present near any number with 'мин' here
        # (description says "Время зарядки: 90 мин" — wrong keyword for talk time)
        assert _get(out, self._RAZG_ID) is None

    def test_talk_time_does_not_fill_charge_time(self):
        """'время разговора 45 мин' MUST NOT fill 'время зарядки'."""
        ctx = _ctx("Время разговора: 45 мин в режиме 4G.")
        target_charge = _numeric(self._ZARYAD_ID, "Время зарядки, мин")
        out = _apply_numeric_spec_from_text([], [target_charge], ctx)
        assert _get(out, self._ZARYAD_ID) is None

    def test_correct_attr_fills_from_its_own_keyword(self):
        """Correct keyword present near number → fills the right attr."""
        ctx = _ctx("Время зарядки: 90 мин при стандартном зарядном устройстве.")
        target_charge = _numeric(self._ZARYAD_ID, "Время зарядки, мин")
        out = _apply_numeric_spec_from_text([], [target_charge], ctx)
        v = _get(out, self._ZARYAD_ID)
        assert v is not None
        assert v.value == "90"
        assert v.source == Source.DESCRIPTION
        assert v.confidence == _NUMSPEC_CONF

    def test_both_attrs_from_separate_sentences(self):
        """Text has both keywords → each fills only its own attr."""
        ctx = _ctx(
            "Время зарядки: 90 мин. Время разговора составляет 600 мин в 4G."
        )
        target_charge = _numeric(self._ZARYAD_ID, "Время зарядки, мин")
        target_talk = _numeric(self._RAZG_ID, "Время разговора, мин")
        out = _apply_numeric_spec_from_text([], [target_charge, target_talk], ctx)
        v_charge = _get(out, self._ZARYAD_ID)
        v_talk = _get(out, self._RAZG_ID)
        assert v_charge is not None and v_charge.value == "90"
        assert v_talk is not None and v_talk.value == "600"


class TestNumericSpecWrongUnit:
    """Wrong unit → no fill (cross-unit bleed guard)."""

    def test_wrong_unit_no_fill(self):
        """Attr expects 'мин' but text has 'Вт' → no fill."""
        ctx = _ctx("Время зарядки: 65 Вт поддерживается.")
        target = _numeric(4001, "Время зарядки, мин")
        out = _apply_numeric_spec_from_text([], [target], ctx)
        assert _get(out, 4001) is None

    def test_number_only_no_unit_no_fill(self):
        """Number present but no unit token near keyword → no fill."""
        ctx = _ctx("Время зарядки 90")  # no unit word → pattern doesn't match
        target = _numeric(4001, "Время зарядки, мин")
        out = _apply_numeric_spec_from_text([], [target], ctx)
        assert _get(out, 4001) is None


class TestServiceLifeBackwardCompat:
    """Service-life backward-compat: _apply_service_life_from_text still works."""

    _SL_ID = 6036

    def test_service_life_fills_from_description(self):
        ctx = _ctx("Срок службы 5 лет при правильной эксплуатации.")
        target = _free_text(self._SL_ID, "Срок службы, лет")
        # Both wrappers should work
        out1 = _apply_service_life_from_text([], [target], ctx)
        out2 = _apply_numeric_spec_from_text([], [target], ctx)
        assert _get(out1, self._SL_ID).value == "5"
        assert _get(out2, self._SL_ID).value == "5"

    def test_service_life_no_phrase_no_fill(self):
        ctx = _ctx("Гарантия 12 месяцев. Отличное качество.")
        target = _free_text(self._SL_ID, "Срок службы, лет")
        out = _apply_numeric_spec_from_text([], [target], ctx)
        assert _get(out, self._SL_ID) is None

    def test_service_life_number_without_year_unit_no_fill(self):
        """'срок службы 5' without year unit → no fill."""
        ctx = _ctx("срок службы 5")
        target = _free_text(self._SL_ID, "Срок службы, лет")
        out = _apply_numeric_spec_from_text([], [target], ctx)
        assert _get(out, self._SL_ID) is None

    def test_service_life_already_filled_not_overwritten(self):
        ctx = _ctx("Срок службы 5 лет.")
        existing = _av(self._SL_ID, "3", source=Source.OZON_CARD)
        target = _free_text(self._SL_ID, "Срок службы, лет")
        out = _apply_numeric_spec_from_text([existing], [target], ctx)
        v = _get(out, self._SL_ID)
        assert v is not None and v.value == "3"
        assert sum(1 for x in out if x.attribute_id == self._SL_ID) == 1


class TestNumspecFromEvidenceString:
    """Number found in evidence string (not description) → fills attr."""

    def test_fills_from_evidence_string(self):
        ctx = _ctx()  # no description
        existing_av = _av(
            999, "some value",
            source=Source.WEB_SEARCH,
            evidence="Время зарядки: 45 мин с адаптером 20Вт",
        )
        target = _numeric(4001, "Время зарядки, мин")
        out = _apply_numeric_spec_from_text([existing_av], [target], ctx)
        v = _get(out, 4001)
        assert v is not None
        assert v.value == "45"


# ===========================================================================
# LEVER 3a: Boolean Да/Нет — stated-only
# ===========================================================================

class TestIsBoolTarget:
    def test_da_net_is_bool(self):
        t = _bool_attr(1, "Наличие серийного номера")
        assert _is_bool_target(t) is True

    def test_no_allowed_values_not_bool(self):
        t = _free_text(1, "Наличие серийного номера")
        assert _is_bool_target(t) is False

    def test_non_bool_enum_not_bool(self):
        t = _enum_attr(1, "Цвет", ["Красный", "Синий"])
        assert _is_bool_target(t) is False


class TestBoolStatedFromText:
    """Integration tests for _apply_boolean_stated_from_text."""

    _SERIAL_ID = 5001
    _SMART_ID = 5002
    _NFC_ID = 5003
    _BT_ID = 5004

    def test_serial_number_stated_fills_da(self):
        """Text explicitly mentions 'серийный номер' → fills 'Да'."""
        ctx = _ctx("Каждое устройство имеет уникальный серийный номер для отслеживания.")
        target = _bool_attr(self._SERIAL_ID, "Наличие серийного номера")
        out = _apply_boolean_stated_from_text([], [target], ctx)
        v = _get(out, self._SERIAL_ID)
        assert v is not None
        assert v.value == "Да"
        assert v.source == Source.DESCRIPTION
        assert v.confidence == _BOOL_STATED_CONF
        assert _BOOL_STATED_EVIDENCE_PREFIX in v.evidence

    def test_no_serial_mention_no_fill(self):
        """No mention of serial number → NOT filled (not even 'Нет')."""
        ctx = _ctx("Отличный товар. Мощность 15 Вт.")
        target = _bool_attr(self._SERIAL_ID, "Наличие серийного номера")
        out = _apply_boolean_stated_from_text([], [target], ctx)
        assert _get(out, self._SERIAL_ID) is None

    def test_smartphone_control_stated_fills_da(self):
        """Text mentions 'управление через приложение' → fills 'Да'."""
        ctx = _ctx("Управление через приложение на вашем смартфоне Android/iOS.")
        target = _bool_attr(self._SMART_ID, "Управление со смартфона")
        out = _apply_boolean_stated_from_text([], [target], ctx)
        v = _get(out, self._SMART_ID)
        assert v is not None
        assert v.value == "Да"

    def test_smartphone_control_via_app_stated(self):
        """'мобильное приложение' → fills 'Да' for smartphone control."""
        ctx = _ctx("Полный контроль через мобильное приложение eHome Pro.")
        target = _bool_attr(self._SMART_ID, "Управление через приложение")
        out = _apply_boolean_stated_from_text([], [target], ctx)
        v = _get(out, self._SMART_ID)
        assert v is not None
        assert v.value == "Да"

    def test_no_app_mention_no_fill(self):
        """No app/smartphone mention → NOT filled."""
        ctx = _ctx("Ручное управление. Мощность 1500 Вт.")
        target = _bool_attr(self._SMART_ID, "Управление со смартфона")
        out = _apply_boolean_stated_from_text([], [target], ctx)
        assert _get(out, self._SMART_ID) is None

    def test_nfc_stated_fills_da(self):
        """NFC mentioned → fills 'Да'."""
        ctx = _ctx("Поддержка NFC для бесконтактных платежей.")
        target = _bool_attr(self._NFC_ID, "NFC")
        out = _apply_boolean_stated_from_text([], [target], ctx)
        v = _get(out, self._NFC_ID)
        assert v is not None and v.value == "Да"

    def test_nfc_not_mentioned_no_fill(self):
        """NFC not mentioned → NOT filled."""
        ctx = _ctx("Bluetooth 5.0. Wi-Fi 802.11ac.")
        target = _bool_attr(self._NFC_ID, "NFC")
        out = _apply_boolean_stated_from_text([], [target], ctx)
        assert _get(out, self._NFC_ID) is None

    def test_never_overwrites_existing_value(self):
        """Already-filled boolean → NOT overwritten."""
        ctx = _ctx("Поддержка NFC для платежей.")
        existing = _av(self._NFC_ID, "Нет", source=Source.OZON_CARD)
        target = _bool_attr(self._NFC_ID, "NFC")
        out = _apply_boolean_stated_from_text([existing], [target], ctx)
        v = _get(out, self._NFC_ID)
        assert v is not None and v.value == "Нет"
        assert sum(1 for x in out if x.attribute_id == self._NFC_ID) == 1

    def test_no_default_da_from_category(self):
        """Absence of evidence → never defaults to 'Да'."""
        ctx = _ctx("Просто хороший товар.")
        targets = [
            _bool_attr(5010, "Наличие серийного номера"),
            _bool_attr(5011, "Управление со смартфона"),
            _bool_attr(5012, "NFC"),
            _bool_attr(5013, "Bluetooth"),
            _bool_attr(5014, "Wi-Fi"),
        ]
        out = _apply_boolean_stated_from_text([], targets, ctx)
        for t in targets:
            assert _get(out, t.id) is None, f"Should not fill {t.name} by default"

    def test_bluetooth_fills_from_evidence(self):
        """Bluetooth mentioned in evidence string → fills 'Да'."""
        ctx = _ctx()
        existing_av = _av(
            999, "something",
            source=Source.WEB_SEARCH,
            evidence="Оснащён модулем Bluetooth 5.2 для беспроводного подключения",
        )
        target = _bool_attr(self._BT_ID, "Bluetooth")
        out = _apply_boolean_stated_from_text([existing_av], [target], ctx)
        v = _get(out, self._BT_ID)
        assert v is not None and v.value == "Да"

    def test_non_boolean_enum_not_touched(self):
        """Non-boolean enum (e.g. colour list) → not touched."""
        ctx = _ctx("Цвет: красный.")
        target = _enum_attr(5020, "Цвет", ["Красный", "Синий", "Зелёный"])
        out = _apply_boolean_stated_from_text([], [target], ctx)
        assert _get(out, 5020) is None


# ===========================================================================
# LEVER 3b: "Комплектация" — verbatim list from description
# ===========================================================================

class TestIsKomplektatsiyaTarget:
    def test_komplektatsiya(self):
        assert _is_komplektatsiya_target("Комплектация") is True

    def test_komplektatsiya_lower(self):
        assert _is_komplektatsiya_target("комплектация") is True

    def test_v_komplekte(self):
        assert _is_komplektatsiya_target("В комплекте") is True

    def test_unrelated(self):
        assert _is_komplektatsiya_target("Мощность") is False

    def test_brand(self):
        assert _is_komplektatsiya_target("Бренд") is False


class TestApplyKomplektatsiyaFromText:
    """Integration tests for _apply_komplektatsiya_from_text."""

    _KOMP_ID = 6001

    def test_fills_from_komplektatsiya_section(self):
        """'комплектация: …' found in description → fills verbatim."""
        ctx = _ctx("Характеристики: мощность 1500 Вт.\nКомплектация: основной блок, инструкция, кабель питания.")
        target = _free_text(self._KOMP_ID, "Комплектация")
        out = _apply_komplektatsiya_from_text([], [target], ctx)
        v = _get(out, self._KOMP_ID)
        assert v is not None
        assert "основной блок" in v.value
        assert v.source == Source.DESCRIPTION
        assert v.evidence == _KOMPLEKTATSIYA_EVIDENCE
        assert v.confidence == _KOMPLEKTATSIYA_CONF

    def test_fills_from_v_komplekte_section(self):
        """'в комплекте: …' found → fills verbatim."""
        ctx = _ctx("Описание товара.\nВ комплекте: наушники, зарядное устройство, чехол.")
        target = _free_text(self._KOMP_ID, "Комплектация")
        out = _apply_komplektatsiya_from_text([], [target], ctx)
        v = _get(out, self._KOMP_ID)
        assert v is not None
        assert "наушники" in v.value
        assert "зарядное устройство" in v.value

    def test_no_section_no_fill(self):
        """Description doesn't have комплектация section → no fill."""
        ctx = _ctx("Мощный пылесос 2000 Вт. Гарантия 1 год.")
        target = _free_text(self._KOMP_ID, "Комплектация")
        out = _apply_komplektatsiya_from_text([], [target], ctx)
        assert _get(out, self._KOMP_ID) is None

    def test_no_description_no_fill(self):
        """No product description → no fill."""
        ctx = _ctx("")
        target = _free_text(self._KOMP_ID, "Комплектация")
        out = _apply_komplektatsiya_from_text([], [target], ctx)
        assert _get(out, self._KOMP_ID) is None

    def test_never_overwrites_existing(self):
        """Already-filled Комплектация → NOT overwritten."""
        ctx = _ctx("Комплектация: провод, вилка.")
        existing = _av(self._KOMP_ID, "Только блок", source=Source.OZON_CARD)
        target = _free_text(self._KOMP_ID, "Комплектация")
        out = _apply_komplektatsiya_from_text([existing], [target], ctx)
        v = _get(out, self._KOMP_ID)
        assert v is not None and v.value == "Только блок"
        assert sum(1 for x in out if x.attribute_id == self._KOMP_ID) == 1

    def test_enum_komplektatsiya_not_filled(self):
        """Комплектация with allowed_values (enum) → not filled by this lever."""
        ctx = _ctx("Комплектация: провод, вилка.")
        target = _enum_attr(self._KOMP_ID, "Комплектация", ["Стандарт", "Расширенная"])
        out = _apply_komplektatsiya_from_text([], [target], ctx)
        assert _get(out, self._KOMP_ID) is None

    def test_case_insensitive_match(self):
        """'КОМПЛЕКТАЦИЯ' (caps) → still matches."""
        ctx = _ctx("КОМПЛЕКТАЦИЯ: кабель, блок питания.")
        target = _free_text(self._KOMP_ID, "Комплектация")
        out = _apply_komplektatsiya_from_text([], [target], ctx)
        v = _get(out, self._KOMP_ID)
        assert v is not None and "кабель" in v.value

    def test_unrelated_target_not_filled(self):
        """Target that is not 'Комплектация' → not filled."""
        ctx = _ctx("Комплектация: кабель, инструкция.")
        target = _free_text(6002, "Мощность")
        out = _apply_komplektatsiya_from_text([], [target], ctx)
        assert _get(out, 6002) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
