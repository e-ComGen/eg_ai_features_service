"""Тест _reconcile_enum_value_ids — инвариант value_ids ↔ value 1:1.

Баг (eg_importer, кроссовки): «Цвет товара» вернулся value='черный' (скаляр), но
value_ids=15 (вся палитра категории, реальные id белый/серый/синий…). Зальётся
неверными цветами. Reconcile сбрасывает рассинхрон → авторитетный резолв пересоберёт
ids строго из текста.
"""
from __future__ import annotations

from app.services.enrichment.base import AttributeValue, TargetAttribute, Source
from app.services.enrichment.pipeline import (
    _reconcile_enum_value_ids,
    _drop_ungrounded_color_guess,
    _is_multivalue_color_value,
    _drop_multivalue_color_premerge,
    _apply_color_source_guard,
)


COLOR_ID = 10096
_TARGETS = [TargetAttribute(id=COLOR_ID, name="Цвет товара", type="enum",
                            allowed_values=["черный", "белый", "серый"])]


def _av(**kw):
    base = dict(attribute_id=COLOR_ID, value="черный", confidence=0.9, source=Source.OZON_CARD)
    base.update(kw)
    return AttributeValue(**base)


def test_scalar_color_with_palette_ids_is_reset():
    """Скаляр 'черный' с 15 value_ids (палитра донора) → ids сброшены."""
    palette = [61574, 61571, 61576, 61581, 61579, 61580, 61583, 61578,
               61585, 61586, 61584, 61575, 61573, 61582, 61610]
    av = _av(value="черный", value_ids=palette, is_collection=False)
    out = _reconcile_enum_value_ids([av], _TARGETS)[0]
    assert out.value == "черный"          # текст сохранён
    assert out.value_ids is None          # рассинхрон-палитра сброшена
    assert out.value_id is None           # под пере-резолв


def test_aligned_collection_untouched():
    """Выровненная коллекция (len value == len value_ids) не трогается."""
    av = _av(value=["черный", "белый"], value_ids=[61574, 61571], is_collection=True)
    out = _reconcile_enum_value_ids([av], _TARGETS)[0]
    assert out.value == ["черный", "белый"]
    assert out.value_ids == [61574, 61571]


def test_clean_scalar_untouched():
    """Чистый скаляр (value_id, без списка value_ids) не трогается."""
    av = _av(value="черный", value_id=61574, value_ids=None, is_collection=False)
    out = _reconcile_enum_value_ids([av], _TARGETS)[0]
    assert out.value_id == 61574
    assert out.value_ids is None


def test_collection_length_mismatch_is_reset():
    """Коллекция с рассинхроном длины (1 значение, 15 ids) → сброс."""
    av = _av(value=["черный"], value_ids=list(range(15)), is_collection=True)
    out = _reconcile_enum_value_ids([av], _TARGETS)[0]
    assert out.value_ids is None


# ── _drop_ungrounded_color_guess: мульти-цвет от ЛЮБОГО источника → no_data ──

def test_ozon_card_multicolor_palette_dropped():
    """Выровненная мульти-цвет палитра от ozon_card (донор) → дроп (no_data)."""
    av = _av(value=["черный", "белый", "синий", "серый", "красный"],
             value_ids=[61574, 61571, 61581, 61576, 61579],
             is_collection=True, source=Source.OZON_CARD)
    out = _drop_ungrounded_color_guess([av], _TARGETS)
    assert out == []  # мульти-цвет не привязан к этому SKU → дроп


def test_scalar_value_with_palette_value_ids_dropped():
    """ozon_card форма: value='черный' СКАЛЯР + value_ids=[10/40] → дроп.

    Реальный кейс eg_importer (Adidas 10 ids, PUMA 40): value не список, но
    value_ids — палитра. Условие на длину value_ids, не на форму value.
    """
    av = _av(value="черный", value_ids=[61574, 61571, 61581, 61576, 61579,
                                        61580, 61583, 61578, 61585, 61586],
             is_collection=True, source=Source.OZON_CARD)
    out = _drop_ungrounded_color_guess([av], _TARGETS)
    assert out == []


def test_single_grounded_color_kept():
    """Одиночный grounded-цвет (primary) НЕ трогаем."""
    av = _av(value=["черный"], value_ids=[61574], is_collection=True, source=Source.OZON_CARD)
    out = _drop_ungrounded_color_guess([av], _TARGETS)
    assert len(out) == 1 and out[0].value == ["черный"]


def test_scalar_color_kept():
    """Скалярный цвет НЕ трогаем."""
    av = _av(value="черный", value_id=61574, source=Source.OZON_CARD)
    out = _drop_ungrounded_color_guess([av], _TARGETS)
    assert len(out) == 1


# ── _is_multivalue_color_value: строка-палитра через ; / , / / ───────────────

def test_string_palette_semicolon_is_multi():
    """WbCard-кейс: 'коричневый; темно-коричневый; белый; ...' (строка) → мульти."""
    assert _is_multivalue_color_value("коричневый; темно-коричневый; белый; зеленый", None) is True


def test_string_palette_comma_and_slash_is_multi():
    assert _is_multivalue_color_value("белый, синий", None) is True
    assert _is_multivalue_color_value("белый/чёрный", None) is True


def test_single_color_string_not_multi():
    """Один цвет (даже с дефисом «темно-синий») — НЕ мульти."""
    assert _is_multivalue_color_value("черный", None) is False
    assert _is_multivalue_color_value("темно-синий", 61574) is False


def test_list_and_valueids_forms_multi():
    assert _is_multivalue_color_value(["белый", "чёрный"], None) is True
    assert _is_multivalue_color_value("черный", [10, 40]) is True


# ── _drop_multivalue_color_premerge: одиночный цвет из имени выживает ──────────

def test_premerge_drops_palette_keeps_single_name_color():
    """WB-палитра (строка ;) дропается ДО merge, одиночный «черный» из имени остаётся.

    Реальный баг: «Nike Air Max 90 чёрные» → WbCard вернул палитру другой расцветки
    'коричневый; ...; бирюзовый' conf 0.93. Без pre-merge дропа она вытесняла бы
    «черный» из описания. После — палитра уходит ДО merge, остаётся верный цвет.
    """
    palette = _av(value="коричневый; темно-коричневый; белый; зеленый; бирюзовый",
                  confidence=0.93, source=Source.WB_CARD)
    name_color = _av(value="черный", confidence=0.8, source=Source.DESCRIPTION)
    out = _drop_multivalue_color_premerge([palette, name_color], _TARGETS)
    assert len(out) == 1
    assert out[0].value == "черный"  # одиночный grounded-цвет выжил


def test_premerge_keeps_single_color_only():
    """Если палитры нет — одиночный цвет не трогаем."""
    av = _av(value="черный", value_id=61574)
    assert _drop_multivalue_color_premerge([av], _TARGETS) == [av]


# ── _apply_color_source_guard: цвет-гадание от web/vision/llm дропается ────────

def test_color_guard_drops_web_search():
    """web_search «чёрный» (гадание колорвея PUMA без цвета в имени) → дроп."""
    av = _av(value="черный", source=Source.WEB_SEARCH)
    assert _apply_color_source_guard([av], _TARGETS) == []


def test_color_guard_drops_vision_and_llm():
    vis = _av(value="коричневый", source=Source.VISION)
    llm = _av(value="серый", source=Source.LLM_KNOWLEDGE)
    assert _apply_color_source_guard([vis, llm], _TARGETS) == []


def test_color_guard_keeps_description_color_from_name():
    """color-from-name (source=DESCRIPTION) — per-SKU, остаётся."""
    av = _av(value="черный", source=Source.DESCRIPTION, evidence="color_from_name")
    out = _apply_color_source_guard([av], _TARGETS)
    assert len(out) == 1 and out[0].value == "черный"


def test_color_guard_ignores_noncolor_targets():
    """Не-цвет атрибут от web_search — не трогаем (гард только для цвета)."""
    other = AttributeValue(attribute_id=9999, value="Демисезон", confidence=0.9,
                           source=Source.WEB_SEARCH)
    assert _apply_color_source_guard([other], _TARGETS) == [other]
