"""Тест _reconcile_enum_value_ids — инвариант value_ids ↔ value 1:1.

Баг (eg_importer, кроссовки): «Цвет товара» вернулся value='черный' (скаляр), но
value_ids=15 (вся палитра категории, реальные id белый/серый/синий…). Зальётся
неверными цветами. Reconcile сбрасывает рассинхрон → авторитетный резолв пересоберёт
ids строго из текста.
"""
from __future__ import annotations

from app.services.enrichment.base import AttributeValue, TargetAttribute, Source
from app.services.enrichment.pipeline import _reconcile_enum_value_ids


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
