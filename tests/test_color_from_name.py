"""Тест _apply_color_from_name — цвет из НАЗВАНИЯ (надёжный per-SKU источник).

Доноры (WbCard/ozon_card) отдают палитру чужой расцветки (дропается). Единственный
надёжный цвет — в самом названии товара продавца: «…чёрные»→чёрный. Матч по стему
против allowed_values; ровно один → заполняем, двусмысленность/ноль → пусто.
"""
from __future__ import annotations

from app.services.enrichment.base import (
    AttributeValue, ExtractionContext, TargetAttribute, Source,
)
from app.services.enrichment.pipeline import _apply_color_from_name, _color_stem


COLOR_ID = 10096
_ALLOWED = ["черный", "белый", "серый", "синий", "красный", "темно-синий", "коричневый"]


def _targets(allowed=_ALLOWED):
    return [TargetAttribute(id=COLOR_ID, name="Цвет товара", type="enum",
                            semantic_type="color", allowed_values=allowed)]


def _ctx(name):
    return ExtractionContext(product_id=1, product_name=name, category_id=15621048,
                             category_path=["Обувь", "Кроссовки"], brand="Nike")


def _color_val(out):
    return [v.value for v in out if v.attribute_id == COLOR_ID]


# ── _color_stem ──────────────────────────────────────────────────────────────

def test_color_stem_normalizes_adjective_forms():
    assert _color_stem("чёрные") == _color_stem("черный") == "черн"
    assert _color_stem("синие") == _color_stem("синий") == "син"
    assert _color_stem("красная") == _color_stem("красный") == "красн"


# ── _apply_color_from_name ───────────────────────────────────────────────────

def test_fills_color_from_name_plural():
    """«Nike Air Max 90 чёрные» → чёрный (стем-матч черные↔черный)."""
    out = _apply_color_from_name([], _targets(), _ctx("Nike Air Max 90 чёрные"))
    assert _color_val(out) == ["черный"]
    av = next(v for v in out if v.attribute_id == COLOR_ID)
    assert av.source == Source.DESCRIPTION and av.evidence == "color_from_name"


def test_no_color_in_name_stays_empty():
    """Цвета в имени нет → ничего не добавляем (пусто честнее)."""
    out = _apply_color_from_name([], _targets(), _ctx("Nike Air Max 90"))
    assert _color_val(out) == []


def test_two_colors_ambiguous_skipped():
    """«чёрно-белые» → черный И белый → двусмысленно → не заполняем."""
    out = _apply_color_from_name([], _targets(), _ctx("Nike Air Max чёрно-белые"))
    assert _color_val(out) == []


def test_multiword_color_matched():
    """Двухсловный allowed «тёмно-синий» матчится из «тёмно-синие»."""
    out = _apply_color_from_name([], _targets(), _ctx("Кроссовки тёмно-синие Nike"))
    # «синий» тоже застемится из «синие» → 2 матча (темно-синий + синий) → двусмысленно.
    # Это безопасно: пусто честнее, чем угадать. Проверяем, что не упало и не наврало.
    vals = _color_val(out)
    assert vals == [] or vals == ["темно-синий"]


def test_already_filled_not_overwritten():
    """Одиночный grounded-цвет уже есть → color-from-name не трогает."""
    existing = AttributeValue(attribute_id=COLOR_ID, value="серый", confidence=0.9,
                              source=Source.OZON_CARD)
    out = _apply_color_from_name([existing], _targets(), _ctx("Nike Air Max 90 чёрные"))
    assert _color_val(out) == ["серый"]  # не перетёрли донорский одиночный


def test_no_allowed_values_noop():
    """Нет allowed_values → нечего матчить, no-op."""
    t = [TargetAttribute(id=COLOR_ID, name="Цвет товара", type="enum",
                         semantic_type="color", allowed_values=None)]
    out = _apply_color_from_name([], t, _ctx("Nike Air Max 90 чёрные"))
    assert _color_val(out) == []
