"""Тест backfill allowed_values из Ozon API для enum-полей без options.

Баг (eg_importer, тостер): «Количество отделений» — закрытый словарь Ozon
{1,2,3,4}, но caller не прислал options → enum-гейт нечем активировать →
verbatim «8» из описания протекало. Движок теперь сам тянет список из
list_values (как для ТН ВЭД) и заполняет allowed_values, после чего
_enforce_allowed_values режет out-of-list.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.services.enrichment.base import ExtractionContext, TargetAttribute
from app.services.enrichment.pipeline import (
    _is_enum_options_candidate,
    _backfill_allowed_values_from_api,
    _ENUM_OPTIONS_CAP,
)

_RUNTIME = "app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup"
_ENV = {"OZON_CLIENT_ID": "x", "OZON_API_KEY": "y"}

SLOTS = TargetAttribute(id=4820, name="Количество отделений", type="numeric")
COLOR = TargetAttribute(id=10096, name="Цвет товара", type="enum",
                        allowed_values=["белый", "черный"])
POWER = TargetAttribute(id=4851, name="Мощность, Вт", type="numeric")
TNVED = TargetAttribute(id=22232, name="ТН ВЭД коды ЕАЭС", type="text")
PARTNO = TargetAttribute(id=4381, name="Партномер", type="text")


def _ctx(type_id=96031):
    return ExtractionContext(product_id=1, product_name="Philips HD2581/90",
                             category_id=17039630, ozon_type_id=type_id)


# ── _is_enum_options_candidate ───────────────────────────────────────────────

def test_empty_options_numeric_is_candidate():
    assert _is_enum_options_candidate(SLOTS) is True


def test_already_has_options_not_candidate():
    assert _is_enum_options_candidate(COLOR) is False


def test_unit_field_not_candidate():
    assert _is_enum_options_candidate(POWER) is False  # «, Вт» → свободное числовое


def test_tnved_not_candidate():
    assert _is_enum_options_candidate(TNVED) is False  # свой резолвер


def test_model_name_not_candidate():
    assert _is_enum_options_candidate(PARTNO) is False  # model_name


# ── _backfill_allowed_values_from_api ────────────────────────────────────────

def _slots_values(*a, **k):
    return [{"id": 41834, "value": "1"}, {"id": 41840, "value": "2"},
            {"id": 41843, "value": "3"}, {"id": 41844, "value": "4"}]


def test_backfill_populates_closed_list():
    async def _run():
        with patch.dict("os.environ", _ENV), \
             patch(f"{_RUNTIME}.list_values", side_effect=_slots_values):
            return await _backfill_allowed_values_from_api([SLOTS], _ctx())

    out = asyncio.run(_run())
    assert out[0].allowed_values == ["1", "2", "3", "4"]
    assert out[0].id == 4820


def test_backfill_skips_when_no_type_id():
    async def _run():
        with patch.dict("os.environ", _ENV), \
             patch(f"{_RUNTIME}.list_values", side_effect=_slots_values) as m:
            res = await _backfill_allowed_values_from_api([SLOTS], _ctx(type_id=None))
            return res, m

    out, mock = asyncio.run(_run())
    assert out[0].allowed_values is None
    mock.assert_not_called()  # без type_id API не дёргаем


def test_backfill_ignores_truncated_dict():
    """Список ≥ капа = усечённый/огромный справочник → НЕ применяем как allowlist."""
    big = [{"id": i, "value": str(i)} for i in range(_ENUM_OPTIONS_CAP)]

    async def _run():
        with patch.dict("os.environ", _ENV), \
             patch(f"{_RUNTIME}.list_values", return_value=big):
            return await _backfill_allowed_values_from_api([SLOTS], _ctx())

    out = asyncio.run(_run())
    assert out[0].allowed_values is None  # усечён → не трогаем


def test_backfill_leaves_candidates_with_no_dict():
    """API вернул [] (поле не словарное / 404) → target без изменений."""
    async def _run():
        with patch.dict("os.environ", _ENV), \
             patch(f"{_RUNTIME}.list_values", return_value=[]):
            return await _backfill_allowed_values_from_api([SLOTS], _ctx())

    out = asyncio.run(_run())
    assert out[0].allowed_values is None


def test_backfill_does_not_touch_noncandidates():
    """Цвет (уже с options) и Мощность (единица) не трогаем; API зовём только для слотов."""
    async def _run():
        with patch.dict("os.environ", _ENV), \
             patch(f"{_RUNTIME}.list_values", side_effect=_slots_values) as m:
            res = await _backfill_allowed_values_from_api([COLOR, POWER, SLOTS], _ctx())
            return res, m

    out, mock = asyncio.run(_run())
    by_id = {t.id: t for t in out}
    assert by_id[10096].allowed_values == ["белый", "черный"]  # не тронут
    assert by_id[4851].allowed_values is None                  # не тронут
    assert by_id[4820].allowed_values == ["1", "2", "3", "4"]  # заполнен
    assert mock.call_count == 1  # API дёрнут только для слотов
