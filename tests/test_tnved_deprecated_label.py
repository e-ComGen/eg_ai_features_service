"""Тест Баг 4 — tnved_resolver не должен отдавать СНЯТУЮ с действия запись ТН ВЭД.

eg_importer (13_Джинсы, cat 200000933:93080): резолв вернул value_id 972056539 =
«6203423100 - (Действие прекращено с 15.09.2024) … брюки … из денима». Код
семантически верный, но запись депрекейтнута — Ozon её отвергнет. Снятые ярлыки
фильтруются из constrained-pick (на входе) и отвергаются в dict-валидации (defense).
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

from app.services.enrichment.base import ExtractionContext
from app.services.enrichment.sources.tnved_source import (
    TnvedSource,
    _is_deprecated_dict_label,
    _ABSTAIN,
)


_RUNTIME = "app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup"
_ENV = {"OZON_CLIENT_ID": "x", "OZON_API_KEY": "y"}

_DEPRECATED = "6203423100 - (Действие прекращено с 15.09.2024) брюки из денима"
_ACTIVE = "6203423900 - Брюки мужские из прочих текстильных материалов"


def _ctx():
    return ExtractionContext(product_id=1, product_name="Джинсы мужские синие",
                             category_id=200000933, ozon_type_id=93080,
                             category_path=["Одежда", "Джинсы"])


# ── _is_deprecated_dict_label ────────────────────────────────────────────────

def test_detects_deistvie_prekrasheno():
    assert _is_deprecated_dict_label(_DEPRECATED) is True


def test_detects_variants():
    assert _is_deprecated_dict_label("1234567890 - Действие приостановлено") is True
    assert _is_deprecated_dict_label("1234567890 - запись утратила силу") is True
    assert _is_deprecated_dict_label("1234567890 - исключён из номенклатуры") is True


def test_active_label_not_flagged():
    assert _is_deprecated_dict_label(_ACTIVE) is False
    assert _is_deprecated_dict_label("6404110000 - Обувь спортивная") is False
    assert _is_deprecated_dict_label("") is False


# ── _fetch_dict_labels: снятые ярлыки отсеиваются на входе constrained-pick ───

def test_fetch_dict_labels_filters_deprecated():
    raw = [
        {"id": 972056539, "value": _DEPRECATED},
        {"id": 972056540, "value": _ACTIVE},
    ]

    async def _run():
        src = TnvedSource()
        with patch.dict("os.environ", _ENV), \
             patch(f"{_RUNTIME}.list_values", return_value=raw):
            return await src._fetch_dict_labels(_ctx(), attr_id=22232)

    out = asyncio.run(_run())
    assert [it["id"] for it in out] == [972056540]  # снятый ярлык ушёл


# ── _validate_against_ozon_dict: defense — снятый хит → abstain ───────────────

def test_validate_rejects_deprecated_hit():
    """Даже если search_value вернул снятую запись (blind-guess путь) → abstain."""
    async def _run():
        src = TnvedSource()
        with patch.dict("os.environ", _ENV), \
             patch(f"{_RUNTIME}.search_value",
                   return_value={"id": 972056539, "value": _DEPRECATED}):
            return await src._validate_against_ozon_dict("6203423100", 22232, _ctx())

    assert asyncio.run(_run()) is _ABSTAIN


def test_validate_accepts_active_hit():
    """Действующая запись → (полный ярлык, value_id)."""
    async def _run():
        src = TnvedSource()
        with patch.dict("os.environ", _ENV), \
             patch(f"{_RUNTIME}.search_value",
                   return_value={"id": 972056540, "value": _ACTIVE}):
            return await src._validate_against_ozon_dict("6203423900", 22232, _ctx())

    out = asyncio.run(_run())
    assert out == (_ACTIVE, 972056540)
