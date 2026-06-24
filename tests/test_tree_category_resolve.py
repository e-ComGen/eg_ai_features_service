"""Тест resolve_description_category_id — живой dcid (родитель type_id) из дерева.

Баг (тостер): шаблонный category_id 47156221 values-API отвергает; правильный
description_category_id типа 96031 = 17039630 (родитель в дереве). Резолвер строит
{type_id: parent_dcid} из дерева Ozon.
"""
from __future__ import annotations

import asyncio

import app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup as rl


# Фрагмент дерева: Бытовая техника > Тепловая обработка(dcid 17039630) > Тостер(type 96031)
_TREE = [
    {"description_category_id": 200, "category_name": "Бытовая техника", "children": [
        {"description_category_id": 17039630, "category_name": "Тепловая обработка", "children": [
            {"type_id": 96031, "type_name": "Тостер"},
            {"type_id": 95614, "type_name": "Ростер"},
        ]},
    ]},
]


def _reset():
    rl.clear_tree_cache()


def test_walk_tree_maps_type_to_parent_dcid():
    _reset()
    rl._walk_tree(_TREE, None)
    assert rl._type_to_dcid[96031] == 17039630
    assert rl._type_to_dcid[95614] == 17039630


def test_resolve_returns_live_dcid_from_cache():
    _reset()
    rl._type_to_dcid.update({96031: 17039630})
    rl._tree_loaded = True  # не ходим в сеть
    out = asyncio.run(rl.resolve_description_category_id(96031))
    assert out == 17039630


def test_resolve_unknown_type_returns_none():
    _reset()
    rl._type_to_dcid.update({96031: 17039630})
    rl._tree_loaded = True
    assert asyncio.run(rl.resolve_description_category_id(999999)) is None
    assert asyncio.run(rl.resolve_description_category_id(None)) is None
    _reset()
