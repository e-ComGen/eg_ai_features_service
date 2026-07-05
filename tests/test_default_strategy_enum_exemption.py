# -*- coding: utf-8 -*-
"""DefaultStrategy (cscart) must be EXEMPT from the enum drop-guards.

Bug: a cscart request carries no marketplace → get_strategy(None) → DefaultStrategy,
whose resolve_value_ids is a no-op, so every select/enum value (Бренд и др.) comes out
with value_id=None by design (the PHP applier binds label→variant_id later). But the
pipeline drops any enum with value_id=None (_drop_unresolved_*_enums) — a guard meant
only for Ozon/WB dictionary binding. Result: every select-field silently vanished
("Ingested 0 values"), while text/number fields filled fine.

Fix: MarketplaceStrategy.requires_dictionary_value_ids (fail-closed True); DefaultStrategy
overrides it to False; the pipeline runs the two enum drop-guards only when the strategy
requires dictionary value_ids. Color-guard stays unconditional; Ozon/WB behavior unchanged.

These tests mirror the exact pipeline branch (`if self._strategy.requires_dictionary_value_ids`)
using the REAL strategy flags and the REAL drop functions.
"""
from app.services.enrichment.pipeline import (
    _drop_unresolved_required_enums,
    _drop_unresolved_optional_enums,
)
from app.services.enrichment.base import AttributeValue, TargetAttribute, Source
from app.services.enrichment.strategies.default_strategy import DefaultStrategy
from app.services.enrichment.strategies.ozon_strategy import OzonStrategy
from app.services.enrichment.strategies.wildberries_strategy import WildberriesStrategy


def _brand_target():
    return TargetAttribute(
        id=31,
        name="Бренд",
        type="text",
        allowed_values=["Dyson", "Bosch", "Xiaomi"],
        is_collection=False,
        is_required=True,
    )


def _brand_value():
    # cscart-style: label from product title, no dictionary value_id (PHP resolves it).
    return AttributeValue(
        attribute_id=31,
        value="Dyson",
        confidence=0.95,
        source=Source.LLM_KNOWLEDGE,
        value_id=None,
        evidence="brand-from-name: 'Dyson' в наименовании товара",
    )


# --- flag wiring ------------------------------------------------------------

def test_default_strategy_is_exempt():
    assert DefaultStrategy().requires_dictionary_value_ids is False


def test_marketplace_strategies_require_dictionary_value_ids():
    assert OzonStrategy().requires_dictionary_value_ids is True
    assert WildberriesStrategy().requires_dictionary_value_ids is True


# --- behavior: same input, guard gated by the real strategy flag ------------

def _apply_enum_drop_guards(strategy, values, targets):
    """Exact replica of the pipeline branch (pipeline.py finalize tail)."""
    if strategy.requires_dictionary_value_ids:
        values = _drop_unresolved_optional_enums(values, targets)
        values = _drop_unresolved_required_enums(values, targets)
    return values


def test_cscart_brand_survives_under_default_strategy():
    out = _apply_enum_drop_guards(DefaultStrategy(), [_brand_value()], [_brand_target()])
    assert [v.value for v in out] == ["Dyson"]


def test_same_brand_still_dropped_under_ozon():
    out = _apply_enum_drop_guards(OzonStrategy(), [_brand_value()], [_brand_target()])
    assert out == []


def test_same_brand_still_dropped_under_wb():
    out = _apply_enum_drop_guards(WildberriesStrategy(), [_brand_value()], [_brand_target()])
    assert out == []
