"""Tests for build_response_model in MarketplaceStrategy.

All tests are pure unit tests — no LLM calls, no external I/O.
Enforcement is case-sensitive: allowed_values must match exactly as stored in dict.
"""
import pytest
from pydantic import BaseModel, Field, AliasChoices
from typing import Optional

from app.services.enrichment.base import TargetAttribute
from app.services.enrichment.strategies.default_strategy import DefaultStrategy
from app.services.enrichment.strategies.ozon_strategy import OzonStrategy


# ---------------------------------------------------------------------------
# Shared pydantic models (mirrors _ExtractionResponse from description_source)
# ---------------------------------------------------------------------------

class _Attr(BaseModel):
    model_config = {"populate_by_name": True}
    attribute_id: int = Field(..., validation_alias=AliasChoices("attribute_id", "id"))
    value: str | int | float | bool | list[str | int | float | bool] = Field(
        ..., validation_alias=AliasChoices("value", "attribute_value")
    )


class _Response(BaseModel):
    extracted: list[_Attr]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _target_enum(attr_id: int, allowed: list[str]) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name="Цвет товара", type="enum", allowed_values=allowed)


def _target_text(attr_id: int) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name="Описание", type="text")


def _target_no_allowed(attr_id: int) -> TargetAttribute:
    """Enum-classified by type but missing allowed_values — no constraint."""
    return TargetAttribute(id=attr_id, name="Цвет", type="enum", allowed_values=[])


def _parse(model, attr_id: int, value):
    """Build a response instance via model constructor (triggers validators)."""
    return model(extracted=[{"attribute_id": attr_id, "value": value}])


# ---------------------------------------------------------------------------
# 1. DefaultStrategy returns base model unchanged
# ---------------------------------------------------------------------------

def test_default_strategy_returns_base_model():
    strategy = DefaultStrategy()
    targets = [_target_enum(1, ["черный", "белый"])]
    result = strategy.build_response_model(_Response, targets)
    assert result is _Response


# ---------------------------------------------------------------------------
# 2. OzonStrategy rejects out-of-dict value (case-sensitive)
# ---------------------------------------------------------------------------

def test_ozon_rejects_wrong_case():
    """'Чёрный' must fail when allowed list contains 'черный' (exact case match)."""
    strategy = OzonStrategy()
    targets = [_target_enum(10096, ["черный", "белый"])]
    model = strategy.build_response_model(_Response, targets)

    with pytest.raises(Exception):  # pydantic ValidationError
        _parse(model, 10096, "Чёрный")


# ---------------------------------------------------------------------------
# 3. OzonStrategy accepts exact match from allowed list
# ---------------------------------------------------------------------------

def test_ozon_accepts_exact_match():
    strategy = OzonStrategy()
    targets = [_target_enum(10096, ["черный", "белый"])]
    model = strategy.build_response_model(_Response, targets)

    obj = _parse(model, 10096, "черный")
    assert obj.extracted[0].value == "черный"


# ---------------------------------------------------------------------------
# 4. OzonStrategy accepts collection where every element is in allowed list
# ---------------------------------------------------------------------------

def test_ozon_accepts_valid_collection():
    strategy = OzonStrategy()
    targets = [_target_enum(10096, ["черный", "белый", "серый"])]
    model = strategy.build_response_model(_Response, targets)

    obj = _parse(model, 10096, ["черный", "белый"])
    assert obj.extracted[0].value == ["черный", "белый"]


# ---------------------------------------------------------------------------
# 5. OzonStrategy rejects collection with at least one invalid element
# ---------------------------------------------------------------------------

def test_ozon_rejects_collection_with_bad_element():
    strategy = OzonStrategy()
    targets = [_target_enum(10096, ["черный", "белый"])]
    model = strategy.build_response_model(_Response, targets)

    with pytest.raises(Exception):
        _parse(model, 10096, ["черный", "Красный"])


# ---------------------------------------------------------------------------
# 6. Non-enum targets bypass enforcement entirely
# ---------------------------------------------------------------------------

def test_non_enum_target_bypasses_enforcement():
    """text-type target (no allowed_values): any value passes."""
    strategy = OzonStrategy()
    targets = [_target_text(999)]
    model = strategy.build_response_model(_Response, targets)
    # model must be base unchanged (no constraints added for non-enum)
    obj = _parse(model, 999, "anything at all")
    assert obj.extracted[0].value == "anything at all"


# ---------------------------------------------------------------------------
# 7. Skip target with empty allowed_values — no constraint, any value passes
# ---------------------------------------------------------------------------

def test_ozon_skip_target_with_empty_allowed():
    strategy = OzonStrategy()
    targets = [_target_no_allowed(10096)]
    model = strategy.build_response_model(_Response, targets)
    # No constraints → base model returned unchanged
    assert model is _Response


# ---------------------------------------------------------------------------
# 8. OzonStrategy: constraint scoped per attribute_id, other attrs unaffected
# ---------------------------------------------------------------------------

def test_ozon_constraint_scoped_to_attr_id():
    """Constrained attr rejects bad value; unrelated attr (text) is always OK."""
    strategy = OzonStrategy()
    targets = [_target_enum(10096, ["черный"]), _target_text(888)]
    model = strategy.build_response_model(_Response, targets)

    # text attr (888) can hold any value
    obj = model(extracted=[{"attribute_id": 888, "value": "free text here"}])
    assert obj.extracted[0].value == "free text here"

    # enum attr (10096) still enforced even when mixed with text attr
    with pytest.raises(Exception):
        model(extracted=[
            {"attribute_id": 888, "value": "ok"},
            {"attribute_id": 10096, "value": "Неверный"},
        ])
