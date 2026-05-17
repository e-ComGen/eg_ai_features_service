"""Tests for PipelineAdapter (step I — integration adapter).

All LLM / orchestrator calls are mocked; no live API calls.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline_adapter import PipelineAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_orchestrator(return_values: list[AttributeValue] | None = None) -> MagicMock:
    orch = MagicMock()
    orch.enrich = AsyncMock(return_value=return_values or [])
    return orch


def _make_av(attr_id: int, value: str = "red", confidence: float = 0.9) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=confidence,
        source=Source.DESCRIPTION,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adapter_builds_context_from_args():
    """ExtractionContext passed to orchestrator contains all provided fields."""
    orch = _make_orchestrator()
    adapter = PipelineAdapter(orchestrator=orch)

    await adapter.run(
        product_id=42,
        product_name="Widget Pro",
        product_description="A great widget",
        category_id=7,
        category_path=["Electronics", "Gadgets"],
        brand="Acme",
        ean="1234567890123",
        source_urls=["https://supplier.example.com"],
        image_urls=["https://cdn.example.com/img.jpg"],
        max_cost_usd=0.05,
    )

    orch.enrich.assert_called_once()
    ctx: ExtractionContext = orch.enrich.call_args[0][0]
    assert ctx.product_id == 42
    assert ctx.product_name == "Widget Pro"
    assert ctx.product_description == "A great widget"
    assert ctx.category_id == 7
    assert ctx.category_path == ["Electronics", "Gadgets"]
    assert ctx.brand == "Acme"
    assert ctx.ean == "1234567890123"
    assert ctx.source_urls == ["https://supplier.example.com"]
    assert ctx.image_urls == ["https://cdn.example.com/img.jpg"]
    assert ctx.max_cost_usd == 0.05


@pytest.mark.asyncio
async def test_adapter_builds_targets_from_raw_dicts():
    """TargetAttribute list is correctly built from targets_raw dicts."""
    orch = _make_orchestrator()
    adapter = PipelineAdapter(orchestrator=orch)

    targets_raw = [
        {
            "id": 10,
            "name": "Colour",
            "type": "text",
            "allowed_values": ["Red", "Blue"],
            "semantic_type": "color",
            "description": "Product colour",
        },
        {
            "attribute_id": 20,
            "name": "Weight",
            "type": "numeric",
        },
    ]

    await adapter.run(
        product_id=1,
        product_name="P",
        product_description=None,
        category_id=1,
        targets_raw=targets_raw,
    )

    targets: list[TargetAttribute] = orch.enrich.call_args[0][1]
    assert len(targets) == 2

    t0 = targets[0]
    assert t0.id == 10
    assert t0.name == "Colour"
    assert t0.type == "text"
    assert t0.allowed_values == ["Red", "Blue"]
    assert t0.semantic_type == "color"
    assert t0.description == "Product colour"

    t1 = targets[1]
    assert t1.id == 20
    assert t1.name == "Weight"
    assert t1.type == "numeric"


@pytest.mark.asyncio
async def test_adapter_handles_missing_optional_fields():
    """Adapter runs without raising when optional args are omitted / None."""
    orch = _make_orchestrator()
    adapter = PipelineAdapter(orchestrator=orch)

    # Minimal call — only required fields
    result = await adapter.run(
        product_id=1,
        product_name="Minimal Product",
        product_description=None,
        category_id=1,
    )

    orch.enrich.assert_called_once()
    ctx: ExtractionContext = orch.enrich.call_args[0][0]
    assert ctx.brand is None
    assert ctx.ean is None
    assert ctx.source_urls == []
    assert ctx.image_urls == []
    assert ctx.category_path == []
    assert result == []  # orchestrator returned []


@pytest.mark.asyncio
async def test_adapter_calls_orchestrator_enrich():
    """adapter.run always calls orchestrator.enrich exactly once."""
    orch = _make_orchestrator()
    adapter = PipelineAdapter(orchestrator=orch)

    await adapter.run(
        product_id=5,
        product_name="Test",
        product_description="desc",
        category_id=3,
        targets_raw=[{"name": "Color", "type": "text"}],
    )

    orch.enrich.assert_called_once()
    # Both context and targets passed positionally
    assert len(orch.enrich.call_args[0]) == 2


@pytest.mark.asyncio
async def test_adapter_returns_orchestrator_result():
    """adapter.run returns exactly what orchestrator.enrich returns."""
    expected = [
        _make_av(1, "blue", 0.95),
        _make_av(2, "15kg", 0.88),
    ]
    orch = _make_orchestrator(return_values=expected)
    adapter = PipelineAdapter(orchestrator=orch)

    result = await adapter.run(
        product_id=1,
        product_name="X",
        product_description=None,
        category_id=1,
    )

    assert result is expected


@pytest.mark.asyncio
async def test_adapter_handles_empty_targets():
    """Empty targets_raw results in empty list passed to orchestrator."""
    orch = _make_orchestrator()
    adapter = PipelineAdapter(orchestrator=orch)

    await adapter.run(
        product_id=1,
        product_name="X",
        product_description=None,
        category_id=1,
        targets_raw=[],
    )

    targets: list[TargetAttribute] = orch.enrich.call_args[0][1]
    assert targets == []


# ---------------------------------------------------------------------------
# convert_to_legacy_dict
# ---------------------------------------------------------------------------

def test_convert_to_legacy_dict_maps_id_to_name():
    """convert_to_legacy_dict uses targets_raw to map attribute_id -> name."""
    targets_raw = [
        {"id": 10, "name": "Colour"},
        {"id": 20, "name": "Weight"},
    ]
    values = [
        _make_av(10, "Red"),
        _make_av(20, "5kg"),
    ]
    result = PipelineAdapter.convert_to_legacy_dict(values, targets_raw)
    assert result == {"Colour": "Red", "Weight": "5kg"}


def test_convert_to_legacy_dict_falls_back_to_str_id_when_no_targets():
    """Without targets_raw, attribute_id (int) is stringified as key."""
    values = [_make_av(99, "blue")]
    result = PipelineAdapter.convert_to_legacy_dict(values)
    assert result == {"99": "blue"}
