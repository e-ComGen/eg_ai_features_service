"""Tests for WbApparelRagSource — off-by-default wiring + safety gates.

Без сети / без HuggingFace / без реального LLM. Qdrant-доступ замокан.
"""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.enrichment.base import (
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.sources.wb_apparel_rag_source import (
    WB_APPAREL_COLLECTION,
    WbApparelRagSource,
)


def _ctx(**overrides) -> ExtractionContext:
    base = dict(product_id=1, product_name="Платье женское летнее", category_id=42)
    base.update(overrides)
    return ExtractionContext(**base)


def _target(attr_id: int, name: str, attr_type: str = "enum") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=attr_type)


# ---------------------------------------------------------------------------
# Identity / source_type
# ---------------------------------------------------------------------------

def test_source_type_reuses_competitor_rag():
    """Решение enum-vs-reuse: шарим Source.COMPETITOR_RAG, не добавляем новый enum."""
    src = WbApparelRagSource()
    assert src.source_type == Source.COMPETITOR_RAG
    assert src._collection_name == WB_APPAREL_COLLECTION
    assert "wb_apparel_rag.qdrant" in src._index_path


# ---------------------------------------------------------------------------
# Gate (a): no-op when collection missing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dormant_when_collection_absent():
    """Если коллекции нет — extract() возвращает [] без вызова parent.extract."""
    src = WbApparelRagSource(index_path="/nonexistent/path.qdrant")
    # collection_exists → False (нет директории индекса, нет QDRANT_URL fallback)
    with patch.object(src, "_collection_exists", return_value=False):
        with patch.object(
            WbApparelRagSource.__mro__[1], "extract", new=AsyncMock()
        ) as parent_extract:
            out = await src.extract(_ctx(), [_target(1, "Цвет")])
    assert out == []
    parent_extract.assert_not_called()


def test_collection_exists_false_for_missing_embedded_index():
    src = WbApparelRagSource(index_path="/definitely/not/here.qdrant")
    src._qdrant_url = None  # форсим embedded-режим
    assert src._collection_exists() is False


def test_collection_exists_true_when_present():
    src = WbApparelRagSource(index_path="/some/path.qdrant")
    src._qdrant_url = None
    with patch("os.path.isdir", return_value=True):
        fake_client = MagicMock()
        coll = MagicMock()
        coll.name = WB_APPAREL_COLLECTION
        fake_client.get_collections.return_value.collections = [coll]
        with patch.object(src, "_get_client", return_value=fake_client):
            assert src._collection_exists() is True


# ---------------------------------------------------------------------------
# Gate (b): material only emitted when it resolves to an Ozon enum value_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_material_dropped_when_no_value_id():
    """'Состав' без резолва в value_id выбрасывается; не-material остаётся."""
    from app.services.enrichment.base import AttributeValue

    src = WbApparelRagSource()
    color_av = AttributeValue(
        attribute_id=1, value="красный", confidence=0.8, source=Source.COMPETITOR_RAG
    )
    mat_av = AttributeValue(
        attribute_id=2, value="хлопок 95%", confidence=0.8, source=Source.COMPETITOR_RAG
    )
    parent_ret = [color_av, mat_av]

    with patch.object(src, "_collection_exists", return_value=True), patch.object(
        WbApparelRagSource.__mro__[1], "extract", new=AsyncMock(return_value=parent_ret)
    ), patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.resolve_value_id",
        return_value=None,
    ):
        out = await src.extract(
            _ctx(), [_target(1, "Цвет"), _target(2, "Состав")]
        )
    ids = {v.attribute_id for v in out}
    assert ids == {1}  # material dropped, color kept


@pytest.mark.asyncio
async def test_material_kept_and_value_id_set_when_resolved():
    """'Состав' с успешным резолвом остаётся и получает value_id."""
    from app.services.enrichment.base import AttributeValue

    src = WbApparelRagSource()
    mat_av = AttributeValue(
        attribute_id=2, value="хлопок", confidence=0.8, source=Source.COMPETITOR_RAG
    )

    with patch.object(src, "_collection_exists", return_value=True), patch.object(
        WbApparelRagSource.__mro__[1], "extract", new=AsyncMock(return_value=[mat_av])
    ), patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.resolve_value_id",
        return_value=777,
    ):
        out = await src.extract(_ctx(), [_target(2, "Состав")])
    assert len(out) == 1
    assert out[0].value_id == 777


# ---------------------------------------------------------------------------
# Pipeline wiring: OFF by default, ON only with env flag
# ---------------------------------------------------------------------------

def test_pipeline_no_wb_apparel_by_default(monkeypatch):
    """Без флага и без явного competitor_rag — _competitor_rag остаётся None."""
    monkeypatch.delenv("WB_APPAREL_RAG_ENABLED", raising=False)
    orch = PipelineOrchestrator()
    assert orch._competitor_rag is None


def test_pipeline_registers_wb_apparel_when_flag_on(monkeypatch):
    """WB_APPAREL_RAG_ENABLED=1 → регистрируется WbApparelRagSource (дормантен)."""
    monkeypatch.setenv("WB_APPAREL_RAG_ENABLED", "1")
    orch = PipelineOrchestrator()
    assert isinstance(orch._competitor_rag, WbApparelRagSource)
    # judge зарегистрирован под Source.COMPETITOR_RAG
    assert Source.COMPETITOR_RAG in orch._judges


def test_pipeline_explicit_source_not_overridden_by_flag(monkeypatch):
    """Явно переданный competitor_rag не перетирается флагом."""
    monkeypatch.setenv("WB_APPAREL_RAG_ENABLED", "1")
    from app.services.enrichment.sources.competitor_rag_source import CompetitorRagSource

    explicit = CompetitorRagSource.__new__(CompetitorRagSource)
    explicit._judge = MagicMock()
    orch = PipelineOrchestrator(competitor_rag_source=explicit)
    assert orch._competitor_rag is explicit
    assert not isinstance(orch._competitor_rag, WbApparelRagSource)
