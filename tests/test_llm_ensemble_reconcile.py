"""Tests for the ported Phase 2a/2c ensemble reconcile (port of eg-importer's
test_llm_ensemble_phase2a.py / phase2c.py), ported alongside
app/services/enrichment/ensemble/reconcile.py + grounding.py + the
LlmKnowledgeSource._extract_ensemble flag-gated branch.

All tests use mocks -- no real LLM calls. LLM_ENSEMBLE_ENABLED stays False
by default (see app/config.py); these tests exercise the ensemble path
directly/explicitly regardless of the global flag.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app import config
from app.services.enrichment.base import AttributeValue, Source, TargetAttribute, ExtractionContext
from app.services.enrichment.ensemble.reconcile import reconcile, values_agree_fast, values_agree_llm
from app.services.enrichment.sources.llm_knowledge_source import LlmKnowledgeSource
import app.services.enrichment.sources.llm_knowledge_source as lks_module


def _mk_target(attr_id: int, name: str = "attr", semantic_type: str | None = None, is_collection: bool = False) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type="text", semantic_type=semantic_type, is_collection=is_collection)


def _mk_parsed(attribute_id: int, value, confidence: float = 0.9, reasoning: str | None = "test"):
    return SimpleNamespace(attribute_id=attribute_id, value=value, confidence=confidence, reasoning=reasoning)


def _mk_context() -> ExtractionContext:
    return ExtractionContext(product_id=1, product_name="Test Product XL", category_id=1, brand="TestBrand")


def _mk_manager(valid: bool):
    m = SimpleNamespace()
    m.structured_request = AsyncMock(return_value=(SimpleNamespace(valid=valid, reason="mock"), 10))
    return m


# ========== values_agree_fast ==========

def test_fast_exact_match_true():
    assert values_agree_fast("Синий", "синий") is True


def test_fast_ambiguous_acronyms_returns_none():
    assert values_agree_fast("IPS", "VA") is None


def test_fast_unit_mismatch_returns_none():
    # "5 m" vs "5 mm" -- equal leading numbers with different non-empty unit
    # remainders must NOT fast-accept (would silently equate different quantities).
    assert values_agree_fast("5 m", "5 mm") is None


# ========== values_agree_llm ==========

@pytest.mark.asyncio
async def test_agree_llm_fail_closed_on_none_response():
    fake_manager = AsyncMock()
    fake_manager.structured_request = AsyncMock(return_value=(None, 0))
    result = await values_agree_llm("A", "B", "attr", fake_manager)
    assert result is False


# ========== reconcile(): agree / disagree / solo branches ==========

@pytest.mark.asyncio
async def test_reconcile_both_agree_exact_emits_value():
    target = _mk_target(1001, name="Экран")
    parsed_a = [_mk_parsed(1001, "Синий")]
    parsed_b = [_mk_parsed(1001, "синий")]
    fake_arbiter = AsyncMock()
    fake_arbiter.structured_request = AsyncMock()
    result = await reconcile(parsed_a, parsed_b, [target], fake_arbiter)
    assert len(result) == 1
    assert result[0].attribute_id == 1001
    assert result[0].source == Source.LLM_KNOWLEDGE
    assert result[0].confidence == pytest.approx(config.LLM_ENSEMBLE_CONFIDENCE)
    fake_arbiter.structured_request.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_fast_none_llm_no_does_not_emit(tmp_path, monkeypatch):
    import importlib
    _rec_mod = importlib.import_module("app.services.enrichment.ensemble.reconcile")
    monkeypatch.setattr(_rec_mod, "_DISAGREEMENT_LOG", tmp_path / "disagreements.jsonl")
    target = _mk_target(1004, name="Panel Type")
    parsed_a = [_mk_parsed(1004, "IPS")]
    parsed_b = [_mk_parsed(1004, "VA")]
    fake_arbiter = AsyncMock()
    fake_arbiter.structured_request = AsyncMock(return_value=(SimpleNamespace(same=False), 10))
    result = await reconcile(parsed_a, parsed_b, [target], fake_arbiter)
    assert len(result) == 0
    fake_arbiter.structured_request.assert_awaited_once()


@pytest.mark.asyncio
async def test_reconcile_only_in_model_a_not_emitted_without_context():
    target = _mk_target(2001, name="Color")
    parsed_a = [_mk_parsed(2001, "Red")]
    parsed_b = []
    fake_arbiter = AsyncMock()
    fake_arbiter.structured_request = AsyncMock()
    # No context/judge passed -> solo branch is a no-op (never emits).
    result = await reconcile(parsed_a, parsed_b, [target], fake_arbiter)
    assert len(result) == 0
    fake_arbiter.structured_request.assert_not_called()


@pytest.mark.asyncio
async def test_solo_from_a_judged_by_manager_b_cross_vendor():
    config.LLM_ENSEMBLE_SOLO_POLICY = "judge"
    config.LLM_ENSEMBLE_SOLO_JUDGE = "cross"
    try:
        target = _mk_target(3001, name="Процессор")
        parsed_a = [_mk_parsed(3001, "Apple M2")]
        parsed_b = []
        manager_a = _mk_manager(True)
        manager_b = _mk_manager(True)
        arbiter = AsyncMock()

        result = await reconcile(parsed_a, parsed_b, [target], arbiter,
                                  context=_mk_context(), manager_a=manager_a, manager_b=manager_b)

        # A-produced solo -> judged by the OPPOSITE vendor (manager_b), never manager_a.
        manager_b.structured_request.assert_awaited_once()
        manager_a.structured_request.assert_not_called()
        assert len(result) == 1
        assert "cross-vendor" in (result[0].evidence or "")
    finally:
        config.LLM_ENSEMBLE_SOLO_JUDGE = "main"


# ========== LlmKnowledgeSource.extract() flag OFF/ON ==========

@pytest.mark.asyncio
async def test_extract_flag_off_uses_single_path_ensemble_not_called(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENSEMBLE_ENABLED", False)
    fake_ensemble_getter = MagicMock(side_effect=AssertionError("get_ensemble_managers must not be called when flag is OFF"))
    monkeypatch.setattr(lks_module, "get_ensemble_managers", fake_ensemble_getter)

    fake_main_manager = AsyncMock()
    fake_main_manager.structured_request = AsyncMock(return_value=(None, 0))

    source = LlmKnowledgeSource(llm_manager=fake_main_manager)
    context = _mk_context()
    targets = [_mk_target(3001, name="Цвет")]

    result = await source.extract(context, targets)

    fake_main_manager.structured_request.assert_awaited()
    fake_ensemble_getter.assert_not_called()
    assert result == []


@pytest.mark.asyncio
async def test_extract_flag_on_calls_ensemble_and_reconcile(monkeypatch):
    monkeypatch.setattr(config, "LLM_ENSEMBLE_ENABLED", True)

    fake_manager_a = AsyncMock()
    fake_manager_a.structured_request = AsyncMock(return_value=(None, 0))
    fake_manager_b = AsyncMock()
    fake_manager_b.structured_request = AsyncMock(return_value=(None, 0))
    monkeypatch.setattr(lks_module, "get_ensemble_managers", lambda: (fake_manager_a, fake_manager_b))
    monkeypatch.setattr(lks_module, "get_grounding_manager", lambda: None)

    fake_reconcile = AsyncMock(return_value=[
        AttributeValue(attribute_id=3001, value="Черный", confidence=config.LLM_ENSEMBLE_CONFIDENCE,
                        source=Source.LLM_KNOWLEDGE, evidence="ensemble consensus (A+B): test"),
    ])
    monkeypatch.setattr(lks_module, "ensemble_reconcile", fake_reconcile)

    fake_main_manager = AsyncMock()
    source = LlmKnowledgeSource(llm_manager=fake_main_manager)
    context = _mk_context()
    targets = [_mk_target(3001, name="Цвет")]

    result = await source.extract(context, targets)

    fake_manager_a.structured_request.assert_awaited()
    fake_manager_b.structured_request.assert_awaited()
    fake_reconcile.assert_awaited()
    assert len(result) == 1
    assert result[0].attribute_id == 3001
