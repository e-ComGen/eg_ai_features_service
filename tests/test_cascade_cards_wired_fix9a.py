"""Functional oracle for FIX-9a (Ozon-card wired live + rich-sources default true).

Spec: docs/MANIFEST_source_cascade_cards_first.md, section FIX-9a.
Encodes the manifest's worked-example oracle O1..O4 verbatim -- does NOT
invent expected values, only asserts the ones the manifest specifies.

``_RICH_SOURCES_ENABLED`` is a module-level constant derived from the
environment at import time, so O3/O4 (which flip PIPELINE_RICH_SOURCES)
reload the module rather than mutate the already-imported constant.
"""
from __future__ import annotations

import importlib

import pytest

import app.services.enrichment.pipeline_adapter as pipeline_adapter_module

_ENV_VARS = ("PIPELINE_RICH_SOURCES", "SCRAPPEY_KEY", "SERPER_KEY")


@pytest.fixture(autouse=True)
def _restore_module_state(monkeypatch):
    """Reload the module back to its real (env-clean) default after every
    test in this file, so other test modules importing pipeline_adapter
    see the actual default, not a state left over from an env flip here."""
    yield
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    importlib.reload(pipeline_adapter_module)


def test_o1_scrappey_key_ozon_card_wired(monkeypatch):
    """O1: SCRAPPEY_KEY in env -> adapter._ozon_card is not None AND the
    orchestrator it builds received it (Stage 0.5 OzonCard is active)."""
    monkeypatch.setenv("SCRAPPEY_KEY", "test-key-123")
    mod = importlib.reload(pipeline_adapter_module)

    adapter = mod.PipelineAdapter()

    assert adapter._ozon_card is not None
    assert adapter._orch._ozon_card is not None
    assert adapter._orch._ozon_card is adapter._ozon_card


def test_o2_no_keys_constructs_without_error(monkeypatch):
    """O2: no SCRAPPEY_KEY/SERPER_KEY -> PipelineAdapter() constructs with no
    error; cards stay []-no-op (self-gated inside OzonCardSource)."""
    monkeypatch.delenv("SCRAPPEY_KEY", raising=False)
    monkeypatch.delenv("SERPER_KEY", raising=False)
    mod = importlib.reload(pipeline_adapter_module)

    adapter = mod.PipelineAdapter()  # must not raise

    # OzonCardSource is now constructed unconditionally (always-on wiring),
    # but self-gates: no key -> its own is_applicable()/extract() are [].
    assert adapter._ozon_card is not None
    inner = adapter._ozon_card._inner
    assert inner._scrappey_key is None


def test_o3_rich_sources_default_true(monkeypatch):
    """O3: PIPELINE_RICH_SOURCES unset -> _RICH_SOURCES_ENABLED is True (new default)."""
    monkeypatch.delenv("PIPELINE_RICH_SOURCES", raising=False)
    mod = importlib.reload(pipeline_adapter_module)

    assert mod._RICH_SOURCES_ENABLED is True


def test_o4_rich_sources_explicit_false(monkeypatch):
    """O4: PIPELINE_RICH_SOURCES=false -> _RICH_SOURCES_ENABLED is False (override still works)."""
    monkeypatch.setenv("PIPELINE_RICH_SOURCES", "false")
    mod = importlib.reload(pipeline_adapter_module)

    assert mod._RICH_SOURCES_ENABLED is False
