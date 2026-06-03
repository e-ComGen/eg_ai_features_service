"""Focused tests for dual-lang wiring + Serper concurrency semaphore.

All collaborators are mocked — NO network calls.

Covers:
  (a) WebSearchSource defaults to 2 languages [ru, en] when context.languages is None.
  (b) env WEBSEARCH_LANGS=ru forces a single language.
  (c) the Serper semaphore getter returns an asyncio.Semaphore and reuses the
      same object within a running loop.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock

from app.services.enrichment.base import ExtractionContext, TargetAttribute
from app.services.enrichment.websearch_producer import WebSearchProducer
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.enrichment.sources.web_search_source import (
    WebSearchSource, _WebExtractionResponse, _WebExtractedAttr,
)
import app.services.enrichment.websearch_producer as wsp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(languages=None) -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name="ACME Widget Pro 5000",
        category_id=10,
        brand="ACME",
        ean="1234567890123",
        languages=languages,
    )


def _make_source():
    mock_producer = AsyncMock(spec=WebSearchProducer)
    mock_producer.produce_summary = AsyncMock(return_value="Weight is 500g.")

    mock_extractor = AsyncMock(spec=StructuredLlmManager)
    resp = _WebExtractionResponse(
        extracted=[_WebExtractedAttr(attribute_id=42, value="500g", confidence=0.9)]
    )
    mock_extractor.structured_request = AsyncMock(return_value=(resp, 100))

    source = WebSearchSource(
        websearch_producer=mock_producer,
        extraction_manager=mock_extractor,
    )
    return source, mock_producer


def _target() -> TargetAttribute:
    return TargetAttribute(id=42, name="Weight", type="text")


# ---------------------------------------------------------------------------
# (a) Default = dual-lang [ru, en]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_defaults_to_dual_lang_when_context_languages_none(monkeypatch):
    monkeypatch.delenv("WEBSEARCH_LANGS", raising=False)
    source, mock_producer = _make_source()
    ctx = _make_context(languages=None)

    await source.extract(ctx, [_target()])

    _, kwargs = mock_producer.produce_summary.await_args
    assert kwargs["languages"] == ["ru", "en"]


# ---------------------------------------------------------------------------
# (b) WEBSEARCH_LANGS=ru forces single language
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_env_forces_single_language(monkeypatch):
    monkeypatch.setenv("WEBSEARCH_LANGS", "ru")
    source, mock_producer = _make_source()
    ctx = _make_context(languages=None)

    await source.extract(ctx, [_target()])

    _, kwargs = mock_producer.produce_summary.await_args
    assert kwargs["languages"] == ["ru"]


@pytest.mark.asyncio
async def test_env_blanks_are_stripped(monkeypatch):
    monkeypatch.setenv("WEBSEARCH_LANGS", "ru, , en,")
    source, mock_producer = _make_source()
    ctx = _make_context(languages=None)

    await source.extract(ctx, [_target()])

    _, kwargs = mock_producer.produce_summary.await_args
    assert kwargs["languages"] == ["ru", "en"]


@pytest.mark.asyncio
async def test_context_languages_override_env(monkeypatch):
    monkeypatch.setenv("WEBSEARCH_LANGS", "ru")
    source, mock_producer = _make_source()
    ctx = _make_context(languages=["en", "de"])

    await source.extract(ctx, [_target()])

    _, kwargs = mock_producer.produce_summary.await_args
    assert kwargs["languages"] == ["en", "de"]


# ---------------------------------------------------------------------------
# (c) Serper semaphore getter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_serper_sem_returns_semaphore_and_is_reused(monkeypatch):
    # Reset module holder so we exercise lazy creation on the running loop.
    monkeypatch.setattr(wsp, "_serper_sem", None)
    monkeypatch.setattr(wsp, "_serper_sem_loop", None)

    sem1 = wsp._get_serper_sem()
    sem2 = wsp._get_serper_sem()

    assert isinstance(sem1, asyncio.Semaphore)
    assert sem1 is sem2  # reused within the same running loop


@pytest.mark.asyncio
async def test_serper_sem_size_from_env(monkeypatch):
    monkeypatch.setattr(wsp, "_serper_sem", None)
    monkeypatch.setattr(wsp, "_serper_sem_loop", None)
    monkeypatch.setenv("WEBSEARCH_SERPER_CONCURRENCY", "3")

    sem = wsp._get_serper_sem()
    assert isinstance(sem, asyncio.Semaphore)
    # Semaphore has no public size attr; _value reflects remaining permits.
    assert sem._value == 3
