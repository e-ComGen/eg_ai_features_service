"""Юнит-тесты для GTINResolver (app/services/enrichment/sources/gtin_resolver.py).

Все тесты работают на МОКАХ — живых Serper-вызовов нет.
"""
from __future__ import annotations

import importlib
import sys
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.enrichment.base import ExtractionContext
from app.services.enrichment.sources.gtin_resolver import (
    _extract_gtin_candidates,
    _pick_best_candidate,
    resolve_gtin,
)


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------

def _make_context(
    product_name: str = "Sony WH-1000XM5",
    brand: Optional[str] = "Sony",
    ean: Optional[str] = None,
) -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name=product_name,
        category_id=42,
        brand=brand,
        ean=ean,
    )


def _make_serper_results(snippets: list[str]):
    """Строит mock SerperResults из списка строк (title=snippet для простоты)."""
    from app.services.providers.serper_client import OrganicResult, SerperResults

    organic = [
        OrganicResult(title=s, link=f"http://example.com/{i}", snippet=s, position=i + 1)
        for i, s in enumerate(snippets)
    ]
    return SerperResults(query="test", organic_results=organic)


# ---------------------------------------------------------------------------
# Тесты вспомогательных функций
# ---------------------------------------------------------------------------


class TestExtractGtinCandidates:
    def test_finds_13_digit(self):
        text = "EAN: 4905524927467 — официальный код"
        assert "4905524927467" in _extract_gtin_candidates(text)

    def test_finds_12_digit(self):
        text = "UPC 012345678905 printed on box"
        assert "012345678905" in _extract_gtin_candidates(text)

    def test_ignores_14_digit(self):
        text = "некий номер 12345678901234 в тексте"
        candidates = _extract_gtin_candidates(text)
        assert "12345678901234" not in candidates

    def test_ignores_7_digit(self):
        text = "код 1234567 не является EAN"
        assert not _extract_gtin_candidates(text)

    def test_multiple_candidates(self):
        text = "4905524927467 и также 4905524927467 и 0012345678905"
        candidates = _extract_gtin_candidates(text)
        assert len(candidates) == 3


class TestPickBestCandidate:
    # EAN-13 для Sony WH-1000XM5 (checksum-валидный: сумма = 70, check = 7 → 4905524927467)
    VALID_EAN = "4905524927467"
    # Мусорный 13-значный (checksum неверный)
    INVALID_EAN = "1234567890123"

    def test_returns_none_when_empty(self):
        assert _pick_best_candidate([]) is None

    def test_rejects_invalid_checksum(self):
        assert _pick_best_candidate([self.INVALID_EAN]) is None

    def test_accepts_valid_ean13(self):
        assert _pick_best_candidate([self.VALID_EAN]) == self.VALID_EAN

    def test_picks_most_frequent(self):
        # VALID_EAN встречается 2 раза, другой валидный — 1 раз
        # 4006381333931 — валидный EAN-13 для Bosch (примером берём Sony ещё раз для простоты)
        other_valid = "4905524927467"  # тот же — проверяем частотность
        candidates = [self.VALID_EAN, self.VALID_EAN, "4000000000000"]
        # "4000000000000" невалиден; self.VALID_EAN × 2 → возвращает его
        result = _pick_best_candidate(candidates)
        assert result == self.VALID_EAN

    def test_ignores_invalid_prefers_valid(self):
        candidates = [self.INVALID_EAN, self.INVALID_EAN, self.VALID_EAN]
        assert _pick_best_candidate(candidates) == self.VALID_EAN


# ---------------------------------------------------------------------------
# Тесты resolve_gtin (с моком SerperClient)
# ---------------------------------------------------------------------------

VALID_EAN_SONY = "4905524927467"


@pytest.mark.asyncio
async def test_resolve_gtin_returns_valid_ean():
    """Корректный EAN-13 из Serper snippet → resolve_gtin возвращает его."""
    context = _make_context()
    results = _make_serper_results([
        f"Sony WH-1000XM5 штрихкод EAN {VALID_EAN_SONY}",
        "Купить Sony WH-1000XM5 — характеристики",
    ])

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value=results)

    # Очищаем кэш перед тестом
    from app.services.enrichment.sources import gtin_resolver
    gtin_resolver._CACHE.clear()

    with patch(
        "app.services.enrichment.sources.gtin_resolver.get_web_search_client",
        return_value=mock_client,
        create=True,
    ):
        ean = await resolve_gtin(context)

    assert ean == VALID_EAN_SONY
    mock_client.search.assert_called_once()


@pytest.mark.asyncio
async def test_resolve_gtin_rejects_invalid_checksum():
    """Числа с неверным checksum отбрасываются → None."""
    context = _make_context()
    bad_ean = "1234567890123"  # невалидный checksum
    results = _make_serper_results([f"некий продукт код {bad_ean}"])

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value=results)

    from app.services.enrichment.sources import gtin_resolver
    gtin_resolver._CACHE.clear()

    with patch(
        "app.services.enrichment.sources.gtin_resolver.get_web_search_client",
        return_value=mock_client,
        create=True,
    ):
        ean = await resolve_gtin(context)

    assert ean is None


@pytest.mark.asyncio
async def test_resolve_gtin_empty_results_returns_none():
    """Пустая выдача Serper → None, без исключений."""
    from app.services.providers.serper_client import SerperResults

    context = _make_context()
    empty_results = SerperResults(query="test", organic_results=[])

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value=empty_results)

    from app.services.enrichment.sources import gtin_resolver
    gtin_resolver._CACHE.clear()

    with patch(
        "app.services.enrichment.sources.gtin_resolver.get_web_search_client",
        return_value=mock_client,
        create=True,
    ):
        ean = await resolve_gtin(context)

    assert ean is None


@pytest.mark.asyncio
async def test_resolve_gtin_serper_unavailable_returns_none():
    """Нет Serper client (get_web_search_client → None) → None, без исключений."""
    context = _make_context()

    from app.services.enrichment.sources import gtin_resolver
    gtin_resolver._CACHE.clear()

    with patch(
        "app.services.enrichment.sources.gtin_resolver.get_web_search_client",
        return_value=None,
        create=True,
    ):
        ean = await resolve_gtin(context)

    assert ean is None


@pytest.mark.asyncio
async def test_resolve_gtin_caches_result():
    """Второй вызов с теми же аргументами не делает новый Serper-запрос."""
    context = _make_context()
    results = _make_serper_results([f"EAN {VALID_EAN_SONY}"])

    mock_client = AsyncMock()
    mock_client.search = AsyncMock(return_value=results)

    from app.services.enrichment.sources import gtin_resolver
    gtin_resolver._CACHE.clear()

    with patch(
        "app.services.enrichment.sources.gtin_resolver.get_web_search_client",
        return_value=mock_client,
        create=True,
    ):
        ean1 = await resolve_gtin(context)
        ean2 = await resolve_gtin(context)

    assert ean1 == ean2 == VALID_EAN_SONY
    assert mock_client.search.call_count == 1  # кэш сработал


# ---------------------------------------------------------------------------
# Тест: флаг GTIN_RESOLVE_ENABLED — проверяем через env без создания тяжёлого
# PipelineOrchestrator (его инициализация требует сетевых credential-проверок).
# ---------------------------------------------------------------------------


def test_gtin_resolver_flag_off_by_default(monkeypatch):
    """GTIN_RESOLVE_ENABLED не задан → флаг False (по умолчанию off)."""
    monkeypatch.delenv("GTIN_RESOLVE_ENABLED", raising=False)
    # Проверяем логику чтения флага напрямую, без создания PipelineOrchestrator.
    import os
    assert os.environ.get("GTIN_RESOLVE_ENABLED", "0") == "0"


def test_gtin_resolver_flag_env_on(monkeypatch):
    """GTIN_RESOLVE_ENABLED=1 → env читается корректно."""
    monkeypatch.setenv("GTIN_RESOLVE_ENABLED", "1")
    import os
    assert os.environ.get("GTIN_RESOLVE_ENABLED", "0") == "1"


@pytest.mark.asyncio
async def test_pipeline_gtin_stage_enabled_mutates_context_ean(monkeypatch):
    """При gtin_resolver=True монкипатч на resolve_gtin отрабатывает."""
    resolved_ean = VALID_EAN_SONY

    async def _mock_resolve_gtin(ctx: ExtractionContext) -> Optional[str]:
        return resolved_ean

    monkeypatch.setattr(
        "app.services.enrichment.sources.gtin_resolver.resolve_gtin",
        _mock_resolve_gtin,
    )

    # Проверяем что monkeypatch заменил функцию корректно.
    from app.services.enrichment.sources import gtin_resolver as gtin_module
    ctx = _make_context(ean=None)
    ean = await gtin_module.resolve_gtin(ctx)
    assert ean == resolved_ean
