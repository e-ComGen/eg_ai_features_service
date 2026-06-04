"""Tests for CompetitorRagSource and CompetitorRagJudge.

Все тесты используют in-memory Qdrant (без файловой персистентности).
Нет вызовов реальных LLM или HuggingFace API.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.competitor_rag_judge import CompetitorRagJudge
from app.services.enrichment.sources.competitor_rag_source import (
    CompetitorRagSource,
    _RelevanceFilter,
)
from app.services.enrichment.pipeline import PipelineOrchestrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**overrides) -> ExtractionContext:
    base = dict(product_id=1, product_name="Блок питания 600W ATX", category_id=42)
    base.update(overrides)
    return ExtractionContext(**base)


def _make_target(attr_id: int, name: str, attr_type: str = "enum") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=attr_type)


def _fake_neighbors(characteristics_list: list[dict], extra_fields: dict | None = None) -> list[dict]:
    """Создать список payload-соседей для мока.

    extra_fields — поля добавляемые к каждому соседу (например imt_name, categories).
    """
    extra = extra_fields or {}
    return [
        {"variantid": i, "imt_name": f"Product {i}", "name": f"Product {i}", "characteristics": ch, **extra}
        for i, ch in enumerate(characteristics_list)
    ]


def _make_rag_source_with_mock(
    neighbors: list[dict],
    llm_manager=None,
    top_k: int = 10,
) -> CompetitorRagSource:
    """Создать CompetitorRagSource с замоканным _search_neighbors, _get_embedding и LLM."""
    source = CompetitorRagSource.__new__(CompetitorRagSource)
    source._index_path = "/fake/path"
    source._collection_name = "ozon_products"
    source._top_k = top_k
    source._min_consensus = 2
    source._embed_model_name = "fake-model"
    source._client = None
    from app.services.enrichment.judges.competitor_rag_judge import CompetitorRagJudge
    source._judge = CompetitorRagJudge()
    source._search_neighbors = MagicMock(return_value=neighbors)
    source._llm_manager = llm_manager
    return source


def _make_llm_manager_returning(relevant_indices: list[int] | None) -> MagicMock:
    """Создать мок LLM-менеджера, возвращающий заданные индексы (или None при ошибке)."""
    manager = MagicMock()
    if relevant_indices is None:
        # Симулируем ошибку LLM (fallback должен сработать)
        manager.structured_request = AsyncMock(return_value=(None, 0))
    else:
        result = _RelevanceFilter(relevant_indices=relevant_indices)
        manager.structured_request = AsyncMock(return_value=(result, 10))
    return manager


# ---------------------------------------------------------------------------
# Test 1: Consensus extraction — ≥2/5 соседей согласны → AV emit (legacy path)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_consensus_emits_av():
    """Если ≥2 из 5 соседей содержат одинаковое значение — emit AttributeValue."""
    neighbors = _fake_neighbors([
        {"Цвет": ["Красный"]},
        {"Цвет": ["Красный"]},
        {"Цвет": ["Синий"]},
        {"Цвет": ["Красный"]},
        {"Цвет": ["Зелёный"]},
    ])
    # LLM-фильтр возвращает все 5 индексов — все релевантны
    llm_mgr = _make_llm_manager_returning([0, 1, 2, 3, 4])
    source = _make_rag_source_with_mock(neighbors, llm_manager=llm_mgr, top_k=5)
    targets = [_make_target(10, "Цвет")]
    ctx = _make_context()

    with patch("app.services.enrichment.sources.competitor_rag_source._get_embedding",
               return_value=[0.1] * 128):
        results = await source.extract(ctx, targets)

    assert len(results) == 1
    av = results[0]
    assert av.attribute_id == 10
    assert av.value == "Красный"
    assert av.source == Source.COMPETITOR_RAG
    assert av.confidence >= 0.7
    assert "3/5" in av.evidence  # 3 из 5 согласны


# ---------------------------------------------------------------------------
# Test 2: Нет консенсуса → пустой список
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_no_consensus_returns_empty():
    """Если все соседи дают разные значения — не emit ничего (избегаем ложный сигнал)."""
    neighbors = _fake_neighbors([
        {"Цвет": ["Красный"]},
        {"Цвет": ["Синий"]},
        {"Цвет": ["Зелёный"]},
        {"Цвет": ["Жёлтый"]},
        {"Цвет": ["Белый"]},
    ])
    llm_mgr = _make_llm_manager_returning([0, 1, 2, 3, 4])
    source = _make_rag_source_with_mock(neighbors, llm_manager=llm_mgr, top_k=5)
    targets = [_make_target(10, "Цвет")]
    ctx = _make_context()

    with patch("app.services.enrichment.sources.competitor_rag_source._get_embedding",
               return_value=[0.1] * 128):
        results = await source.extract(ctx, targets)

    assert results == []


# ---------------------------------------------------------------------------
# Test 3: Нет соседей → пустой список
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_no_neighbors_returns_empty():
    """Если Qdrant вернул пустой список — возвращаем пустой список."""
    source = _make_rag_source_with_mock([], llm_manager=_make_llm_manager_returning([]))
    targets = [_make_target(10, "Цвет")]
    ctx = _make_context()

    with patch("app.services.enrichment.sources.competitor_rag_source._get_embedding",
               return_value=[0.1] * 128):
        results = await source.extract(ctx, targets)

    assert results == []


# ---------------------------------------------------------------------------
# Test 4: already_filled → пропустить заполненные атрибуты
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_skips_already_filled():
    """already_filled targets не должны попасть в extract — не emit AV для них."""
    neighbors = _fake_neighbors([
        {"Цвет": ["Красный"]},
        {"Цвет": ["Красный"]},
        {"Цвет": ["Красный"]},
    ])
    llm_mgr = _make_llm_manager_returning([0, 1, 2])
    source = _make_rag_source_with_mock(neighbors, llm_manager=llm_mgr)
    targets = [_make_target(10, "Цвет")]

    # Атрибут уже заполнен
    already = [AttributeValue(attribute_id=10, value="Синий", confidence=0.9, source=Source.DESCRIPTION)]
    ctx = _make_context()

    with patch("app.services.enrichment.sources.competitor_rag_source._get_embedding",
               return_value=[0.1] * 128):
        results = await source.extract(ctx, targets, already_filled=already)

    assert results == []


# ---------------------------------------------------------------------------
# Test 5: Judge — принять confidence ≥ 0.6
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_judge_accepts_high_confidence():
    """CompetitorRagJudge принимает значение с confidence ≥ 0.6."""
    judge = CompetitorRagJudge()
    av = AttributeValue(attribute_id=10, value="Красный", confidence=0.75, source=Source.COMPETITOR_RAG)
    ctx = _make_context()
    result = await judge.validate(av, ctx)
    assert result is True


@pytest.mark.asyncio
async def test_judge_rejects_low_confidence():
    """CompetitorRagJudge отклоняет значение с confidence < 0.6."""
    judge = CompetitorRagJudge()
    av = AttributeValue(attribute_id=10, value="Красный", confidence=0.5, source=Source.COMPETITOR_RAG)
    ctx = _make_context()
    result = await judge.validate(av, ctx)
    assert result is False


# ---------------------------------------------------------------------------
# Test 6: Source enum — новое значение существует
# ---------------------------------------------------------------------------

def test_source_enum_competitor_rag():
    """Source.COMPETITOR_RAG должен быть зарегистрирован в enum."""
    assert Source.COMPETITOR_RAG == "competitor_rag"
    assert Source.COMPETITOR_RAG in Source


# ---------------------------------------------------------------------------
# Test 7: Pipeline integration — CompetitorRagSource wire-in
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_with_competitor_rag():
    """PipelineOrchestrator с CompetitorRagSource → AVs из RAG появляются в output."""
    from app.services.enrichment.base import LlmJudge
    from unittest.mock import AsyncMock, MagicMock

    # Мокаем все LLM-based sources
    def _mock_source(src_type):
        s = MagicMock()
        s.source_type = src_type
        s.is_applicable = MagicMock(return_value=False)  # LLM sources неприменимы
        s.extract = AsyncMock(return_value=[])
        j = MagicMock(spec=LlmJudge)
        j.source = src_type
        j.validate = AsyncMock(return_value=True)
        s.get_judge = MagicMock(return_value=j)
        return s

    # Мокаем CompetitorRagSource чтобы вернуть 1 AV
    rag_av = AttributeValue(
        attribute_id=10,
        value="Красный",
        confidence=0.8,
        source=Source.COMPETITOR_RAG,
        evidence="seen in 3/5 similar Ozon cards (LLM-filtered)",
    )
    rag_source = MagicMock(spec=CompetitorRagSource)
    rag_source.source_type = Source.COMPETITOR_RAG
    rag_source.is_applicable = MagicMock(return_value=True)
    rag_source.extract = AsyncMock(return_value=[rag_av])
    rag_judge = MagicMock(spec=CompetitorRagJudge)
    rag_judge.source = Source.COMPETITOR_RAG
    rag_judge.validate = AsyncMock(return_value=True)
    rag_source.get_judge = MagicMock(return_value=rag_judge)

    # Мокаем стратегию
    strategy = MagicMock()
    strategy.name = "test"
    strategy.filter_unsupported_attributes = MagicMock(side_effect=lambda t: t)
    strategy.filter_by_dictionary = MagicMock(side_effect=lambda t, c: t)
    strategy.normalize_target_with_context = MagicMock(side_effect=lambda t, c: t)
    strategy.force_websearch_targets = MagicMock(return_value=set())
    strategy.resolve_value_ids = MagicMock(side_effect=lambda v, c: v)
    strategy.post_process_values = MagicMock(side_effect=lambda vals, t, c: vals)
    strategy.validate_value = MagicMock(return_value=MagicMock(is_valid=True, normalized_value=None))
    strategy.build_response_model = MagicMock(side_effect=lambda m, t: m)
    strategy.llm_resolve_tail = AsyncMock(side_effect=lambda vals, t, c: vals)

    # Мокаем FinishingExtractor
    with patch("app.services.enrichment.pipeline.FinishingExtractor") as MockFinishing:
        mock_finisher = MagicMock()
        mock_finisher.extract_missing = AsyncMock(return_value=[])
        MockFinishing.return_value = mock_finisher

        # Мокаем LlmClassifier и CostPredictor
        classifier = MagicMock()
        classifier.classify = AsyncMock(return_value={})
        cost_pred = MagicMock()
        cost_pred.is_web_search_worth = AsyncMock(return_value=False)

        pipeline = PipelineOrchestrator(
            description_source=_mock_source(Source.DESCRIPTION),
            knowledge_source=_mock_source(Source.LLM_KNOWLEDGE),
            vision_source=_mock_source(Source.VISION),
            websearch_source=_mock_source(Source.WEB_SEARCH),
            competitor_rag_source=rag_source,
            classifier=classifier,
            cost_predictor=cost_pred,
            strategy=strategy,
        )

    ctx = _make_context()
    targets = [_make_target(10, "Цвет")]
    results = await pipeline.enrich(ctx, targets)

    assert any(av.source == Source.COMPETITOR_RAG for av in results), (
        f"Expected COMPETITOR_RAG AV in results, got: {results}"
    )


# ---------------------------------------------------------------------------
# Test 8 (NEW): LLM-фильтр возвращает [0, 2] из 10 → только эти 2 участвуют в consensus
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_filter_returns_subset_indices():
    """LLM-фильтр возвращает [0, 2] из 10 кандидатов → только они участвуют в голосовании."""
    # 10 кандидатов: только 0 и 2 релевантны (БП), остальные — мотозапчасти
    characteristics_list = [
        {"Мощность": ["750W"]},   # 0 — БП (релевантный)
        {"Тип": ["Цепь"]},         # 1 — мото (нерелевантный)
        {"Мощность": ["750W"]},   # 2 — БП (релевантный)
        {"Тип": ["Гайка"]},        # 3 — мото
        {"Тип": ["Цепь"]},         # 4 — мото
        {"Тип": ["Цепь"]},         # 5 — мото
        {"Тип": ["Гайка"]},        # 6 — мото
        {"Тип": ["Цепь"]},         # 7 — мото
        {"Тип": ["Гайка"]},        # 8 — мото
        {"Тип": ["Цепь"]},         # 9 — мото
    ]
    neighbors = _fake_neighbors(characteristics_list)
    # LLM возвращает только индексы 0 и 2
    llm_mgr = _make_llm_manager_returning([0, 2])
    source = _make_rag_source_with_mock(neighbors, llm_manager=llm_mgr)
    targets = [_make_target(20, "Мощность"), _make_target(21, "Тип")]
    ctx = _make_context(product_name="Блок питания 750W ATX")

    with patch("app.services.enrichment.sources.competitor_rag_source._get_embedding",
               return_value=[0.1] * 128):
        results = await source.extract(ctx, targets)

    # Только "Мощность" должна иметь consensus (2/2 = 100%), "Тип" — нет (0 голосов от кандидатов 0,2)
    attr_ids = {av.attribute_id for av in results}
    assert 20 in attr_ids, "Мощность должна быть в результатах"
    assert 21 not in attr_ids, "Тип (мото) не должен попасть в результаты"

    moshnost_av = next(av for av in results if av.attribute_id == 20)
    assert moshnost_av.value == "750W"
    assert "LLM-filtered" in moshnost_av.evidence

    # Убеждаемся что мусор от мото-кандидатов не прошёл
    all_values = [av.value for av in results]
    assert "Цепь" not in all_values
    assert "Гайка" not in all_values


# ---------------------------------------------------------------------------
# Test 9 (NEW): LLM-фильтр возвращает [] → CompetitorRagSource возвращает []
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_filter_returns_empty_yields_no_results():
    """Если LLM-фильтр отсеял все кандидаты → возвращаем [] (лучше пропустить)."""
    neighbors = _fake_neighbors([
        {"Тип": ["Цепь"]},
        {"Тип": ["Гайка"]},
        {"Тип": ["Цепь"]},
    ])
    llm_mgr = _make_llm_manager_returning([])  # нет релевантных кандидатов
    source = _make_rag_source_with_mock(neighbors, llm_manager=llm_mgr)
    targets = [_make_target(10, "Тип")]
    ctx = _make_context(product_name="Блок питания 750W ATX")

    with patch("app.services.enrichment.sources.competitor_rag_source._get_embedding",
               return_value=[0.1] * 128):
        results = await source.extract(ctx, targets)

    assert results == [], "Нулевые кандидаты после фильтра → пустой результат"


# ---------------------------------------------------------------------------
# Test 10 (NEW): LLM-фильтр возвращает все 10 → все участвуют в consensus
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_filter_returns_all_10_candidates():
    """Если LLM-фильтр одобряет все 10 кандидатов → все участвуют в consensus."""
    # 10 кандидатов, 8 согласны на "Красный"
    characteristics_list = [{"Цвет": ["Красный"]}] * 8 + [{"Цвет": ["Синий"]}] * 2
    neighbors = _fake_neighbors(characteristics_list)
    llm_mgr = _make_llm_manager_returning(list(range(10)))  # все 10
    source = _make_rag_source_with_mock(neighbors, llm_manager=llm_mgr)
    targets = [_make_target(30, "Цвет")]
    ctx = _make_context()

    with patch("app.services.enrichment.sources.competitor_rag_source._get_embedding",
               return_value=[0.1] * 128):
        results = await source.extract(ctx, targets)

    assert len(results) == 1
    av = results[0]
    assert av.value == "Красный"
    assert "8/10" in av.evidence
    # Confidence: 0.7 + 0.05*8 = 1.1 → capped to 0.95
    assert av.confidence == pytest.approx(0.95, abs=0.01)


# ---------------------------------------------------------------------------
# Test 11 (NEW): Consensus с 1 релевантным категорийно-точным кандидатом → emit AV
# (поведение изменено намеренно: _MIN_FILTERED_CANDIDATES снижен с 2 → 1,
#  т.к. category filter в retrieval режет шум — см. competitor_rag_source.py)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_relevant_candidate_emits_av():
    """Если после LLM-фильтра остался 1 категорийно-точный кандидат → emit AV (min=1)."""
    neighbors = _fake_neighbors([
        {"Мощность": ["650W"]},   # 0 — единственный релевантный
        {"Тип": ["Цепь"]},         # 1 — нерелевантный
        {"Тип": ["Гайка"]},        # 2 — нерелевантный
    ])
    llm_mgr = _make_llm_manager_returning([0])  # только один кандидат прошёл
    source = _make_rag_source_with_mock(neighbors, llm_manager=llm_mgr)
    targets = [_make_target(20, "Мощность")]
    ctx = _make_context(product_name="Блок питания 650W ATX")

    with patch("app.services.enrichment.sources.competitor_rag_source._get_embedding",
               return_value=[0.1] * 128):
        results = await source.extract(ctx, targets)

    # _MIN_FILTERED_CANDIDATES=1: одиночный категорийно-точный голос принимается
    assert len(results) == 1, "1 кандидат теперь достаточен для consensus (min=1)"
    assert results[0].attribute_id == 20
    assert results[0].value == "650W"
    assert "1/1" in results[0].evidence


# ---------------------------------------------------------------------------
# Test 12 (NEW): Graceful degradation — LLM-фильтр вернул None → fallback на raw top-5
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_filter_failure_falls_back_to_legacy():
    """При ошибке LLM-фильтра (None) → graceful degradation: consensus на raw top-5."""
    # 7 соседей, 3 из первых 5 согласны
    neighbors = _fake_neighbors([
        {"Цвет": ["Красный"]},
        {"Цвет": ["Красный"]},
        {"Цвет": ["Красный"]},
        {"Цвет": ["Синий"]},
        {"Цвет": ["Зелёный"]},
        {"Цвет": ["Фиолетовый"]},  # эти два не должны войти в top-5
        {"Цвет": ["Чёрный"]},
    ])
    # LLM-фильтр падает (возвращает None)
    llm_mgr = _make_llm_manager_returning(None)
    source = _make_rag_source_with_mock(neighbors, llm_manager=llm_mgr)
    targets = [_make_target(10, "Цвет")]
    ctx = _make_context()

    with patch("app.services.enrichment.sources.competitor_rag_source._get_embedding",
               return_value=[0.1] * 128):
        results = await source.extract(ctx, targets)

    # Fallback должен найти consensus в top-5: 3/5 = "Красный"
    assert len(results) == 1
    assert results[0].value == "Красный"
    # Evidence без "(LLM-filtered)" — это legacy evidence
    assert "LLM-filtered" not in results[0].evidence
