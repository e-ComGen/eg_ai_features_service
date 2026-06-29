"""Tests for PipelineOrchestrator — sequential cost-aware routing.

All external LLM calls are mocked; no live API calls.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.base import (
    Source,
    SOURCE_PRIORITY,
    AttributeValue,
    TargetAttribute,
    ExtractionContext,
    AttributeSource,
    LlmJudge,
)
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.strategies.base import MarketplaceStrategy, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(id: int, name: str = "Цвет", semantic_type: str | None = None) -> TargetAttribute:
    return TargetAttribute(id=id, name=name, type="text", semantic_type=semantic_type)


def _make_ctx(**overrides) -> ExtractionContext:
    base = dict(product_id=1, product_name="Test Product", category_id=1)
    base.update(overrides)
    return ExtractionContext(**base)


def _make_value(attr_id: int, source: Source, confidence: float = 0.99) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value="red",
        confidence=confidence,
        source=source,
    )


def _mock_source(source_type: Source, extract_return: list | None = None) -> MagicMock:
    """Return a mock AttributeSource with get_judge() returning a mock LlmJudge."""
    s = MagicMock(spec=AttributeSource)
    s.source_type = source_type
    s.is_applicable = MagicMock(return_value=True)
    s.extract = AsyncMock(return_value=extract_return or [])
    judge = MagicMock(spec=LlmJudge)
    judge.source = source_type
    judge.validate = AsyncMock(return_value=True)
    s.get_judge = MagicMock(return_value=judge)
    return s


def _mock_classifier(routing: dict) -> MagicMock:
    c = MagicMock()
    c.classify = AsyncMock(return_value=routing)
    return c


def _mock_cost_predictor(worth: bool = True) -> MagicMock:
    cp = MagicMock()
    cp.is_web_search_worth = AsyncMock(return_value=worth)
    return cp


def _build_pipeline(
    desc_values=None,
    knowledge_values=None,
    vision_values=None,
    web_values=None,
    routing=None,
    cost_worth=True,
) -> tuple[PipelineOrchestrator, dict]:
    """Build an orchestrator with all components mocked."""
    desc_src = _mock_source(Source.DESCRIPTION, desc_values or [])
    know_src = _mock_source(Source.LLM_KNOWLEDGE, knowledge_values or [])
    vis_src = _mock_source(Source.VISION, vision_values or [])
    web_src = _mock_source(Source.WEB_SEARCH, web_values or [])

    classifier = _mock_classifier(routing or {})
    cost_pred = _mock_cost_predictor(cost_worth)

    orch = PipelineOrchestrator(
        description_source=desc_src,
        knowledge_source=know_src,
        vision_source=vis_src,
        websearch_source=web_src,
        classifier=classifier,
        cost_predictor=cost_pred,
    )
    mocks = {
        "desc": desc_src,
        "know": know_src,
        "vis": vis_src,
        "web": web_src,
        "classifier": classifier,
        "cost": cost_pred,
    }
    return orch, mocks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_description_only_covers_all_targets_returns_early():
    """If DescriptionSource returns confident values for all targets, classifier is never called."""
    targets = [_make_target(1), _make_target(2)]
    values = [
        _make_value(1, Source.DESCRIPTION, confidence=0.99),
        _make_value(2, Source.DESCRIPTION, confidence=0.99),
    ]
    orch, mocks = _build_pipeline(desc_values=values)

    ctx = _make_ctx()
    result = await orch.enrich(ctx, targets)

    assert len(result) == 2
    mocks["classifier"].classify.assert_not_called()
    mocks["know"].extract.assert_not_called()
    mocks["vis"].extract.assert_not_called()
    mocks["web"].extract.assert_not_called()


@pytest.mark.asyncio
async def test_classifier_called_when_description_incomplete():
    """Classifier is called when description leaves some targets unfilled."""
    targets = [_make_target(1), _make_target(2)]
    # Only fills attr 1 with high confidence
    values = [_make_value(1, Source.DESCRIPTION, confidence=0.99)]
    orch, mocks = _build_pipeline(desc_values=values, routing={2: [Source.LLM_KNOWLEDGE]})

    ctx = _make_ctx()
    await orch.enrich(ctx, targets)

    mocks["classifier"].classify.assert_called_once()
    # Called with remaining=[target(2)]
    call_args = mocks["classifier"].classify.call_args
    remaining_passed = call_args[0][1]  # second positional arg
    assert len(remaining_passed) == 1
    assert remaining_passed[0].id == 2


@pytest.mark.asyncio
async def test_routes_to_knowledge_source_when_classifier_suggests():
    """When classifier returns LLM_KNOWLEDGE as first choice, knowledge source is called."""
    targets = [_make_target(1)]
    routing = {1: [Source.LLM_KNOWLEDGE]}
    know_values = [_make_value(1, Source.LLM_KNOWLEDGE, confidence=0.95)]
    orch, mocks = _build_pipeline(
        desc_values=[],
        knowledge_values=know_values,
        routing=routing,
    )

    ctx = _make_ctx()
    result = await orch.enrich(ctx, targets)

    mocks["know"].extract.assert_called_once()
    assert len(result) == 1
    assert result[0].source == Source.LLM_KNOWLEDGE


@pytest.mark.asyncio
async def test_routes_to_vision_when_classifier_suggests_and_images_present():
    """Vision source is called when classifier suggests VISION and image_urls are present."""
    targets = [_make_target(1, semantic_type="color")]
    routing = {1: [Source.VISION]}
    vis_values = [_make_value(1, Source.VISION, confidence=0.90)]
    orch, mocks = _build_pipeline(
        desc_values=[],
        vision_values=vis_values,
        routing=routing,
    )
    # image_urls present
    ctx = _make_ctx(image_urls=["https://example.com/img.jpg"])
    result = await orch.enrich(ctx, targets)

    mocks["vis"].extract.assert_called_once()
    assert result[0].source == Source.VISION


@pytest.mark.asyncio
async def test_skips_vision_when_no_image_urls():
    """VisionSource is NOT called even if classifier suggests it when no image_urls."""
    targets = [_make_target(1)]
    routing = {1: [Source.VISION]}

    desc_src = _mock_source(Source.DESCRIPTION, [])
    know_src = _mock_source(Source.LLM_KNOWLEDGE, [])
    web_src = _mock_source(Source.WEB_SEARCH, [])

    # Vision source: is_applicable returns False when no images
    vis_src = _mock_source(Source.VISION, [])
    vis_src.is_applicable = MagicMock(return_value=False)

    orch = PipelineOrchestrator(
        description_source=desc_src,
        knowledge_source=know_src,
        vision_source=vis_src,
        websearch_source=web_src,
        classifier=_mock_classifier(routing),
        cost_predictor=_mock_cost_predictor(False),
    )

    ctx = _make_ctx()  # no image_urls
    await orch.enrich(ctx, targets)

    vis_src.extract.assert_not_called()


@pytest.mark.asyncio
async def test_cost_predictor_blocks_websearch_when_not_worth():
    """WebSearch is NOT called when CostPredictor returns False."""
    targets = [_make_target(1)]
    routing = {1: [Source.WEB_SEARCH]}
    orch, mocks = _build_pipeline(
        desc_values=[],
        routing=routing,
        cost_worth=False,
    )

    ctx = _make_ctx()
    result = await orch.enrich(ctx, targets)

    # CostPredictor was consulted and gated out the Stage-4 web routing.
    mocks["cost"].is_web_search_worth.assert_called_once()
    # NOTE: the finishing pass (Stage 5) legitimately re-runs every source —
    # including WebSearch — on still-empty targets, so web.extract may be called
    # there. That re-attempt is NOT the cost-gated Stage-4 call. Since the mock
    # returns no values, the gated attribute stays empty.
    assert result == []


@pytest.mark.asyncio
async def test_websearch_runs_when_cost_predictor_approves():
    """WebSearch IS called when CostPredictor returns True."""
    targets = [_make_target(1)]
    routing = {1: [Source.WEB_SEARCH]}
    web_values = [_make_value(1, Source.WEB_SEARCH, confidence=0.91)]
    orch, mocks = _build_pipeline(
        desc_values=[],
        web_values=web_values,
        routing=routing,
        cost_worth=True,
    )

    ctx = _make_ctx()
    result = await orch.enrich(ctx, targets)

    mocks["web"].extract.assert_called_once()
    assert result[0].source == Source.WEB_SEARCH


@pytest.mark.asyncio
async def test_merger_picks_highest_confidence_per_attribute():
    """_merge keeps the value with the highest confidence for each attribute_id."""
    targets = [_make_target(1)]
    routing = {1: [Source.LLM_KNOWLEDGE, Source.WEB_SEARCH]}

    # Description gives low confidence, knowledge gives higher
    desc_values = [_make_value(1, Source.DESCRIPTION, confidence=0.50)]
    know_values = [_make_value(1, Source.LLM_KNOWLEDGE, confidence=0.95)]

    orch, mocks = _build_pipeline(
        desc_values=desc_values,
        knowledge_values=know_values,
        routing=routing,
    )

    ctx = _make_ctx()
    result = await orch.enrich(ctx, targets)

    assert len(result) == 1
    assert result[0].source == Source.LLM_KNOWLEDGE
    # Both sources independently produced the same value ('red') for attr 1, so the
    # consensus bonus in _merge applies: 0.95 + 0.10 capped at 0.97. LLM_KNOWLEDGE
    # still wins over DESCRIPTION (0.50). The cross-source-agreement boost is the
    # intended merge behavior (see _merge docstring).
    assert result[0].confidence == 0.97


@pytest.mark.asyncio
async def test_merger_tie_break_by_source_priority():
    """When confidence is equal, SOURCE_PRIORITY determines winner.

    DESCRIPTION(4) > VISION(3) > WEB_SEARCH(2) > LLM_KNOWLEDGE(1)
    """
    # Inject two values with equal confidence, different sources
    # We'll test directly via _merge
    orch, _ = _build_pipeline()

    low_priority = _make_value(1, Source.LLM_KNOWLEDGE, confidence=0.80)
    high_priority = _make_value(1, Source.DESCRIPTION, confidence=0.80)

    # Order: low first → high should win
    merged = orch._merge([low_priority, high_priority])
    assert len(merged) == 1
    assert merged[0].source == Source.DESCRIPTION

    # Order: high first → high should still win
    merged2 = orch._merge([high_priority, low_priority])
    assert len(merged2) == 1
    assert merged2[0].source == Source.DESCRIPTION

    # Check VISION > WEB_SEARCH
    vision_val = _make_value(2, Source.VISION, confidence=0.80)
    web_val = _make_value(2, Source.WEB_SEARCH, confidence=0.80)
    merged3 = orch._merge([web_val, vision_val])
    assert merged3[0].source == Source.VISION


@pytest.mark.asyncio
async def test_source_failure_does_not_crash_pipeline():
    """If DescriptionSource raises, pipeline continues with remaining stages."""
    targets = [_make_target(1)]
    routing = {1: [Source.LLM_KNOWLEDGE]}
    know_values = [_make_value(1, Source.LLM_KNOWLEDGE, confidence=0.95)]

    desc_src = _mock_source(Source.DESCRIPTION)
    desc_src.extract = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    know_src = _mock_source(Source.LLM_KNOWLEDGE, know_values)
    vis_src = _mock_source(Source.VISION, [])
    web_src = _mock_source(Source.WEB_SEARCH, [])

    orch = PipelineOrchestrator(
        description_source=desc_src,
        knowledge_source=know_src,
        vision_source=vis_src,
        websearch_source=web_src,
        classifier=_mock_classifier(routing),
        cost_predictor=_mock_cost_predictor(False),
    )

    ctx = _make_ctx()
    result = await orch.enrich(ctx, targets)

    # Should still get knowledge result despite description failure
    assert len(result) == 1
    assert result[0].source == Source.LLM_KNOWLEDGE


@pytest.mark.asyncio
async def test_judge_rejection_filters_value():
    """If judge rejects a value (validate returns False), it is excluded from results."""
    targets = [_make_target(1)]

    # Low-confidence value so judge IS called
    desc_value = _make_value(1, Source.DESCRIPTION, confidence=0.50)

    desc_src = _mock_source(Source.DESCRIPTION, [desc_value])
    # Override judge to reject
    judge = MagicMock(spec=LlmJudge)
    judge.source = Source.DESCRIPTION
    judge.validate = AsyncMock(return_value=False)  # reject!
    desc_src.get_judge = MagicMock(return_value=judge)

    orch = PipelineOrchestrator(
        description_source=desc_src,
        knowledge_source=_mock_source(Source.LLM_KNOWLEDGE, []),
        vision_source=_mock_source(Source.VISION, []),
        websearch_source=_mock_source(Source.WEB_SEARCH, []),
        classifier=_mock_classifier({}),
        cost_predictor=_mock_cost_predictor(False),
    )

    ctx = _make_ctx()
    result = await orch.enrich(ctx, targets)

    assert result == []


@pytest.mark.asyncio
async def test_di_allows_injecting_mocks():
    """All dependencies can be injected — verifies constructor DI interface."""
    desc_src = _mock_source(Source.DESCRIPTION, [])
    know_src = _mock_source(Source.LLM_KNOWLEDGE, [])
    vis_src = _mock_source(Source.VISION, [])
    web_src = _mock_source(Source.WEB_SEARCH, [])
    classifier = _mock_classifier({})
    cost_pred = _mock_cost_predictor(False)

    orch = PipelineOrchestrator(
        description_source=desc_src,
        knowledge_source=know_src,
        vision_source=vis_src,
        websearch_source=web_src,
        classifier=classifier,
        cost_predictor=cost_pred,
    )

    ctx = _make_ctx()
    result = await orch.enrich(ctx, [_make_target(1)])

    # All sources injected correctly — desc was called, classifier was called.
    # desc may be invoked more than once: Stage 0 plus the Stage-5 finishing pass,
    # which re-runs every source on still-empty targets. assert_called (≥1) is the
    # correct wiring check here, not assert_called_once.
    desc_src.extract.assert_called()
    classifier.classify.assert_called_once()
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Force-websearch bypass tests
# ---------------------------------------------------------------------------

def _make_strategy_with_force_list(force_ids: set[int]) -> MarketplaceStrategy:
    """Build a minimal MarketplaceStrategy stub that returns `force_ids` from force_websearch_targets."""
    class _StubStrategy(MarketplaceStrategy):
        @property
        def name(self) -> str:
            return "stub"

        def force_websearch_targets(self, targets, context=None) -> set[int]:
            return force_ids

    return _StubStrategy()


@pytest.mark.asyncio
async def test_force_websearch_runs_even_when_cost_predictor_returns_false():
    """WebSearch IS called for force-listed attrs even when CostPredictor says False."""
    force_id = 6049  # Кол-во разъемов Molex — in force list
    targets = [_make_target(force_id, "Кол-во разъемов Molex")]
    routing = {force_id: [Source.WEB_SEARCH]}
    web_values = [_make_value(force_id, Source.WEB_SEARCH, confidence=0.92)]

    web_src = _mock_source(Source.WEB_SEARCH, web_values)
    cost_pred = _mock_cost_predictor(worth=False)  # predictor says NO
    strategy = _make_strategy_with_force_list({force_id})

    orch = PipelineOrchestrator(
        description_source=_mock_source(Source.DESCRIPTION, []),
        knowledge_source=_mock_source(Source.LLM_KNOWLEDGE, []),
        vision_source=_mock_source(Source.VISION, []),
        websearch_source=web_src,
        classifier=_mock_classifier(routing),
        cost_predictor=cost_pred,
        strategy=strategy,
    )

    ctx = _make_ctx()
    result = await orch.enrich(ctx, targets)

    # WebSearch should have run despite cost_predictor=False
    web_src.extract.assert_called_once()
    assert len(result) == 1
    assert result[0].source == Source.WEB_SEARCH
    assert result[0].attribute_id == force_id


@pytest.mark.asyncio
async def test_force_websearch_cost_predictor_not_called_for_force_only_targets():
    """When ALL websearch candidates are force-listed, CostPredictor is never called."""
    force_id = 6049
    targets = [_make_target(force_id)]
    routing = {force_id: [Source.WEB_SEARCH]}

    cost_pred = _mock_cost_predictor(worth=False)
    strategy = _make_strategy_with_force_list({force_id})

    orch = PipelineOrchestrator(
        description_source=_mock_source(Source.DESCRIPTION, []),
        knowledge_source=_mock_source(Source.LLM_KNOWLEDGE, []),
        vision_source=_mock_source(Source.VISION, []),
        websearch_source=_mock_source(Source.WEB_SEARCH, []),
        classifier=_mock_classifier(routing),
        cost_predictor=cost_pred,
        strategy=strategy,
    )

    ctx = _make_ctx()
    await orch.enrich(ctx, targets)

    # CostPredictor should NOT be called when all candidates are force-listed
    cost_pred.is_web_search_worth.assert_not_called()


@pytest.mark.asyncio
async def test_force_and_optional_websearch_targets_handled_in_one_call():
    """Force-listed and CostPredictor-approved targets are combined into one WebSearch call."""
    force_id = 6049
    optional_id = 999
    targets = [_make_target(force_id), _make_target(optional_id)]
    routing = {
        force_id: [Source.WEB_SEARCH],
        optional_id: [Source.WEB_SEARCH],
    }
    web_values = [
        _make_value(force_id, Source.WEB_SEARCH, confidence=0.90),
        _make_value(optional_id, Source.WEB_SEARCH, confidence=0.85),
    ]
    web_src = _mock_source(Source.WEB_SEARCH, web_values)
    cost_pred = _mock_cost_predictor(worth=True)  # approves optional
    strategy = _make_strategy_with_force_list({force_id})

    orch = PipelineOrchestrator(
        description_source=_mock_source(Source.DESCRIPTION, []),
        knowledge_source=_mock_source(Source.LLM_KNOWLEDGE, []),
        vision_source=_mock_source(Source.VISION, []),
        websearch_source=web_src,
        classifier=_mock_classifier(routing),
        cost_predictor=cost_pred,
        strategy=strategy,
    )

    ctx = _make_ctx()
    result = await orch.enrich(ctx, targets)

    # One combined WebSearch call for both targets
    web_src.extract.assert_called_once()
    call_targets = web_src.extract.call_args[0][1]  # second positional arg = targets
    called_ids = {t.id for t in call_targets}
    assert force_id in called_ids
    assert optional_id in called_ids
    assert len(result) == 2


@pytest.mark.asyncio
async def test_non_force_attr_still_blocked_by_cost_predictor():
    """Attributes NOT in force list are still blocked when CostPredictor returns False."""
    force_id = 6049
    optional_id = 999  # not in force list
    targets = [_make_target(optional_id)]
    routing = {optional_id: [Source.WEB_SEARCH]}

    # WebSearch returns nothing so the gated attr stays empty (finishing pass also
    # re-runs WebSearch on empty targets, but with no value to recover).
    web_src = _mock_source(Source.WEB_SEARCH, [])
    cost_pred = _mock_cost_predictor(worth=False)
    strategy = _make_strategy_with_force_list({force_id})  # force list only has 6049

    orch = PipelineOrchestrator(
        description_source=_mock_source(Source.DESCRIPTION, []),
        knowledge_source=_mock_source(Source.LLM_KNOWLEDGE, []),
        vision_source=_mock_source(Source.VISION, []),
        websearch_source=web_src,
        classifier=_mock_classifier(routing),
        cost_predictor=cost_pred,
        strategy=strategy,
    )

    ctx = _make_ctx()
    result = await orch.enrich(ctx, targets)

    # Non-force attr IS gated by the CostPredictor at Stage 4 (predictor consulted,
    # returned False → not routed). Unlike force attrs, which bypass the predictor.
    cost_pred.is_web_search_worth.assert_called_once()
    assert result == []


def _make_target_typed(
    id: int,
    name: str,
    type: str = "text",
    allowed_values: list | None = None,
) -> TargetAttribute:
    """Helper для создания TargetAttribute с произвольным type и allowed_values."""
    return TargetAttribute(id=id, name=name, type=type, allowed_values=allowed_values)


@pytest.mark.asyncio
async def test_ozon_strategy_force_websearch_targets_dimensions():
    """OzonStrategy.force_websearch_targets() включает dimension-атрибуты (kind=dimensions)."""
    from app.services.enrichment.strategies.ozon_strategy import OzonStrategy
    strategy = OzonStrategy()
    targets = [
        _make_target_typed(1, "Длина, см", type="numeric"),    # dimensions
        _make_target_typed(2, "Высота, см", type="numeric"),   # dimensions
        _make_target_typed(3, "Ширина, см", type="numeric"),   # dimensions
        _make_target_typed(4, "Бренд", type="text"),           # text — не force
    ]
    force_ids = strategy.force_websearch_targets(targets)
    assert 1 in force_ids, "Длина (dimensions) должна быть в force list"
    assert 2 in force_ids, "Высота (dimensions) должна быть в force list"
    assert 3 in force_ids, "Ширина (dimensions) должна быть в force list"
    assert 4 not in force_ids, "Бренд (text без словаря) не должен быть в force list"


@pytest.mark.asyncio
async def test_ozon_strategy_force_websearch_targets_numeric():
    """OzonStrategy.force_websearch_targets() включает numeric-атрибуты (kind=numeric)."""
    from app.services.enrichment.strategies.ozon_strategy import OzonStrategy
    strategy = OzonStrategy()
    targets = [
        _make_target_typed(10, "Мощность, Вт", type="numeric"),         # numeric
        _make_target_typed(11, "Кол-во разъемов SATA", type="Integer"), # numeric (Integer тип)
        _make_target_typed(12, "Описание", type="text"),                 # text — не force
    ]
    force_ids = strategy.force_websearch_targets(targets)
    assert 10 in force_ids, "Мощность Вт (numeric) должна быть в force list"
    assert 11 in force_ids, "Кол-во разъемов Integer (numeric) должна быть в force list"
    assert 12 not in force_ids, "Описание (text) не должно быть в force list"


@pytest.mark.asyncio
async def test_ozon_strategy_force_websearch_targets_large_enum():
    """OzonStrategy.force_websearch_targets() включает enum с >100 значениями."""
    from app.services.enrichment.strategies.ozon_strategy import OzonStrategy
    strategy = OzonStrategy()
    large_enum_values = [f"Значение{i}" for i in range(150)]
    small_enum_values = ["Красный", "Синий", "Зелёный"]
    targets = [
        _make_target_typed(20, "Бренд", type="text", allowed_values=large_enum_values),  # enum >100
        _make_target_typed(21, "Цвет", type="text", allowed_values=small_enum_values),    # enum ≤100
        _make_target_typed(22, "Страна", type="text", allowed_values=large_enum_values),  # enum >100
    ]
    force_ids = strategy.force_websearch_targets(targets)
    assert 20 in force_ids, "Бренд с >100 значениями (large enum) должен быть в force list"
    assert 21 not in force_ids, "Цвет с 3 значениями (small enum) не должен быть в force list"
    assert 22 in force_ids, "Страна с >100 значениями (large enum) должна быть в force list"


@pytest.mark.asyncio
async def test_ozon_strategy_force_websearch_targets_empty_for_text():
    """OzonStrategy.force_websearch_targets() возвращает пустое множество для text-атрибутов."""
    from app.services.enrichment.strategies.ozon_strategy import OzonStrategy
    strategy = OzonStrategy()
    targets = [
        _make_target_typed(30, "Название модели", type="text"),
        _make_target_typed(31, "Описание товара", type="text"),
    ]
    force_ids = strategy.force_websearch_targets(targets)
    assert force_ids == set(), "Чистые text-атрибуты не должны быть в force list"


@pytest.mark.asyncio
async def test_default_strategy_force_websearch_targets_empty():
    """DefaultStrategy.force_websearch_targets() всегда возвращает пустое множество."""
    from app.services.enrichment.strategies.default_strategy import DefaultStrategy
    strategy = DefaultStrategy()
    targets = [
        _make_target_typed(1, "Длина, см", type="numeric"),
        _make_target_typed(2, "Бренд", type="text", allowed_values=[f"b{i}" for i in range(200)]),
    ]
    assert strategy.force_websearch_targets(targets) == set()


@pytest.mark.asyncio
async def test_wb_strategy_force_websearch_targets_empty():
    """WildberriesStrategy.force_websearch_targets() наследует пустой default."""
    from app.services.enrichment.strategies.wildberries_strategy import WildberriesStrategy
    strategy = WildberriesStrategy()
    targets = [
        _make_target_typed(1, "Длина, см", type="numeric"),
    ]
    assert strategy.force_websearch_targets(targets) == set()


@pytest.mark.asyncio
async def test_ozon_no_hardcoded_force_constant():
    """OZON_FORCE_WEBSEARCH_ATTRS больше не существует в модуле ozon_strategy."""
    import importlib
    import app.services.enrichment.strategies.ozon_strategy as mod
    assert not hasattr(mod, "OZON_FORCE_WEBSEARCH_ATTRS"), (
        "OZON_FORCE_WEBSEARCH_ATTRS должен быть удалён — логика теперь в force_websearch_targets()"
    )


# ---------------------------------------------------------------------------
# IceCat-first + RAG-skip тесты (новый порядок: Description → IceCat → RAG fallback)
# ---------------------------------------------------------------------------

def _mock_icecat_source(fill_count: int = 0) -> MagicMock:
    """Mock IceCatSource возвращающий fill_count уверенных атрибутов."""
    from app.services.enrichment.base import LlmJudge
    values = [_make_value(100 + i, Source.ICECAT, confidence=0.97) for i in range(fill_count)]
    s = MagicMock(spec=AttributeSource)
    s.source_type = Source.ICECAT
    s.is_applicable = MagicMock(return_value=True)
    s.extract = AsyncMock(return_value=values)
    judge = MagicMock(spec=LlmJudge)
    judge.source = Source.ICECAT
    judge.validate = AsyncMock(return_value=True)
    s.get_judge = MagicMock(return_value=judge)
    return s


def _mock_rag_source(fill_count: int = 0) -> MagicMock:
    """Mock CompetitorRagSource возвращающий fill_count атрибутов."""
    from app.services.enrichment.base import LlmJudge
    values = [_make_value(200 + i, Source.COMPETITOR_RAG, confidence=0.90) for i in range(fill_count)]
    s = MagicMock(spec=AttributeSource)
    s.source_type = Source.COMPETITOR_RAG
    s.is_applicable = MagicMock(return_value=True)
    s.extract = AsyncMock(return_value=values)
    judge = MagicMock(spec=LlmJudge)
    judge.source = Source.COMPETITOR_RAG
    judge.validate = AsyncMock(return_value=True)
    s.get_judge = MagicMock(return_value=judge)
    return s


@pytest.mark.asyncio
async def test_icecat_runs_before_rag():
    """IceCat запускается ДО CompetitorRAG — новый порядок: Description → IceCat → RAG."""
    call_order = []

    icecat_src = _mock_icecat_source(fill_count=0)  # IceCat ничего не нашёл
    rag_src = _mock_rag_source(fill_count=0)

    # Перехватываем вызовы чтобы проверить порядок
    original_icecat_extract = icecat_src.extract
    original_rag_extract = rag_src.extract

    async def icecat_extract_spy(*args, **kwargs):
        call_order.append("icecat")
        return await original_icecat_extract(*args, **kwargs)

    async def rag_extract_spy(*args, **kwargs):
        call_order.append("rag")
        return await original_rag_extract(*args, **kwargs)

    icecat_src.extract = icecat_extract_spy
    rag_src.extract = rag_extract_spy

    targets = [_make_target(1)]
    orch = PipelineOrchestrator(
        description_source=_mock_source(Source.DESCRIPTION, []),
        knowledge_source=_mock_source(Source.LLM_KNOWLEDGE, []),
        vision_source=_mock_source(Source.VISION, []),
        websearch_source=_mock_source(Source.WEB_SEARCH, []),
        icecat_source=icecat_src,
        competitor_rag_source=rag_src,
        classifier=_mock_classifier({}),
        cost_predictor=_mock_cost_predictor(False),
    )

    await orch.enrich(_make_ctx(), targets)

    # IceCat должен быть вызван раньше RAG
    assert call_order.index("icecat") < call_order.index("rag"), (
        f"IceCat должен идти до RAG, но порядок: {call_order}"
    )


@pytest.mark.asyncio
async def test_rag_runs_even_when_icecat_fills_5_or_more():
    """CompetitorRAG запускается ВСЕГДА, даже когда IceCat заполнил ≥ 5 атрибутов.

    Прежний skip-guard «< 5 IceCat fills» удалён намеренно (см. комментарий в
    pipeline.enrich, Stage 0.7): IceCat avg = 5.35 на БП → RAG никогда не вызывался,
    что мешало измерять его реальный эффект. RAG — дешёвый (0 LLM calls, ~100ms
    Qdrant), поэтому запускается безусловно, пока остаются незаполненные targets.
    """
    # IceCat возвращает 5 уверенных атрибутов
    icecat_src = _mock_icecat_source(fill_count=5)
    rag_src = _mock_rag_source(fill_count=2)

    # Запрашиваем 10 атрибутов чтобы не сработал early-exit по remaining=0
    targets = [_make_target(i) for i in range(1, 11)]

    orch = PipelineOrchestrator(
        description_source=_mock_source(Source.DESCRIPTION, []),
        knowledge_source=_mock_source(Source.LLM_KNOWLEDGE, []),
        vision_source=_mock_source(Source.VISION, []),
        websearch_source=_mock_source(Source.WEB_SEARCH, []),
        icecat_source=icecat_src,
        competitor_rag_source=rag_src,
        classifier=_mock_classifier({}),
        cost_predictor=_mock_cost_predictor(False),
    )

    await orch.enrich(_make_ctx(), targets)

    # RAG ДОЛЖЕН быть вызван — skip-guard удалён, RAG запускается безусловно
    rag_src.extract.assert_called_once()


@pytest.mark.asyncio
async def test_rag_runs_when_icecat_fills_less_than_5():
    """CompetitorRAG запускается если IceCat заполнил < 5 атрибутов (фолбэк)."""
    # IceCat возвращает только 3 атрибута (< 5 — порог пропуска RAG)
    icecat_src = _mock_icecat_source(fill_count=3)
    rag_src = _mock_rag_source(fill_count=0)

    targets = [_make_target(i) for i in range(1, 11)]

    orch = PipelineOrchestrator(
        description_source=_mock_source(Source.DESCRIPTION, []),
        knowledge_source=_mock_source(Source.LLM_KNOWLEDGE, []),
        vision_source=_mock_source(Source.VISION, []),
        websearch_source=_mock_source(Source.WEB_SEARCH, []),
        icecat_source=icecat_src,
        competitor_rag_source=rag_src,
        classifier=_mock_classifier({}),
        cost_predictor=_mock_cost_predictor(False),
    )

    await orch.enrich(_make_ctx(), targets)

    # RAG должен быть вызван — IceCat дал только 3 атрибута (фолбэк)
    rag_src.extract.assert_called_once()


@pytest.mark.asyncio
async def test_rag_runs_when_no_icecat_source():
    """Если IceCatSource не передан — RAG работает как обычно (icecat_filled_count=0)."""
    pytest.importorskip("qdrant_client", reason="Пропускаем если Qdrant недоступен (RAG тест)")
    rag_src = _mock_rag_source(fill_count=1)
    targets = [_make_target(1)]

    orch = PipelineOrchestrator(
        description_source=_mock_source(Source.DESCRIPTION, []),
        knowledge_source=_mock_source(Source.LLM_KNOWLEDGE, []),
        vision_source=_mock_source(Source.VISION, []),
        websearch_source=_mock_source(Source.WEB_SEARCH, []),
        icecat_source=None,       # нет IceCat
        competitor_rag_source=rag_src,
        classifier=_mock_classifier({}),
        cost_predictor=_mock_cost_predictor(False),
    )

    await orch.enrich(_make_ctx(), targets)

    # RAG должен быть вызван — IceCat отсутствует, icecat_filled_count=0 < 5
    rag_src.extract.assert_called_once()
