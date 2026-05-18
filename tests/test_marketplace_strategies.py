"""Tests for MarketplaceStrategy pattern — WB, Ozon, Default, Factory.

All tests are pure unit tests — no LLM calls, no external I/O.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.strategies.base import MarketplaceStrategy, ValidationResult
from app.services.enrichment.strategies.default_strategy import DefaultStrategy
from app.services.enrichment.strategies.wildberries_strategy import (
    WildberriesStrategy,
    WB_SKIP_SEMANTIC_TYPES,
    WB_BANNED_PHRASES,
)
from app.services.enrichment.strategies.ozon_strategy import OzonStrategy, OZON_SKIP_SEMANTIC_TYPES
from app.services.enrichment.strategies.factory import get_strategy
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.strategies.base import MarketplaceStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(
    id: int = 1,
    name: str = "Цвет",
    type: str = "text",
    semantic_type: str | None = None,
) -> TargetAttribute:
    return TargetAttribute(id=id, name=name, type=type, semantic_type=semantic_type)


def _make_ctx() -> ExtractionContext:
    return ExtractionContext(product_id=1, product_name="Test Product", category_id=1)


def _make_value(attr_id: int = 1, value: str = "red", source: Source = Source.DESCRIPTION) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=0.9,
        source=source,
    )


def _mock_source(source_type: Source, extract_return=None):
    from app.services.enrichment.base import AttributeSource, LlmJudge
    s = MagicMock(spec=AttributeSource)
    s.source_type = source_type
    s.is_applicable = MagicMock(return_value=True)
    s.extract = AsyncMock(return_value=extract_return or [])
    judge = MagicMock(spec=LlmJudge)
    judge.source = source_type
    judge.validate = AsyncMock(return_value=True)
    s.get_judge = MagicMock(return_value=judge)
    return s


def _build_pipeline(strategy=None, desc_values=None):
    desc_src = _mock_source(Source.DESCRIPTION, desc_values or [])
    know_src = _mock_source(Source.LLM_KNOWLEDGE, [])
    vis_src = _mock_source(Source.VISION, [])
    web_src = _mock_source(Source.WEB_SEARCH, [])

    classifier = MagicMock()
    classifier.classify = AsyncMock(return_value={})
    cost_pred = MagicMock()
    cost_pred.is_web_search_worth = AsyncMock(return_value=False)

    orch = PipelineOrchestrator(
        description_source=desc_src,
        knowledge_source=know_src,
        vision_source=vis_src,
        websearch_source=web_src,
        classifier=classifier,
        cost_predictor=cost_pred,
        strategy=strategy,
    )
    return orch


# ---------------------------------------------------------------------------
# 1. DefaultStrategy — filters nothing
# ---------------------------------------------------------------------------

def test_default_strategy_noop_filters_nothing():
    strategy = DefaultStrategy()
    targets = [
        _make_target(1, semantic_type="ean"),
        _make_target(2, semantic_type="article"),
        _make_target(3, semantic_type="color"),
    ]
    result = strategy.filter_unsupported_attributes(targets)
    assert result == targets  # no-op: all kept


# ---------------------------------------------------------------------------
# 2. DefaultStrategy — validates everything as valid
# ---------------------------------------------------------------------------

def test_default_strategy_validates_everything_as_valid():
    strategy = DefaultStrategy()
    target = _make_target(1, type="text")
    ctx = _make_ctx()

    for value in ["лучший", "№1", "some random text", "", 42, None]:
        result = strategy.validate_value(target, value, ctx)
        assert result.is_valid, f"DefaultStrategy should accept value: {value!r}"


# ---------------------------------------------------------------------------
# 3. WB — skips EAN attributes
# ---------------------------------------------------------------------------

def test_wb_strategy_skips_ean_attributes():
    strategy = WildberriesStrategy()
    targets = [
        _make_target(1, semantic_type="ean"),
        _make_target(2, semantic_type="color"),
    ]
    result = strategy.filter_unsupported_attributes(targets)
    assert len(result) == 1
    assert result[0].id == 2


# ---------------------------------------------------------------------------
# 4. WB — skips article/sku attributes
# ---------------------------------------------------------------------------

def test_wb_strategy_skips_article_attributes():
    strategy = WildberriesStrategy()
    targets = [
        _make_target(1, semantic_type="article"),
        _make_target(2, semantic_type="sku"),
        _make_target(3, semantic_type="material"),
    ]
    result = strategy.filter_unsupported_attributes(targets)
    ids = [t.id for t in result]
    assert 1 not in ids
    assert 2 not in ids
    assert 3 in ids


# ---------------------------------------------------------------------------
# 5. WB — rejects banned phrases
# ---------------------------------------------------------------------------

def test_wb_strategy_rejects_banned_phrases():
    strategy = WildberriesStrategy()
    target = _make_target(1, type="text")
    ctx = _make_ctx()

    for phrase in WB_BANNED_PHRASES:
        text = f"Это {phrase} товар на рынке"
        result = strategy.validate_value(target, text, ctx)
        assert not result.is_valid, f"WB should reject text containing '{phrase}'"
        assert phrase in (result.reason or "")


# ---------------------------------------------------------------------------
# 6. WB — accepts normal text
# ---------------------------------------------------------------------------

def test_wb_strategy_accepts_normal_text():
    strategy = WildberriesStrategy()
    target = _make_target(1, type="text")
    ctx = _make_ctx()

    result = strategy.validate_value(target, "Хлопковая футболка синего цвета", ctx)
    assert result.is_valid
    assert result.normalized_value == "Хлопковая футболка синего цвета"


# ---------------------------------------------------------------------------
# 7. OzonStrategy — basic init and name
# ---------------------------------------------------------------------------

def test_ozon_strategy_basic_init():
    strategy = OzonStrategy()
    assert strategy.name == "ozon"
    # filter_unsupported_attributes is callable
    targets = [_make_target(1, semantic_type="color"), _make_target(2, semantic_type="ean")]
    result = strategy.filter_unsupported_attributes(targets)
    # EAN should be filtered
    ids = [t.id for t in result]
    assert 2 not in ids
    assert 1 in ids


# ---------------------------------------------------------------------------
# 8. Factory — returns DefaultStrategy for None
# ---------------------------------------------------------------------------

def test_factory_returns_default_for_none():
    strategy = get_strategy(None)
    assert isinstance(strategy, DefaultStrategy)
    assert strategy.name == "default"


# ---------------------------------------------------------------------------
# 9. Factory — returns DefaultStrategy for unknown name
# ---------------------------------------------------------------------------

def test_factory_returns_default_for_unknown_name():
    strategy = get_strategy("unknown_marketplace_xyz")
    assert isinstance(strategy, DefaultStrategy)
    assert strategy.name == "default"


# ---------------------------------------------------------------------------
# 10. Factory — returns WbStrategy for "wb" and "wildberries"
# ---------------------------------------------------------------------------

def test_factory_returns_wb_strategy_for_wb_or_wildberries():
    wb1 = get_strategy("wb")
    assert isinstance(wb1, WildberriesStrategy)
    assert wb1.name == "wb"

    wb2 = get_strategy("wildberries")
    assert isinstance(wb2, WildberriesStrategy)

    # Case-insensitive
    wb3 = get_strategy("WB")
    assert isinstance(wb3, WildberriesStrategy)


# ---------------------------------------------------------------------------
# 11. Factory — returns OzonStrategy for "ozon"
# ---------------------------------------------------------------------------

def test_factory_returns_ozon_strategy():
    strategy = get_strategy("ozon")
    assert isinstance(strategy, OzonStrategy)
    assert strategy.name == "ozon"

    # Case-insensitive
    strategy2 = get_strategy("OZON")
    assert isinstance(strategy2, OzonStrategy)


# ---------------------------------------------------------------------------
# 12. PipelineOrchestrator — uses DefaultStrategy when not specified
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_uses_default_strategy_when_not_specified():
    orch = _build_pipeline(strategy=None)
    assert isinstance(orch._strategy, DefaultStrategy)
    # Ensure enrich runs without error
    ctx = _make_ctx()
    result = await orch.enrich(ctx, [_make_target(1)])
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 13. PipelineOrchestrator — uses strategy.filter_unsupported_attributes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_uses_strategy_filter_unsupported():
    # WB strategy filters out ean semantic_type
    strategy = WildberriesStrategy()
    orch = _build_pipeline(strategy=strategy)

    targets = [
        _make_target(1, semantic_type="ean"),   # should be filtered
        _make_target(2, semantic_type="color"),  # should remain
    ]
    ctx = _make_ctx()

    # The description source mock returns value for attr 2
    desc_values = [_make_value(attr_id=2, value="blue")]
    desc_src = _mock_source(Source.DESCRIPTION, desc_values)
    orch._sources[Source.DESCRIPTION] = desc_src
    from app.services.enrichment.confidence_aware_judge import ConfidenceAwareJudgeWrapper
    orch._judges[Source.DESCRIPTION] = ConfidenceAwareJudgeWrapper(desc_src.get_judge())

    result = await orch.enrich(ctx, targets)
    returned_ids = {v.attribute_id for v in result}
    # attr 1 (ean) should never appear — WB filters it before pipeline
    assert 1 not in returned_ids


# ---------------------------------------------------------------------------
# 14. PipelineOrchestrator — strategy.validate_value returning False drops value
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_uses_strategy_validate_value():
    """Mock strategy that rejects everything; pipeline should return empty list."""

    class RejectAllStrategy(MarketplaceStrategy):
        @property
        def name(self) -> str:
            return "reject_all"

        def validate_value(self, target, value, context) -> ValidationResult:
            return ValidationResult(is_valid=False, reason="test rejection")

    strategy = RejectAllStrategy()
    desc_values = [_make_value(attr_id=1, value="red")]
    orch = _build_pipeline(strategy=strategy, desc_values=desc_values)

    ctx = _make_ctx()
    result = await orch.enrich(ctx, [_make_target(1)])
    # All values should be dropped by validation
    assert result == []
