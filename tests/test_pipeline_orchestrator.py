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

    mocks["cost"].is_web_search_worth.assert_called_once()
    mocks["web"].extract.assert_not_called()
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
    assert result[0].confidence == 0.95


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

    # All sources injected correctly — desc was called, classifier was called
    desc_src.extract.assert_called_once()
    classifier.classify.assert_called_once()
    assert isinstance(result, list)
