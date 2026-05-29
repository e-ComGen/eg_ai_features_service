"""Tests for skip-filled cooperative prompting between sources.

Task: verify that already_filled AVs are:
  - excluded from targets passed to LLM (reducing output tokens)
  - forwarded through the pipeline (2nd source sees 1st source's high-conf results)
  - never forwarded below confidence threshold 0.85
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, call

from app.services.enrichment.base import (
    AttributeValue, ExtractionContext, Source, TargetAttribute,
)
from app.services.enrichment.sources.description_source import (
    DescriptionSource, _ExtractionResponse, _ExtractedAttr,
)
from app.services.enrichment.sources.llm_knowledge_source import (
    LlmKnowledgeSource, _KnowledgeResponse, _KnowledgeAttr,
)
from app.services.enrichment.sources.vision_source import VisionSource
from app.services.enrichment.sources.web_search_source import (
    WebSearchSource, _WebExtractionResponse, _WebExtractedAttr,
)
from app.services.enrichment.prompt_router import (
    build_already_filled_block, filter_already_filled_targets,
    SKIP_FILLED_CONFIDENCE_THRESHOLD,
)
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.base import LlmJudge, AttributeSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(**overrides) -> ExtractionContext:
    base = dict(
        product_id=1,
        product_name="Блок питания Cooler Master MWE Gold 750W",
        product_description="Блок питания мощностью 750 Вт, бренд Cooler Master.",
        category_id=42,
        brand="Cooler Master",
    )
    base.update(overrides)
    return ExtractionContext(**base)


def _target(attr_id: int, name: str = "Мощность, Вт") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type="numeric")


def _av(attr_id: int, source: Source, confidence: float, value: str = "750") -> AttributeValue:
    return AttributeValue(attribute_id=attr_id, value=value, confidence=confidence, source=source)


def _mock_source_factory(source_type: Source, extract_return=None):
    """Создаёт mock AttributeSource для pipeline."""
    s = MagicMock(spec=AttributeSource)
    s.source_type = source_type
    s.is_applicable = MagicMock(return_value=True)
    s.extract = AsyncMock(return_value=extract_return or [])
    judge = MagicMock(spec=LlmJudge)
    judge.source = source_type
    judge.validate = AsyncMock(return_value=True)
    s.get_judge = MagicMock(return_value=judge)
    return s


# ---------------------------------------------------------------------------
# Test 1: filter_already_filled_targets removes high-conf attrs
# ---------------------------------------------------------------------------

def test_filter_already_filled_targets_removes_high_conf():
    """Атрибуты с confidence ≥ 0.85 убираются из targets."""
    targets = [_target(1, "Бренд"), _target(2, "Мощность, Вт"), _target(3, "Цвет")]
    already_filled = [
        _av(1, Source.DESCRIPTION, confidence=0.90),  # выше порога — убираем
        _av(2, Source.DESCRIPTION, confidence=0.84),  # ниже порога — оставляем
    ]
    result = filter_already_filled_targets(targets, already_filled)
    ids = [t.id for t in result]
    assert 1 not in ids, "attr_id=1 (conf=0.90) должен быть убран"
    assert 2 in ids, "attr_id=2 (conf=0.84) должен остаться"
    assert 3 in ids, "attr_id=3 (не в already_filled) должен остаться"


# ---------------------------------------------------------------------------
# Test 2: source with empty already_filled behaves exactly as before
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_description_source_empty_already_filled_unchanged():
    """При пустом already_filled behaviour не меняется — все targets передаются в LLM."""
    mock_llm = AsyncMock()
    mock_llm.structured_request.return_value = (
        _ExtractionResponse(extracted=[_ExtractedAttr(attribute_id=10, value="750", confidence=0.9)]),
        100,
    )
    source = DescriptionSource(llm_manager=mock_llm)
    ctx = _ctx()
    targets = [_target(10, "Мощность, Вт")]

    # Вызов без already_filled (default)
    result_default = await source.extract(ctx, targets)
    # Вызов с пустым списком — должен дать тот же результат
    ctx2 = _ctx()
    result_empty = await source.extract(ctx2, targets, already_filled=[])

    assert len(result_default) == 1
    assert len(result_empty) == 1
    # Оба вызова: prompt не содержит ALREADY RESOLVED
    for call_args in mock_llm.structured_request.call_args_list:
        user_text = call_args.kwargs.get("user_text", "")
        assert "ALREADY RESOLVED" not in user_text


# ---------------------------------------------------------------------------
# Test 3: source reduces targets passed to LLM when already_filled is non-empty
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_knowledge_source_skips_already_filled_in_prompt():
    """LlmKnowledgeSource: target с confidence ≥ 0.85 не попадает в targets_block."""
    mock_llm = AsyncMock()
    mock_llm.structured_request.return_value = (
        _KnowledgeResponse(known_attributes=[]),
        50,
    )
    source = LlmKnowledgeSource(llm_manager=mock_llm)
    ctx = _ctx()

    targets = [_target(1, "Бренд"), _target(2, "Мощность, Вт")]
    already_filled = [_av(1, Source.DESCRIPTION, confidence=0.92, value="Cooler Master")]

    await source.extract(ctx, targets, already_filled=already_filled)

    mock_llm.structured_request.assert_called_once()
    call_kwargs = mock_llm.structured_request.call_args.kwargs
    user_text = call_kwargs["user_text"]
    system_prompt = call_kwargs["system_prompt"]

    # attr_id=1 убран из targets → не должен появляться как target
    # attr_id=2 должен быть в targets_block
    assert "id=2" in user_text, "attr_id=2 должен быть в targets_block"
    # ALREADY RESOLVED секция присутствует
    assert "ALREADY RESOLVED" in user_text
    # system_prompt содержит правило пропуска
    assert "ALREADY RESOLVED" in system_prompt or "already known" in system_prompt.lower()


# ---------------------------------------------------------------------------
# Test 4: pipeline passes 1st source's high-conf AVs to 2nd source
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_second_source_receives_first_source_avs():
    """Pipeline: 2-й source получает high-conf AVs от 1-го source в already_filled."""
    # Description source возвращает один high-conf AV
    desc_av = _av(1, Source.DESCRIPTION, confidence=0.95)
    desc_src = _mock_source_factory(Source.DESCRIPTION, [desc_av])

    know_src = _mock_source_factory(Source.LLM_KNOWLEDGE, [])
    vis_src = _mock_source_factory(Source.VISION, [])
    web_src = _mock_source_factory(Source.WEB_SEARCH, [])

    classifier = MagicMock()
    # Classifier отправляет attr_id=2 к LLM_KNOWLEDGE
    classifier.classify = AsyncMock(return_value={2: [Source.LLM_KNOWLEDGE]})
    cost_pred = MagicMock()
    cost_pred.is_web_search_worth = AsyncMock(return_value=False)

    targets = [_target(1, "Бренд"), _target(2, "Мощность, Вт")]

    orch = PipelineOrchestrator(
        description_source=desc_src,
        knowledge_source=know_src,
        vision_source=vis_src,
        websearch_source=web_src,
        classifier=classifier,
        cost_predictor=cost_pred,
    )
    await orch.enrich(_ctx(), targets)

    # know_src.extract должен был быть вызван с already_filled содержащим desc_av
    know_src.extract.assert_called_once()
    call_kwargs = know_src.extract.call_args.kwargs
    already_filled_passed = call_kwargs.get("already_filled") or []
    filled_ids = [av.attribute_id for av in already_filled_passed]
    assert 1 in filled_ids, "attr_id=1 от DescriptionSource должен быть в already_filled для KnowledgeSource"


# ---------------------------------------------------------------------------
# Test 5: low-confidence AVs (<0.85) are NOT forwarded
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_low_conf_avs_not_forwarded():
    """AVs с confidence < 0.85 не передаются в already_filled следующего source."""
    # Description source возвращает один LOW-conf AV
    desc_av_low = _av(1, Source.DESCRIPTION, confidence=0.70)
    desc_src = _mock_source_factory(Source.DESCRIPTION, [desc_av_low])

    know_src = _mock_source_factory(Source.LLM_KNOWLEDGE, [])
    vis_src = _mock_source_factory(Source.VISION, [])
    web_src = _mock_source_factory(Source.WEB_SEARCH, [])

    classifier = MagicMock()
    classifier.classify = AsyncMock(return_value={1: [Source.LLM_KNOWLEDGE]})
    cost_pred = MagicMock()
    cost_pred.is_web_search_worth = AsyncMock(return_value=False)

    orch = PipelineOrchestrator(
        description_source=desc_src,
        knowledge_source=know_src,
        vision_source=vis_src,
        websearch_source=web_src,
        classifier=classifier,
        cost_predictor=cost_pred,
    )
    targets = [_target(1, "Бренд")]
    await orch.enrich(_ctx(), targets)

    # already_filled для know_src должен быть пустым (low conf не проходит)
    know_src.extract.assert_called_once()
    call_kwargs = know_src.extract.call_args.kwargs
    already_filled_passed = call_kwargs.get("already_filled") or []
    assert already_filled_passed == [], (
        f"Low-conf AVs не должны попадать в already_filled, но получили: {already_filled_passed}"
    )


# ---------------------------------------------------------------------------
# Test 6: build_already_filled_block returns empty strings for no high-conf AVs
# ---------------------------------------------------------------------------

def test_build_already_filled_block_empty_when_all_low_conf():
    """Если все AVs ниже порога — возвращаем пустые строки (нет пreamble)."""
    avs = [
        _av(1, Source.DESCRIPTION, confidence=0.80),
        _av(2, Source.DESCRIPTION, confidence=0.50),
    ]
    preamble, rule = build_already_filled_block(avs)
    assert preamble == ""
    assert rule == ""


def test_build_already_filled_block_nonempty_for_high_conf():
    """При наличии high-conf AVs — preamble и rule непустые."""
    avs = [
        _av(85, Source.DESCRIPTION, confidence=0.95, value="Cooler Master"),
    ]
    preamble, rule = build_already_filled_block(avs)
    assert "ALREADY RESOLVED" in preamble
    assert "attribute_id=85" in preamble
    assert rule != ""
