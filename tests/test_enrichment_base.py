"""Тесты для базовых типов architecture spec."""
import pytest
from pydantic import ValidationError

from app.services.enrichment.base import (
    Source, SOURCE_PRIORITY, SOURCE_CONFIDENCE_THRESHOLDS,
    AttributeValue, TargetAttribute, ExtractionContext,
    LlmJudge, AttributeSource,
)


# --- Source enum ---

def test_source_enum_has_all_four_values():
    assert Source.DESCRIPTION == "description"
    assert Source.LLM_KNOWLEDGE == "llm_knowledge"
    assert Source.VISION == "vision"
    assert Source.WEB_SEARCH == "web_search"


def test_source_priority_covers_all_sources():
    for src in Source:
        assert src in SOURCE_PRIORITY


def test_source_priority_ordering_description_highest():
    assert SOURCE_PRIORITY[Source.DESCRIPTION] > SOURCE_PRIORITY[Source.VISION]
    assert SOURCE_PRIORITY[Source.VISION] > SOURCE_PRIORITY[Source.WEB_SEARCH]
    assert SOURCE_PRIORITY[Source.WEB_SEARCH] > SOURCE_PRIORITY[Source.LLM_KNOWLEDGE]


def test_source_confidence_thresholds_in_valid_range():
    for src, threshold in SOURCE_CONFIDENCE_THRESHOLDS.items():
        assert 0.0 < threshold < 1.0


# --- AttributeValue ---

def test_attribute_value_basic():
    av = AttributeValue(
        attribute_id=1, value="red", confidence=0.95, source=Source.DESCRIPTION
    )
    assert av.value == "red"
    assert av.judge_validated is False  # default


def test_attribute_value_rejects_confidence_above_1():
    with pytest.raises(ValidationError):
        AttributeValue(attribute_id=1, value="x", confidence=1.5, source=Source.VISION)


def test_attribute_value_rejects_negative_confidence():
    with pytest.raises(ValidationError):
        AttributeValue(attribute_id=1, value="x", confidence=-0.1, source=Source.VISION)


def test_attribute_value_truncates_long_evidence():
    long_text = "x" * 2000
    av = AttributeValue(attribute_id=1, value="v", confidence=0.5, source=Source.WEB_SEARCH, evidence=long_text)
    assert len(av.evidence) <= 1004  # 1000 + "..."


def test_attribute_value_accepts_numeric_value():
    av = AttributeValue(attribute_id=1, value=42, confidence=0.9, source=Source.DESCRIPTION)
    assert av.value == 42


def test_attribute_value_is_confident_above_threshold():
    av = AttributeValue(attribute_id=1, value="red", confidence=0.96, source=Source.DESCRIPTION)
    assert av.is_confident() is True


def test_attribute_value_not_confident_below_threshold():
    av = AttributeValue(attribute_id=1, value="red", confidence=0.80, source=Source.DESCRIPTION)
    # threshold для DESCRIPTION = 0.95
    assert av.is_confident() is False


def test_attribute_value_is_confident_uses_source_specific_threshold():
    # Vision threshold 0.85 — 0.86 проходит, 0.84 нет
    high_vision = AttributeValue(attribute_id=1, value="red", confidence=0.86, source=Source.VISION)
    assert high_vision.is_confident() is True
    low_vision = AttributeValue(attribute_id=1, value="red", confidence=0.84, source=Source.VISION)
    assert low_vision.is_confident() is False


# --- TargetAttribute ---

def test_target_attribute_basic():
    t = TargetAttribute(id=42, name="Цвет", type="enum", allowed_values=["red", "blue"])
    assert t.semantic_type is None


def test_target_attribute_with_semantic_type():
    t = TargetAttribute(id=42, name="Цвет", type="enum", semantic_type="color")
    assert t.semantic_type == "color"


# --- ExtractionContext ---

def test_extraction_context_defaults():
    ctx = ExtractionContext(product_id=1, product_name="Test", category_id=42)
    assert ctx.cost_so_far_usd == 0.0
    assert ctx.llm_calls_so_far == 0
    assert ctx.max_cost_usd == 0.10
    assert ctx.source_urls == []
    assert ctx.image_urls == []


# --- AttributeSource abstract ---

def test_attribute_source_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        AttributeSource()


def test_concrete_attribute_source_works():
    class DummySource(AttributeSource):
        @property
        def source_type(self) -> Source:
            return Source.DESCRIPTION
        def is_applicable(self, context, target) -> bool:
            return True
        async def extract(self, context, targets):
            return []
        def get_judge(self) -> LlmJudge:
            return DummyJudge()

    class DummyJudge(LlmJudge):
        source = Source.DESCRIPTION
        async def validate(self, value, context):
            return True

    s = DummySource()
    assert s.source_type == Source.DESCRIPTION


def test_concrete_missing_method_cannot_be_instantiated():
    """Если concrete class забыл реализовать метод — TypeError."""
    class IncompleteSource(AttributeSource):
        @property
        def source_type(self) -> Source:
            return Source.VISION
        # забыли is_applicable, extract, get_judge

    with pytest.raises(TypeError):
        IncompleteSource()


# --- LlmJudge abstract ---

def test_llm_judge_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        LlmJudge()
