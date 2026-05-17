"""Unit tests for DescriptionJudge."""
import pytest
from unittest.mock import AsyncMock

from app.services.enrichment.base import AttributeValue, ExtractionContext, Source
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.enrichment.judges.description_judge import DescriptionJudge, _JudgeVerdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(description: str | None = "This product has red color and weighs 500g.") -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name="Test Product",
        product_description=description,
        category_id=42,
    )


def _make_value(attr_id: int = 10, value: str = "red", evidence: str | None = "red color") -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=0.9,
        source=Source.DESCRIPTION,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# DescriptionJudge tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_validate_returns_false_without_description():
    """Judge returns False immediately when context has no description."""
    mock_llm = AsyncMock(spec=StructuredLlmManager)
    judge = DescriptionJudge(llm_manager=mock_llm)

    result = await judge.validate(_make_value(), _make_context(None))

    assert result is False
    mock_llm.structured_request.assert_not_called()


@pytest.mark.asyncio
async def test_validate_calls_llm():
    """Judge must call LLM when description is present."""
    mock_llm = AsyncMock(spec=StructuredLlmManager)
    mock_llm.structured_request.return_value = (_JudgeVerdict(valid=True, reason="Found in text"), 50)
    judge = DescriptionJudge(llm_manager=mock_llm)

    await judge.validate(_make_value(), _make_context())

    mock_llm.structured_request.assert_called_once()
    call_kwargs = mock_llm.structured_request.call_args
    user_text = call_kwargs.kwargs.get("user_text") or call_kwargs.args[1]
    assert "red color" in user_text or "red" in user_text


@pytest.mark.asyncio
async def test_validate_returns_llm_verdict():
    """validate() returns True/False based on verdict.valid from LLM."""
    mock_llm = AsyncMock(spec=StructuredLlmManager)
    judge = DescriptionJudge(llm_manager=mock_llm)

    # valid=True → returns True
    mock_llm.structured_request.return_value = (_JudgeVerdict(valid=True, reason="Supported"), 50)
    result_true = await judge.validate(_make_value(), _make_context())
    assert result_true is True

    # valid=False → returns False
    mock_llm.structured_request.return_value = (_JudgeVerdict(valid=False, reason="Not found"), 50)
    result_false = await judge.validate(_make_value(), _make_context())
    assert result_false is False


@pytest.mark.asyncio
async def test_validate_returns_false_on_llm_failure():
    """If LLM returns (None, 0) → validate returns False."""
    mock_llm = AsyncMock(spec=StructuredLlmManager)
    mock_llm.structured_request.return_value = (None, 0)
    judge = DescriptionJudge(llm_manager=mock_llm)

    result = await judge.validate(_make_value(), _make_context())

    assert result is False
