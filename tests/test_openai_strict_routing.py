"""Tests for OpenAI strict json_schema routing.

Проверяем:
- response_model с __has_enum_constraints__=True + OPENAI_API_KEY → OpenAI strict manager
- OPENAI_API_KEY отсутствует → fallback на DeepSeek
- нет enum constraints → DeepSeek как прежде
- OpenAIStrictProvider строит правильный strict json_schema payload
- Literal-поле переживает round-trip через _make_strict_schema
"""

from __future__ import annotations

import json
import pytest
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleModel(BaseModel):
    color: str
    weight: float


class _EnumModel(BaseModel):
    color: Literal["red", "blue", "green"]
    size: str


# Имитируем constrained model как OzonStrategy её создаёт
class _ConstrainedModel(_EnumModel):
    __has_enum_constraints__ = True


# ---------------------------------------------------------------------------
# 1. С __has_enum_constraints__=True + ключ → использует OpenAIStrictProvider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_description_source_routes_to_openai_strict_when_flag_set():
    """DescriptionSource должен вызвать OpenAIStrictProvider когда модель с флагом."""
    from app.services.enrichment.sources.description_source import DescriptionSource
    from app.services.enrichment.base import ExtractionContext, TargetAttribute

    mock_main_llm = AsyncMock()
    mock_main_llm.structured_request.return_value = (None, 0)

    mock_strict = AsyncMock()
    # Возвращаем успешный результат от strict provider
    from app.services.enrichment.sources.description_source import _ExtractionResponse
    mock_strict.structured_request.return_value = (_ExtractionResponse(extracted=[]), 10)

    mock_strategy = MagicMock()
    mock_strategy.build_response_model.return_value = _ConstrainedModel

    source = DescriptionSource(llm_manager=mock_main_llm, strategy=mock_strategy)

    ctx = ExtractionContext(
        product_id=1,
        product_name="Test",
        product_description="A red widget, 500g.",
        category_id=10,
        category_path=[],
    )
    targets = [TargetAttribute(id=1, name="Color", type="enum", allowed_values=["red", "blue"])]

    with patch(
        "app.services.enrichment.sources.description_source.get_openai_strict_manager",
        return_value=mock_strict,
    ):
        await source.extract(ctx, targets)

    # strict provider должен был быть вызван, main llm — нет
    mock_strict.structured_request.assert_called_once()
    mock_main_llm.structured_request.assert_not_called()


# ---------------------------------------------------------------------------
# 2. OPENAI_API_KEY не задан → fallback на основной llm
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_description_source_falls_back_when_no_openai_key():
    """Когда get_openai_strict_manager() возвращает None — используем основной LLM."""
    from app.services.enrichment.sources.description_source import DescriptionSource, _ExtractionResponse
    from app.services.enrichment.base import ExtractionContext, TargetAttribute

    mock_main_llm = AsyncMock()
    mock_main_llm.structured_request.return_value = (_ExtractionResponse(extracted=[]), 5)

    mock_strategy = MagicMock()
    mock_strategy.build_response_model.return_value = _ConstrainedModel

    source = DescriptionSource(llm_manager=mock_main_llm, strategy=mock_strategy)

    ctx = ExtractionContext(
        product_id=2,
        product_name="Test",
        product_description="A blue widget.",
        category_id=10,
        category_path=[],
    )
    targets = [TargetAttribute(id=1, name="Color", type="enum", allowed_values=["red", "blue"])]

    # get_openai_strict_manager возвращает None (ключ не задан)
    with patch(
        "app.services.enrichment.sources.description_source.get_openai_strict_manager",
        return_value=None,
    ):
        await source.extract(ctx, targets)

    mock_main_llm.structured_request.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Без enum constraints → всегда DeepSeek
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_description_source_uses_main_llm_when_no_constraints():
    """Модель без __has_enum_constraints__ → вызываем основной LLM, strict не трогаем."""
    from app.services.enrichment.sources.description_source import DescriptionSource, _ExtractionResponse
    from app.services.enrichment.base import ExtractionContext, TargetAttribute

    mock_main_llm = AsyncMock()
    mock_main_llm.structured_request.return_value = (_ExtractionResponse(extracted=[]), 5)

    mock_strategy = MagicMock()
    # Обычная модель без флага
    mock_strategy.build_response_model.return_value = _SimpleModel

    source = DescriptionSource(llm_manager=mock_main_llm, strategy=mock_strategy)

    ctx = ExtractionContext(
        product_id=3,
        product_name="Test",
        product_description="A product description with text.",
        category_id=10,
        category_path=[],
    )
    targets = [TargetAttribute(id=1, name="Color", type="text")]

    mock_strict_factory = MagicMock()
    with patch(
        "app.services.enrichment.sources.description_source.get_openai_strict_manager",
        mock_strict_factory,
    ):
        await source.extract(ctx, targets)

    mock_main_llm.structured_request.assert_called_once()
    # Фабрика strict provider не вызывалась совсем
    mock_strict_factory.assert_not_called()


# ---------------------------------------------------------------------------
# 4. OpenAIStrictProvider строит правильный strict json_schema payload
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_strict_provider_builds_strict_payload():
    """Проверяем что strict=True и additionalProperties=false присутствуют в запросе."""
    from app.services.providers.openai_strict_provider import OpenAIStrictProvider

    captured_kwargs: dict = {}

    async def fake_create(**kwargs):
        captured_kwargs.update(kwargs)
        # Возвращаем минимальный ответ
        resp = MagicMock()
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        resp.choices[0].message.content = '{"color": "red", "size": "M"}'
        resp.model = "gpt-4.1-mini"
        return resp

    provider = OpenAIStrictProvider(api_key="sk-fake", model="gpt-4.1-mini")
    provider._client = MagicMock()
    provider._client.chat.completions.create = fake_create

    result, tokens = await provider.structured_request(
        system_prompt="Extract attributes.",
        user_text="A red medium shirt.",
        response_model=_EnumModel,
    )

    assert "response_format" in captured_kwargs
    rf = captured_kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    schema = rf["json_schema"]["schema"]
    assert schema.get("additionalProperties") is False
    assert "required" in schema


# ---------------------------------------------------------------------------
# 5. Literal-поле переживает round-trip через _make_strict_schema
# ---------------------------------------------------------------------------

def test_make_strict_schema_preserves_enum_literals():
    """_make_strict_schema не должна уничтожать enum/Literal значения в schema."""
    from app.services.providers.openai_strict_provider import _make_strict_schema

    schema = _EnumModel.model_json_schema()
    _make_strict_schema(schema)

    # additionalProperties и required должны быть добавлены
    assert schema.get("additionalProperties") is False
    assert "color" in schema.get("required", [])
    assert "size" in schema.get("required", [])


# ---------------------------------------------------------------------------
# 6. get_openai_strict_manager — возвращает None без ключа, provider с ключом
# ---------------------------------------------------------------------------

def test_get_openai_strict_manager_returns_none_without_key():
    """Без OPENAI_API_KEY фабрика возвращает None."""
    from app.services.providers.factory import get_openai_strict_manager

    with patch("app.services.providers.factory.config") as mock_cfg:
        mock_cfg.OPENAI_API_KEY = ""
        result = get_openai_strict_manager()

    assert result is None


def test_get_openai_strict_manager_returns_provider_with_key():
    """С OPENAI_API_KEY фабрика возвращает OpenAIStrictProvider."""
    from app.services.providers.factory import get_openai_strict_manager
    from app.services.providers.openai_strict_provider import OpenAIStrictProvider

    with patch("app.services.providers.factory.config") as mock_cfg:
        mock_cfg.OPENAI_API_KEY = "sk-test-key"
        mock_cfg.OPENAI_STRUCTURED_MODEL = "gpt-4.1-mini"
        result = get_openai_strict_manager()

    assert isinstance(result, OpenAIStrictProvider)
