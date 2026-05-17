"""Tests for provider factory — config-driven provider selection.

All tests mock config values and provider constructors — no live API calls.
Run with: pytest tests/test_provider_factory.py -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.base import LlmProvider


# ---------------------------------------------------------------------------
# get_main_manager
# ---------------------------------------------------------------------------

class TestGetMainManager:
    def test_returns_openai_manager_when_provider_is_openai(self):
        """When PROVIDER_MAIN=openai, the openai_manager is returned as-is."""
        from app.services.providers.factory import get_main_manager

        with patch("app.services.providers.factory.config") as mock_cfg:
            mock_cfg.PROVIDER_MAIN = "openai"
            mock_cfg.MAIN_MODEL = "deepseek-v4-flash"
            mock_cfg.OPENAI_API_KEY = "sk-fake"

            fake_manager = MagicMock()
            result = get_main_manager(openai_manager=fake_manager)

        assert result is fake_manager

    def test_returns_structured_manager_for_deepseek(self):
        """When PROVIDER_MAIN=deepseek, a StructuredLlmManager wrapping DeepSeekProvider is returned."""
        from app.services.providers.factory import get_main_manager
        from app.services.providers.deepseek_provider import DeepSeekProvider

        with patch("app.services.providers.factory.config") as mock_cfg, \
             patch("app.services.providers.factory.DeepSeekProvider") as MockDS:
            mock_cfg.PROVIDER_MAIN = "deepseek"
            mock_cfg.MAIN_MODEL = "deepseek-v4-flash"
            MockDS.return_value = MagicMock(spec=LlmProvider)

            result = get_main_manager()

        assert isinstance(result, StructuredLlmManager)
        MockDS.assert_called_once()

    def test_returns_structured_manager_for_openrouter(self):
        """When PROVIDER_MAIN=openrouter, a StructuredLlmManager wrapping OpenRouterProvider is returned."""
        from app.services.providers.factory import get_main_manager

        with patch("app.services.providers.factory.config") as mock_cfg, \
             patch("app.services.providers.factory.OpenRouterProvider") as MockOR:
            mock_cfg.PROVIDER_MAIN = "openrouter"
            mock_cfg.MAIN_MODEL = "google/gemini-2.5-flash"
            MockOR.return_value = MagicMock(spec=LlmProvider)

            result = get_main_manager()

        assert isinstance(result, StructuredLlmManager)
        MockOR.assert_called_once()

    def test_raises_for_unknown_provider(self):
        """Unknown PROVIDER_MAIN should raise ValueError."""
        from app.services.providers.factory import get_main_manager

        with patch("app.services.providers.factory.config") as mock_cfg:
            mock_cfg.PROVIDER_MAIN = "unknown_llm"
            mock_cfg.MAIN_MODEL = "x"

            with pytest.raises(ValueError, match="unknown_llm"):
                get_main_manager()


# ---------------------------------------------------------------------------
# get_vision_provider
# ---------------------------------------------------------------------------

class TestGetVisionProvider:
    def test_returns_openrouter_for_openrouter(self):
        """PROVIDER_VISION=openrouter → OpenRouterProvider."""
        from app.services.providers.factory import get_vision_provider
        from app.services.providers.openrouter_provider import OpenRouterProvider

        with patch("app.services.providers.factory.config") as mock_cfg, \
             patch("app.services.providers.factory.OpenRouterProvider") as MockOR:
            mock_cfg.PROVIDER_VISION = "openrouter"
            mock_cfg.VISION_MODEL = "google/gemini-2.5-flash"
            MockOR.return_value = MagicMock(spec=LlmProvider)

            result = get_vision_provider()

        MockOR.assert_called_once()
        assert result is MockOR.return_value

    def test_returns_openai_adapter_for_openai(self):
        """PROVIDER_VISION=openai → OpenAIProviderAdapter wrapping OpenAIManager."""
        from app.services.providers.factory import get_vision_provider
        from app.services.providers.openai_adapter import OpenAIProviderAdapter

        fake_client = MagicMock()
        fake_manager = MagicMock()
        fake_manager.client = fake_client

        with patch("app.services.providers.factory.config") as mock_cfg, \
             patch("app.services.llm_manager.AsyncOpenAI", return_value=fake_client):
            mock_cfg.PROVIDER_VISION = "openai"
            mock_cfg.VISION_MODEL = "gpt-4o"
            mock_cfg.OPENAI_API_KEY = "sk-fake"

            result = get_vision_provider()

        assert isinstance(result, OpenAIProviderAdapter)

    def test_raises_for_unknown_vision_provider(self):
        """Unknown PROVIDER_VISION should raise ValueError."""
        from app.services.providers.factory import get_vision_provider

        with patch("app.services.providers.factory.config") as mock_cfg:
            mock_cfg.PROVIDER_VISION = "anthropic"

            with pytest.raises(ValueError, match="anthropic"):
                get_vision_provider()


# ---------------------------------------------------------------------------
# get_web_search_client
# ---------------------------------------------------------------------------

class TestGetWebSearchClient:
    def test_returns_serper_client_when_serper(self):
        """PROVIDER_WEB_SEARCH=serper → SerperClient."""
        from app.services.providers.factory import get_web_search_client

        with patch("app.services.providers.factory.config") as mock_cfg, \
             patch("app.services.providers.factory.SerperClient") as MockSerper:
            mock_cfg.PROVIDER_WEB_SEARCH = "serper"
            MockSerper.return_value = MagicMock()

            result = get_web_search_client()

        MockSerper.assert_called_once()
        assert result is MockSerper.return_value

    def test_returns_none_when_openai(self):
        """PROVIDER_WEB_SEARCH=openai → None (caller uses existing Responses API)."""
        from app.services.providers.factory import get_web_search_client

        with patch("app.services.providers.factory.config") as mock_cfg:
            mock_cfg.PROVIDER_WEB_SEARCH = "openai"

            result = get_web_search_client()

        assert result is None


# ---------------------------------------------------------------------------
# StructuredLlmManager — unit tests (no live LLM)
# ---------------------------------------------------------------------------

class TestStructuredLlmManager:
    @pytest.mark.asyncio
    async def test_returns_parsed_pydantic_object_on_valid_json(self):
        """Happy path: valid JSON response → parsed Pydantic object + tokens."""
        from unittest.mock import AsyncMock
        from pydantic import BaseModel
        from app.services.providers.base import LlmResponse

        class DummyModel(BaseModel):
            color: str
            weight: float

        fake_response = LlmResponse(
            content='{"color": "red", "weight": 1.5}',
            model="deepseek-v4-flash",
            input_tokens=10,
            output_tokens=20,
            cost_usd=0.001,
            raw={},
        )
        mock_provider = MagicMock(spec=LlmProvider)
        mock_provider.complete = AsyncMock(return_value=fake_response)

        manager = StructuredLlmManager(provider=mock_provider, model="deepseek-v4-flash")
        result, tokens = await manager.structured_request(
            system_prompt="Extract color and weight.",
            user_text="Red item, 1.5kg",
            response_model=DummyModel,
        )

        assert result is not None
        assert result.color == "red"
        assert result.weight == pytest.approx(1.5)
        assert tokens == 30  # 10 + 20

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self):
        """Malformed JSON → returns (None, tokens) without raising."""
        from unittest.mock import AsyncMock
        from pydantic import BaseModel
        from app.services.providers.base import LlmResponse

        class DummyModel(BaseModel):
            color: str

        fake_response = LlmResponse(
            content="not valid json {{{",
            model="deepseek-v4-flash",
            input_tokens=5,
            output_tokens=5,
            cost_usd=0.0,
            raw={},
        )
        mock_provider = MagicMock(spec=LlmProvider)
        mock_provider.complete = AsyncMock(return_value=fake_response)

        manager = StructuredLlmManager(provider=mock_provider, model="deepseek-v4-flash")
        result, tokens = await manager.structured_request(
            system_prompt="sys",
            user_text="user",
            response_model=DummyModel,
        )

        assert result is None
        assert tokens == 10

    @pytest.mark.asyncio
    async def test_returns_none_zero_tokens_on_provider_exception(self):
        """Provider exception → returns (None, 0) without raising."""
        from unittest.mock import AsyncMock
        from pydantic import BaseModel

        class DummyModel(BaseModel):
            color: str

        mock_provider = MagicMock(spec=LlmProvider)
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("API down"))

        manager = StructuredLlmManager(provider=mock_provider, model="deepseek-v4-flash")
        result, tokens = await manager.structured_request(
            system_prompt="sys",
            user_text="user",
            response_model=DummyModel,
        )

        assert result is None
        assert tokens == 0
