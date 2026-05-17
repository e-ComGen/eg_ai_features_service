"""Unit tests for LLM provider adapters and Serper client.

All tests use mocked HTTP — no live API calls are made.
Run with:  pytest tests/test_providers.py -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import RateLimitError, APIStatusError

from app.services.providers.base import LlmResponse
from app.services.providers.deepseek_provider import DeepSeekProvider
from app.services.providers.openrouter_provider import OpenRouterProvider
from app.services.providers.serper_client import SerperClient, SerperResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_response(
    content: str = "test output",
    model: str = "deepseek-v4-flash",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
    extra_usage: dict | None = None,
) -> MagicMock:
    """Build a fake openai ChatCompletion response object."""
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    raw_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    if extra_usage:
        raw_usage.update(extra_usage)

    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    resp = MagicMock()
    resp.model = model
    resp.usage = usage
    resp.choices = [choice]
    resp.model_dump.return_value = {
        "model": model,
        "usage": raw_usage,
        "choices": [{"message": {"content": content}}],
    }
    return resp


# ---------------------------------------------------------------------------
# DeepSeek provider tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deepseek_provider_completion():
    """Happy-path: verify request payload and normalised LlmResponse fields."""
    fake_response = _make_openai_response(
        content='{"color": "red"}',
        model="deepseek-v4-flash",
        prompt_tokens=50,
        completion_tokens=10,
    )

    with patch("app.services.providers.deepseek_provider.config") as mock_cfg, \
         patch("app.services.providers.deepseek_provider.AsyncOpenAI") as MockClient:

        mock_cfg.DEEPSEEK_API_KEY = "ds-fake-key"
        mock_cfg.DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"

        instance = MockClient.return_value
        instance.chat = MagicMock()
        instance.chat.completions = MagicMock()
        instance.chat.completions.create = AsyncMock(return_value=fake_response)

        provider = DeepSeekProvider(api_key="ds-fake-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "hello"}],
            model="deepseek-v4-flash",
            response_format={"type": "json_object"},
        )

    assert isinstance(result, LlmResponse)
    assert result.content == '{"color": "red"}'
    assert result.model == "deepseek-v4-flash"
    assert result.input_tokens == 50
    assert result.output_tokens == 10
    assert result.cost_usd > 0  # price estimated from hardcoded table

    # Verify create was called with the right arguments
    call_kwargs = instance.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-v4-flash"
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_deepseek_handles_429():
    """Provider should re-raise RateLimitError on 429 without swallowing it."""
    with patch("app.services.providers.deepseek_provider.config") as mock_cfg, \
         patch("app.services.providers.deepseek_provider.AsyncOpenAI") as MockClient:

        mock_cfg.DEEPSEEK_API_KEY = "ds-fake-key"
        mock_cfg.DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-flash"

        # Build a minimal RateLimitError (openai SDK wraps the raw httpx response)
        fake_response = MagicMock()
        fake_response.status_code = 429
        fake_response.headers = {}
        rate_limit_exc = RateLimitError(
            message="rate limit exceeded",
            response=fake_response,
            body={"error": {"message": "rate limit exceeded"}},
        )

        instance = MockClient.return_value
        instance.chat = MagicMock()
        instance.chat.completions = MagicMock()
        instance.chat.completions.create = AsyncMock(side_effect=rate_limit_exc)

        provider = DeepSeekProvider(api_key="ds-fake-key")

        with pytest.raises(RateLimitError):
            await provider.complete(
                messages=[{"role": "user", "content": "hi"}],
                model="deepseek-v4-flash",
            )


@pytest.mark.asyncio
async def test_deepseek_missing_api_key_raises():
    """Instantiating without an API key should raise ValueError immediately."""
    with patch("app.services.providers.deepseek_provider.config") as mock_cfg:
        mock_cfg.DEEPSEEK_API_KEY = ""
        with pytest.raises(ValueError, match="DEEP_SEEK_API_KEY"):
            DeepSeekProvider(api_key="")


# ---------------------------------------------------------------------------
# OpenRouter provider tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_openrouter_provider_completion():
    """Verify that the model_id is forwarded verbatim to the API call."""
    fake_response = _make_openai_response(
        content="gemini says hello",
        model="google/gemini-2.5-flash",
        prompt_tokens=30,
        completion_tokens=15,
        extra_usage={"total_cost": 0.000042},
    )

    with patch("app.services.providers.openrouter_provider.config") as mock_cfg, \
         patch("app.services.providers.openrouter_provider.AsyncOpenAI") as MockClient:

        mock_cfg.OPENROUTER_API_KEY = "or-fake-key"

        instance = MockClient.return_value
        instance.chat = MagicMock()
        instance.chat.completions = MagicMock()
        instance.chat.completions.create = AsyncMock(return_value=fake_response)

        provider = OpenRouterProvider(api_key="or-fake-key")
        result = await provider.complete(
            messages=[{"role": "user", "content": "ping"}],
            model="google/gemini-2.5-flash",
        )

    assert isinstance(result, LlmResponse)
    assert result.content == "gemini says hello"
    assert result.model == "google/gemini-2.5-flash"
    assert result.input_tokens == 30
    assert result.output_tokens == 15
    # cost parsed from total_cost field
    assert result.cost_usd == pytest.approx(0.000042, rel=1e-3)

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "google/gemini-2.5-flash"


@pytest.mark.asyncio
async def test_openrouter_forwards_response_format():
    """response_format kwarg must be passed through to the API call."""
    fake_response = _make_openai_response(model="openai/gpt-4o")

    with patch("app.services.providers.openrouter_provider.config") as mock_cfg, \
         patch("app.services.providers.openrouter_provider.AsyncOpenAI") as MockClient:

        mock_cfg.OPENROUTER_API_KEY = "or-fake-key"

        instance = MockClient.return_value
        instance.chat = MagicMock()
        instance.chat.completions = MagicMock()
        instance.chat.completions.create = AsyncMock(return_value=fake_response)

        provider = OpenRouterProvider(api_key="or-fake-key")
        await provider.complete(
            messages=[{"role": "user", "content": "json please"}],
            model="openai/gpt-4o",
            response_format={"type": "json_object"},
        )

    call_kwargs = instance.chat.completions.create.call_args.kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}


# ---------------------------------------------------------------------------
# Serper client tests
# ---------------------------------------------------------------------------

_SERPER_FIXTURE = {
    "organic": [
        {
            "title": "Example Product",
            "link": "https://example.com/product",
            "snippet": "A great product with features.",
            "position": 1,
        },
        {
            "title": "Another Result",
            "link": "https://example.com/another",
            "snippet": "More info here.",
            "position": 2,
        },
    ],
    "knowledgeGraph": {"title": "Example", "type": "Product"},
    "relatedSearches": [
        {"query": "example product review"},
        {"query": "buy example product"},
    ],
}


@pytest.mark.asyncio
async def test_serper_search_returns_results():
    """Happy-path: verify request body and parsed SerperResults."""
    import httpx
    import respx

    with patch("app.services.providers.serper_client.config") as mock_cfg:
        mock_cfg.SERPER_API_KEY = "serper-fake-key"

        with respx.mock() as mock_serper:
            mock_serper.post("https://google.serper.dev/search").mock(
                return_value=httpx.Response(200, json=_SERPER_FIXTURE)
            )

            client = SerperClient(api_key="serper-fake-key", gl="ru", hl="ru")
            results = await client.search("test query", num_results=5)

            assert isinstance(results, SerperResults)
            assert results.query == "test query"
            assert len(results.organic_results) == 2
            assert results.organic_results[0].title == "Example Product"
            assert results.organic_results[0].link == "https://example.com/product"
            assert results.organic_results[0].position == 1
            assert results.knowledge_graph == {"title": "Example", "type": "Product"}
            assert "example product review" in results.related_searches

            # Verify the outgoing request body
            request = mock_serper.calls.last.request
            body = json.loads(request.content)
            assert body["q"] == "test query"
            assert body["num"] == 5
            assert body["gl"] == "ru"
            assert body["hl"] == "ru"

            # Verify auth header
            assert request.headers["x-api-key"] == "serper-fake-key"


@pytest.mark.asyncio
async def test_serper_handles_quota_exceeded():
    """429 from Serper should raise httpx.HTTPStatusError."""
    import httpx
    import respx

    with patch("app.services.providers.serper_client.config") as mock_cfg:
        mock_cfg.SERPER_API_KEY = "serper-fake-key"

        with respx.mock() as mock_serper:
            mock_serper.post("https://google.serper.dev/search").mock(
                return_value=httpx.Response(429, text="Too Many Requests")
            )

            client = SerperClient(api_key="serper-fake-key")

            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await client.search("quota test", num_results=3)

            assert exc_info.value.response.status_code == 429


@pytest.mark.asyncio
async def test_serper_missing_api_key_raises():
    """Instantiating SerperClient without an API key should raise ValueError."""
    with patch("app.services.providers.serper_client.config") as mock_cfg:
        mock_cfg.SERPER_API_KEY = ""
        with pytest.raises(ValueError, match="SERPER_API_KEY"):
            SerperClient(api_key="")
