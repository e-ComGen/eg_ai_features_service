"""Unit tests for FallbackProvider.

All tests use mocked providers — no live API calls.
Run with: pytest tests/test_fallback_provider.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from app.services.providers.base import LlmProvider, LlmResponse
from app.services.providers.fallback_provider import FallbackProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_provider(name: str) -> MagicMock:
    """Build a mock LlmProvider with a given name."""
    mock = MagicMock(spec=LlmProvider)
    mock.name = name
    return mock


def _make_response(model: str = "test-model") -> LlmResponse:
    return LlmResponse(
        content="ok",
        model=model,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        raw={},
    )


def _make_api_status_error(status_code: int) -> APIStatusError:
    """Build a minimal APIStatusError with the given status code."""
    fake_response = MagicMock()
    fake_response.status_code = status_code
    fake_response.headers = {}
    return APIStatusError(
        message=f"HTTP {status_code}",
        response=fake_response,
        body={"error": {"message": f"HTTP {status_code}"}},
    )


def _make_rate_limit_error() -> RateLimitError:
    fake_response = MagicMock()
    fake_response.status_code = 429
    fake_response.headers = {}
    return RateLimitError(
        message="rate limit exceeded",
        response=fake_response,
        body={"error": {"message": "rate limit exceeded"}},
    )


def _make_connection_error() -> APIConnectionError:
    return APIConnectionError(request=MagicMock())


def _make_timeout_error() -> APITimeoutError:
    return APITimeoutError(request=MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_primary_succeeds_returns_primary_response():
    """When primary succeeds, fallback is never called."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    expected = _make_response("deepseek-v4-flash")
    primary.complete = AsyncMock(return_value=expected)
    fallback.complete = AsyncMock()  # should not be called

    provider = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")
    result = await provider.complete(messages=[], model="deepseek-v4-flash")

    assert result is expected
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_primary_throws_network_error_fallback_called():
    """APIConnectionError on primary triggers fallback; response comes from fallback."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_connection_error())
    fallback_resp = _make_response("gpt-4o-mini")
    fallback.complete = AsyncMock(return_value=fallback_resp)

    provider = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")
    result = await provider.complete(messages=[], model="deepseek-v4-flash")

    fallback.complete.assert_called_once()
    assert "[FALLBACK]" in result.model


@pytest.mark.asyncio
async def test_primary_throws_timeout_fallback_called():
    """APITimeoutError on primary triggers fallback."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_timeout_error())
    fallback.complete = AsyncMock(return_value=_make_response("gpt-4o-mini"))

    provider = FallbackProvider(primary, fallback)
    result = await provider.complete(messages=[], model="deepseek-v4-flash")

    fallback.complete.assert_called_once()
    assert "[FALLBACK]" in result.model


@pytest.mark.asyncio
async def test_primary_throws_rate_limit_fallback_called():
    """RateLimitError (429) on primary triggers fallback."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_rate_limit_error())
    fallback.complete = AsyncMock(return_value=_make_response("gpt-4o-mini"))

    provider = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")
    result = await provider.complete(messages=[], model="deepseek-v4-flash")

    fallback.complete.assert_called_once()
    assert "[FALLBACK]" in result.model


@pytest.mark.asyncio
async def test_primary_throws_5xx_fallback_called():
    """500 Server Error on primary triggers fallback."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_api_status_error(500))
    fallback.complete = AsyncMock(return_value=_make_response("gpt-4o-mini"))

    provider = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")
    result = await provider.complete(messages=[], model="deepseek-v4-flash")

    fallback.complete.assert_called_once()
    assert "[FALLBACK]" in result.model


@pytest.mark.asyncio
async def test_primary_throws_503_fallback_called():
    """503 Service Unavailable also triggers fallback (5xx family)."""
    primary = _make_provider("openrouter")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_api_status_error(503))
    fallback.complete = AsyncMock(return_value=_make_response("gpt-4o-mini"))

    provider = FallbackProvider(primary, fallback)
    await provider.complete(messages=[], model="openai/gpt-4o")

    fallback.complete.assert_called_once()


@pytest.mark.asyncio
async def test_primary_throws_400_no_fallback():
    """400 Bad Request is a client-side error; fallback must NOT be triggered."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_api_status_error(400))
    fallback.complete = AsyncMock()

    provider = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")

    with pytest.raises(APIStatusError) as exc_info:
        await provider.complete(messages=[], model="deepseek-v4-flash")

    assert exc_info.value.status_code == 400
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_primary_throws_401_no_fallback():
    """401 Unauthorized is an auth misconfiguration; fallback must NOT be triggered."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_api_status_error(401))
    fallback.complete = AsyncMock()

    provider = FallbackProvider(primary, fallback)

    with pytest.raises(APIStatusError) as exc_info:
        await provider.complete(messages=[], model="deepseek-v4-flash")

    assert exc_info.value.status_code == 401
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_primary_throws_403_no_fallback():
    """403 Forbidden propagates without fallback."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_api_status_error(403))
    fallback.complete = AsyncMock()

    provider = FallbackProvider(primary, fallback)

    with pytest.raises(APIStatusError) as exc_info:
        await provider.complete(messages=[], model="deepseek-v4-flash")

    assert exc_info.value.status_code == 403
    fallback.complete.assert_not_called()


@pytest.mark.asyncio
async def test_both_throw_propagates():
    """When both primary and fallback fail, the fallback exception is propagated."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_connection_error())
    fallback.complete = AsyncMock(side_effect=RuntimeError("fallback also dead"))

    provider = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")

    with pytest.raises(RuntimeError, match="fallback also dead"):
        await provider.complete(messages=[], model="deepseek-v4-flash")


@pytest.mark.asyncio
async def test_fallback_uses_override_model():
    """fallback_model parameter must be used in the fallback call, not the original model."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_connection_error())
    fallback.complete = AsyncMock(return_value=_make_response("gpt-4o-mini"))

    provider = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")
    await provider.complete(messages=[], model="deepseek-v4-flash")

    call_kwargs = fallback.complete.call_args.kwargs
    assert call_kwargs.get("model") == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_fallback_uses_original_model_when_no_override():
    """When fallback_model is None, the original model string is passed to fallback."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_connection_error())
    fallback.complete = AsyncMock(return_value=_make_response("deepseek-v4-flash"))

    provider = FallbackProvider(primary, fallback, fallback_model=None)
    await provider.complete(messages=[], model="deepseek-v4-flash")

    call_kwargs = fallback.complete.call_args.kwargs
    assert call_kwargs.get("model") == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_response_marked_with_fallback_tag():
    """Fallback response.model must contain '[FALLBACK]' for observability."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    primary.complete = AsyncMock(side_effect=_make_rate_limit_error())
    fallback.complete = AsyncMock(return_value=_make_response("gpt-4o-mini"))

    provider = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")
    result = await provider.complete(messages=[], model="deepseek-v4-flash")

    assert "[FALLBACK]" in result.model
    # Original model name should still be present
    assert "gpt-4o-mini" in result.model


@pytest.mark.asyncio
async def test_provider_name_combines_primary_and_fallback():
    """FallbackProvider.name should reflect both primary and fallback names."""
    primary = _make_provider("deepseek")
    fallback = _make_provider("openai")

    provider = FallbackProvider(primary, fallback)

    assert "deepseek" in provider.name
    assert "openai" in provider.name
    assert "fallback" in provider.name
