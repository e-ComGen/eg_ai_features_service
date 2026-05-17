"""FallbackProvider — wraps a primary LlmProvider and falls back to a secondary
on transient errors (network issues, timeouts, rate limits, 5xx responses).

Does NOT fallback on:
- 400 Bad Request (our bug, not the provider's fault)
- 401 Unauthorized (auth misconfiguration)
- 403 Forbidden
- Pydantic validation errors (our schema problem)
"""

from __future__ import annotations

import logging

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from .base import LlmProvider, LlmResponse

logger = logging.getLogger(__name__)

# Errors that TRIGGER fallback (transient / availability problems)
FALLBACK_ERRORS = (
    APIConnectionError,      # network problem
    APITimeoutError,         # timeout
    RateLimitError,          # 429
    httpx.HTTPError,         # generic HTTP issues (covers HTTPStatusError too)
    httpx.TimeoutException,
    httpx.NetworkError,
)


def _is_server_error(exc: Exception) -> bool:
    """Return True if exc is an APIStatusError with a 5xx status code."""
    if isinstance(exc, APIStatusError):
        return 500 <= exc.status_code < 600
    return False


class FallbackProvider(LlmProvider):
    """Wraps a primary LlmProvider and automatically falls back to a secondary
    provider when transient errors occur.

    Usage::

        primary = DeepSeekProvider()
        fallback = OpenAIProviderAdapter(OpenAIManager(api_key=...))
        provider = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")

    The fallback response is tagged with "[FALLBACK]" in response.model for
    observability / billing attribution.
    """

    def __init__(
        self,
        primary: LlmProvider,
        fallback: LlmProvider,
        fallback_model: str | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        # When set, overrides the model name used on the fallback call.
        # Useful when the fallback provider uses different model IDs
        # (e.g. "gpt-4o-mini" instead of "deepseek-v4-flash").
        self.fallback_model = fallback_model
        self.name = f"{primary.name}+fallback({fallback.name})"

    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: dict | None = None,
        timeout: int = 60,
    ) -> LlmResponse:
        """Attempt primary provider; on transient error switch to fallback."""
        _should_fallback = False

        try:
            return await self.primary.complete(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=timeout,
            )
        except FALLBACK_ERRORS as exc:
            logger.warning(
                "[FALLBACK] Primary provider %s failed (%s: %s), switching to %s",
                self.primary.name,
                type(exc).__name__,
                exc,
                self.fallback.name,
            )
            _should_fallback = True
        except APIStatusError as exc:
            if _is_server_error(exc):
                logger.warning(
                    "[FALLBACK] Primary %s returned 5xx (%d), switching to %s",
                    self.primary.name,
                    exc.status_code,
                    self.fallback.name,
                )
                _should_fallback = True
            else:
                # 4xx (400, 401, 403, …) — caller's fault, propagate immediately.
                raise

        if not _should_fallback:
            # Should not reach here, but defensive guard.
            raise RuntimeError("FallbackProvider: unexpected state — no response and no fallback trigger")

        fallback_model_to_use = self.fallback_model or model
        try:
            response = await self.fallback.complete(
                messages=messages,
                model=fallback_model_to_use,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=timeout,
            )
            # Tag the response so observability / logs can detect fallback usage.
            response.model = f"{response.model} [FALLBACK]"
            return response
        except Exception as exc:
            logger.error(
                "[FALLBACK] BOTH providers failed. Primary %s → fallback %s: %s",
                self.primary.name,
                self.fallback.name,
                exc,
            )
            raise
