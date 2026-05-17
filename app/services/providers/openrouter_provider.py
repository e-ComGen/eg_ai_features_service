"""OpenRouter provider — access any model through one OpenAI-compatible endpoint.

Supported model_id examples:
    google/gemini-2.5-flash
    anthropic/claude-sonnet-4.6
    openai/gpt-4o
    deepseek/deepseek-chat-v3.2

OpenRouter returns the actual incurred cost in ``response.usage`` as
``total_cost`` (USD), so we use that directly instead of hardcoded prices.
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI, RateLimitError, APIStatusError

from app import config
from .base import LlmProvider, LlmResponse

logger = logging.getLogger(__name__)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(LlmProvider):
    """Adapter for OpenRouter (multi-model gateway, OpenAI-compatible)."""

    name = "openrouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        site_url: str = "https://github.com/e-comgen/cpAiFeatures",
        app_name: str = "e-comgen AI attributes",
    ) -> None:
        self._api_key = api_key or config.OPENROUTER_API_KEY
        if not self._api_key:
            raise ValueError(
                "OpenRouter API key is not set. "
                "Set OPEN_ROUTER_API_KEY in your .env file."
            )
        # OpenRouter recommends passing site URL + app name as extra headers
        # for better rate-limit tiers and usage attribution.
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=_OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
        )

    async def complete(
        self,
        messages: list[dict],
        model: str = "google/gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
        timeout: int = 60,
    ) -> LlmResponse:
        kwargs: dict = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        if response_format:
            kwargs["response_format"] = response_format

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            logger.warning("OpenRouter rate-limited (429): %s", exc)
            raise
        except APIStatusError as exc:
            logger.error(
                "OpenRouter API error %s: %s", exc.status_code, exc.message
            )
            raise

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # OpenRouter may return actual cost via usage extensions.
        # The field is non-standard; fall back to 0.0 if absent.
        raw_dict = response.model_dump()
        cost_usd: float = 0.0
        try:
            cost_usd = float(
                raw_dict.get("usage", {}).get("total_cost", 0.0) or 0.0
            )
        except (TypeError, ValueError):
            pass

        content = response.choices[0].message.content or ""

        return LlmResponse(
            content=content,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            raw=raw_dict,
        )
