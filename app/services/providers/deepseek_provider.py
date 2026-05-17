"""DeepSeek V4 provider using the OpenAI-compatible API.

DeepSeek implements the same REST protocol as OpenAI, so we use the official
``openai`` SDK with a custom ``base_url``.  This gives us structured outputs
(JSON mode) and streaming for free.

Official pricing (api.deepseek.com, май 2026):
    deepseek-v4-flash:
        input (cache miss): $0.14 / 1M tokens
        input (cache hit):  $0.0028 / 1M tokens  (98% discount)
        output:             $0.28 / 1M tokens

    deepseek-v4-pro (PROMO до 31.05.2026, после ×4):
        input (cache miss): $0.435 / 1M tokens   (после promo: $1.74)
        input (cache hit):  $0.003625 / 1M tokens (после promo: $0.0145)
        output:             $0.87 / 1M tokens    (после promo: $3.48)

Через OpenRouter v4-flash дешевле: $0.112 / $0.224 (~20% объёмная скидка).
Source: https://api-docs.deepseek.com/quick_start/pricing
"""

from __future__ import annotations

import logging
from typing import Optional

from openai import AsyncOpenAI, RateLimitError, APIStatusError

from app import config
from .base import LlmProvider, LlmResponse

logger = logging.getLogger(__name__)

# Hardcoded price table (USD per 1 million tokens) — официальные цены DeepSeek май 2026.
# Cache hit pricing применяется автоматически когда DeepSeek распознаёт identical prefix.
# ⚠️ V4-Pro в промо до 31.05.2026 — после цена вырастет ×4. Проверить и обновить.
_PRICES: dict[str, dict[str, float]] = {
    # Direct V4 names — на 17.05.2026 API возвращает HTTP 200 но пустой content.
    # Возможно V4 в beta/preview. Используем алиасы которые маршрутизируются на V4 моделях
    # до отключения алиасов 24.07.2026.
    "deepseek-v4-flash": {"input": 0.14, "output": 0.28},
    "deepseek-v4-pro":   {"input": 0.435, "output": 0.87},  # промо до 31.05.2026
    # Production aliases (РАБОТАЮТ, default в config):
    "deepseek-chat":     {"input": 0.14, "output": 0.28},  # → routes to V4-flash
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},  # V3.1 reasoning model
}
_DEFAULT_PRICE = {"input": 0.14, "output": 0.28}  # fallback (= flash, безопаснее занизить)


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = _PRICES.get(model, _DEFAULT_PRICE)
    return (
        input_tokens * prices["input"] / 1_000_000
        + output_tokens * prices["output"] / 1_000_000
    )


class DeepSeekProvider(LlmProvider):
    """Adapter for DeepSeek V4 via api.deepseek.com (OpenAI-compatible)."""

    name = "deepseek"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or config.DEEPSEEK_API_KEY
        if not self._api_key:
            raise ValueError(
                "DeepSeek API key is not set. "
                "Set DEEP_SEEK_API_KEY in your .env file."
            )
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url="https://api.deepseek.com/v1",
        )

    async def complete(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,
        timeout: int = 60,
    ) -> LlmResponse:
        model = model or config.DEEPSEEK_DEFAULT_MODEL

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
            logger.warning("DeepSeek rate-limited (429): %s", exc)
            raise
        except APIStatusError as exc:
            logger.error("DeepSeek API error %s: %s", exc.status_code, exc.message)
            raise

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        content = response.choices[0].message.content or ""

        return LlmResponse(
            content=content,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_estimate_cost(model, input_tokens, output_tokens),
            raw=response.model_dump(),
        )
