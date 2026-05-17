"""Thin adapter that wraps the existing OpenAIManager into the LlmProvider interface.

This keeps OpenAI as a valid PROVIDER_MAIN="openai" fallback without changing
any existing OpenAIManager code.  The adapter bridges two different APIs:

* LlmProvider.complete(messages, model, ...) → LlmResponse   (new interface)
* OpenAIManager.structured_request(sys, user, model) → (parsed, tokens)   (old)

For the provider-swap use-case we only need `complete()`, so that's what we
implement here.  structured_request() is left on OpenAIManager and called
directly from AiPipeline / HallucinationJudge when the provider is openai.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import LlmProvider, LlmResponse
from ..llm_manager import OpenAIManager

logger = logging.getLogger(__name__)


class OpenAIProviderAdapter(LlmProvider):
    """Wraps OpenAIManager.client.chat.completions.create into LlmProvider.complete().

    Used when PROVIDER_MAIN="openai" so the factory can return a uniform
    LlmProvider regardless of which backend is selected.
    """

    name = "openai"

    def __init__(self, manager: OpenAIManager) -> None:
        self._manager = manager
        self._client = manager.client

    async def complete(
        self,
        messages: list[dict],
        model: str = "gpt-4o-mini",
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

        response = await self._client.chat.completions.create(**kwargs)

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        content = response.choices[0].message.content or ""

        return LlmResponse(
            content=content,
            model=response.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,  # OpenAI billing tracked externally
            raw=response.model_dump(),
        )

    # ------------------------------------------------------------------
    # Convenience: expose structured_request so callers that hold an
    # OpenAIProviderAdapter can still call structured_request() without
    # knowing the underlying type.
    # ------------------------------------------------------------------
    async def structured_request(self, system_prompt, user_text, response_model):
        """Delegate to the wrapped OpenAIManager.structured_request()."""
        return await self._manager.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=response_model,
        )
