"""Base interface for all LLM provider adapters."""
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel


class LlmResponse(BaseModel):
    """Normalised response returned by every LlmProvider."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float  # estimated; OpenRouter returns actual cost, others are hardcoded
    raw: Any  # full provider response dict, kept for debug / billing audit


class LlmProvider(ABC):
    """Abstract base class that every LLM provider adapter must implement."""

    name: str  # human-readable name, e.g. "deepseek", "openrouter"

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2000,
        response_format: Optional[dict] = None,  # e.g. {"type": "json_object"}
        timeout: int = 60,
    ) -> LlmResponse:
        """Send a chat-completion request and return a normalised LlmResponse."""
        ...
