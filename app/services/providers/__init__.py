"""Provider adapters for cheap LLM inference (Tier 1 cost reduction)."""
from .base import LlmProvider, LlmResponse
from .deepseek_provider import DeepSeekProvider
from .openrouter_provider import OpenRouterProvider
from .serper_client import SerperClient
from .openai_adapter import OpenAIProviderAdapter
from .structured_adapter import StructuredLlmManager
from .factory import (
    get_main_manager,
    get_vision_provider,
    get_web_search_client,
    get_extraction_manager,
)

__all__ = [
    "LlmProvider",
    "LlmResponse",
    "DeepSeekProvider",
    "OpenRouterProvider",
    "SerperClient",
    "OpenAIProviderAdapter",
    "StructuredLlmManager",
    "get_main_manager",
    "get_vision_provider",
    "get_web_search_client",
    "get_extraction_manager",
]
