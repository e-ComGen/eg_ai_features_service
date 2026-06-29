"""Provider factory — returns the correct backend based on config settings.

All provider selection is config-driven (PROVIDER_MAIN, PROVIDER_VISION,
PROVIDER_WEB_SEARCH env vars).  Callers import get_main_manager(),
get_vision_provider(), get_web_search_provider() — never instantiate
providers directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app import config
from .base import LlmProvider
from .deepseek_provider import DeepSeekProvider
from .openrouter_provider import OpenRouterProvider
from .serper_client import SerperClient
from .structured_adapter import StructuredLlmManager
from .openai_strict_provider import OpenAIStrictProvider
from .gemini_pdf_provider import GeminiPdfProvider

if TYPE_CHECKING:
    from ..llm_manager import OpenAIManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fallback helper
# ---------------------------------------------------------------------------

def _wrap_with_fallback(primary: LlmProvider, is_vision: bool = False) -> LlmProvider:
    """Optionally wrap *primary* with FallbackProvider pointing at OpenAI.

    Returns *primary* unchanged when:
    - ENABLE_OPENAI_FALLBACK is False
    - primary is already the OpenAI adapter (no point wrapping OpenAI with itself)
    """
    if not config.ENABLE_OPENAI_FALLBACK:
        return primary
    if primary.name == "openai":
        return primary

    from .fallback_provider import FallbackProvider
    from .openai_adapter import OpenAIProviderAdapter
    from ..llm_manager import OpenAIManager

    openai_mgr = OpenAIManager(api_key=config.OPENAI_API_KEY)
    fallback_provider = OpenAIProviderAdapter(openai_mgr)
    model = config.OPENAI_FALLBACK_MODEL_VISION if is_vision else config.OPENAI_FALLBACK_MODEL
    logger.debug(
        "Wrapping %s with FallbackProvider → openai/%s (is_vision=%s)",
        primary.name, model, is_vision,
    )
    return FallbackProvider(primary=primary, fallback=fallback_provider, fallback_model=model)


# ---------------------------------------------------------------------------
# Main LLM — used by AiPipeline (parser, deduction, judge, extraction)
# Returns a StructuredLlmManager that exposes structured_request()
# compatible with the existing pipeline interface.
# ---------------------------------------------------------------------------

def get_main_manager(openai_manager: "OpenAIManager | None" = None) -> "StructuredLlmManager | OpenAIManager":
    """Return a structured_request()-compatible manager for the main pipeline.

    When PROVIDER_MAIN="openai" the existing openai_manager is returned as-is
    (it already has structured_request()).  For other providers, a
    StructuredLlmManager wrapping the selected LlmProvider is returned.

    Args:
        openai_manager: The existing OpenAIManager instance.  Required only
            when PROVIDER_MAIN="openai".
    """
    provider_name = config.PROVIDER_MAIN
    model = config.MAIN_MODEL

    if provider_name == "openai":
        if openai_manager is None:
            # Lazy import to avoid circular dependency
            from ..llm_manager import OpenAIManager
            openai_manager = OpenAIManager(api_key=config.OPENAI_API_KEY)
        logger.info("Main provider: openai (gpt-4o-mini via structured parse)")
        return openai_manager

    provider = _make_raw_provider(provider_name)
    provider = _wrap_with_fallback(provider, is_vision=False)
    logger.info("Main provider: %s, model: %s", provider_name, model)
    return StructuredLlmManager(provider=provider, model=model)


# ---------------------------------------------------------------------------
# Vision LLM — used by VisionProducer
# Returns a plain LlmProvider (complete() only — no structured parse needed)
# ---------------------------------------------------------------------------

def get_vision_provider() -> LlmProvider:
    """Return a vision-capable LlmProvider based on PROVIDER_VISION."""
    provider_name = config.PROVIDER_VISION

    if provider_name == "openrouter":
        logger.info("Vision provider: openrouter, model: %s", config.VISION_MODEL)
        provider = OpenRouterProvider()
        return _wrap_with_fallback(provider, is_vision=True)

    if provider_name == "openai":
        # Wrap OpenAIManager client into a thin LlmProvider
        from ..llm_manager import OpenAIManager
        from .openai_adapter import OpenAIProviderAdapter
        manager = OpenAIManager(api_key=config.OPENAI_API_KEY)
        logger.info("Vision provider: openai (gpt-4o via chat.completions.create)")
        return OpenAIProviderAdapter(manager)

    raise ValueError(
        f"Unknown PROVIDER_VISION={provider_name!r}. "
        "Valid values: 'openrouter', 'openai'."
    )


# ---------------------------------------------------------------------------
# Web search — used by WebSearchProducer
# Returns either SerperClient or None (None means use existing OpenAI path)
# ---------------------------------------------------------------------------

def get_web_search_client() -> "SerperClient | None":
    """Return SerperClient when PROVIDER_WEB_SEARCH='serper', else None.

    When None is returned, WebSearchProducer falls back to the existing
    OpenAI Responses API path (web_search.py).

    TODO: add Serper → OpenAI web_search fallback when Serper is unavailable.
    """
    if config.PROVIDER_WEB_SEARCH == "serper":
        logger.info("Web search provider: serper")
        return SerperClient()
    logger.info("Web search provider: openai (existing Responses API)")
    return None


# ---------------------------------------------------------------------------
# Router LLM — used by TreeRouter for traversal decisions
# By default shares PROVIDER_MAIN; can be overridden via PROVIDER_ROUTER env var.
# ---------------------------------------------------------------------------

def get_router_manager() -> "StructuredLlmManager":
    """Return a StructuredLlmManager for TreeRouter traversal decisions.

    Uses PROVIDER_MAIN backend unless PROVIDER_ROUTER is set explicitly.
    Always returns a StructuredLlmManager (never a bare OpenAIManager).
    Fallback wrapping applied identically to the main manager.
    """
    provider_name = config.PROVIDER_ROUTER
    model = config.ROUTER_MODEL

    if provider_name == "openai":
        from ..llm_manager import OpenAIManager
        from .openai_adapter import OpenAIProviderAdapter
        mgr = OpenAIManager(api_key=config.OPENAI_API_KEY)
        adapter = OpenAIProviderAdapter(mgr)
        return StructuredLlmManager(provider=adapter, model=model)

    provider = _make_raw_provider(provider_name)
    provider = _wrap_with_fallback(provider, is_vision=False)
    logger.info("Router provider: %s, model: %s", provider_name, model)
    return StructuredLlmManager(provider=provider, model=model)


# ---------------------------------------------------------------------------
# Extraction LLM — used for text extracted from vision/web search results
# (same provider as main but model may differ)
# ---------------------------------------------------------------------------

def get_extraction_manager() -> "StructuredLlmManager | OpenAIManager":
    """Return a structured_request()-compatible manager for post-enrichment extraction.

    Uses PROVIDER_MAIN backend but EXTRACTION_FROM_TEXT_MODEL model.
    """
    provider_name = config.PROVIDER_MAIN
    model = config.EXTRACTION_FROM_TEXT_MODEL

    if provider_name == "openai":
        from ..llm_manager import OpenAIManager
        return OpenAIManager(api_key=config.OPENAI_API_KEY)

    provider = _make_raw_provider(provider_name)
    provider = _wrap_with_fallback(provider, is_vision=False)
    return StructuredLlmManager(provider=provider, model=model)


# ---------------------------------------------------------------------------
# OpenAI strict json_schema — для enum-heavy structured extraction
# ---------------------------------------------------------------------------

def get_openai_strict_manager() -> "OpenAIStrictProvider | None":
    """Вернуть OpenAIStrictProvider если OPENAI_API_KEY задан, иначе None.

    Используется sources для маршрутизации вызовов с __has_enum_constraints__=True
    через OpenAI gpt-4.1-mini strict mode вместо DeepSeek JSON mode.
    """
    if not config.USE_OPENAI_STRICT:
        logger.debug("get_openai_strict_manager: USE_OPENAI_STRICT=false, returning None (DeepSeek fallback)")
        return None
    if not config.OPENAI_API_KEY:
        logger.debug("get_openai_strict_manager: OPENAI_API_KEY not set, returning None")
        return None
    return OpenAIStrictProvider(
        api_key=config.OPENAI_API_KEY,
        model=config.OPENAI_STRUCTURED_MODEL,
    )


# ---------------------------------------------------------------------------
# Gemini PDF — native PDF input для PdfDatasheetSource
# ---------------------------------------------------------------------------

def get_gemini_pdf_provider() -> "GeminiPdfProvider | None":
    """Return GeminiPdfProvider если OPEN_ROUTER_API_KEY задан, иначе None."""
    if not config.OPENROUTER_API_KEY:
        logger.debug("get_gemini_pdf_provider: OPEN_ROUTER_API_KEY not set, returning None")
        return None
    return GeminiPdfProvider(
        api_key=config.OPENROUTER_API_KEY,
        model=config.VISION_MODEL,
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _make_raw_provider(provider_name: str) -> LlmProvider:
    if provider_name == "deepseek":
        return DeepSeekProvider()
    if provider_name == "openrouter":
        return OpenRouterProvider()
    raise ValueError(
        f"Unknown provider {provider_name!r}. "
        "Valid values: 'deepseek', 'openrouter', 'openai'."
    )
