"""
VisionProducer — 1 LLM call: image_urls → textual product description.

Produces plain text only. Structured attribute extraction is done by a
separate extraction step downstream (see job_processor._extract_attrs_from_text).

Provider selection is config-driven (PROVIDER_VISION / VISION_MODEL).
Default: OpenRouter → google/gemini-2.5-flash (cheap, multimodal).
Fallback: PROVIDER_VISION=openai → existing gpt-4o path.
"""

import asyncio
import logging
from typing import Optional

from app import config
from app.services.providers.base import LlmProvider

logger = logging.getLogger(__name__)

# Hard limit: cost / latency control.  Vision tokens are expensive.
MAX_IMAGES = 4

_VISION_SYSTEM_PROMPT = (
    "You are a meticulous product analyst. "
    "Describe the product shown in the image(s) as factually as possible. "
    "Focus on: colour, material (by appearance), shape, approximate size "
    "(only if a reference object is visible), any visible text / markings / logos, "
    "and the product's visible condition. "
    "Be concise and factual — no marketing language, no assumptions beyond what is "
    "visually evident."
)

_VISION_USER_TEMPLATE = (
    "Product name (for context only): {product_name}\n\n"
    "Describe every image in detail following the system instructions. "
    "Combine observations into a single coherent paragraph."
)


class VisionProducer:
    """1 LLM call: image_urls → textual product description.

    The returned text is suitable for passing to an extraction step that
    pulls structured attribute values out of it.  This class intentionally
    does NOT return structured attrs — that is a separate concern.

    Accepts either:
    * An LlmProvider instance (new API — used when PROVIDER_VISION is set)
    * A legacy llm_manager with .client attribute (old OpenAI path — backward compat)
    """

    def __init__(
        self,
        # First positional arg accepts either an LlmProvider or a legacy llm_manager.
        # Legacy tests pass an object with a .client attribute as the first arg.
        provider_or_manager=None,
        model: Optional[str] = None,
    ):
        if provider_or_manager is None:
            # Auto-construct from config (preferred for production)
            from app.services.providers.factory import get_vision_provider
            self._provider = get_vision_provider()
            self._model = model or config.VISION_MODEL
            self._legacy_client = None
        elif isinstance(provider_or_manager, LlmProvider):
            # New path: explicit LlmProvider (OpenRouter, OpenAI adapter, etc.)
            self._provider = provider_or_manager
            self._model = model or config.VISION_MODEL
            self._legacy_client = None
        elif hasattr(provider_or_manager, 'client'):
            # Legacy path: llm_manager with .client (old OpenAI direct call)
            self._provider = None
            self._legacy_client = provider_or_manager.client
            self._model = model or "gpt-4o"
        else:
            raise TypeError(
                f"VisionProducer: expected LlmProvider or llm_manager with .client, "
                f"got {type(provider_or_manager).__name__}"
            )

    async def produce_description(
        self,
        image_urls: list[str],
        product_name: str = "",
        timeout: int = 30,
    ) -> Optional[str]:
        """Send up to MAX_IMAGES image URLs to the vision model.

        Returns:
            A plain-text description string, or None on failure / empty list.
        """
        if not image_urls:
            logger.debug("VisionProducer: no image_urls provided, skipping.")
            return None

        # Cost / latency guard — take only the first MAX_IMAGES.
        urls_to_use = image_urls[:MAX_IMAGES]
        if len(image_urls) > MAX_IMAGES:
            logger.info(
                "VisionProducer: truncated image list from %d to %d.",
                len(image_urls),
                MAX_IMAGES,
            )

        # Build multi-content user message: text + image_url blocks.
        user_content: list[dict] = [
            {
                "type": "text",
                "text": _VISION_USER_TEMPLATE.format(
                    product_name=product_name or "unknown"
                ),
            }
        ]
        for url in urls_to_use:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url, "detail": "low"},
                }
            )

        messages = [
            {"role": "system", "content": _VISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            if self._provider is not None:
                # New path: use LlmProvider.complete()
                # Gemini 2.5 Flash via OpenRouter accepts OpenAI-format image_url content.
                llm_resp = await asyncio.wait_for(
                    self._provider.complete(
                        messages=messages,
                        model=self._model,
                        temperature=0.1,
                        max_tokens=512,
                        timeout=timeout,
                    ),
                    timeout=timeout,
                )
                text = (llm_resp.content or "").strip()
            else:
                # Legacy path: direct AsyncOpenAI client call
                response = await asyncio.wait_for(
                    self._legacy_client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=512,
                    ),
                    timeout=timeout,
                )
                text = (response.choices[0].message.content or "").strip()

            if not text:
                logger.warning("VisionProducer: empty response from model.")
                return None
            logger.debug("VisionProducer: produced %d chars.", len(text))
            return text

        except asyncio.TimeoutError:
            logger.warning(
                "VisionProducer: timed out after %ds for product %r.",
                timeout,
                product_name,
            )
            return None
        except Exception as exc:
            logger.error("VisionProducer: LLM call failed: %s", exc)
            return None
