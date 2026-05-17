"""
WebSearchProducer — 1 LLM call with web_search tool: product identifiers →
textual summary of web-found characteristics.

Produces plain text only.  Structured attribute extraction is done
by a separate extraction step downstream.

Reuses the OpenAI Responses API (same approach as web_search.py) but
with a product-level query rather than a per-feature one.
"""

import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_WS_USER_TEMPLATE = (
    "Find online the technical specifications and characteristics of the following product.\n\n"
    "Product name: {product_name}\n"
    "{brand_line}"
    "{ean_line}"
    "\n"
    "Search manufacturer websites, official retailer pages, or trusted review sites. "
    "Return a plain-text summary of the product characteristics you found "
    "(dimensions, weight, material, colour, capacity, power, etc.). "
    "If you cannot find reliable information, state that explicitly — do NOT invent specs."
)


class WebSearchProducer:
    """1 LLM call with web_search tool: product identifiers → textual summary.

    The returned text is suitable for passing to an extraction step that
    pulls structured attribute values out of it.  This class intentionally
    does NOT return structured attrs — that is a separate concern.
    """

    def __init__(self, llm_manager, model: str = "gpt-4o"):
        """
        Args:
            llm_manager: OpenAIManager instance (provides .client: AsyncOpenAI).
            model: Model that supports the web_search tool (e.g. gpt-4o).
        """
        self.client = llm_manager.client
        self.model = model

    async def produce_summary(
        self,
        product_name: str,
        brand: Optional[str] = None,
        ean: Optional[str] = None,
        timeout: int = 60,
    ) -> Optional[str]:
        """Search the web for product info and return a plain-text summary.

        Returns:
            A plain-text summary string, or None on failure.
        """
        if not product_name:
            logger.debug("WebSearchProducer: no product_name provided, skipping.")
            return None

        brand_line = f"Brand: {brand}\n" if brand else ""
        ean_line = f"EAN / barcode: {ean}\n" if ean else ""

        query = _WS_USER_TEMPLATE.format(
            product_name=product_name,
            brand_line=brand_line,
            ean_line=ean_line,
        )

        try:
            response = await asyncio.wait_for(
                self.client.responses.create(
                    model=self.model,
                    input=query,
                    tools=[{"type": "web_search"}],
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "WebSearchProducer: timed out after %ds for product %r.",
                timeout,
                product_name,
            )
            return None
        except Exception as exc:
            logger.error("WebSearchProducer: Responses API call failed: %s", exc)
            return None

        # Extract plain text answer from the Responses API output.
        answer_text = (getattr(response, "output_text", "") or "").strip()
        if not answer_text:
            logger.warning(
                "WebSearchProducer: empty response for product %r.", product_name
            )
            return None

        logger.debug("WebSearchProducer: produced %d chars.", len(answer_text))
        return answer_text
