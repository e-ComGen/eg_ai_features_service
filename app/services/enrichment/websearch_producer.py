"""
WebSearchProducer — product identifiers → textual summary of web-found characteristics.

Produces plain text only.  Structured attribute extraction is done
by a separate extraction step downstream.

Provider selection is config-driven (PROVIDER_WEB_SEARCH):
  "serper"  → Serper Google Search API + LLM extraction (cheap, ~$0.001/query)
  "openai"  → existing OpenAI Responses API with web_search tool (fallback)
"""

import asyncio
import logging
from typing import Optional

from app import config

logger = logging.getLogger(__name__)

_WS_USER_TEMPLATE = (
    "Find online the technical specifications and characteristics of the following product.\n\n"
    "Product name: {product_name}\n"
    "{brand_line}"
    "{ean_line}"
    "\n"
    "Search manufacturer websites, official retailer pages, or trusted review sites. "
    "Return a plain-text summary of the product characteristics you found "
    "(dimensions, weight, material, colour, capacity, power, etc.; "
    "for apparel/footwear/textiles also include: fabric/material composition, season, "
    "fit/cut/style, care instructions if available). "
    "If you cannot find reliable information, state that explicitly — do NOT invent specs."
)

_SERPER_EXTRACTION_SYSTEM = (
    "You are a product data extractor. "
    "Below are web search snippets about a specific product. "
    "Extract and summarise the technical characteristics and specifications you find "
    "(dimensions, weight, material, colour, capacity, power, connectivity, etc.; "
    "for apparel/footwear/textiles also include: fabric/material composition, season, "
    "fit/cut/style, care instructions, gender, and age group if mentioned). "
    "Write a concise plain-text summary. "
    "Do NOT invent any specs not present in the snippets. "
    "If the snippets don't contain useful data, say so explicitly."
)


class WebSearchProducer:
    """product identifiers → plain-text summary of web-found characteristics.

    The returned text is suitable for passing to an extraction step that
    pulls structured attribute values out of it.  This class intentionally
    does NOT return structured attrs — that is a separate concern.

    Accepts:
    * provider=None, serper_client=None → auto-detect from config
    * serper_client + extractor_manager → Serper path
    * llm_manager (legacy) → OpenAI Responses API path
    """

    def __init__(
        self,
        # Legacy param: llm_manager with .client (OpenAI Responses API).
        # When provided explicitly, always uses the legacy OpenAI Responses API path,
        # regardless of config.PROVIDER_WEB_SEARCH (backward compat for tests/old callers).
        llm_manager=None,
        model: str = "gpt-4o",
        # New params: explicit provider objects (used in tests / production)
        serper_client=None,
        extractor_manager=None,
    ):
        if serper_client is not None:
            # Explicitly injected Serper client (e.g. from tests)
            self._serper = serper_client
            self._extractor = extractor_manager
            self._use_serper = True
            self._legacy_client = None
            self._legacy_model = model
        elif llm_manager is not None:
            # Legacy path: explicit llm_manager — always use OpenAI Responses API.
            # This preserves backward compatibility (existing tests, old callers).
            self._legacy_client = llm_manager.client
            self._legacy_model = model
            self._use_serper = False
            self._serper = None
            self._extractor = None
        elif config.PROVIDER_WEB_SEARCH == "serper":
            # Auto-construct Serper path from config
            from app.services.providers.factory import get_web_search_client, get_main_manager
            self._serper = get_web_search_client()
            self._extractor = get_main_manager()
            self._use_serper = True
            self._legacy_client = None
            self._legacy_model = model
        else:
            # Auto-construct legacy path (OpenAI Responses API)
            from app.services.llm_manager import OpenAIManager
            mgr = OpenAIManager(api_key=config.OPENAI_API_KEY)
            self._legacy_client = mgr.client
            self._legacy_model = model
            self._use_serper = False
            self._serper = None
            self._extractor = None

    async def produce_summary(
        self,
        product_name: str,
        brand: Optional[str] = None,
        ean: Optional[str] = None,
        mpn: Optional[str] = None,
        timeout: int = 60,
    ) -> Optional[str]:
        """Search the web for product info and return a plain-text summary.

        Returns:
            A plain-text summary string, or None on failure.
        """
        if not product_name:
            logger.debug("WebSearchProducer: no product_name provided, skipping.")
            return None

        if self._use_serper:
            return await self._produce_via_serper(product_name, brand, ean, mpn, timeout)
        else:
            return await self._produce_via_openai(product_name, brand, ean, mpn, timeout)

    # ------------------------------------------------------------------
    # Serper path: Google search + LLM extraction
    # ------------------------------------------------------------------
    async def _produce_via_serper(
        self,
        product_name: str,
        brand: Optional[str],
        ean: Optional[str],
        mpn: Optional[str],
        timeout: int,
    ) -> Optional[str]:
        # Build search query — MPN first (highest signal: точный код производителя),
        # then product_name + brand + ean (любые secondary identifiers).
        parts: list[str] = []
        if mpn:
            parts.append(mpn)
        parts.append(product_name)
        if brand:
            parts.append(brand)
        if ean:
            parts.append(ean)
        query = " ".join(parts) + " характеристики технические"

        try:
            results = await asyncio.wait_for(
                self._serper.search(query, num_results=5),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "WebSearchProducer (serper): search timed out after %ds for %r.",
                timeout,
                product_name,
            )
            return None
        except Exception as exc:
            logger.error("WebSearchProducer (serper): search failed: %s", exc)
            return None

        if not results.organic_results:
            logger.warning(
                "WebSearchProducer (serper): no organic results for %r.", product_name
            )
            return None

        # --- Full-page fetch: top-1-2 HTTPS links from organic results ---
        _PAGE_FETCH_TIMEOUT = 20      # seconds total for all page fetches
        _PAGE_TEXT_CAP = 6000         # chars to keep per product (LLM context guard)

        top_urls = [
            r.link
            for r in results.organic_results[:5]
            if getattr(r, "link", None) and str(r.link).startswith("https://")
        ][:2]

        page_section = ""
        if top_urls:
            try:
                from app.services.url_fetcher import fetch_all as _fetch_all
                raw_page_text = await asyncio.wait_for(
                    _fetch_all(top_urls),
                    timeout=_PAGE_FETCH_TIMEOUT,
                )
                if raw_page_text:
                    # Каталог-страницы (asus.com/techspec и т.п.) повторяют спеки
                    # для каждого SKU линейки 5-10 раз → boilerplate вытесняет
                    # ключевые спеки за cap. Дедуп строк (order-preserving) даёт
                    # ~38% сжатия, ключевое (USB/Wi-Fi/HDMI) влезает в окно.
                    raw_page_text = "\n".join(dict.fromkeys(raw_page_text.split("\n")))
                    page_section = (
                        "=== Текст страницы со спецификациями ===\n"
                        + raw_page_text[:_PAGE_TEXT_CAP]
                        + "\n"
                    )
                    logger.debug(
                        "WebSearchProducer (serper): fetched %d chars from %d page(s) for %r.",
                        len(raw_page_text),
                        len(top_urls),
                        product_name,
                    )
            except asyncio.TimeoutError:
                logger.warning(
                    "WebSearchProducer (serper): page fetch timed out (%ds) for %r — using snippets only.",
                    _PAGE_FETCH_TIMEOUT,
                    product_name,
                )
            except Exception as exc:
                logger.warning(
                    "WebSearchProducer (serper): page fetch failed for %r: %s — using snippets only.",
                    product_name,
                    exc,
                )
        # -------------------------------------------------------------------

        # Build context from top snippets
        snippets = [
            f"[{r.position}] {r.title}\n{r.snippet}"
            for r in results.organic_results[:5]
        ]
        search_context = page_section + "\n\n".join(snippets)

        # 1 LLM call: extract characteristics from snippets
        user_text = (
            f"Товар: {product_name}"
            + (f"\nБренд: {brand}" if brand else "")
            + (f"\nMPN: {mpn}" if mpn else "")
            + (f"\nEAN: {ean}" if ean else "")
            + f"\n\nНайденные фрагменты:\n{search_context}"
        )

        if self._extractor is None:
            logger.error(
                "WebSearchProducer (serper): extractor_manager is None, cannot extract."
            )
            return None

        try:
            # extractor_manager exposes structured_request() OR complete()
            if hasattr(self._extractor, "complete"):
                from app.services.providers.base import LlmProvider
                if isinstance(self._extractor, LlmProvider):
                    llm_resp = await asyncio.wait_for(
                        self._extractor.complete(
                            messages=[
                                {"role": "system", "content": _SERPER_EXTRACTION_SYSTEM},
                                {"role": "user", "content": user_text},
                            ],
                            model=config.EXTRACTION_FROM_TEXT_MODEL,
                            temperature=0.1,
                            max_tokens=1024,
                        ),
                        timeout=timeout,
                    )
                    text = (llm_resp.content or "").strip()
                    if not text:
                        logger.warning(
                            "WebSearchProducer (serper): empty extraction result for %r.",
                            product_name,
                        )
                        return None
                    return text

            # StructuredLlmManager / OpenAIManager path — use complete() via provider
            # Fall through to direct provider call
            from app.services.providers.factory import _make_raw_provider
            from app import config as _cfg
            raw_provider = _make_raw_provider(_cfg.PROVIDER_MAIN)
            llm_resp = await asyncio.wait_for(
                raw_provider.complete(
                    messages=[
                        {"role": "system", "content": _SERPER_EXTRACTION_SYSTEM},
                        {"role": "user", "content": user_text},
                    ],
                    model=config.EXTRACTION_FROM_TEXT_MODEL,
                    temperature=0.1,
                    max_tokens=1024,
                ),
                timeout=timeout,
            )
            text = (llm_resp.content or "").strip()
            if not text:
                return None
            return text

        except asyncio.TimeoutError:
            logger.warning(
                "WebSearchProducer (serper): extraction timed out for %r.", product_name
            )
            return None
        except Exception as exc:
            logger.error("WebSearchProducer (serper): extraction failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Legacy path: OpenAI Responses API with web_search tool
    # ------------------------------------------------------------------
    async def _produce_via_openai(
        self,
        product_name: str,
        brand: Optional[str],
        ean: Optional[str],
        mpn: Optional[str],
        timeout: int,
    ) -> Optional[str]:
        brand_line = f"Brand: {brand}\n" if brand else ""
        # Кладём MPN в product_name строку чтобы не ломать существующий template.
        product_line = product_name
        if mpn:
            product_line = f"{product_name} (MPN: {mpn})"
        ean_line = f"EAN / barcode: {ean}\n" if ean else ""

        query = _WS_USER_TEMPLATE.format(
            product_name=product_line,
            brand_line=brand_line,
            ean_line=ean_line,
        )

        try:
            response = await asyncio.wait_for(
                self._legacy_client.responses.create(
                    model=self._legacy_model,
                    input=query,
                    tools=[{"type": "web_search"}],
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "WebSearchProducer (openai): timed out after %ds for product %r.",
                timeout,
                product_name,
            )
            return None
        except Exception as exc:
            logger.error("WebSearchProducer (openai): Responses API call failed: %s", exc)
            return None

        answer_text = (getattr(response, "output_text", "") or "").strip()
        if not answer_text:
            logger.warning(
                "WebSearchProducer (openai): empty response for product %r.", product_name
            )
            return None

        logger.debug("WebSearchProducer: produced %d chars.", len(answer_text))
        return answer_text
