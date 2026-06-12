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
import os
import re
from typing import Optional

from app import config

logger = logging.getLogger(__name__)

# Caps CONCURRENT Serper HTTP calls across the whole run. Eval runs up to 8
# products concurrently × dual-lang = up to 16 concurrent Serper calls → 429.
# Created lazily on the RUNNING loop (a module-level Semaphore bound at import
# time latches onto a dead/foreign loop and raises "bound to a different loop").
_serper_sem: "asyncio.Semaphore | None" = None
_serper_sem_loop: "asyncio.AbstractEventLoop | None" = None


def _get_serper_sem() -> asyncio.Semaphore:
    """Return a process-wide Serper concurrency semaphore, created lazily on the
    currently running event loop and reused for that loop."""
    global _serper_sem, _serper_sem_loop
    loop = asyncio.get_running_loop()
    if _serper_sem is None or _serper_sem_loop is not loop:
        size = int(os.environ.get("WEBSEARCH_SERPER_CONCURRENCY", "8"))
        _serper_sem = asyncio.Semaphore(size)
        _serper_sem_loop = loop
    return _serper_sem

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
        # Raw page text from last _produce_via_serper call (before LLM summarisation).
        # Set during each serper call; read by WebSearchSource right after produce_summary
        # to extract verbatim spec pairs (OemSpecHarvest).  Thread-safety: asyncio
        # pipeline processes one product at a time per WebSearchProducer instance.
        self._last_page_text: Optional[str] = None

    # Language-specific query suffixes — anchor Serper towards spec-like pages
    # in each language. For unknown languages we fall back to English (which
    # Google's mixed-language ranking handles best).
    _LANG_QUERY_SUFFIXES = {
        "ru": "характеристики технические",
        "en": "specifications datasheet",
        "de": "technische daten datenblatt",
        "fr": "caractéristiques techniques fiche",
        "es": "especificaciones técnicas ficha",
    }

    async def produce_summary(
        self,
        product_name: str,
        brand: Optional[str] = None,
        ean: Optional[str] = None,
        mpn: Optional[str] = None,
        timeout: int = 60,
        languages: Optional[list[str]] = None,
    ) -> Optional[str]:
        """Search the web for product info and return a plain-text summary.

        When ``languages`` has more than one entry, runs an independent Serper
        search per language in parallel and concatenates the resulting summaries.
        This typically lifts coverage for international brands (e.g. Dyson,
        Samsung) where the EN manufacturer site holds authoritative specs while
        the RU retail listings reveal local availability/variants.

        Returns:
            A plain-text summary string, or None on failure.
        """
        if not product_name:
            logger.debug("WebSearchProducer: no product_name provided, skipping.")
            return None

        if not self._use_serper:
            # Legacy OpenAI path doesn't support multilang yet — single call.
            return await self._produce_via_openai(product_name, brand, ean, mpn, timeout)

        langs = [(lang or "ru").lower() for lang in (languages or ["ru"])]
        # Dedup while preserving order
        langs = list(dict.fromkeys(langs))

        if len(langs) == 1:
            return await self._produce_via_serper(
                product_name, brand, ean, mpn, timeout, lang=langs[0]
            )

        # Multi-lang: run each in parallel, concatenate non-empty.
        results = await asyncio.gather(*[
            self._produce_via_serper(product_name, brand, ean, mpn, timeout, lang=lang)
            for lang in langs
        ], return_exceptions=True)

        chunks = []
        for lang, res in zip(langs, results):
            if isinstance(res, Exception):
                logger.warning("WebSearchProducer: lang=%s failed: %s", lang, res)
                continue
            if res:
                chunks.append(f"--- {lang.upper()} ---\n{res}")

        if not chunks:
            return None
        if len(chunks) == 1:
            # Strip the "--- LANG ---" header when only one succeeded.
            return chunks[0].split("\n", 1)[1]
        return "\n\n".join(chunks)

    # ------------------------------------------------------------------
    # Boilerplate guard (generic, no domain hardcode)
    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_boilerplate(text: str) -> bool:
        """Generic heuristic: True if `text` is mostly markup/scripts or has
        almost no «spec signal», so it should NOT be fed to the LLM.

        Two checks (either trips the guard):
          1. Markup ratio: high density of `<`, `{`, `}`, `function`/`var `/
             stylesheet tokens relative to length → trafilatura failed and we
             got raw JS/CSS instead of content.
          2. Spec-signal density: real specs contain digits + colons + units
             («Состав: 95% хлопок»). Near-zero of those over a long blob → noise.

        Pure heuristic, no hardcoded domains.
        """
        if not text:
            return True

        n = len(text)
        lowered = text.lower()

        # --- 1. Markup / script density ---
        markup_chars = lowered.count("<") + lowered.count("{") + lowered.count("}")
        markup_tokens = (
            lowered.count("function")
            + lowered.count("var ")
            + lowered.count("</")
            + lowered.count("px;")
            + lowered.count("rgba(")
            + lowered.count("@media")
            + lowered.count("script")
            + lowered.count("stylesheet")
        )
        # Each token ~ several "junk" chars; weight them.
        junk_score = markup_chars + markup_tokens * 8
        # Threshold tuned for modern Shopify/Wix sites that ship some inline JS
        # alongside real product specs. 0.04 was too strict — flagged genuine
        # spec pages with embedded analytics widgets. 0.06 keeps obvious
        # antibot-JS responses out but lets normal mixed pages through.
        if n > 0 and (junk_score / n) > 0.06:
            return True

        # --- 2. Spec-signal density ---
        # Real specs are dense with digits and key:value colons.
        digits = sum(c.isdigit() for c in text)
        colons = text.count(":")
        # For a non-trivial blob, expect some digits and at least a few colons.
        if n >= 400:
            if digits == 0 or colons < 2:
                return True
            if (digits / n) < 0.004 and colons < 4:
                return True

        return False

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
        lang: str = "ru",
    ) -> Optional[str]:
        # Reset page-text cache only for the RU pass (preferred language for OEM
        # spec-harvest).  In dual-lang mode both calls run concurrently; resetting
        # only on RU prevents EN from clearing a RU result that finished first,
        # and the write guard below ensures EN never overwrites RU.
        if lang == "ru":
            self._last_page_text = None
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
        suffix = self._LANG_QUERY_SUFFIXES.get(lang, self._LANG_QUERY_SUFFIXES["en"])
        query = " ".join(parts) + " " + suffix

        try:
            sem = _get_serper_sem()
            async with sem:
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

        # --- Snippets are the PRIMARY signal ---
        # Serper organic snippets уже содержат реальные спеки
        # («Сезон: …; Материал: …; Состав: …»). Retail-HTML-фетч часто = 403
        # или JS/CSS-мусор, который РАЗБАВЛЯЕТ полезные сниппеты и вытесняет их
        # за cap. Поэтому сниппеты идут ПЕРВЫМИ в context (гарантированно доходят
        # до LLM), а текст страницы добавляется ПОСЛЕ и только если прошёл гард
        # от boilerplate.
        snippets = [
            f"[{r.position}] {r.title}\n{r.snippet}"
            for r in results.organic_results[:5]
        ]
        snippet_section = "\n\n".join(snippets)

        # --- Full-page fetch: top-1-2 HTTPS links from organic results ---
        # Per-URL timeout so a slow Scrappey fallback on one URL cannot cancel
        # a fast/cached result from another URL — partial results are always saved.
        _PAGE_FETCH_TIMEOUT = 20      # seconds per individual URL fetch
        _PAGE_TEXT_CAP = 6000         # chars to keep per product (LLM context guard)

        top_urls = [
            r.link
            for r in results.organic_results[:5]
            if getattr(r, "link", None) and str(r.link).startswith("https://")
        ][:2]

        page_section = ""
        if top_urls:
            from app.services.url_fetcher import fetch_url_content as _fetch_one

            async def _fetch_with_timeout(url: str) -> Optional[str]:
                """Fetch a single URL with per-URL timeout; return extracted text or None."""
                try:
                    result = await asyncio.wait_for(
                        _fetch_one(url),
                        timeout=_PAGE_FETCH_TIMEOUT,
                    )
                    return result.content if result is not None else None
                except asyncio.TimeoutError:
                    logger.warning(
                        "WebSearchProducer (serper): fetch timed out (%ds) for %r.",
                        _PAGE_FETCH_TIMEOUT,
                        url,
                    )
                    return None
                except Exception as exc:
                    logger.warning(
                        "WebSearchProducer (serper): fetch failed for %r: %s",
                        url,
                        exc,
                    )
                    return None

            per_url_results = await asyncio.gather(*[_fetch_with_timeout(u) for u in top_urls])
            raw_page_text = "\n".join(r for r in per_url_results if r)

            if raw_page_text:
                # Каталог-страницы (asus.com/techspec и т.п.) повторяют спеки
                # для каждого SKU линейки 5-10 раз → boilerplate вытесняет
                # ключевые спеки за cap. Дедуп строк (order-preserving) даёт
                # ~38% сжатия, ключевое (USB/Wi-Fi/HDMI) влезает в окно.
                raw_page_text = "\n".join(dict.fromkeys(raw_page_text.split("\n")))
                # Boilerplate-гард: если страница — преимущественно
                # разметка/скрипты (антибот вернул JS/CSS-мусор вместо
                # контента) ИЛИ почти нет «спек-сигнала» — ОТБРОСИТЬ, чтобы
                # не разбавлять сниппеты. Эвристика генеричная, без хардкода.
                if self._looks_like_boilerplate(raw_page_text):
                    logger.debug(
                        "WebSearchProducer (serper): page text looks like boilerplate "
                        "for %r — dropped, using snippets only.",
                        product_name,
                    )
                else:
                    page_section = (
                        "\n\n=== Текст страницы со спецификациями ===\n"
                        + raw_page_text[:_PAGE_TEXT_CAP]
                        + "\n"
                    )
                    # Expose raw page text for OemSpecHarvest verbatim pass.
                    # Stored here (before LLM summarisation) so spec lines are
                    # intact; the LLM summary loses the «Key: Value» structure.
                    # RU-preference: only write when lang=="ru" OR when no RU
                    # result has been stored yet (prevents EN from overwriting RU
                    # in dual-lang concurrent execution).
                    if lang == "ru" or self._last_page_text is None:
                        self._last_page_text = raw_page_text[:_PAGE_TEXT_CAP]
                    logger.debug(
                        "WebSearchProducer (serper): fetched %d chars from %d page(s) for %r.",
                        len(raw_page_text),
                        len(top_urls),
                        product_name,
                    )
        # -------------------------------------------------------------------

        # Snippets first (primary), page text appended after (secondary, guarded).
        search_context = snippet_section + page_section

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

    # ------------------------------------------------------------------
    # Composition mining: targeted Serper search → raw HTML → extractor
    # ------------------------------------------------------------------

    # Material words that must appear in an evidence_quote to count as a
    # composition signal. Uses a shared pattern from composition_extractor
    # but we keep a lightweight inline check to avoid circular import at
    # module level.
    _COMPOSITION_SIGNAL_RE = re.compile(
        r"(?:"
        r"\d{1,3}\s*%"          # digit + percent sign
        r"|"
        r"\b(?:хлопок|cotton|полиэстер|polyester|эластан|elastane|spandex"
        r"|вискоза|viscose|шерсть|wool|полиамид|polyamide|нейлон|nylon"
        r"|лиоцелл|lyocell|акрил|acrylic|лён|linen|кашемир|cashmere"
        r"|модал|modal|бамбук|bamboo|шёлк|silk|флис|fleece)\b"
        r")",
        re.IGNORECASE,
    )

    # Structured LLM response for composition mining
    # Defined as a module-level type below to allow isinstance checks in tests.

    async def mine_composition(
        self,
        product_name: str,
        brand: Optional[str] = None,
        timeout: int = 30,
        llm_provider=None,
        llm_calls_budget: int = 99,
        llm_calls_so_far: int = 0,
    ) -> list[str]:
        """Mine fabric/material composition from the web.

        Strategy (cost-aware, "пусто честнее мусора"):
        1. Issue 2-3 query variants (RU + EN) via Serper, dedup URLs.
        2. Fetch up to 5 HTTPS result pages via fetch_all_results.
        3. REGEX FIRST (free): run extract_composition on each page's raw HTML.
           If regex finds a composition AND brand/model verification passes →
           return immediately with no LLM call (cost saved).
        4. LLM RECALL (only when regex found nothing across all pages):
           Make ONE LLM call over truncated page texts with a strict prompt.
           Acceptance gate:
             - is_our_product == True
             - evidence_quote is a VERBATIM substring of the actually-fetched text
             - evidence_quote contains a composition signal (material word or %)
           This accepts blogs/reviews/wholesalers while blocking hallucination.

        Returns [] on any failure (fail-closed).
        Only functional when self._use_serper is True; returns [] for the legacy
        OpenAI path (no page fetching there).
        """
        if not self._use_serper or self._serper is None:
            return []
        if not product_name:
            return []

        from app.services.enrichment.composition_extractor import (
            extract_composition,
            page_matches_brand,
        )
        from app.services.url_fetcher import fetch_all_results

        # ------------------------------------------------------------------
        # Step 1 — Broader multi-query search (2-3 variants, deduped URLs)
        # ------------------------------------------------------------------
        parts_base = []
        if brand:
            parts_base.append(brand)
        parts_base.append(product_name)
        base = " ".join(parts_base)

        # Determine if brand is Latin (heuristic: majority ASCII letters)
        brand_str = brand or ""
        latin_ratio = (
            sum(1 for c in brand_str if c.isascii() and c.isalpha()) / max(len(brand_str), 1)
        )
        is_latin_brand = latin_ratio > 0.5

        queries: list[str] = [
            base + " состав",
            base + " материал",
        ]
        if is_latin_brand:
            queries.append(base + " material composition")

        seen_urls: set[str] = set()
        ordered_urls: list[str] = []

        _SKIP_DOMAINS = ("wildberries.ru", "ozon.ru", "aliexpress")

        for query in queries:
            try:
                sem = _get_serper_sem()
                async with sem:
                    results = await asyncio.wait_for(
                        self._serper.search(query, num_results=5),
                        timeout=timeout,
                    )
            except (asyncio.TimeoutError, Exception) as exc:
                logger.warning(
                    "WebSearchProducer.mine_composition: search failed for query %r: %s",
                    query, exc,
                )
                continue

            for r in results.organic_results or []:
                link = getattr(r, "link", None)
                if not link:
                    continue
                link = str(link)
                if not link.startswith("https://"):
                    continue
                if any(d in link for d in _SKIP_DOMAINS):
                    continue
                if link not in seen_urls:
                    seen_urls.add(link)
                    ordered_urls.append(link)

        # Limit to top-5 unique URLs
        top_urls = ordered_urls[:5]

        if not top_urls:
            logger.debug(
                "WebSearchProducer.mine_composition: no HTTPS URLs for %r", product_name
            )
            return []

        # ------------------------------------------------------------------
        # Step 2 — Fetch pages
        # ------------------------------------------------------------------
        try:
            fetch_results = await asyncio.wait_for(
                fetch_all_results(top_urls),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "WebSearchProducer.mine_composition: page fetch timed out for %r", product_name
            )
            return []
        except Exception as exc:
            logger.warning(
                "WebSearchProducer.mine_composition: page fetch failed for %r: %s",
                product_name, exc,
            )
            return []

        # ------------------------------------------------------------------
        # Step 3 — REGEX FIRST PASS (free, no LLM)
        # ------------------------------------------------------------------
        # Collect (url, text) pairs for potential LLM pass
        page_texts: list[tuple[str, str]] = []

        for fr in fetch_results:
            source_text = fr.raw_html if fr.raw_html else fr.content
            if not source_text:
                continue
            page_texts.append((fr.url, source_text))

            # Old gate was "must be PDP only" → caused Levi's false-reject.
            # New gate: require brand token anywhere in page (URL, title, h1,
            # first 2k text) AND at least one significant product_name word.
            if not page_matches_brand(source_text, fr.url, brand, product_name):
                logger.debug(
                    "WebSearchProducer.mine_composition: brand/model mismatch on %r — "
                    "skipping regex pass",
                    fr.url,
                )
                continue

            compositions = extract_composition(source_text)
            if compositions:
                logger.info(
                    "WebSearchProducer.mine_composition: REGEX HIT on %r — %s",
                    fr.url, compositions,
                )
                return _dedup_list(compositions)

        # ------------------------------------------------------------------
        # Step 4 — LLM RECALL BOOSTER (only when regex found nothing)
        # ------------------------------------------------------------------
        # Budget guard
        if llm_calls_so_far >= llm_calls_budget:
            logger.debug(
                "WebSearchProducer.mine_composition: LLM budget exhausted "
                "(%d/%d) for %r — skip LLM recall",
                llm_calls_so_far, llm_calls_budget, product_name,
            )
            return []

        if not page_texts:
            return []

        # Resolve LLM provider: injected > self._extractor > raw factory provider
        provider = llm_provider
        if provider is None:
            provider = self._extractor

        if provider is None:
            logger.debug(
                "WebSearchProducer.mine_composition: no LLM provider — skip LLM recall"
            )
            return []

        # Build context: top 3 pages, 5000 chars each
        _PAGE_CHAR_CAP = 5000
        _MAX_PAGES_FOR_LLM = 3

        context_blocks: list[str] = []
        full_texts: dict[str, str] = {}  # url → full_text for verbatim check

        for url, raw in page_texts[:_MAX_PAGES_FOR_LLM]:
            # Flatten HTML for LLM readability
            from app.services.enrichment.composition_extractor import _flatten_html
            flat = _flatten_html(raw)[:_PAGE_CHAR_CAP]
            context_blocks.append(f"=== SOURCE: {url} ===\n{flat}")
            full_texts[url] = _flatten_html(raw)  # full version for verbatim check

        pages_context = "\n\n".join(context_blocks)

        system_prompt = (
            "You are a product data specialist. Your task is to find the EXACT "
            "fabric/material composition of a specific product from provided web page excerpts.\n\n"
            "Rules:\n"
            "1. Only answer if the page clearly mentions THIS brand AND this model/line name.\n"
            "2. Blogs, reviews, wholesalers, and fan sites are acceptable sources — "
            "composition is product-invariant.\n"
            "3. NEVER guess or invent a composition not stated in the text.\n"
            "4. The evidence_quote field MUST be copied VERBATIM from the provided text.\n"
            "5. Set is_our_product=false if you are not sure the page is about THIS product.\n"
            "6. If no composition is found in the text, set composition=null.\n\n"
            "Return a JSON object with exactly these fields:\n"
            "{\n"
            '  "composition": string | null,\n'
            '  "material_primary": string | null,\n'
            '  "evidence_quote": string (VERBATIM substring from provided text),\n'
            '  "source_hint": string (URL or site name),\n'
            '  "is_our_product": bool,\n'
            '  "confidence": number between 0 and 1\n'
            "}"
        )

        user_text = (
            f'Find the material composition of: "{product_name}"'
            + (f" (brand: {brand})" if brand else "")
            + "\n\nWeb page excerpts:\n"
            + pages_context
        )

        try:
            # Duck-type dispatch: prefer direct .complete() when available
            # (works for LlmProvider subclasses AND test mocks).
            # Fall back to StructuredLlmManager path via _llm_complete_raw.
            if callable(getattr(provider, "complete", None)):
                llm_resp = await asyncio.wait_for(
                    provider.complete(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_text},
                        ],
                        model=_get_extraction_model(),
                        temperature=0.0,
                        max_tokens=512,
                        response_format={"type": "json_object"},
                    ),
                    timeout=timeout,
                )
                raw_json = (llm_resp.content or "").strip()
            else:
                # StructuredLlmManager path
                raw_json = await _llm_complete_raw(
                    provider, system_prompt, user_text, timeout
                )
        except asyncio.TimeoutError:
            logger.warning(
                "WebSearchProducer.mine_composition: LLM call timed out for %r", product_name
            )
            return []
        except Exception as exc:
            logger.warning(
                "WebSearchProducer.mine_composition: LLM call failed for %r: %s",
                product_name, exc,
            )
            return []

        # ------------------------------------------------------------------
        # Step 5 — ACCEPTANCE GATE (verbatim-quote + is_our_product)
        # ------------------------------------------------------------------
        parsed = _parse_composition_llm_response(raw_json)
        if parsed is None:
            logger.debug(
                "WebSearchProducer.mine_composition: LLM response parse failed for %r",
                product_name,
            )
            return []

        composition = parsed.get("composition")
        is_our = parsed.get("is_our_product", False)
        evidence_quote = parsed.get("evidence_quote", "") or ""
        source_hint = parsed.get("source_hint", "") or ""

        # Gate 1: LLM must believe the page is about our product
        if not is_our:
            logger.debug(
                "WebSearchProducer.mine_composition: LLM says is_our_product=False "
                "for %r — rejected",
                product_name,
            )
            return []

        # Gate 2: composition must be non-null
        if not composition:
            logger.debug(
                "WebSearchProducer.mine_composition: LLM returned null composition for %r",
                product_name,
            )
            return []

        # Gate 3: evidence_quote must be a VERBATIM substring of actually-fetched text
        # (blocks hallucination — the quote must really exist on the page)
        quote_verified = False
        if evidence_quote:
            for url, full_text in full_texts.items():
                if evidence_quote in full_text:
                    quote_verified = True
                    logger.debug(
                        "WebSearchProducer.mine_composition: verbatim quote verified on %r",
                        url,
                    )
                    break

        if not quote_verified:
            logger.warning(
                "WebSearchProducer.mine_composition: evidence_quote NOT found verbatim "
                "in any fetched page for %r — REJECTED (hallucination guard). "
                "Quote: %r",
                product_name, evidence_quote[:120],
            )
            return []

        # Gate 4: evidence_quote must contain a composition signal
        if not self._COMPOSITION_SIGNAL_RE.search(evidence_quote):
            logger.debug(
                "WebSearchProducer.mine_composition: evidence_quote has no composition "
                "signal for %r — rejected. Quote: %r",
                product_name, evidence_quote[:80],
            )
            return []

        logger.info(
            "WebSearchProducer.mine_composition: LLM HIT for %r — %r (source: %s)",
            product_name, composition, source_hint,
        )
        return [composition]


# ---------------------------------------------------------------------------
# Module-level helpers (outside the class — no self needed)
# ---------------------------------------------------------------------------


def _dedup_list(items: list[str]) -> list[str]:
    """Deduplicate while preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _get_extraction_model() -> str:
    from app import config as _cfg
    return _cfg.EXTRACTION_FROM_TEXT_MODEL


def _parse_composition_llm_response(raw_json: str) -> Optional[dict]:
    """Parse JSON from LLM response; returns None on any failure."""
    import json
    if not raw_json:
        return None
    try:
        # Strip markdown code fences if present
        text = raw_json.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # drop first and last fence lines
            inner = lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:]
            text = "\n".join(inner)
        return json.loads(text)
    except Exception:
        return None


async def _llm_complete_raw(provider, system_prompt: str, user_text: str, timeout: int) -> str:
    """Call a StructuredLlmManager-style provider and return raw text."""
    # StructuredLlmManager.structured_request() requires a Pydantic model.
    # Fall through to the underlying raw provider's complete() instead.
    if hasattr(provider, "_provider") and hasattr(provider._provider, "complete"):
        from app import config as _cfg
        resp = await asyncio.wait_for(
            provider._provider.complete(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                model=_cfg.EXTRACTION_FROM_TEXT_MODEL,
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            ),
            timeout=timeout,
        )
        return (resp.content or "").strip()
    return ""
