"""LamodaScrapflySource — last-resort clothing attribute gap-filler via Lamoda + Scrapfly.

COST-CONTROL DESIGN (critical):
  This source fires ONLY when ALL of the following hold simultaneously:
    (a) The product is clothing/footwear/fashion accessories — gated by a cheap
        LLM classifier ("Is product '<name>' clothing/footwear/fashion accessory?")
        before any Serper or Scrapfly call.
    (b) The env flag LAMODA_SCRAPFLY_ENABLED is set to "true" / "1".
    (c) There are still-empty target attributes after previous sources.

  If the LLM gate says "no" (not clothing) — returns [] immediately, zero credits spent.

ALGORITHM:
  1. LLM gate: cheap да/нет call on product_name only. Returns [] on "нет".
  2. Serper search: ``site:lamoda.ru/p <brand> <product_name>``.
     First organic result with ``/p/`` in the path is the product URL.
  3. Scrapfly scrape: GET https://api.scrapfly.io/scrape with params
     key=<key>, url=<lamoda_url>, asp=true, render_js=true, country=ru,
     proxy_pool=public_residential_pool. (Residential REQUIRED — DataDome.)
     Cost: 30 credits/request. ~376KB valid HTML; ~4KB = DataDome block.
  4. Parse attributes: regex over triples
     ``"key":"<k>","title":"<title>","value":"<value>"`` → dict {title: value}.
     Yields ~21 attributes (Состав %, Сезон, Цвет, Фасон, Страна производства, …).
  5. Map characteristics: pass parsed pairs through OzonCardSource._map_characteristics
     for consistent name-match → value_id resolution (same machinery as WbCardSource /
     ScrapflyOzonSource). Emit with source=Source.LAMODA.

SOURCE TAG:
  emits Source.LAMODA with evidence f"lamoda:scrapfly:{title[:40]}"

PARSER:
  regex ``"key":"([^"]*)","title":"([^"]*)","value":"([^"]*)"`` on raw HTML.
  Returns {title: value} dict directly — no BeautifulSoup needed.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional

from pydantic import BaseModel

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.wb_card_judge import WbCardJudge
from app.services.enrichment.sources.ozon_card_source import OzonCardSource
from app.services.providers import scrapfly_client as _scrapfly_mod
from app.services.providers.factory import get_main_manager, get_web_search_client
from app.services.providers.zenrows_client import zenrows_fetch as _zenrows_fetch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scrapfly params (verified by owner — DO NOT change)
_SCRAPFLY_COUNTRY = "ru"
_SCRAPFLY_PROXY_POOL = "public_residential_pool"

# Lamoda URL pattern for product pages
_LAMODA_PRODUCT_URL_RE = re.compile(
    r"https?://(?:www\.)?lamoda\.ru/p/[^/?#\s]+",
    re.IGNORECASE,
)

# Attribute triple regex — Lamoda JSON-in-HTML pattern (owner-verified)
# Matches: "key":"<k>","title":"<title>","value":"<value>"
_ATTR_TRIPLE_RE = re.compile(
    r'"key":"([^"]*)","title":"([^"]*)","value":"([^"]*)"'
)

# Minimum HTML length to consider a valid Scrapfly response.
# ~376KB valid page vs ~4KB DataDome block.
_MIN_VALID_HTML_LEN = 10_000

# Total timeout for the entire source (Serper + 1 Scrapfly call).
_TOTAL_TIMEOUT = 90.0

# Confidence mirrors WbCardSource (exact-level for verbatim card copy).
_CONF = 0.90

# Max Serper results to scan for a lamoda /p/ URL.
_SERPER_NUM_RESULTS = 5


# ---------------------------------------------------------------------------
# LLM gate model
# ---------------------------------------------------------------------------

class _ClothingGateResponse(BaseModel):
    """LLM response: is this product clothing/footwear/fashion accessory?"""
    is_clothing: bool


# ---------------------------------------------------------------------------
# Attribute parser
# ---------------------------------------------------------------------------

def _parse_lamoda_attributes(html: str) -> dict[str, str]:
    """Extract {title: value} pairs from Lamoda product page HTML.

    Lamoda embeds attribute data as JSON-like triples in the page:
      "key":"<k>","title":"<title>","value":"<value>"
    This regex extracts all such triples and returns a deduped dict.
    ~21 attributes typically found (Состав, %, Сезон, Цвет, Фасон, etc.).
    """
    result: dict[str, str] = {}
    for match in _ATTR_TRIPLE_RE.finditer(html):
        title = match.group(2).strip()
        value = match.group(3).strip()
        if title and value and title not in result:
            result[title] = value
    return result


# Vocabulary bridge: Lamoda's fixed attribute-title vocabulary → Ozon canonical
# attribute name(s). WITHOUT this, fuzzy name-match drops the richest fields:
# Lamoda "Состав, %" never matches Ozon "Состав материала" by name. ONE-to-MANY:
# the same Lamoda title can target several Ozon attrs that live in DIFFERENT
# categories (e.g. clothing has "Состав материала", footwear has "Материал
# верха") — we emit ALL candidates; only the one whose target exists in this
# product's category matches, the rest are harmlessly dropped. GENERAL mapping
# over Lamoda's fixed vocabulary (NOT per-product hardcode).
_LAMODA_TO_OZON_ALIASES: dict[str, list[str]] = {
    "Состав, %": ["Состав материала", "Материал верха", "Материал"],
    "Материал": ["Материал", "Материал верха"],
    "Материал верха": ["Материал верха"],
    "Материал подкладки, %": ["Материал подкладки", "Материал подкладки обуви"],
    "Материал подошвы": ["Материал подошвы обуви"],
    "Материал стельки": ["Материал стельки"],
    "Застежка": ["Вид застежки"],
    "Страна производства": ["Страна-изготовитель"],
    "Фасон": ["Модель", "Фасон"],
    "Цвет": ["Цвет товара", "Название цвета"],
    "Гарантийный срок": ["Гарантия", "Гарантийный срок"],
}


def _is_model_reference(title: str) -> bool:
    """True for Lamoda titles describing the MODEL/photo, not the product.

    Lamoda lists the photo model's measurements ("Рост модели на фото",
    "Параметры модели", "Размер товара на модели", "Рост") as attributes —
    these are about the person, NOT the garment, and pollute product fills.
    """
    t = title.strip().lower()
    return "модел" in t or "на модели" in t or t == "рост"


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

class LamodaScrapflySource(AttributeSource):
    """Last-resort clothing attribute gap-filler via Lamoda + Scrapfly.

    Only fires when LAMODA_SCRAPFLY_ENABLED=true AND the product passes the
    LLM clothing gate. Cost: 1 Serper call + 30 Scrapfly credits/product.
    Latency: 15-60s (Scrapfly residential render).
    """

    def __init__(
        self,
        scrapfly_key: Optional[str] = None,
        enabled: Optional[bool] = None,
        llm_manager=None,
        web_search_client=None,
    ) -> None:
        from app import config as _cfg

        # Scrapfly key: explicit arg > env > config
        self._key = (
            scrapfly_key
            or os.environ.get("SCRAPFLY_API_KEY")
            or os.environ.get("SCRAPFLY_KEY")
            or _cfg.SCRAPFLY_API_KEY
        )
        # Feature flag: explicit arg > env
        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = _cfg.LAMODA_SCRAPFLY_ENABLED

        if not self._key:
            logger.warning(
                "[LamodaScrapfly] SCRAPFLY_API_KEY not configured — source will always return []."
            )

        # LLM for clothing gate (cheapest main model — just да/нет)
        self._llm = llm_manager or get_main_manager()

        # Serper for product URL discovery
        self._search_client = web_search_client
        if self._search_client is None:
            try:
                self._search_client = get_web_search_client()
            except Exception as exc:
                logger.warning(
                    "[LamodaScrapfly] web search client unavailable (%s) — extract() will return [].",
                    exc,
                )
                self._search_client = None

        # Judge mirrors WbCardJudge: pre-moderated card copy, no LLM needed
        self._judge = WbCardJudge()

        # OzonCardSource helper for _map_characteristics (parsing/mapping only, no HTTP)
        self._ozon_helper = OzonCardSource(scrappey_key=None)

    @property
    def source_type(self) -> Source:
        return Source.LAMODA

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Applicable when enabled, key present, search client available, product_name non-trivial."""
        return bool(
            self._enabled
            and self._key
            and self._search_client
            and context.product_name
            and len(context.product_name.strip()) >= 5
        )

    def get_judge(self) -> LlmJudge:
        return self._judge

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Extract clothing attributes from Lamoda via Scrapfly.

        Returns [] immediately when:
          - disabled or no API key / search client
          - LLM gate says product is NOT clothing/footwear/fashion accessory
          - no remaining gap attributes
        """
        if not targets or not context.product_name:
            return []
        if not self._enabled or not self._key or not self._search_client:
            return []

        # GATE: no remaining gaps
        already_filled = already_filled or []
        filled_attr_ids = {av.attribute_id for av in already_filled if av.confidence >= 0.80}
        gap_targets = [t for t in targets if t.id not in filled_attr_ids]
        if not gap_targets:
            logger.debug(
                "[LamodaScrapfly] skip — no gap attributes for '%s'",
                context.product_name[:60],
            )
            return []

        logger.info(
            "[LamodaScrapfly] triggered for '%s' (%d gap attrs)",
            context.product_name[:60], len(gap_targets),
        )

        try:
            return await asyncio.wait_for(
                self._do_extract(context, gap_targets),
                timeout=_TOTAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[LamodaScrapfly] total timeout (%.0fs) for '%s' — giving up",
                _TOTAL_TIMEOUT, context.product_name[:60],
            )
            return []
        except Exception as exc:
            logger.warning(
                "[LamodaScrapfly] unexpected error for '%s': %s",
                context.product_name[:60], exc,
            )
            return []

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    async def _do_extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        product_name = context.product_name.strip()

        # ---- Step 1: LLM clothing gate ----
        is_clothing = await self._check_is_clothing(product_name)
        if not is_clothing:
            logger.info(
                "[LamodaScrapfly] LLM gate: '%s' is NOT clothing — skip (0 credits spent)",
                product_name[:60],
            )
            return []

        logger.info(
            "[LamodaScrapfly] LLM gate: '%s' IS clothing — proceeding",
            product_name[:60],
        )

        # ---- Step 2: Serper search → Lamoda product URL ----
        lamoda_url = await self._find_lamoda_url(product_name, context.brand)
        if not lamoda_url:
            logger.info(
                "[LamodaScrapfly] no Lamoda product URL found for '%s'",
                product_name[:60],
            )
            return []

        logger.info("[LamodaScrapfly] product URL: %s", lamoda_url)

        # ---- Step 3: ZenRows fetch (replaced Scrapfly — multi-account ban) ----
        fetch_result = await _zenrows_fetch(
            lamoda_url,
            wait_ms=5000,   # let Lamoda's SPA hydrate the attribute JSON
        )

        if not fetch_result.success or not fetch_result.content:
            logger.info(
                "[LamodaScrapfly] ZenRows fetch failed: %s (credits_used=%d)",
                fetch_result.error, fetch_result.credits_used,
            )
            return []

        html = fetch_result.content
        if len(html) < _MIN_VALID_HTML_LEN:
            logger.info(
                "[LamodaScrapfly] HTML too short (%d chars) — likely DataDome block/underhyd (credits_used=%d)",
                len(html), fetch_result.credits_used,
            )
            return []

        logger.info(
            "[LamodaScrapfly] got %d chars (credits_used=%d)",
            len(html), fetch_result.credits_used,
        )

        # ---- Step 4: Parse attributes ----
        attr_dict = _parse_lamoda_attributes(html)
        if not attr_dict:
            logger.info("[LamodaScrapfly] 0 attributes extracted from page")
            return []

        logger.info(
            "[LamodaScrapfly] extracted %d attributes: %s",
            len(attr_dict),
            list(attr_dict.keys())[:10],
        )

        # ---- Step 5: Map to AttributeValues via OzonCardSource machinery ----
        # Convert dict → list of dicts compatible with _map_characteristics.
        #  • Fix 1: drop MODEL/photo-reference titles (Рост модели, Параметры
        #    модели…) — they describe the person, not the product (mud).
        #  • Fix 2: for aliased titles emit ONLY the canonical Ozon name(s)
        #    (one-to-many, covers clothing + footwear attrs). NOT the original —
        #    the raw Lamoda title fuzzy-mis-matches (e.g. "Материал подкладки, %"
        #    → "Материал"). Unaliased titles keep the original (Сезон, Узор… match fine).
        chars = []
        for title, value in attr_dict.items():
            if _is_model_reference(title):
                continue
            canonicals = _LAMODA_TO_OZON_ALIASES.get(title)
            if canonicals:
                for cname in canonicals:
                    chars.append({"name": cname, "value": value, "value_ids": []})
            else:
                chars.append({"name": title, "value": value, "value_ids": []})

        title_for_evidence = lamoda_url.rstrip("/").split("/")[-1] or product_name[:40]
        evidence_suffix = f"lamoda:scrapfly:{title_for_evidence[:40]}"

        raw_avs = self._ozon_helper._map_characteristics(
            chars,
            targets,
            context,
            mode="brand_line",   # Lamoda card = same brand, possibly different model variant
            title=title_for_evidence,
            score=85.0,
            evidence_override=evidence_suffix,
        )

        # _map_characteristics hard-codes source=OZON_CARD — patch to Source.LAMODA
        # so audit trail, judge dispatch, and merge priority are correct.
        return [av.model_copy(update={"source": Source.LAMODA}) for av in raw_avs]

    async def _check_is_clothing(self, product_name: str) -> bool:
        """LLM gate: is this product clothing/footwear/fashion accessory?

        Sends a minimal да/нет prompt on the product name alone (no Serper/Scrapfly).
        Returns True when the LLM says yes, False on no or any error (fail-closed = no wasted credits).
        """
        system_prompt = (
            "Ты классификатор товаров. Отвечай ТОЛЬКО JSON {'is_clothing': true} или {'is_clothing': false}.\n"
            "is_clothing=true: одежда, обувь, аксессуары одежды (ремни, шарфы, перчатки, шапки, сумки, рюкзаки).\n"
            "is_clothing=false: электроника, продукты питания, мебель, спорттовары без одежды, и всё остальное."
        )
        user_text = f"Товар: \"{product_name}\"\nЯвляется ли он одеждой/обувью/аксессуаром одежды?"

        try:
            parsed, _ = await self._llm.structured_request(
                system_prompt=system_prompt,
                user_text=user_text,
                response_model=_ClothingGateResponse,
                temperature=0.0,
            )
            if parsed is None:
                logger.warning("[LamodaScrapfly] LLM gate returned None — fail-closed (skip)")
                return False
            return parsed.is_clothing
        except Exception as exc:
            logger.warning("[LamodaScrapfly] LLM gate error: %s — fail-closed (skip)", exc)
            return False

    async def _find_lamoda_url(
        self,
        product_name: str,
        brand: Optional[str],
    ) -> Optional[str]:
        """Serper search → first Lamoda /p/ product URL.

        Query: ``site:lamoda.ru/p <brand> <product_name>``
        """
        brand_part = brand.strip() if brand and brand.strip() else ""
        query_parts = ["site:lamoda.ru/p"]
        if brand_part:
            query_parts.append(brand_part)
        query_parts.append(product_name)
        query = " ".join(query_parts)

        logger.info("[LamodaScrapfly] Serper query: '%s'", query[:120])

        try:
            results = await self._search_client.search(query, num_results=_SERPER_NUM_RESULTS)
        except Exception as exc:
            logger.warning("[LamodaScrapfly] Serper error: %s", exc)
            return None

        organic = getattr(results, "organic_results", None) or []
        for item in organic:
            link = getattr(item, "link", "") or ""
            if _LAMODA_PRODUCT_URL_RE.match(link) and "/p/" in link:
                return link

        logger.info("[LamodaScrapfly] no Lamoda /p/ URL in Serper results")
        return None
