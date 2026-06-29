"""app/services/enrichment/marketplaces/base.py

OOP skeleton for the "marketplace pool per product type" pattern.

TEMPLATE METHOD pattern:
  MarketplaceSource.enrich() is the fixed orchestration skeleton.
  Subclasses implement find_url / fetch / extract.

DESIGN NOTE — Ozon/WB are ALWAYS-ON:
  OzonCardSource and WbCardSource are NOT in this pool. They are native
  sources that always fire first in the main pipeline regardless of product
  type. This pool contains ADDITIONAL web marketplaces (Lamoda, Yandex.Market,
  etc.) that fire only when there are remaining attribute gaps and the product
  type matches.
"""
from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Product type taxonomy
# ---------------------------------------------------------------------------

class ProductType(Enum):
    """Coarse product type for marketplace pool routing."""
    CLOTHING = "clothing"
    FOOTWEAR = "footwear"
    ELECTRONICS = "electronics"
    FOOD = "food"
    BEAUTY = "beauty"
    HOME = "home"
    TOYS = "toys"
    OTHER = "other"


# Keyword patterns per type (checked case-insensitively against category_path tokens).
# Each pattern is a compiled regex tested against the full lowercased path string.
_TYPE_PATTERNS: list[tuple[ProductType, re.Pattern[str]]] = [
    (ProductType.CLOTHING,    re.compile(r"одежд", re.IGNORECASE)),
    (ProductType.FOOTWEAR,    re.compile(r"обув", re.IGNORECASE)),
    (ProductType.ELECTRONICS, re.compile(r"электроник|смартфон|ноутбук|телевизор|наушник", re.IGNORECASE)),
    (ProductType.BEAUTY,      re.compile(r"красот|парфюм|косметик", re.IGNORECASE)),
    (ProductType.FOOD,        re.compile(r"продукт|еда|бакалея", re.IGNORECASE)),
    (ProductType.HOME,        re.compile(r"дом|мебель|кухн", re.IGNORECASE)),
    (ProductType.TOYS,        re.compile(r"игрушк|конструктор", re.IGNORECASE)),
]


def classify_product_type(context: ExtractionContext) -> ProductType:
    """Rule-based product type classification from category_path.

    Joins all path segments into a single string and tests each type's
    regex pattern in priority order. Returns OTHER when nothing matches.
    """
    if not context.category_path:
        return ProductType.OTHER

    path_str = " ".join(context.category_path)
    for ptype, pattern in _TYPE_PATTERNS:
        if pattern.search(path_str):
            return ptype
    return ProductType.OTHER


# ---------------------------------------------------------------------------
# Abstract MarketplaceSource — TEMPLATE METHOD
# ---------------------------------------------------------------------------

class MarketplaceSource(ABC):
    """Abstract base for a single web-marketplace enrichment source.

    Subclasses implement the three primitives:
      find_url  — discover the product URL on this marketplace
      fetch     — retrieve the raw page content
      extract   — parse content into AttributeValues

    The template method enrich() glues them together with logging and
    fail-soft error handling.

    Attributes
    ----------
    name : str
        Human-readable short name, used in log prefixes ([MP:lamoda]).
    domain : str
        Domain of the marketplace (e.g. "lamoda.ru").
    serves_types : set[ProductType] | None
        Product types this source can serve. None means all types.
    source_tag : Source
        Source enum value for AttributeValue.source tagging.
    """

    name: str
    domain: str
    serves_types: Optional[set[ProductType]]
    source_tag: Source

    # ------------------------------------------------------------------
    # Primitives — subclasses MUST override
    # ------------------------------------------------------------------

    @abstractmethod
    async def find_url(
        self,
        context: ExtractionContext,
    ) -> Optional[str]:
        """Discover the product page URL on this marketplace.

        Returns None if no URL can be found (no search results, wrong domain).
        """

    @abstractmethod
    async def fetch(self, url: str) -> Optional[str]:
        """Fetch raw page content for the given URL.

        Returns HTML/text string, or None on fetch failure.
        """

    @abstractmethod
    def extract(
        self,
        html: str,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Parse page content into AttributeValues for the given targets.

        Pure/sync: all async work (LLM calls) should be done via a separate
        async helper and awaited BEFORE calling extract. Subclasses that need
        async extraction should override enrich() or use an async helper that
        gets called in the enrich() implementation.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def applies_to(self, ptype: ProductType) -> bool:
        """True when this source handles the given product type."""
        return self.serves_types is None or ptype in self.serves_types

    # ------------------------------------------------------------------
    # Shared object-search URL discovery (barcode/article → name fallback)
    # ------------------------------------------------------------------

    def _build_object_queries(self, context: ExtractionContext) -> list[str]:
        """Ordered Serper terms, STRONGEST object identifier first.

        Mirrors the Ozon/WB approach: an exact code pins the one product
        object far more reliably than a fuzzy name search. Each code is
        QUOTED so Serper keeps it as a single token (без кавычек Serper
        дробит код и находит посторонние карточки):
          1. EAN/barcode          — globally unique physical product key
                                     (best cross-marketplace match)
          2. article / mpn + brand — manufacturer model code (как WB article-path)
          3. brand + product_name  — loose fallback (legacy behaviour)
        The caller prepends ``site:<domain>``. Terms are deduped, order kept.
        """
        brand = (context.brand or "").strip()
        terms: list[str] = []

        ean = (context.ean or "").strip()
        if ean:
            terms.append(f'"{ean}"')

        for code in (context.article, context.mpn):
            code = (code or "").strip()
            if code:
                term = f'"{code}" {brand}'.strip() if brand else f'"{code}"'
                terms.append(term)

        name = context.product_name.strip()
        terms.append(f"{brand} {name}".strip() if brand else name)

        seen: set[str] = set()
        out: list[str] = []
        for t in terms:
            key = t.lower()
            if t and key not in seen:
                seen.add(key)
                out.append(t)
        return out

    async def _find_url_by_object_search(
        self,
        context: ExtractionContext,
        search_client,
        *,
        site_filter: str,
        accept,
        num_results: int,
        log_prefix: str,
    ) -> Optional[str]:
        """Discover the product-object URL on this marketplace.

        Tries each identifier query in priority order (``site:<filter> <term>``)
        and returns the FIRST organic link that ``accept(link)`` approves. The
        precise barcode/article query short-circuits the loose name query when
        it works; if a site doesn't index the code, we fall through to the next
        term. Returns None when no query yields an accepted URL.

        Parameters
        ----------
        site_filter : str
            Serper ``site:`` value, e.g. ``"market.yandex.ru"`` or ``"lamoda.ru/p"``.
        accept : Callable[[str], bool]
            Predicate that validates a candidate link is a real product page.
        """
        queries = self._build_object_queries(context)
        for term in queries:
            query = f"site:{site_filter} {term}"
            logger.info("%s Serper query: '%s'", log_prefix, query[:140])
            try:
                results = await search_client.search(query, num_results=num_results)
            except Exception as exc:
                logger.warning("%s Serper error on '%s': %s", log_prefix, query[:80], exc)
                continue
            organic = getattr(results, "organic_results", None) or []
            for item in organic:
                link = getattr(item, "link", "") or ""
                if accept(link):
                    logger.info("%s matched URL via '%s': %s", log_prefix, term[:60], link)
                    return link

        logger.info("%s no product URL across %d queries", log_prefix, len(queries))
        return None

    # ------------------------------------------------------------------
    # TEMPLATE METHOD — concrete orchestration skeleton
    # ------------------------------------------------------------------

    async def enrich(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Full enrichment cycle: find → fetch → extract, with fail-soft handling.

        Steps:
          1. find_url  — if None, log and return []
          2. fetch     — if None or too-short, log and return []
          3. extract   — parse attributes; any exception → log + return []

        All exceptions are caught at each step so one failing marketplace
        never blocks the others.
        """
        prefix = f"[MP:{self.name}]"
        try:
            url = await self.find_url(context)
        except Exception as exc:
            logger.warning("%s find_url error for '%s': %s", prefix, context.product_name[:60], exc)
            return []

        if not url:
            logger.info("%s no URL found for '%s'", prefix, context.product_name[:60])
            return []

        logger.info("%s URL: %s", prefix, url)

        try:
            html = await self.fetch(url)
        except Exception as exc:
            logger.warning("%s fetch error for %s: %s", prefix, url[:80], exc)
            return []

        if not html or len(html) < 1000:
            logger.info("%s fetch returned empty/too-short content (%d chars)", prefix, len(html) if html else 0)
            return []

        logger.info("%s fetched %d chars", prefix, len(html))

        try:
            results = self.extract(html, context, targets)
        except Exception as exc:
            logger.warning("%s extract error: %s", prefix, exc)
            return []

        logger.info("%s extracted %d attribute values", prefix, len(results))
        return results
