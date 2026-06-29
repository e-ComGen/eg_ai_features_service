"""app/services/enrichment/marketplaces/generic.py

GenericMarketplace — universal specialist web-marketplace enrichment source.

Each instance is configured with a (name, domain) pair. URL discovery uses
the shared _find_url_by_object_search (EAN → article/mpn → brand+name), the
page is fetched via ZenRows, and attributes are extracted by the grounded LLM
extractor.  No per-category hardcode: if the site does not carry the product,
the object search returns no URL and the instance contributes [].
"""
from __future__ import annotations

import logging
from typing import Optional

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.marketplaces.base import MarketplaceSource, ProductType
from app.services.enrichment.marketplaces.llm_extractor import llm_extract_attributes
from app.services.providers.factory import get_web_search_client
from app.services.providers.scrapedo_client import scrapedo_fetch

logger = logging.getLogger(__name__)

_SERPER_NUM_RESULTS = 5
_MIN_VALID_HTML_LEN = 10_000

# Non-HTML asset extensions — a Serper result pointing at one of these is a
# file (datasheet PDF, product image), never a product page worth extracting.
_NON_HTML_EXT = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".ico",
    ".zip", ".rar", ".7z", ".xml", ".json", ".csv", ".css", ".js",
    ".mp4", ".webm", ".mov", ".mp3", ".doc", ".docx", ".xls", ".xlsx",
)


class GenericMarketplace(MarketplaceSource):
    """Universal specialist web-marketplace (ZenRows fetch + grounded LLM extract).

    Configured purely by (name, domain) — no per-site logic. URL discovery
    uses the inherited _find_url_by_object_search with site:<domain> and
    the instance-level _accept_url predicate.

    Args:
        name: Short human-readable name used in log prefixes ([MP:<name>]).
        domain: Marketplace domain, e.g. "zdravcity.ru".
        wait_ms: ZenRows JS render wait in milliseconds (default 5000).
        web_search_client: Optional pre-built Serper client.
        llm_manager: Optional pre-built LLM manager for the extractor.
    """

    serves_types: Optional[set[ProductType]] = None  # all types
    source_tag: Source = Source.WEB_MARKETPLACE

    def __init__(
        self,
        *,
        name: str,
        domain: str,
        wait_ms: int = 9000,  # 9s: lazy-hydrated spec tables (divan/chitai-gorod/
                              # vseinstrumenti) didn't load within 5s on Scrape.do
        web_search_client=None,
        llm_manager=None,
    ) -> None:
        self.name = name
        self.domain = domain
        self.wait_ms = wait_ms
        self._search_client = web_search_client
        self._llm_manager = llm_manager

    def _get_search_client(self):
        if self._search_client is None:
            try:
                self._search_client = get_web_search_client()
            except Exception as exc:
                logger.warning("[MP:%s] search client unavailable: %s", self.name, exc)
        return self._search_client

    # ------------------------------------------------------------------
    # Accept predicate — any non-root product page on this domain
    # ------------------------------------------------------------------

    def _accept_url(self, link: str) -> bool:
        """Accept any non-root page on self.domain.

        The object-identifier Serper query (site:<domain> "<article>") provides
        precision; we just need to confirm the link is a real page, not the
        domain root or an unrelated subdomain.
        """
        if self.domain not in link:
            return False
        # Everything after the domain (strip leading slash)
        path_part = link.split(self.domain, 1)[1].strip("/")
        if not path_part:
            return False
        # Reject asset/file URLs (e.g. .../datasheet.pdf, .../image.jpg) — these
        # are not product pages. Compare the path before any query string.
        path_no_query = path_part.split("?", 1)[0].split("#", 1)[0].lower()
        if path_no_query.endswith(_NON_HTML_EXT):
            return False
        return True

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    async def find_url(self, context: ExtractionContext) -> Optional[str]:
        """Find the product URL on this marketplace using EAN/article/name search."""
        client = self._get_search_client()
        if client is None:
            logger.warning("[MP:%s] no search client — cannot discover URL", self.name)
            return None

        return await self._find_url_by_object_search(
            context,
            client,
            site_filter=self.domain,
            accept=self._accept_url,
            num_results=_SERPER_NUM_RESULTS,
            log_prefix=f"[MP:{self.name}]",
        )

    async def fetch(self, url: str) -> Optional[str]:
        """ZenRows fetch for this marketplace (JS rendering required)."""
        prefix = f"[MP:{self.name}]"
        result = await scrapedo_fetch(url, wait_ms=self.wait_ms)
        if not result.success or not result.content:
            logger.info(
                "%s ZenRows failed: %s (credits=%d)",
                prefix, result.error, result.credits_used,
            )
            return None

        html = result.content
        if len(html) < _MIN_VALID_HTML_LEN:
            logger.info(
                "%s HTML too short (%d chars) — possible block (credits=%d)",
                prefix, len(html), result.credits_used,
            )
            return None

        logger.info(
            "%s fetched %d chars (credits=%d)",
            prefix, len(html), result.credits_used,
        )
        return html

    def extract(
        self,
        html: str,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Sync stub — extraction is async (see enrich override)."""
        raise NotImplementedError(
            f"GenericMarketplace({self.name}).extract() must not be called directly; "
            "use enrich() which handles async LLM extraction."
        )

    # ------------------------------------------------------------------
    # Override enrich() — async extraction via LLM
    # ------------------------------------------------------------------

    async def enrich(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Full enrichment cycle with async LLM extraction.

        Overrides the base template method to support async extraction
        (llm_extract_attributes is async; extract() ABC is sync).
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
            logger.warning("%s fetch error: %s", prefix, exc)
            return []

        if not html or len(html) < 1000:
            logger.info(
                "%s empty/short content (%d chars)",
                prefix, len(html) if html else 0,
            )
            return []

        try:
            results = await llm_extract_attributes(
                html=html,
                context=context,
                targets=targets,
                source_tag=self.source_tag,
                llm_manager=self._llm_manager,
            )
        except Exception as exc:
            logger.warning("%s llm_extract error: %s", prefix, exc)
            return []

        logger.info("%s extracted %d attribute values", prefix, len(results))
        return results
