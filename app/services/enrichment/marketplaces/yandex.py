"""app/services/enrichment/marketplaces/yandex.py

YandexMarketMarketplace — universal attribute enrichment via market.yandex.ru.

Serves all product types (serves_types=None).  Uses ZenRows for page fetch
and the generic LLM extractor (llm_extract_attributes) since Yandex.Market's
JS-rendered DOM is not regex-friendly.

URL discovery: Serper ``site:market.yandex.ru <brand> <name>``,
first result containing ``/product`` or ``--`` in the path.
"""
from __future__ import annotations

import asyncio
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

_PREFIX = "[MP:yandex_market]"
_SERPER_NUM_RESULTS = 5
_MIN_VALID_HTML_LEN = 10_000


class YandexMarketMarketplace(MarketplaceSource):
    """Yandex.Market product attribute source (all product types).

    Uses ZenRows for JS-rendered page fetch and the shared LLM extractor
    for attribute parsing (no regex parser — Yandex DOM changes frequently).

    Args:
        web_search_client: Optional pre-built Serper client.
        llm_manager: Optional pre-built LLM manager for the extractor.
    """

    name = "yandex_market"
    domain = "market.yandex.ru"
    serves_types: Optional[set[ProductType]] = None  # all types
    source_tag: Source = Source.YANDEX_MARKET

    def __init__(self, web_search_client=None, llm_manager=None) -> None:
        self._search_client = web_search_client
        self._llm_manager = llm_manager

    def _get_search_client(self):
        if self._search_client is None:
            try:
                self._search_client = get_web_search_client()
            except Exception as exc:
                logger.warning("%s search client unavailable: %s", _PREFIX, exc)
        return self._search_client

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _accept_url(link: str) -> bool:
        """A Yandex.Market product page (slug contains /product or --)."""
        return "market.yandex.ru" in link and ("/product" in link or "--" in link)

    async def find_url(self, context: ExtractionContext) -> Optional[str]:
        """Find the SAME product object on Yandex.Market — barcode/article first.

        Like Ozon/WB, we anchor on an exact identifier when available
        (EAN → article/mpn) and fall back to brand+name. See
        MarketplaceSource._build_object_queries / _find_url_by_object_search.
        """
        client = self._get_search_client()
        if client is None:
            logger.warning("%s no search client — cannot discover URL", _PREFIX)
            return None

        return await self._find_url_by_object_search(
            context,
            client,
            site_filter="market.yandex.ru",
            accept=self._accept_url,
            num_results=_SERPER_NUM_RESULTS,
            log_prefix=_PREFIX,
        )

    async def fetch(self, url: str) -> Optional[str]:
        """ZenRows fetch for Yandex.Market (JS rendering required)."""
        result = await scrapedo_fetch(url, wait_ms=5000)
        if not result.success or not result.content:
            logger.info(
                "%s ZenRows failed: %s (credits=%d)",
                _PREFIX, result.error, result.credits_used,
            )
            return None

        html = result.content
        if len(html) < _MIN_VALID_HTML_LEN:
            logger.info(
                "%s HTML too short (%d chars) — possible block (credits=%d)",
                _PREFIX, len(html), result.credits_used,
            )
            return None

        logger.info(
            "%s fetched %d chars (credits=%d)",
            _PREFIX, len(html), result.credits_used,
        )
        return html

    def extract(
        self,
        html: str,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Sync stub — Yandex extraction is async (see enrich override)."""
        # This method is required by the ABC but the real work happens in
        # enrich() override below via the async llm_extract_attributes call.
        raise NotImplementedError(
            "YandexMarketMarketplace.extract() must not be called directly; "
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
        """Full enrichment cycle with async LLM extraction for Yandex.Market.

        Overrides the base template method to support async extraction
        (llm_extract_attributes is async; extract() ABC is sync).
        """
        prefix = _PREFIX
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
