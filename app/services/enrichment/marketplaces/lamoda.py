"""app/services/enrichment/marketplaces/lamoda.py

LamodaMarketplace — clothing/footwear attribute enrichment via Lamoda.ru.

Reuses parsing logic from lamoda_scrapfly_source WITHOUT the Scrapfly
dependency (uses ZenRows instead). Fetching and URL discovery are the same
approach; parsing is fully shared via direct imports.

Product types served: CLOTHING, FOOTWEAR.

URL discovery: Serper ``site:lamoda.ru/p <brand> <name>``, first /p/ result.
Fetch: ZenRows residential proxy (same as updated lamoda_scrapfly_source).
Extract: _parse_lamoda_attributes + alias mapping + OzonCardSource._map_characteristics.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.marketplaces.base import MarketplaceSource, ProductType
from app.services.enrichment.sources.lamoda_scrapfly_source import (
    _LAMODA_TO_OZON_ALIASES,
    _LAMODA_PRODUCT_URL_RE,
    _MIN_VALID_HTML_LEN,
    _parse_lamoda_attributes,
    _is_model_reference,
    _SERPER_NUM_RESULTS,
)
from app.services.providers.factory import get_web_search_client
from app.services.providers.scrapedo_client import scrapedo_fetch

logger = logging.getLogger(__name__)

_PREFIX = "[MP:lamoda]"


class LamodaMarketplace(MarketplaceSource):
    """Lamoda.ru clothing/footwear attribute source.

    Serves CLOTHING and FOOTWEAR product types.  Does NOT require
    LAMODA_SCRAPFLY_ENABLED — this is the new standalone module.

    Args:
        web_search_client: Optional pre-built Serper client.  If None,
            get_web_search_client() is called lazily.
    """

    name = "lamoda"
    domain = "lamoda.ru"
    # Universal: no product-type hardcode. We simply search for the object on
    # Lamoda (site:lamoda.ru/p). If it isn't sold there (e.g. a TV), the search
    # finds no /p/ URL and enrich() returns [] — the search IS the type filter.
    serves_types: Optional[set[ProductType]] = None
    source_tag: Source = Source.LAMODA

    def __init__(self, web_search_client=None) -> None:
        self._search_client = web_search_client
        self._ozon_helper = None  # lazy-init to avoid import cost at module load

    def _get_search_client(self):
        if self._search_client is None:
            try:
                self._search_client = get_web_search_client()
            except Exception as exc:
                logger.warning("%s search client unavailable: %s", _PREFIX, exc)
        return self._search_client

    def _get_ozon_helper(self):
        if self._ozon_helper is None:
            from app.services.enrichment.sources.ozon_card_source import OzonCardSource
            self._ozon_helper = OzonCardSource(scrappey_key=None)
        return self._ozon_helper

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    @staticmethod
    def _accept_url(link: str) -> bool:
        """A Lamoda product page (/p/ slug)."""
        return bool(_LAMODA_PRODUCT_URL_RE.match(link)) and "/p/" in link

    async def find_url(self, context: ExtractionContext) -> Optional[str]:
        """Find the SAME product object on Lamoda — barcode/article first.

        Like Ozon/WB, anchor on an exact identifier when available
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
            site_filter="lamoda.ru/p",
            accept=self._accept_url,
            num_results=_SERPER_NUM_RESULTS,
            log_prefix=_PREFIX,
        )

    async def fetch(self, url: str) -> Optional[str]:
        """ZenRows residential proxy fetch for Lamoda (DataDome bypass)."""
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
                "%s HTML too short (%d chars) — likely block (credits=%d)",
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
        """Parse Lamoda JSON-in-HTML attributes and map to target AttributeValues.

        Reuses _parse_lamoda_attributes + _LAMODA_TO_OZON_ALIASES from the
        existing lamoda_scrapfly_source without any duplication.
        """
        attr_dict = _parse_lamoda_attributes(html)
        if not attr_dict:
            logger.info("%s 0 attributes extracted from page", _PREFIX)
            return []

        logger.info(
            "%s extracted %d raw attributes: %s",
            _PREFIX, len(attr_dict), list(attr_dict.keys())[:10],
        )

        # Build chars list: apply alias mapping, drop model-reference titles.
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

        ozon_helper = self._get_ozon_helper()
        evidence = f"lamoda:zenrows:{context.product_name[:40]}"
        raw_avs = ozon_helper._map_characteristics(
            chars,
            targets,
            context,
            mode="brand_line",
            title=context.product_name[:50],
            score=85.0,
            evidence_override=evidence,
        )

        # Patch source to LAMODA (OzonCardSource stamps OZON_CARD by default)
        return [av.model_copy(update={"source": Source.LAMODA}) for av in raw_avs]
