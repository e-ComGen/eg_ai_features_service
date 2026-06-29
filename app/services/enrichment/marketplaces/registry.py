"""app/services/enrichment/marketplaces/registry.py

Marketplace pool registry and cost-aware router.

DESIGN — every web marketplace is UNIVERSAL; we just search for the object:
  There is NO product-type hardcode (no "clothing → Lamoda" routing). Exactly
  like Ozon/WB, each marketplace simply tries to FIND THE SAME PRODUCT OBJECT
  by brand+name. The search is the natural filter:
    • Yandex.Market sells everything → finds the object for any product.
    • Lamoda sells apparel → finds a /p/ URL for clothing, finds NOTHING for a
      TV, so it self-returns [] without any type check.
  (Ozon/WB are also universal but still live as native pipeline sources that
  run first; unifying them into this module is a TODO.)

POOL = flat ordered list (MARKETPLACES). Add a new site = append one instance.

MarketplaceRouter.fill_gaps():
  Cost-aware: iterates the pool in order, enriches only still-empty targets,
  stops early when all targets are filled (so cheaper/universal sites run
  first and specialists only fire when gaps remain).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    TargetAttribute,
)
from app.services.enrichment.marketplaces.base import MarketplaceSource
from app.services.enrichment.marketplaces.generic import GenericMarketplace
from app.services.enrichment.marketplaces.lamoda import LamodaMarketplace
from app.services.enrichment.marketplaces.yandex import YandexMarketMarketplace

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost knob: max number of marketplaces to attempt per product.
# 0 = no cap (try all). Set MARKETPLACE_MAX_DISCOVERY env var to limit.
# ---------------------------------------------------------------------------

_MAX_DISCOVERY: int = int(os.getenv("MARKETPLACE_MAX_DISCOVERY", "0"))

# ---------------------------------------------------------------------------
# Singleton instances — one per marketplace (stateless beyond injected deps)
# ---------------------------------------------------------------------------

_lamoda = LamodaMarketplace()
_yandex = YandexMarketMarketplace()

# ---------------------------------------------------------------------------
# Universal specialist web-marketplaces (ZenRows fetch + grounded LLM extract).
# Each just searches for the object by article/EAN; the per-site object search
# is the natural type filter (a fridge won't be found on zdravcity → []).
# ---------------------------------------------------------------------------

_SPECIALIST_DOMAINS = [
    ("zdravcity",      "zdravcity.ru"),       # аптека/БАД
    ("eapteka",        "eapteka.ru"),         # аптека
    ("4lapy",          "4lapy.ru"),           # зоотовары
    ("iledebeaute",    "iledebeaute.ru"),     # косметика/парфюм
    ("goldapple",      "goldapple.ru"),       # косметика
    ("kant",           "kant.ru"),            # спорт/туризм
    ("sportmaster",    "sportmaster.ru"),     # спорт
    ("exist",          "exist.ru"),           # автозапчасти (OEM)
    ("emex",           "emex.ru"),            # автозапчасти (OEM)
    ("divan",          "divan.ru"),           # мебель
    ("hoff",           "hoff.ru"),            # мебель/дом
    ("detmir",         "detmir.ru"),          # детские товары
    ("book24",         "book24.ru"),          # книги (ISBN)
    ("chitai_gorod",   "chitai-gorod.ru"),    # книги
    ("vseinstrumenti", "vseinstrumenti.ru"),  # инструмент/DIY
    ("petrovich",      "petrovich.ru"),       # стройматериалы
    ("sunlight",       "sunlight.net"),       # ювелирка
    ("alltime",        "alltime.ru"),         # часы
]
_specialists = [GenericMarketplace(name=n, domain=d) for n, d in _SPECIALIST_DOMAINS]

# ---------------------------------------------------------------------------
# Pool definition — flat, universal, no product-type routing
# ---------------------------------------------------------------------------

# Ordered list of universal web marketplaces. Each one searches for the SAME
# product object (by brand+name); a site that doesn't carry it just finds no
# URL and contributes nothing. Order = cost/coverage priority: the broadest,
# always-relevant site (Yandex.Market) first, then specialist catalogues whose
# search naturally no-ops on out-of-domain products (Lamoda for a TV → []).
# Adding a marketplace is a one-line append — no per-type wiring.
MARKETPLACES: list[MarketplaceSource] = [_yandex, _lamoda, *_specialists]


def get_marketplace_pool(context: ExtractionContext) -> list[MarketplaceSource]:
    """Return the ordered universal marketplace pool.

    No product-type classification: every marketplace is tried in order and
    self-filters via its own object search (find_url). The `context` argument
    is kept for API stability / future per-product ordering hooks.
    """
    return list(MARKETPLACES)


# ---------------------------------------------------------------------------
# Cost-aware router
# ---------------------------------------------------------------------------

class MarketplaceRouter:
    """Routes enrichment requests across the marketplace pool, cost-aware.

    Usage:
        router = MarketplaceRouter()
        new_values = await router.fill_gaps(context, targets, already_filled_ids)

    The router:
      1. Gets the pool for the product type.
      2. For each marketplace in order:
         a. Computes still-empty target attributes.
         b. If none remain — stops immediately (early exit).
         c. Calls marketplace.enrich(context, gap_targets).
         d. Accumulates results; updates filled-id set.
      3. Returns all accumulated AttributeValues from supplemental sources.
    """

    async def fill_gaps(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled_ids: Optional[set[int]] = None,
    ) -> list[AttributeValue]:
        """Fill attribute gaps using supplemental marketplace sources.

        Parameters
        ----------
        context:
            Extraction context for this product.
        targets:
            All target attributes to fill.
        already_filled_ids:
            Set of attribute IDs already filled with high-confidence values
            by earlier pipeline stages (Ozon, WB, description, etc.).
            These are excluded from gap targets passed to each marketplace.

        Returns
        -------
        List of AttributeValues from supplemental marketplaces.
        Empty list if pool is empty or all targets already filled.
        """
        filled_ids: set[int] = set(already_filled_ids or set())
        pool = get_marketplace_pool(context)

        # Apply discovery cap if set (MARKETPLACE_MAX_DISCOVERY env var).
        if _MAX_DISCOVERY > 0:
            pool = pool[:_MAX_DISCOVERY]

        if not pool:
            logger.debug(
                "[MarketplaceRouter] empty pool for product='%s'",
                context.product_name[:60],
            )
            return []

        accumulated: list[AttributeValue] = []

        for marketplace in pool:
            # Compute gap: targets not yet filled
            gap_targets = [t for t in targets if t.id not in filled_ids]
            if not gap_targets:
                logger.info(
                    "[MarketplaceRouter] all %d targets filled — stopping after %s",
                    len(targets), marketplace.name,
                )
                break

            logger.info(
                "[MarketplaceRouter] trying %s for %d gap targets (product='%s')",
                marketplace.name, len(gap_targets), context.product_name[:60],
            )

            try:
                new_avs = await marketplace.enrich(context, gap_targets)
            except Exception as exc:
                logger.warning(
                    "[MarketplaceRouter] %s.enrich() raised unexpectedly: %s",
                    marketplace.name, exc,
                )
                new_avs = []

            count = len(new_avs)
            logger.info(
                "[MarketplaceRouter] %s contributed %d values",
                marketplace.name, count,
            )

            # Update filled set with newly found attribute IDs
            for av in new_avs:
                filled_ids.add(av.attribute_id)

            accumulated.extend(new_avs)

        return accumulated


# ---------------------------------------------------------------------------
# STANDALONE TEST INSTRUCTIONS (для Opus)
# ---------------------------------------------------------------------------
# Запуск LamodaMarketplace на одном товаре (без реальных сетевых вызовов):
#
#   python -c "
#   import asyncio
#   from unittest.mock import AsyncMock, MagicMock
#   from app.services.enrichment.marketplaces.lamoda import LamodaMarketplace
#   from app.services.enrichment.base import ExtractionContext, TargetAttribute
#
#   async def test():
#       mp = LamodaMarketplace(web_search_client=None)
#       mp._search_client = MagicMock()
#       mp._search_client.search = AsyncMock(return_value=MagicMock(
#           organic_results=[MagicMock(link='https://www.lamoda.ru/p/test-slug/')]
#       ))
#       # Patch fetch to return stub HTML with Lamoda attribute triple
#       mp.fetch = AsyncMock(return_value='x'*50000 + '\"key\":\"k1\",\"title\":\"Цвет\",\"value\":\"Чёрный\"')
#       ctx = ExtractionContext(product_id=1, product_name='Футболка Nike', category_id=123,
#                               category_path=['Одежда'], brand='Nike')
#       targets = [TargetAttribute(id=10, name='Цвет товара', type='enum')]
#       result = await mp.enrich(ctx, targets)
#       print('LamodaMarketplace result:', result)
#   asyncio.run(test())
#   "
#
# Запуск YandexMarketMarketplace (мок LLM + fetch):
#
#   python -c "
#   import asyncio
#   from unittest.mock import AsyncMock, MagicMock, patch
#   from app.services.enrichment.marketplaces.yandex import YandexMarketMarketplace
#   from app.services.enrichment.base import ExtractionContext, TargetAttribute
#
#   async def test():
#       mp = YandexMarketMarketplace(web_search_client=None)
#       mp._search_client = MagicMock()
#       mp._search_client.search = AsyncMock(return_value=MagicMock(
#           organic_results=[MagicMock(link='https://market.yandex.ru/product--iphone/123')]
#       ))
#       mp.fetch = AsyncMock(return_value='x'*11000)
#       with patch('app.services.enrichment.marketplaces.llm_extractor.get_main_manager') as mock_llm:
#           from app.services.enrichment.marketplaces.llm_extractor import _ExtractedAttributes, _ExtractedPair
#           mgr = AsyncMock()
#           mgr.structured_request = AsyncMock(return_value=(
#               _ExtractedAttributes(attributes=[_ExtractedPair(name='Бренд', value='Apple')]), None
#           ))
#           mock_llm.return_value = mgr
#           ctx = ExtractionContext(product_id=2, product_name='iPhone 15', category_id=456,
#                                   category_path=['Смартфон'], brand='Apple')
#           targets = [TargetAttribute(id=20, name='Бренд', type='enum')]
#           result = await mp.enrich(ctx, targets)
#           print('YandexMarketMarketplace result:', result)
#   asyncio.run(test())
#   "
#
# Обе команды запускать из директории CpAiFeatures-web-fetch с активным venv.
