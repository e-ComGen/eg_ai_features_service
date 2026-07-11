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
from app.services.enrichment.marketplaces.generic import ExtractorMarketplace
from app.services.enrichment.marketplaces.extractors.base import Extractor
from app.services.enrichment.marketplaces.extractors.embedded_json_family import EmbeddedJsonExtractor
from app.services.enrichment.marketplaces.extractors.llm_extractor_family import LlmSpecExtractor
from app.services.enrichment.marketplaces.extractors.static_table_family import StaticTableExtractor
from app.services.enrichment.marketplaces.extractors.suburl_family import SubUrlExtractor
from app.services.enrichment.marketplaces.brandshop import BrandshopMarketplace
from app.services.enrichment.marketplaces.fhby import FhByMarketplace
from app.services.enrichment.marketplaces.kupivip import KupiVipMarketplace
from app.services.enrichment.marketplaces.lamoda import LamodaMarketplace
from app.services.enrichment.marketplaces.vipavenue import VipavenueMarketplace
from app.services.enrichment.marketplaces.yandex import YandexMarketMarketplace

logger = logging.getLogger(__name__)

_llm = LlmSpecExtractor(slug="llm")

# Domain -> extractor chain. P1 populates eldorado.ru (EmbeddedJson family, __NEXT_DATA__
# Redux dump); P2 populates holodilnik.ru + nix.ru (StaticTable family); P3 populates
# dns-shop.ru (SubUrl family wrapping an inner StaticTable against /characteristics/).
# Remaining domains stay on DEFAULT_CHAIN until their own slice lands.
DOMAIN_EXTRACTORS: dict[str, list[Extractor]] = {
    "eldorado.ru": [
        EmbeddedJsonExtractor(
            slug="json:eldorado",
            specs_path=lambda root: next(iter(root["props"]["initialState"]["products-store-module"]["products"].values()))["attributeGroups"],
            grouped=True,
            group_items_key="propertyValues",
            name_key="name",
            value_key="propertyValues",
            unit_key="units",
        ),
        _llm,
    ],
    "holodilnik.ru": [
        StaticTableExtractor(
            slug="table:holodilnik",
            row_selector=".params-list--in-product .params-list__item:not(.params-list__item--caption)",
            name_selector=".params-list__item-name",
            value_selector=".params-list__item-value",
            name_strip_selectors=[".params-list__item-name-widget", ".d-none"],
        ),
        _llm,
    ],
    "nix.ru": [
        StaticTableExtractor(
            slug="table:nix",
            row_selector='table#PriceTable tr[id^="trs"]',
            name_selector='td[id^="tds"]:not([id^="tdsa"])',
            value_selector='td[id^="tdsa"] div',
            value_take_first=True,
        ),
        _llm,
    ],
    "dns-shop.ru": [
        SubUrlExtractor(
            slug="suburl:dns",
            url_rule=lambda u: u.split("?")[0].split("#")[0].rstrip("/") + "/characteristics/",
            inner=StaticTableExtractor(
                slug="table:dns",
                row_selector="li.product-characteristics__spec",
                name_selector=".product-characteristics__spec-title",
                value_selector=".product-characteristics__spec-value",
            ),
        ),
        _llm,
    ],
}

# Fallback chain for any domain not in DOMAIN_EXTRACTORS. llm-only for P0 -- this
# reproduces current GenericMarketplace behavior byte-for-byte (the parity gate).
DEFAULT_CHAIN: list[Extractor] = [_llm]

# Family whitelist: comma-separated family prefixes (slug before ":", or the whole slug
# when it has no ":"). Default "llm" ONLY -- deterministic families (json/table/suburl/
# ldjson) stay opt-in until their own A/B slice lands. Point of this flag: kill-switch
# per family without code changes.
MP_EXTRACTOR_FAMILIES: set[str] = {
    fam.strip() for fam in os.environ.get("MP_EXTRACTOR_FAMILIES", "llm").split(",") if fam.strip()
}

# Point kill-switch: comma-separated exact extractor slugs to disable regardless of family
# whitelist state (e.g. "json:eldorado" broke on a vendor markup change -- disable just it).
MP_EXTRACTOR_DISABLE: set[str] = {
    slug.strip() for slug in os.environ.get("MP_EXTRACTOR_DISABLE", "").split(",") if slug.strip()
}


def _extractor_family(slug: str) -> str:
    """Family prefix of an extractor slug: text before ':' or the whole slug if no ':'."""
    return slug.split(":", 1)[0]


def _build_extractor_chain(domain: str) -> list[Extractor]:
    """Resolve a domain's configured extractor chain, filtered by the family whitelist
    and the point-disable set. Falls back to DEFAULT_CHAIN when the domain has no
    dedicated entry in DOMAIN_EXTRACTORS."""
    chain = DOMAIN_EXTRACTORS.get(domain, DEFAULT_CHAIN)
    return [
        ex for ex in chain
        if _extractor_family(ex.slug) in MP_EXTRACTOR_FAMILIES
        and ex.slug not in MP_EXTRACTOR_DISABLE
    ]

# ---------------------------------------------------------------------------
# Cost knob: max number of marketplaces to attempt per product.
# 0 = no cap (try all). Set MARKETPLACE_MAX_DISCOVERY env var to limit.
# ---------------------------------------------------------------------------

_MAX_DISCOVERY: int = int(os.getenv("MARKETPLACE_MAX_DISCOVERY", "0"))

# Extra CIS marketplace domains, flag-gated (default OFF — pool byte-identical
# to today unless explicitly enabled).
EXTRA_MARKETPLACES_ENABLED: bool = os.environ.get("EXTRA_MARKETPLACES_ENABLED", "0") == "1"

# Lamoda.ru is NO-GO (DECISIONS.md 2026-07-10) — independent flag, default OFF, kept OFF.
# LAMODA_SCRAPFLY_ENABLED now only gates the rest of the web-marketplace router stage.
LAMODA_MARKETPLACE_ENABLED: bool = os.environ.get("LAMODA_MARKETPLACE_ENABLED", "0") == "1"

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
    ("jnsonline",      "jnsonline.ru"),       # одежда
    ("aromacode",      "aromacode.ru"),       # косметика-парфюм
    ("akusherstvo",    "akusherstvo.ru"),     # мягкие игрушки
    ("votonia",        "votonia.ru"),         # мягкие игрушки
]
_specialists = [
    ExtractorMarketplace(name=n, domain=d, extractors=_build_extractor_chain(d))
    for n, d in _SPECIALIST_DOMAINS
]

# vipavenue.ru — plain httpx GET (no anti-bot), bespoke accordion parser. Not gated by a
# separate flag: rides the general web-marketplace router stage (LAMODA_SCRAPFLY_ENABLED),
# same as goldapple/sportmaster/etc.
_vipavenue = VipavenueMarketplace()
# kupivip.ru — plain httpx GET (no anti-bot), bespoke detail-block text parser. Not gated
# by a separate flag: rides the general web-marketplace router stage (LAMODA_SCRAPFLY_ENABLED),
# same as goldapple/sportmaster/vipavenue.
_kupivip = KupiVipMarketplace()
# brandshop.ru — plain httpx GET (no anti-bot), product-data block + description-list
# parser. Not gated by a separate flag: rides the general web-marketplace router stage
# (LAMODA_SCRAPFLY_ENABLED), same as goldapple/sportmaster/vipavenue/kupivip.
_brandshop = BrandshopMarketplace()
# fh.by — plain httpx GET (no anti-bot), __NEXT_DATA__ JSON parser (Next.js app, not
# CSS-selector based). Not gated by a separate flag: rides the general web-marketplace
# router stage (LAMODA_SCRAPFLY_ENABLED), same as goldapple/sportmaster/vipavenue/kupivip/
# brandshop. Confirmed GO 2026-07-10 (_reports/probe_4_new_analogs.txt).
_fhby = FhByMarketplace()

# Reorder: fast plain-httpx specialists first, then slow proxy-backed scrapedo specialists.
# Previously the 22 scrapedo specialists ran first, consuming the request timeout budget
# before fast hits could fire (see _reports/diag_prada_timeout.log).
_specialists = [_vipavenue, _kupivip, _brandshop, _fhby, *_specialists]

# ---------------------------------------------------------------------------
# Extra CIS marketplace domains (flag-gated — see EXTRA_MARKETPLACES_ENABLED).
# scrape.do-verified 2026-07-07, rich-spec, see cis_pool coverage map
# ---------------------------------------------------------------------------

EXTRA_MARKETPLACE_DOMAINS = [
    ("bakuelectronics", "bakuelectronics.az"),
    ("kontakt", "kontakt.az"),
    ("umico", "umico.az"),
    ("5element", "5element.by"),
    ("zoommer", "zoommer.ge"),
    ("mechta", "mechta.kz"),
    ("sulpak", "sulpak.kz"),
    ("technodom", "technodom.kz"),
    ("flip", "flip.kz"),
    ("kaspi", "kaspi.kz"),
    ("satu", "satu.kz"),
    ("price", "price.ru"),
    ("holodilnik", "holodilnik.ru"),
    ("armtek", "armtek.ru"),
    ("autodoc", "autodoc.ru"),
    ("zzap", "zzap.ru"),
    ("letu", "letu.ru"),
    ("randewoo", "randewoo.ru"),
    ("rivegauche", "rivegauche.ru"),
    ("citilink", "citilink.ru"),
    ("dns_shop", "dns-shop.ru"),
    ("eldorado", "eldorado.ru"),
    ("mvideo", "mvideo.ru"),
    ("nix", "nix.ru"),
    ("notik", "notik.ru"),
    ("pult", "pult.ru"),
    ("askona", "askona.ru"),
    ("sima_land", "sima-land.ru"),
    ("lemanapro", "lemanapro.ru"),
    ("585zolotoy", "585zolotoy.ru"),
    ("apteka", "apteka.ru"),
    ("bestwatch", "bestwatch.ru"),
    ("texnomart", "texnomart.uz"),
    ("uzum", "uzum.uz"),
]

# Scrape.do geo by TLD — non-RU TLDs get their own geo; everything else
# (incl. .by, no dedicated hook here) defaults to "ru" — acceptable, the
# probe confirmed these domains fetch fine under the default geo.
_TLD_GEO: dict[str, str] = {"kz": "kz", "az": "az", "ge": "ge", "uz": "uz"}


def _geo_for_domain(domain: str) -> str:
    """Scrape.do geo code by TLD (see _TLD_GEO); default 'ru'."""
    tld = domain.rsplit(".", 1)[-1]
    return _TLD_GEO.get(tld, "ru")


_extra_specialists = [
    ExtractorMarketplace(
        name=n, domain=d, geo=_geo_for_domain(d), extractors=_build_extractor_chain(d),
    )
    for n, d in EXTRA_MARKETPLACE_DOMAINS
]

# ---------------------------------------------------------------------------
# Pool definition — flat, universal, no product-type routing
# ---------------------------------------------------------------------------

# Ordered list of universal web marketplaces. Each one searches for the SAME
# product object (by brand+name); a site that doesn't carry it just finds no
# URL and contributes nothing. Order = cost/coverage priority: the broadest,
# always-relevant site (Yandex.Market) first, then specialist catalogues whose
# search naturally no-ops on out-of-domain products (Lamoda for a TV → []).
# Adding a marketplace is a one-line append — no per-type wiring.
MARKETPLACES: list[MarketplaceSource] = [_yandex]
if LAMODA_MARKETPLACE_ENABLED:
    # Lamoda.ru NO-GO (DECISIONS.md 2026-07-10) — excluded from the pool unless this
    # flag is explicitly set (default OFF, independent of LAMODA_SCRAPFLY_ENABLED).
    MARKETPLACES.append(_lamoda)
MARKETPLACES.extend(_specialists)
if EXTRA_MARKETPLACES_ENABLED:
    # Appended, not interleaved — preserves existing cost/coverage order.
    MARKETPLACES = [*MARKETPLACES, *_extra_specialists]


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
