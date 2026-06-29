"""app/services/enrichment/marketplaces

Supplemental web-marketplace pool for attribute gap-filling.

ALWAYS-ON sources (Ozon, WB) are NOT here — they live in the main pipeline.
This package adds universal web marketplaces (Yandex.Market, Lamoda) that each
just SEARCH FOR THE PRODUCT OBJECT (by brand+name), exactly like Ozon/WB. There
is no product-type routing: the per-site object search is the natural filter.

Quick import:
    from app.services.enrichment.marketplaces.registry import (
        MarketplaceRouter,
        get_marketplace_pool,
        MARKETPLACES,
    )
"""
