"""Unit tests for the adaptive multi-site composition harvester.

Covers:
  - open-before-browser URL ordering
  - dead-domain skip (domain_health gate)
  - first-verified-hit is returned and loop stops
  - BrowserFetcher is reused (not re-created) across multiple browser-site calls
  - graceful fallthrough when all sites yield empty/mismatched content
  - brand-gate rejects pages that don't match the brand
  - Serper dedup: same URL from both queries appears once in candidate list
"""

from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.services.enrichment.sources.multisite_composition import (
    APPAREL_SITE_POOL,
    _extract_host,
    _order_by_pool_preference,
    _pool_rank,
    harvest_composition,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_organic_result(link: str) -> MagicMock:
    r = MagicMock()
    r.link = link
    return r


def _make_serper_results(urls: list[str]) -> MagicMock:
    res = MagicMock()
    res.organic_results = [_make_organic_result(u) for u in urls]
    return res


def run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# Pool structure sanity
# ---------------------------------------------------------------------------


class TestPoolStructure:
    def test_open_sites_present(self):
        open_domains = {e.domain for e in APPAREL_SITE_POOL if not e.needs_browser}
        for expected in ("kixbox.ru", "sneakerhead.ru", "basketshop.ru", "brandshop.ru",
                         "street-beat.ru", "blankstyle.com"):
            assert expected in open_domains, f"{expected} missing from open pool"

    def test_browser_sites_present(self):
        browser_domains = {e.domain for e in APPAREL_SITE_POOL if e.needs_browser}
        assert "sportmaster.ru" in browser_domains
        assert "lamoda.ru" in browser_domains

    def test_browser_sites_have_warmup_url(self):
        for entry in APPAREL_SITE_POOL:
            if entry.needs_browser:
                assert entry.warmup_url, f"{entry.domain} browser-site must have warmup_url"

    def test_pool_domains_unique(self):
        domains = [e.domain for e in APPAREL_SITE_POOL]
        assert len(domains) == len(set(domains)), "duplicate domains in APPAREL_SITE_POOL"


# ---------------------------------------------------------------------------
# URL ordering: open sites first, then browser sites, unknown last
# ---------------------------------------------------------------------------


class TestUrlOrdering:
    def test_open_sites_ranked_before_browser(self):
        """Pool open-site rank < browser-site rank."""
        open_entry = next(e for e in APPAREL_SITE_POOL if not e.needs_browser)
        browser_entry = next(e for e in APPAREL_SITE_POOL if e.needs_browser)
        assert _pool_rank(f"https://{open_entry.domain}/p/1") < _pool_rank(
            f"https://{browser_entry.domain}/p/1"
        )

    def test_unknown_site_has_max_rank(self):
        assert _pool_rank("https://random-unknown-site.ru/p/1") == 9999

    def test_ordering_puts_open_first(self):
        urls = [
            "https://lamoda.ru/p/123/brand-slug/",       # browser
            "https://kixbox.ru/champion-hoodie/",         # open
            "https://sneakerhead.ru/nike-tee/",           # open
            "https://sportmaster.ru/product/1234/",       # browser
        ]
        ordered = _order_by_pool_preference(urls)
        open_set = {_extract_host(u) for u in ordered
                    if not any(e.domain == _extract_host(u) and e.needs_browser
                               for e in APPAREL_SITE_POOL)}
        browser_set = {_extract_host(u) for u in ordered
                       if any(e.domain == _extract_host(u) and e.needs_browser
                              for e in APPAREL_SITE_POOL)}
        if open_set and browser_set:
            last_open_idx = max(
                i for i, u in enumerate(ordered) if _extract_host(u) in open_set
            )
            first_browser_idx = min(
                i for i, u in enumerate(ordered) if _extract_host(u) in browser_set
            )
            assert last_open_idx < first_browser_idx, (
                "All open sites must appear before browser sites in ordered list"
            )

    def test_www_prefix_stripped_correctly(self):
        assert _extract_host("https://www.kixbox.ru/p/1") == "kixbox.ru"
        assert _extract_host("https://kixbox.ru/p/1") == "kixbox.ru"


# ---------------------------------------------------------------------------
# Dead-domain skip
# ---------------------------------------------------------------------------


class TestDeadDomainSkip:
    def test_dead_domain_is_skipped(self):
        """should_skip_scrappey → True means the URL is never attempted."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://kixbox.ru/champ/"])
        )

        fetch_calls: list[str] = []

        async def mock_fetch(url):
            fetch_calls.append(url)
            return None

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=True,
            ),
            patch("app.services.url_fetcher.fetch_url_content", new=mock_fetch),
        ):
            result = run(
                harvest_composition(
                    "Champion Hoodie", "Champion", serper_client=serper
                )
            )

        assert result is None
        assert fetch_calls == [], "fetch must NOT be called for a dead domain"

    def test_live_domain_is_not_skipped(self):
        """When domain_health says alive, the URL IS fetched."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://kixbox.ru/champ/"])
        )

        fetch_called: list[bool] = []

        async def mock_fetch_url(url):
            fetch_called.append(True)
            return None  # no content — fallthrough

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=False,
            ),
            patch(
                "app.services.url_fetcher.fetch_url_content",
                new=mock_fetch_url,
            ),
            patch(
                "app.services.enrichment.sources.multisite_composition._INTER_REQUEST_DELAY",
                0,
            ),
        ):
            result = run(
                harvest_composition(
                    "Champion Hoodie", "Champion", serper_client=serper
                )
            )

        assert fetch_called, "fetch MUST be called for a live domain"


# ---------------------------------------------------------------------------
# First verified hit is returned and loop stops early
# ---------------------------------------------------------------------------


class TestFirstHitStopsLoop:
    def test_returns_first_successful_site(self):
        """Second URL should NOT be fetched once first URL yields a composition."""
        page_html = (
            "<html><head><title>Champion Reverse Weave Hoodie</title></head>"
            "<body><p>Состав: 80% хлопок, 20% полиэстер</p></body></html>"
        )

        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results([
                "https://kixbox.ru/champion-hoodie/",
                "https://sneakerhead.ru/champion/",
            ])
        )

        fetch_call_urls: list[str] = []

        async def mock_fetch_url(url):
            fetch_call_urls.append(url)
            fr = MagicMock()
            fr.raw_html = page_html
            fr.content = page_html
            return fr

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=False,
            ),
            patch(
                "app.services.url_fetcher.fetch_url_content",
                new=mock_fetch_url,
            ),
            patch(
                "app.services.enrichment.sources.multisite_composition._INTER_REQUEST_DELAY",
                0,
            ),
        ):
            result = run(
                harvest_composition(
                    "Champion Reverse Weave Hoodie",
                    "Champion",
                    max_sites=4,
                    serper_client=serper,
                )
            )

        assert result is not None, "Should return composition from first site"
        assert "хлопок" in result["composition"]
        assert result["site"] == "kixbox.ru"
        # Second URL should NOT have been fetched (early exit)
        assert len(fetch_call_urls) == 1, (
            f"Loop must stop after first hit, got {len(fetch_call_urls)} fetches"
        )

    def test_result_has_required_keys(self):
        """Result dict must include all expected keys."""
        page_html = (
            "<html><head><title>Nike Sportswear Club T-Shirt</title></head>"
            "<body><p>Материал: 100% хлопок</p></body></html>"
        )

        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://kixbox.ru/nike-tee/"])
        )

        async def mock_fetch_url(url):
            fr = MagicMock()
            fr.raw_html = page_html
            fr.content = page_html
            return fr

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=False,
            ),
            patch(
                "app.services.url_fetcher.fetch_url_content",
                new=mock_fetch_url,
            ),
            patch(
                "app.services.enrichment.sources.multisite_composition._INTER_REQUEST_DELAY",
                0,
            ),
        ):
            result = run(
                harvest_composition(
                    "Футболка мужская Nike Sportswear Club",
                    "Nike",
                    serper_client=serper,
                )
            )

        assert result is not None
        for key in ("composition", "material", "source_url", "site", "evidence"):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Graceful fallthrough (no hit)
# ---------------------------------------------------------------------------


class TestGracefulFallthrough:
    def test_returns_none_when_no_site_yields_composition(self):
        """All sites return empty content → harvest returns None gracefully."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results([
                "https://kixbox.ru/levis/",
                "https://sneakerhead.ru/levis/",
            ])
        )

        async def mock_fetch_url(url):
            return None  # every site fails to return content

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=False,
            ),
            patch(
                "app.services.url_fetcher.fetch_url_content",
                new=mock_fetch_url,
            ),
            patch(
                "app.services.enrichment.sources.multisite_composition._INTER_REQUEST_DELAY",
                0,
            ),
        ):
            result = run(
                harvest_composition(
                    "Джинсы мужские Levis 501 Original",
                    "Levis",
                    serper_client=serper,
                )
            )

        assert result is None

    def test_returns_none_when_brand_gate_rejects_all_pages(self):
        """Brand mismatch on every page → None returned."""
        wrong_brand_html = (
            "<html><head><title>Adidas Hoodie</title></head>"
            "<body><p>Состав: 70% полиэстер, 30% хлопок</p></body></html>"
        )

        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://kixbox.ru/adidas/"])
        )

        async def mock_fetch_url(url):
            fr = MagicMock()
            fr.raw_html = wrong_brand_html
            fr.content = wrong_brand_html
            return fr

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=False,
            ),
            patch(
                "app.services.url_fetcher.fetch_url_content",
                new=mock_fetch_url,
            ),
            patch(
                "app.services.enrichment.sources.multisite_composition._INTER_REQUEST_DELAY",
                0,
            ),
        ):
            result = run(
                harvest_composition(
                    "Champion Hoodie",
                    "Champion",   # page has Adidas, not Champion
                    serper_client=serper,
                )
            )

        assert result is None

    def test_returns_none_on_empty_product_name(self):
        result = run(harvest_composition("", "Nike"))
        assert result is None

    def test_returns_none_on_empty_brand(self):
        result = run(harvest_composition("Nike Tee", ""))
        assert result is None


# ---------------------------------------------------------------------------
# BrowserFetcher is reused (NOT recreated) across browser-site calls
# ---------------------------------------------------------------------------


class TestBrowserFetcherReuse:
    def test_browser_fetcher_created_once_for_multiple_browser_sites(self):
        """Two browser-site URLs → only ONE BrowserFetcher instance created."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results([
                "https://sportmaster.ru/product/12345/",
                "https://lamoda.ru/p/ab123/brand-slug/",
            ])
        )

        mock_bf_instance = MagicMock()
        mock_bf_instance.fetch_with_warmup = AsyncMock(return_value=None)
        mock_bf_instance.close = AsyncMock()
        mock_bf_instance._closed = False

        constructor_call_count = [0]

        def mock_bf_constructor():
            constructor_call_count[0] += 1
            return mock_bf_instance

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=False,
            ),
            patch(
                "app.services.providers.browser_fetcher.BrowserFetcher",
                side_effect=mock_bf_constructor,
            ),
            patch(
                "app.services.enrichment.sources.multisite_composition._INTER_REQUEST_DELAY",
                0,
            ),
        ):
            result = run(
                harvest_composition(
                    "Толстовка Champion",
                    "Champion",
                    max_sites=4,
                    serper_client=serper,
                )
            )

        # BrowserFetcher must be instantiated at most once
        assert constructor_call_count[0] <= 1, (
            f"BrowserFetcher created {constructor_call_count[0]} times; expected ≤1"
        )

    def test_injected_browser_fetcher_is_not_recreated(self):
        """When browser_fetcher is injected, harvest must NOT create a new one."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results([
                "https://lamoda.ru/p/xyz/brand-product/",
            ])
        )

        injected_bf = MagicMock()
        injected_bf.fetch_with_warmup = AsyncMock(return_value=None)
        injected_bf.close = AsyncMock()
        injected_bf._closed = False

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=False,
            ),
            patch(
                "app.services.providers.browser_fetcher.BrowserFetcher",
            ) as mock_bf_cls,
            patch(
                "app.services.enrichment.sources.multisite_composition._INTER_REQUEST_DELAY",
                0,
            ),
        ):
            run(
                harvest_composition(
                    "Толстовка Nike",
                    "Nike",
                    serper_client=serper,
                    browser_fetcher=injected_bf,
                )
            )

        # The BrowserFetcher class must NOT have been instantiated
        mock_bf_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Serper URL deduplication
# ---------------------------------------------------------------------------


class TestSerperDedup:
    def test_same_url_from_both_queries_fetched_once(self):
        """URL appearing in both Serper queries must only be tried once."""
        shared_url = "https://kixbox.ru/champion-hoodie/"

        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results([shared_url])
        )

        fetch_call_urls: list[str] = []

        async def mock_fetch_url(url):
            fetch_call_urls.append(url)
            return None

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=False,
            ),
            patch(
                "app.services.url_fetcher.fetch_url_content",
                new=mock_fetch_url,
            ),
            patch(
                "app.services.enrichment.sources.multisite_composition._INTER_REQUEST_DELAY",
                0,
            ),
        ):
            run(
                harvest_composition(
                    "Champion Hoodie",
                    "Champion",
                    serper_client=serper,
                )
            )

        url_count = fetch_call_urls.count(shared_url)
        assert url_count <= 1, (
            f"Shared URL was fetched {url_count} times; must be deduplicated to 1"
        )


# ---------------------------------------------------------------------------
# Routing: open URLs go through url_fetcher, browser URLs through BrowserFetcher
# ---------------------------------------------------------------------------


class TestRouting:
    def test_open_url_uses_url_fetcher(self):
        """kixbox.ru (open) must use url_fetcher, NOT BrowserFetcher."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://kixbox.ru/nike-tee/"])
        )

        url_fetcher_calls: list[str] = []
        bf_calls: list[str] = []

        async def mock_url_fetcher(url):
            url_fetcher_calls.append(url)
            return None

        mock_bf = MagicMock()
        mock_bf.fetch_with_warmup = AsyncMock(side_effect=lambda *a, **kw: bf_calls.append(a[0]) or None)
        mock_bf.fetch = AsyncMock(side_effect=lambda *a, **kw: bf_calls.append(a[0]) or None)

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=False,
            ),
            patch(
                "app.services.url_fetcher.fetch_url_content",
                new=mock_url_fetcher,
            ),
            patch(
                "app.services.enrichment.sources.multisite_composition._INTER_REQUEST_DELAY",
                0,
            ),
        ):
            run(
                harvest_composition(
                    "Nike Tee",
                    "Nike",
                    serper_client=serper,
                    browser_fetcher=mock_bf,  # injected so it won't be auto-created
                )
            )

        assert url_fetcher_calls, "url_fetcher must be called for open site"
        assert bf_calls == [], "BrowserFetcher must NOT be called for open site"

    def test_browser_url_uses_browser_fetcher(self):
        """lamoda.ru (browser) must use BrowserFetcher.fetch_with_warmup."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://lamoda.ru/p/abc/brand-slug/"])
        )

        url_fetcher_calls: list[str] = []
        bf_warmup_calls: list[str] = []

        async def mock_url_fetcher(url):
            url_fetcher_calls.append(url)
            return None

        mock_bf = MagicMock()
        mock_bf._closed = False
        mock_bf.fetch_with_warmup = AsyncMock(
            side_effect=lambda url, warmup_url=None, **kw: bf_warmup_calls.append(url) or None
        )

        with (
            patch(
                "app.services.enrichment.sources.multisite_composition.should_skip_scrappey",
                return_value=False,
            ),
            patch(
                "app.services.url_fetcher.fetch_url_content",
                new=mock_url_fetcher,
            ),
            patch(
                "app.services.enrichment.sources.multisite_composition._INTER_REQUEST_DELAY",
                0,
            ),
        ):
            run(
                harvest_composition(
                    "Толстовка Lamoda",
                    "Lamoda",
                    serper_client=serper,
                    browser_fetcher=mock_bf,
                )
            )

        assert bf_warmup_calls, "BrowserFetcher.fetch_with_warmup must be called for browser site"
        assert url_fetcher_calls == [], "url_fetcher must NOT be called for browser site"
