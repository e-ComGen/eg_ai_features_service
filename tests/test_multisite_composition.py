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

        async def mock_fetch_url(url, **kwargs):
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

        async def mock_fetch_url(url, **kwargs):
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
        # The specific winning site depends on pool ordering; what matters is early exit.
        assert result["site"] in ("kixbox.ru", "sneakerhead.ru")
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

        async def mock_fetch_url(url, **kwargs):
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

        async def mock_fetch_url(url, **kwargs):
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

        async def mock_fetch_url(url, **kwargs):
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

        async def mock_fetch_url(url, **kwargs):
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

        async def mock_url_fetcher(url, **kwargs):
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

    def test_open_url_passes_force_scrappey_to_url_fetcher(self):
        """harvest_composition must pass force_scrappey=True when fetching open sites."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://kixbox.ru/nike-tee/"])
        )

        captured_kwargs: list[dict] = []

        async def mock_url_fetcher(url, **kwargs):
            captured_kwargs.append(kwargs)
            return None

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
                )
            )

        assert captured_kwargs, "fetch_url_content must have been called"
        assert captured_kwargs[0].get("force_scrappey") is True, (
            "harvest_composition must pass force_scrappey=True for open sites"
        )

    def test_browser_url_does_not_call_url_fetcher(self):
        """lamoda.ru (browser) must use BrowserFetcher, url_fetcher NOT called."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://lamoda.ru/p/abc/brand-slug/"])
        )

        url_fetcher_calls: list[str] = []

        async def mock_url_fetcher(url, **kwargs):
            url_fetcher_calls.append(url)
            return None

        mock_bf = MagicMock()
        mock_bf._closed = False
        mock_bf.fetch_with_warmup = AsyncMock(return_value=None)

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
                    "Толстовка Nike",
                    "Nike",
                    serper_client=serper,
                    browser_fetcher=mock_bf,
                )
            )

        assert url_fetcher_calls == [], (
            "url_fetcher (and thus Scrappey) must NOT be called for browser-strategy sites"
        )

    def test_browser_url_uses_browser_fetcher(self):
        """lamoda.ru (browser) must use BrowserFetcher.fetch_with_warmup."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://lamoda.ru/p/abc/brand-slug/"])
        )

        url_fetcher_calls: list[str] = []
        bf_warmup_calls: list[str] = []

        async def mock_url_fetcher(url, **kwargs):
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


# ---------------------------------------------------------------------------
# SPA auto-escalation: httpx shell → browser re-render → composition found
# ---------------------------------------------------------------------------


class TestSpaAutoEscalation:
    """When httpx returns a JS-SPA shell with no composition, harvest_composition
    must auto-escalate to BrowserFetcher.fetch on the SAME URL, then re-run
    extract_composition on the fully-rendered DOM."""

    def test_spa_escalation_uses_browser_when_httpx_empty_and_spa_shell(self):
        """httpx returns an SPA shell (no composition) → BrowserFetcher.fetch called."""
        # Minimal SPA shell: has an app-root marker, little visible text
        spa_shell_html = (
            "<html><head><title>Nike Sportswear Club Футболка - Street Beat</title></head>"
            "<body><div id='app'></div>"
            "<script>window.__initial_state__={}</script></body></html>"
        )
        # Browser-rendered page includes the spec block with composition
        rendered_html = (
            "<html><head><title>Nike Sportswear Club Футболка - Street Beat</title></head>"
            "<body><div id='app'>"
            "<div class='product-specs'><p>Состав: 100% хлопок</p></div>"
            "</div></body></html>"
        )

        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://street-beat.ru/nike-tee/123/"])
        )

        # httpx returns the SPA shell
        async def mock_fetch_url(url, **kwargs):
            fr = MagicMock()
            fr.raw_html = spa_shell_html
            fr.content = spa_shell_html
            return fr

        # BrowserFetcher.fetch returns the rendered DOM
        mock_bf = MagicMock()
        mock_bf._closed = False
        mock_bf.fetch = AsyncMock(return_value=rendered_html)
        mock_bf.fetch_with_warmup = AsyncMock(return_value=None)
        mock_bf.close = AsyncMock()

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
                    browser_fetcher=mock_bf,
                )
            )

        assert result is not None, "SPA escalation must yield a composition"
        assert "хлопок" in result["composition"]
        assert result["site"] == "street-beat.ru"
        # BrowserFetcher.fetch must have been invoked (SPA escalation path)
        mock_bf.fetch.assert_called_once()

    def test_spa_escalation_not_triggered_when_httpx_has_composition(self):
        """If httpx already found composition, browser escalation is NOT triggered."""
        good_html = (
            "<html><head><title>Nike Tee</title></head>"
            "<body><p>Состав: 80% хлопок, 20% полиэстер</p></body></html>"
        )

        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://kixbox.ru/nike-tee/"])
        )

        async def mock_fetch_url(url, **kwargs):
            fr = MagicMock()
            fr.raw_html = good_html
            fr.content = good_html
            return fr

        mock_bf = MagicMock()
        mock_bf._closed = False
        mock_bf.fetch = AsyncMock(return_value=None)
        mock_bf.fetch_with_warmup = AsyncMock(return_value=None)
        mock_bf.close = AsyncMock()

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
                    "Футболка Nike",
                    "Nike",
                    serper_client=serper,
                    browser_fetcher=mock_bf,
                )
            )

        assert result is not None
        # Browser fetch must NOT have been called (httpx was sufficient)
        mock_bf.fetch.assert_not_called()

    def test_spa_escalation_not_triggered_when_page_is_not_spa_shell(self):
        """Plain page (no SPA markers, no composition) does NOT escalate to browser."""
        plain_no_composition_html = (
            "<html><head><title>Nike Tee Page</title></head>"
            "<body><p>Купить футболку Nike по лучшей цене. Бесплатная доставка.</p>"
            "<p>Скидки до 50% на все товары Nike. Оригинальная продукция.</p>"
            "</body></html>"
        )

        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://kixbox.ru/nike-tee/"])
        )

        async def mock_fetch_url(url, **kwargs):
            fr = MagicMock()
            fr.raw_html = plain_no_composition_html
            fr.content = plain_no_composition_html
            return fr

        mock_bf = MagicMock()
        mock_bf._closed = False
        mock_bf.fetch = AsyncMock(return_value=None)
        mock_bf.close = AsyncMock()

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
                    "Футболка Nike",
                    "Nike",
                    serper_client=serper,
                    browser_fetcher=mock_bf,
                )
            )

        assert result is None
        # No SPA markers → browser escalation must NOT fire
        mock_bf.fetch.assert_not_called()

    def test_spa_escalation_returns_none_gracefully_when_browser_also_misses(self):
        """Browser re-render also finds no composition → returns None (no crash)."""
        spa_shell_html = (
            "<html><head><title>Nike Tee</title></head>"
            "<body><div id='app'></div><script>window.__initial_state__={}</script></body></html>"
        )
        # Browser renders a page but still no composition block
        rendered_no_comp = (
            "<html><head><title>Nike Tee</title></head>"
            "<body><div id='app'><p>Описание товара. Нет состава.</p></div></body></html>"
        )

        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://street-beat.ru/nike-tee/"])
        )

        async def mock_fetch_url(url, **kwargs):
            fr = MagicMock()
            fr.raw_html = spa_shell_html
            fr.content = spa_shell_html
            return fr

        mock_bf = MagicMock()
        mock_bf._closed = False
        mock_bf.fetch = AsyncMock(return_value=rendered_no_comp)
        mock_bf.close = AsyncMock()

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
                    "Футболка Nike",
                    "Nike",
                    serper_client=serper,
                    browser_fetcher=mock_bf,
                )
            )

        # Graceful None — no composition from either httpx or browser
        assert result is None
        mock_bf.fetch.assert_called_once()  # escalation DID fire, just found nothing


# ---------------------------------------------------------------------------
# sneakerhead.ru structured extractor
# ---------------------------------------------------------------------------

from app.services.enrichment.sources.multisite_composition import extract_sneakerhead_fields


class TestExtractSneakerheadFields:
    """Unit tests for extract_sneakerhead_fields — no network, fixture HTML only."""

    # Realistic sneakerhead PDP spec block (server-rendered)
    SNEAKERHEAD_HTML = """
    <html>
    <head><title>Nike Air Force 1 '07 - sneakerhead.ru</title></head>
    <body>
    <div class="product-detail">
      <h1>Nike Air Force 1 '07</h1>
      <div class="product-specs">
        <div class="spec-row">Состав: Кожа, синтетика, текстиль, резина</div>
        <div class="spec-row">Пол: Унисекс</div>
        <div class="spec-row">Страна: Китай</div>
        <div class="spec-row">Артикул: DH2987-102</div>
        <div class="spec-row">Цвет: Белый/Чёрный</div>
        <div class="spec-row">Сезон: Демисезон</div>
      </div>
    </div>
    </body>
    </html>
    """

    def test_extracts_composition(self):
        fields = extract_sneakerhead_fields(self.SNEAKERHEAD_HTML)
        assert "Состав" in fields
        assert "кожа" in fields["Состав"].lower()
        assert "синтетика" in fields["Состав"].lower()

    def test_extracts_pol(self):
        fields = extract_sneakerhead_fields(self.SNEAKERHEAD_HTML)
        assert "Пол" in fields
        assert "Унисекс" in fields["Пол"]

    def test_extracts_strana(self):
        fields = extract_sneakerhead_fields(self.SNEAKERHEAD_HTML)
        assert "Страна" in fields
        assert "Китай" in fields["Страна"]

    def test_extracts_artikul(self):
        fields = extract_sneakerhead_fields(self.SNEAKERHEAD_HTML)
        assert "Артикул" in fields
        assert "DH2987-102" in fields["Артикул"]

    def test_extracts_tsvet(self):
        fields = extract_sneakerhead_fields(self.SNEAKERHEAD_HTML)
        assert "Цвет" in fields
        assert "Белый" in fields["Цвет"]

    def test_extracts_sezon(self):
        fields = extract_sneakerhead_fields(self.SNEAKERHEAD_HTML)
        assert "Сезон" in fields
        assert "Демисезон" in fields["Сезон"]

    def test_returns_empty_on_no_match(self):
        html = "<html><body><p>Купить кроссовки Nike Air Force 1. Бесплатная доставка.</p></body></html>"
        fields = extract_sneakerhead_fields(html)
        assert fields == {}

    def test_returns_empty_on_empty_input(self):
        assert extract_sneakerhead_fields("") == {}
        assert extract_sneakerhead_fields(None) == {}

    def test_no_duplicate_keys(self):
        """Duplicate label lines → only first value kept."""
        html = """
        <div>Состав: Кожа</div>
        <div>Состав: Резина</div>
        """
        fields = extract_sneakerhead_fields(html)
        assert fields.get("Состав") == "Кожа"

    def test_plain_text_works(self):
        """Works on pre-flattened plain text (no HTML tags)."""
        text = "Состав: 100% хлопок\nПол: Мужской\nСтрана: Россия"
        fields = extract_sneakerhead_fields(text)
        assert fields["Состав"] == "100% хлопок"
        assert fields["Пол"] == "Мужской"
        assert fields["Страна"] == "Россия"


class TestSneakerheadInHarvestComposition:
    """Integration: harvest_composition returns extra_fields from sneakerhead."""

    SNEAKERHEAD_HTML = """
    <html>
    <head><title>Nike Air Force 1 '07 - sneakerhead.ru</title></head>
    <body>
    <div class="product-specs">
      <div>Состав: Кожа, синтетика, текстиль, резина</div>
      <div>Пол: Унисекс</div>
      <div>Страна: Китай</div>
      <div>Артикул: DH2987-102</div>
    </div>
    </body>
    </html>
    """

    def test_extra_fields_present_in_result(self):
        """harvest_composition on sneakerhead URL must include extra_fields dict."""
        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://sneakerhead.ru/nike-af1/"])
        )

        async def mock_fetch_url(url, **kwargs):
            fr = MagicMock()
            fr.raw_html = self.SNEAKERHEAD_HTML
            fr.content = self.SNEAKERHEAD_HTML
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
                    "Nike Air Force 1 07",
                    "Nike",
                    serper_client=serper,
                )
            )

        assert result is not None, "sneakerhead hit must return a result"
        assert "кожа" in result["composition"].lower()
        assert result["site"] == "sneakerhead.ru"
        # extra_fields must carry the non-composition structured fields
        extra = result.get("extra_fields", {})
        assert "Пол" in extra, f"Пол missing from extra_fields: {extra}"
        assert "Страна" in extra, f"Страна missing from extra_fields: {extra}"
        assert "Артикул" in extra, f"Артикул missing from extra_fields: {extra}"

    def test_no_extra_fields_key_on_non_sneakerhead(self):
        """Non-sneakerhead sites must NOT have extra_fields key in the result."""
        html = (
            "<html><head><title>Nike Tee - kixbox.ru</title></head>"
            "<body><p>Состав: 100% хлопок</p></body></html>"
        )

        serper = MagicMock()
        serper.search = AsyncMock(
            return_value=_make_serper_results(["https://kixbox.ru/nike-tee/"])
        )

        async def mock_fetch_url(url, **kwargs):
            fr = MagicMock()
            fr.raw_html = html
            fr.content = html
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
                    "Футболка Nike",
                    "Nike",
                    serper_client=serper,
                )
            )

        assert result is not None
        assert "extra_fields" not in result, (
            "extra_fields must only appear for sneakerhead.ru, not other sites"
        )
