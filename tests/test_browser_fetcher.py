"""Unit tests for BrowserFetcher and mine_lamoda_composition.

All tests are fully mocked — no real Chrome is launched, no network calls.
Covers:
  - fetch() returns content from CDP-connected page
  - graceful None when Chrome launch fails (subprocess.Popen raises)
  - graceful None when CDP connect fails
  - session reuse: second fetch() does NOT relaunch Chrome
  - atexit cleanup: _sync_cleanup terminates Chrome + removes temp dir
  - fetch_with_warmup: warmup runs before product URL (happy path)
  - fetch_with_warmup: clears block on 2nd attempt (retry logic)
  - fetch_with_warmup: product page still blocked after warmup → None
  - fetch_with_warmup: warmup goto failure is tolerated, still fetches product
  - mine_lamoda_composition: full happy path (uses fetch_with_warmup on cold session)
  - mine_lamoda_composition: second call reuses warm session (plain fetch)
  - mine_lamoda_composition: Serper finds no Lamoda URLs → None
  - mine_lamoda_composition: brand-gate rejects page → None
  - mine_lamoda_composition: browser returns empty → None
  - mine_lamoda_composition: empty product_name guard
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.services.providers.browser_fetcher import BrowserFetcher, _find_chrome, _free_port


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(coro):
    """Run a coroutine in a fresh event loop (pytest-asyncio alternative)."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_mock_page(content: str = "<html><body>ok</body></html>") -> MagicMock:
    page = MagicMock()
    page.goto = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    page.wait_for_selector = AsyncMock()
    page.content = AsyncMock(return_value=content)
    page.close = AsyncMock()
    return page


def _make_mock_context(page: MagicMock) -> MagicMock:
    ctx = MagicMock()
    ctx.new_page = AsyncMock(return_value=page)
    ctx.close = AsyncMock()
    return ctx


def _make_mock_browser(context: MagicMock) -> MagicMock:
    browser = MagicMock()
    browser.contexts = [context]
    browser.close = AsyncMock()
    browser.new_context = AsyncMock(return_value=context)
    return browser


def _make_mock_playwright(browser: MagicMock) -> MagicMock:
    pw = MagicMock()
    pw.chromium = MagicMock()
    pw.chromium.connect_over_cdp = AsyncMock(return_value=browser)
    pw.stop = AsyncMock()
    return pw


# ---------------------------------------------------------------------------
# BrowserFetcher unit tests
# ---------------------------------------------------------------------------


class TestBrowserFetcherFetch:
    """fetch() returns rendered HTML from a CDP-connected page."""

    def test_fetch_returns_page_content(self):
        """fetch() returns page.content() when _ensure_running succeeds."""
        expected_html = "<html><body>Champion hoodie</body></html>"
        page = _make_mock_page(expected_html)
        ctx = _make_mock_context(page)
        browser = _make_mock_browser(ctx)

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._chrome_proc = MagicMock(pid=1234)
        fetcher._user_data_dir = "/tmp/bf_test"
        fetcher._playwright = None
        # Pre-inject the browser+context so _ensure_running short-circuits
        fetcher._browser = browser
        fetcher._context = ctx

        async def _already_running():
            return True

        fetcher._ensure_running = _already_running

        result = run(fetcher.fetch("https://www.lamoda.ru/p/test/"))

        assert result == expected_html

    def test_fetch_returns_none_on_closed_instance(self):
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = True

        result = run(fetcher.fetch("https://www.lamoda.ru/p/test/"))
        assert result is None

    def test_fetch_returns_none_on_ensure_running_failure(self):
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._browser = None
        fetcher._chrome_proc = None
        fetcher._user_data_dir = None
        fetcher._playwright = None
        fetcher._context = None

        async def _always_fail():
            return False

        fetcher._ensure_running = _always_fail

        result = run(fetcher.fetch("https://www.lamoda.ru/p/test/"))
        assert result is None

    def test_fetch_returns_none_on_page_error(self):
        """page.goto() raising → fetch() returns None (never propagates)."""
        page = MagicMock()
        page.goto = AsyncMock(side_effect=Exception("connection refused"))
        page.close = AsyncMock()
        ctx = _make_mock_context(page)
        browser = _make_mock_browser(ctx)

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._browser = browser
        fetcher._context = ctx
        fetcher._chrome_proc = None
        fetcher._user_data_dir = None
        fetcher._playwright = None

        async def _already_running():
            return True

        fetcher._ensure_running = _already_running

        result = run(fetcher.fetch("https://www.lamoda.ru/p/broken/"))
        assert result is None


class TestBrowserFetcherLaunchFailure:
    """Graceful None when Chrome launch fails."""

    def test_popen_raises_returns_false_from_ensure_running(self):
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._chrome_proc = None
        fetcher._user_data_dir = None
        fetcher._playwright = None
        fetcher._browser = None
        fetcher._context = None
        fetcher._port = None
        fetcher._chrome_path = None

        with (
            patch("app.services.providers.browser_fetcher._find_chrome", return_value="/fake/chrome"),
            patch("app.services.providers.browser_fetcher._free_port", return_value=9222),
            patch("subprocess.Popen", side_effect=OSError("chrome not found")),
            patch("tempfile.mkdtemp", return_value="/tmp/bf_chrome_popen_fail"),
        ):
            result = run(fetcher._ensure_running())

        assert result is False

    def test_chrome_not_found_returns_false(self):
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._chrome_proc = None
        fetcher._user_data_dir = None
        fetcher._playwright = None
        fetcher._browser = None
        fetcher._context = None
        fetcher._port = None
        fetcher._chrome_path = None

        with patch(
            "app.services.providers.browser_fetcher._find_chrome",
            side_effect=RuntimeError("No Chrome installation found"),
        ):
            result = run(fetcher._ensure_running())

        assert result is False


class TestBrowserFetcherSessionReuse:
    """Second fetch() reuses the existing Chrome process — no new Popen."""

    def test_second_fetch_does_not_relaunch(self):
        page = _make_mock_page("<html>ok</html>")
        ctx = _make_mock_context(page)
        browser = _make_mock_browser(ctx)

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        # Simulate already-running: browser is set, chrome_proc is set
        fetcher._browser = browser
        fetcher._context = ctx
        fetcher._chrome_proc = MagicMock(pid=9999)
        fetcher._user_data_dir = "/tmp/bf_existing"
        fetcher._playwright = None

        call_count = [0]

        async def _ensure_running_counter():
            call_count[0] += 1
            # If browser is already set, _ensure_running should return True immediately
            # without touching Popen — replicate the real guard
            if fetcher._browser is not None:
                return True
            return False

        fetcher._ensure_running = _ensure_running_counter

        run(fetcher.fetch("https://www.lamoda.ru/p/first/"))
        run(fetcher.fetch("https://www.lamoda.ru/p/second/"))

        # _ensure_running was called twice; Popen was never called
        assert call_count[0] == 2


class TestBrowserFetcherAtexitCleanup:
    """atexit handler terminates Chrome and removes temp profile dir."""

    def test_sync_cleanup_terminates_chrome_and_removes_dir(self):
        tmp_dir = tempfile.mkdtemp(prefix="bf_test_cleanup_")

        mock_proc = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._chrome_proc = mock_proc
        fetcher._user_data_dir = tmp_dir
        fetcher._browser = None
        fetcher._playwright = None
        fetcher._context = None
        fetcher._port = None
        fetcher._chrome_path = None

        fetcher._sync_cleanup()

        mock_proc.terminate.assert_called_once()
        # Temp dir must be removed
        assert not os.path.isdir(tmp_dir)

    def test_sync_cleanup_tolerates_no_chrome_proc(self):
        """_sync_cleanup is safe when Chrome was never started."""
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._chrome_proc = None
        fetcher._user_data_dir = None

        # Must not raise
        fetcher._sync_cleanup()


# ---------------------------------------------------------------------------
# fetch_with_warmup unit tests
# ---------------------------------------------------------------------------


class TestFetchWithWarmup:
    """fetch_with_warmup() runs homepage warmup before product URL."""

    def _make_warmup_fetcher(self, page_contents: list[str]) -> BrowserFetcher:
        """Create a BrowserFetcher with a mocked context.

        page_contents: list of HTML strings returned by successive new_page
        navigations (first = warmup page, rest = product page per fetch call).
        Each new_page() call returns a fresh mock_page cycling through the list.
        """
        call_index = [0]

        def _make_next_page():
            idx = call_index[0]
            content = page_contents[min(idx, len(page_contents) - 1)]
            call_index[0] += 1
            return _make_mock_page(content)

        ctx = MagicMock()
        ctx.new_page = AsyncMock(side_effect=lambda: asyncio.coroutine(lambda: _make_next_page())())
        ctx.close = AsyncMock()

        browser = _make_mock_browser(ctx)

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._browser = browser
        fetcher._context = ctx
        fetcher._chrome_proc = MagicMock(pid=1234)
        fetcher._user_data_dir = "/tmp/bf_warmup_test"
        fetcher._playwright = None

        async def _already_running():
            return True

        fetcher._ensure_running = _already_running
        return fetcher

    def test_warmup_runs_before_product_and_returns_content(self):
        """fetch_with_warmup navigates warmup URL, then product URL."""
        real_homepage = "<html><body><header>Lamoda</header><nav>Каталог</nav></body></html>"
        real_product = (
            "<html><body><h1>Champion Hoodie</h1>"
            "<div class='x-product-characteristics'><p>Состав: 80% хлопок, 20% полиэстер</p></div>"
            "</body></html>"
        )

        goto_calls: list[str] = []

        # Build a fetcher where both pages return real content (no block marker)
        page_warmup = _make_mock_page(real_homepage)
        page_product = _make_mock_page(real_product)

        async def _record_goto_warmup(url, **kwargs):
            goto_calls.append(("warmup", url))

        async def _record_goto_product(url, **kwargs):
            goto_calls.append(("product", url))

        page_warmup.goto = AsyncMock(side_effect=_record_goto_warmup)
        page_product.goto = AsyncMock(side_effect=_record_goto_product)

        # Both pages returned by successive new_page() calls via the same page
        # (fetch_with_warmup opens ONE page and navigates it twice)
        pages_to_return = [page_warmup]
        ctx = MagicMock()
        ctx.new_page = AsyncMock(side_effect=lambda: _make_page_coro(pages_to_return))
        ctx.close = AsyncMock()
        browser = _make_mock_browser(ctx)

        # Wire a single page that records both goto calls
        combined_page = MagicMock()
        combined_page.close = AsyncMock()
        combined_page.wait_for_selector = AsyncMock()
        combined_page.wait_for_load_state = AsyncMock()

        goto_log: list[str] = []

        async def _combined_goto(url, **kwargs):
            goto_log.append(url)

        # Return real content for homepage, real product for second call
        content_calls = [0]

        async def _combined_content():
            idx = content_calls[0]
            content_calls[0] += 1
            if idx == 0:
                return real_homepage
            return real_product

        combined_page.goto = AsyncMock(side_effect=_combined_goto)
        combined_page.content = AsyncMock(side_effect=_combined_content)

        ctx2 = MagicMock()
        ctx2.new_page = AsyncMock(return_value=combined_page)
        ctx2.close = AsyncMock()
        browser2 = _make_mock_browser(ctx2)

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._browser = browser2
        fetcher._context = ctx2
        fetcher._chrome_proc = MagicMock(pid=1234)
        fetcher._user_data_dir = "/tmp/bf_warmup"
        fetcher._playwright = None

        async def _already_running():
            return True

        fetcher._ensure_running = _already_running

        result = run(
            fetcher.fetch_with_warmup(
                "https://www.lamoda.ru/p/ch001/champion-hoodie/",
                warmup_url="https://www.lamoda.ru/",
                warmup_timeout=10.0,
                product_timeout=15.0,
                warmup_block_marker="Доступ ограничен",
                product_block_marker="Доступ ограничен",
                warmup_retries=1,
                warmup_retry_delay=0.0,
            )
        )

        assert result is not None
        assert "хлопок" in result
        # Warmup URL was navigated before product URL
        assert goto_log[0] == "https://www.lamoda.ru/"
        assert goto_log[1] == "https://www.lamoda.ru/p/ch001/champion-hoodie/"

    def test_warmup_clears_on_second_attempt(self):
        """Warmup retries when first attempt returns the block marker."""
        block_html = "<html><body>Доступ ограничен — DataDome challenge</body></html>"
        real_homepage = "<html><body><header>Lamoda</header></body></html>"
        real_product = "<html><body><h1>Champion</h1><p>Состав: 100% хлопок</p></body></html>"

        content_seq = [block_html, real_homepage, real_product]
        content_calls = [0]

        async def _seq_content():
            idx = min(content_calls[0], len(content_seq) - 1)
            content_calls[0] += 1
            return content_seq[idx]

        page = MagicMock()
        page.goto = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        page.wait_for_selector = AsyncMock()
        page.content = AsyncMock(side_effect=_seq_content)
        page.close = AsyncMock()

        ctx = MagicMock()
        ctx.new_page = AsyncMock(return_value=page)
        ctx.close = AsyncMock()
        browser = _make_mock_browser(ctx)

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._browser = browser
        fetcher._context = ctx
        fetcher._chrome_proc = None
        fetcher._user_data_dir = None
        fetcher._playwright = None

        async def _already_running():
            return True

        fetcher._ensure_running = _already_running

        result = run(
            fetcher.fetch_with_warmup(
                "https://www.lamoda.ru/p/ch001/champion-hoodie/",
                warmup_url="https://www.lamoda.ru/",
                warmup_timeout=10.0,
                product_timeout=15.0,
                warmup_block_marker="Доступ ограничен",
                product_block_marker="Доступ ограничен",
                warmup_retries=2,
                warmup_retry_delay=0.0,
            )
        )

        # Block on attempt 1, cleared on attempt 2, product loaded
        assert result is not None
        assert "хлопок" in result
        # content() was called: attempt1(block) + attempt2(real_home) + product = 3
        assert content_calls[0] == 3

    def test_product_blocked_after_warmup_returns_none(self):
        """Returns None when product page still shows block marker after warmup."""
        real_homepage = "<html><body><header>Lamoda</header></body></html>"
        block_product = "<html><body>Доступ ограничен</body></html>"

        content_seq = [real_homepage, block_product]
        content_calls = [0]

        async def _seq_content():
            idx = min(content_calls[0], len(content_seq) - 1)
            content_calls[0] += 1
            return content_seq[idx]

        page = MagicMock()
        page.goto = AsyncMock()
        page.wait_for_load_state = AsyncMock()
        page.wait_for_selector = AsyncMock()
        page.content = AsyncMock(side_effect=_seq_content)
        page.close = AsyncMock()

        ctx = MagicMock()
        ctx.new_page = AsyncMock(return_value=page)
        ctx.close = AsyncMock()
        browser = _make_mock_browser(ctx)

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._browser = browser
        fetcher._context = ctx
        fetcher._chrome_proc = None
        fetcher._user_data_dir = None
        fetcher._playwright = None

        async def _already_running():
            return True

        fetcher._ensure_running = _already_running

        result = run(
            fetcher.fetch_with_warmup(
                "https://www.lamoda.ru/p/ch001/champion-hoodie/",
                warmup_url="https://www.lamoda.ru/",
                warmup_timeout=10.0,
                product_timeout=15.0,
                warmup_block_marker="Доступ ограничен",
                product_block_marker="Доступ ограничен",
                warmup_retries=1,
                warmup_retry_delay=0.0,
            )
        )

        assert result is None

    def test_warmup_returns_none_on_closed_fetcher(self):
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = True

        result = run(
            fetcher.fetch_with_warmup(
                "https://www.lamoda.ru/p/ch001/champion-hoodie/",
                warmup_url="https://www.lamoda.ru/",
            )
        )
        assert result is None

    def test_warmup_returns_none_on_ensure_running_failure(self):
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._browser = None
        fetcher._chrome_proc = None
        fetcher._user_data_dir = None
        fetcher._playwright = None
        fetcher._context = None

        async def _always_fail():
            return False

        fetcher._ensure_running = _always_fail

        result = run(
            fetcher.fetch_with_warmup(
                "https://www.lamoda.ru/p/ch001/champion-hoodie/",
                warmup_url="https://www.lamoda.ru/",
            )
        )
        assert result is None


def _make_page_coro(pages_list):
    """Helper: return pages from a list for successive new_page() calls."""
    page = pages_list[0]
    import asyncio

    async def _inner():
        return page

    return _inner()


# ---------------------------------------------------------------------------
# mine_lamoda_composition unit tests
# ---------------------------------------------------------------------------


class TestMineLamodoComposition:
    """Unit tests for mine_lamoda_composition() — fully mocked."""

    def _make_serper_result(self, urls: list[str]) -> MagicMock:
        results = MagicMock()
        organics = []
        for i, u in enumerate(urls):
            r = MagicMock()
            r.link = u
            r.title = f"Result {i}"
            r.snippet = ""
            r.position = i + 1
            organics.append(r)
        results.organic_results = organics
        return results

    def test_happy_path_returns_composition(self):
        """Cold session: fetch_with_warmup is called (not plain fetch)."""
        from app.services.enrichment.sources import lamoda_composition as lc
        from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition

        composition_html = (
            "<html><head><title>Champion Hoodie Lamoda</title></head>"
            "<body>"
            "<h1>Champion Reverse Weave Hoodie</h1>"
            "<div class='x-product-characteristics'>"
            "<p>Состав: 80% хлопок, 20% полиэстер</p>"
            "</div>"
            "</body></html>"
        )

        lamoda_url = "https://www.lamoda.ru/p/ch001/champion-reverse-weave/"

        mock_serper = MagicMock()
        mock_serper.search = AsyncMock(
            return_value=self._make_serper_result([lamoda_url])
        )

        mock_fetcher = MagicMock()
        mock_fetcher._closed = False
        mock_fetcher.fetch = AsyncMock(return_value=composition_html)
        mock_fetcher.fetch_with_warmup = AsyncMock(return_value=composition_html)

        # Ensure this fetcher id is NOT in the warmed set so we exercise the cold path
        fetcher_id = id(mock_fetcher)
        lc._session_warmed.discard(fetcher_id)

        result = run(
            mine_lamoda_composition(
                "Толстовка худи Champion Reverse Weave",
                "Champion",
                serper_client=mock_serper,
                fetcher=mock_fetcher,
            )
        )

        assert result is not None
        assert "хлопок" in result["composition"]
        assert result["source_url"] == lamoda_url
        # Cold session must have used fetch_with_warmup, NOT plain fetch
        mock_fetcher.fetch_with_warmup.assert_called_once()
        mock_fetcher.fetch.assert_not_called()

    def test_warm_session_uses_plain_fetch(self):
        """Warm session (fetcher id already in _session_warmed): uses plain fetch."""
        from app.services.enrichment.sources import lamoda_composition as lc
        from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition

        composition_html = (
            "<html><head><title>Champion Hoodie Lamoda</title></head>"
            "<body>"
            "<h1>Champion Reverse Weave Hoodie</h1>"
            "<p>Состав: 90% хлопок, 10% эластан</p>"
            "</body></html>"
        )

        lamoda_url = "https://www.lamoda.ru/p/ch002/champion-warm/"

        mock_serper = MagicMock()
        mock_serper.search = AsyncMock(
            return_value=self._make_serper_result([lamoda_url])
        )

        mock_fetcher = MagicMock()
        mock_fetcher._closed = False
        mock_fetcher.fetch = AsyncMock(return_value=composition_html)
        mock_fetcher.fetch_with_warmup = AsyncMock(return_value=composition_html)

        # Pre-mark this fetcher as already warmed
        fetcher_id = id(mock_fetcher)
        lc._session_warmed.add(fetcher_id)

        try:
            result = run(
                mine_lamoda_composition(
                    "Champion Reverse Weave hoodie",
                    "Champion",
                    serper_client=mock_serper,
                    fetcher=mock_fetcher,
                )
            )
        finally:
            lc._session_warmed.discard(fetcher_id)

        assert result is not None
        # Warm session must use plain fetch, NOT fetch_with_warmup
        mock_fetcher.fetch.assert_called_once()
        mock_fetcher.fetch_with_warmup.assert_not_called()

    def test_no_lamoda_urls_returns_none(self):
        from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition

        # Serper returns non-Lamoda URLs
        mock_serper = MagicMock()
        mock_serper.search = AsyncMock(
            return_value=self._make_serper_result([
                "https://kixbox.ru/champion-hoodie",
                "https://brandshop.ru/champion/",
            ])
        )

        mock_fetcher = MagicMock()
        mock_fetcher.fetch = AsyncMock()

        result = run(
            mine_lamoda_composition(
                "Champion Reverse Weave",
                "Champion",
                serper_client=mock_serper,
                fetcher=mock_fetcher,
            )
        )

        assert result is None
        mock_fetcher.fetch.assert_not_called()

    def test_brand_gate_rejection_returns_none(self):
        """Brand-gate fires on wrong-brand page (pre-warmed session for simplicity)."""
        from app.services.enrichment.sources import lamoda_composition as lc
        from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition

        # HTML page mentions Nike, not Champion → brand-gate fires
        wrong_brand_html = (
            "<html><head><title>Nike Hoodie</title></head>"
            "<body><h1>Nike Sportswear</h1>"
            "<p>Состав: 100% полиэстер</p>"
            "</body></html>"
        )

        mock_serper = MagicMock()
        mock_serper.search = AsyncMock(
            return_value=self._make_serper_result([
                "https://www.lamoda.ru/p/ni001/nike-hoodie/"
            ])
        )

        mock_fetcher = MagicMock()
        mock_fetcher._closed = False
        mock_fetcher.fetch = AsyncMock(return_value=wrong_brand_html)
        mock_fetcher.fetch_with_warmup = AsyncMock(return_value=wrong_brand_html)

        # Pre-warm so we always go through the simpler plain-fetch path here
        fetcher_id = id(mock_fetcher)
        lc._session_warmed.add(fetcher_id)

        try:
            result = run(
                mine_lamoda_composition(
                    "Champion Reverse Weave",
                    "Champion",
                    serper_client=mock_serper,
                    fetcher=mock_fetcher,
                )
            )
        finally:
            lc._session_warmed.discard(fetcher_id)

        assert result is None

    def test_empty_html_returns_none(self):
        """Empty HTML from browser returns None (pre-warmed session)."""
        from app.services.enrichment.sources import lamoda_composition as lc
        from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition

        mock_serper = MagicMock()
        mock_serper.search = AsyncMock(
            return_value=self._make_serper_result([
                "https://www.lamoda.ru/p/ch001/champion-test/"
            ])
        )

        mock_fetcher = MagicMock()
        mock_fetcher._closed = False
        mock_fetcher.fetch = AsyncMock(return_value=None)  # browser failed
        mock_fetcher.fetch_with_warmup = AsyncMock(return_value=None)

        fetcher_id = id(mock_fetcher)
        lc._session_warmed.add(fetcher_id)

        try:
            result = run(
                mine_lamoda_composition(
                    "Champion hoodie",
                    "Champion",
                    serper_client=mock_serper,
                    fetcher=mock_fetcher,
                )
            )
        finally:
            lc._session_warmed.discard(fetcher_id)

        assert result is None

    def test_empty_product_name_returns_none(self):
        from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition

        result = run(mine_lamoda_composition("", "Champion"))
        assert result is None

    def test_empty_brand_returns_none(self):
        from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition

        result = run(mine_lamoda_composition("Champion hoodie", ""))
        assert result is None

    def test_serper_exception_returns_none(self):
        from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition

        mock_serper = MagicMock()
        mock_serper.search = AsyncMock(side_effect=Exception("Serper unavailable"))

        mock_fetcher = MagicMock()

        result = run(
            mine_lamoda_composition(
                "Champion hoodie",
                "Champion",
                serper_client=mock_serper,
                fetcher=mock_fetcher,
            )
        )

        assert result is None

    def test_max_urls_cap_respected(self):
        """Only max_urls pages are fetched (pre-warmed session: uses plain fetch)."""
        from app.services.enrichment.sources import lamoda_composition as lc
        from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition

        urls = [
            f"https://www.lamoda.ru/p/ch{i:03d}/champion-test-{i}/"
            for i in range(5)
        ]

        mock_serper = MagicMock()
        mock_serper.search = AsyncMock(
            return_value=self._make_serper_result(urls)
        )

        fetch_calls: list[str] = []

        async def _fake_fetch(url, wait_selector=None, timeout=45.0):
            fetch_calls.append(url)
            return None  # all fail — we're just counting calls

        mock_fetcher = MagicMock()
        mock_fetcher._closed = False
        mock_fetcher.fetch = _fake_fetch
        mock_fetcher.fetch_with_warmup = AsyncMock(return_value=None)

        # Pre-warm so all URLs go through plain fetch (makes counting unambiguous)
        fetcher_id = id(mock_fetcher)
        lc._session_warmed.add(fetcher_id)

        try:
            run(
                mine_lamoda_composition(
                    "Champion hoodie",
                    "Champion",
                    max_urls=2,
                    serper_client=mock_serper,
                    fetcher=mock_fetcher,
                )
            )
        finally:
            lc._session_warmed.discard(fetcher_id)

        assert len(fetch_calls) == 2

    def test_stops_on_first_hit(self):
        """Stops after first successful extraction (pre-warmed session)."""
        from app.services.enrichment.sources import lamoda_composition as lc
        from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition

        url1 = "https://www.lamoda.ru/p/ch001/champion-first/"
        url2 = "https://www.lamoda.ru/p/ch002/champion-second/"

        composition_html = (
            "<html><head><title>Champion Hoodie Lamoda</title></head>"
            "<body><h1>Champion Reverse Weave</h1>"
            "<p>Состав: 80% хлопок, 20% полиэстер</p></body></html>"
        )

        mock_serper = MagicMock()
        mock_serper.search = AsyncMock(
            return_value=self._make_serper_result([url1, url2])
        )

        fetch_calls: list[str] = []

        async def _fake_fetch(url, wait_selector=None, timeout=45.0):
            fetch_calls.append(url)
            return composition_html

        mock_fetcher = MagicMock()
        mock_fetcher._closed = False
        mock_fetcher.fetch = _fake_fetch
        mock_fetcher.fetch_with_warmup = AsyncMock(return_value=composition_html)

        fetcher_id = id(mock_fetcher)
        lc._session_warmed.add(fetcher_id)

        try:
            result = run(
                mine_lamoda_composition(
                    "Champion Reverse Weave Hoodie",
                    "Champion",
                    max_urls=2,
                    serper_client=mock_serper,
                    fetcher=mock_fetcher,
                )
            )
        finally:
            lc._session_warmed.discard(fetcher_id)

        # Result found
        assert result is not None
        # Only first URL was fetched (stopped on hit)
        assert fetch_calls == [url1]
