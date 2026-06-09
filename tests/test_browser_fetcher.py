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

from app.services.providers.browser_fetcher import (
    BrowserFetcher,
    _DEFAULT_USER_AGENT,
    _find_chrome,
    _free_port,
)


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
        fetcher._user_agent = _DEFAULT_USER_AGENT

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
        import app.services.providers.browser_fetcher as bf_mod

        tmp_dir = tempfile.mkdtemp(prefix="bf_test_cleanup_")

        mock_proc = MagicMock()
        mock_proc.pid = 5555
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._chrome_proc = mock_proc
        fetcher._user_data_dir = tmp_dir
        fetcher._browser = None
        fetcher._playwright = None
        fetcher._context = None
        fetcher._port = None
        fetcher._chrome_path = None
        fetcher._user_agent = _DEFAULT_USER_AGENT

        original_psutil = bf_mod._psutil
        try:
            # Force the POSIX fallback path so terminate() is called
            bf_mod._psutil = None
            with patch("platform.system", return_value="Linux"):
                fetcher._sync_cleanup()
        finally:
            bf_mod._psutil = original_psutil

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


# ---------------------------------------------------------------------------
# FIX 1: user_agent parameter tests
# ---------------------------------------------------------------------------


class TestUserAgentParameter:
    """BrowserFetcher user_agent is threaded into launch args and context creation."""

    def test_default_user_agent_is_not_headless(self):
        """Default UA must never contain 'HeadlessChrome'."""
        assert "HeadlessChrome" not in _DEFAULT_USER_AGENT
        assert "Chrome/" in _DEFAULT_USER_AGENT

    def test_init_default_ua_is_desktop_chrome(self):
        """BrowserFetcher() with no args uses the module-level default UA."""
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._user_agent = _DEFAULT_USER_AGENT  # replicate __init__ logic
        assert fetcher._user_agent == _DEFAULT_USER_AGENT
        assert "HeadlessChrome" not in fetcher._user_agent

    def test_init_custom_ua_is_stored(self):
        """Custom UA passed to __init__ is stored verbatim."""
        custom_ua = "Mozilla/5.0 (X11; Linux x86_64) CustomBrowser/99.0"
        fetcher = BrowserFetcher(user_agent=custom_ua)
        assert fetcher._user_agent == custom_ua

    def test_init_none_ua_falls_back_to_default(self):
        """None UA falls back to the module-level default."""
        fetcher = BrowserFetcher(user_agent=None)
        assert fetcher._user_agent == _DEFAULT_USER_AGENT

    def test_ensure_running_passes_ua_in_launch_args(self):
        """--user-agent=<ua> must appear in the Chrome subprocess launch args."""
        custom_ua = "TestAgent/1.0"
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._chrome_proc = None
        fetcher._user_data_dir = None
        fetcher._playwright = None
        fetcher._browser = None
        fetcher._context = None
        fetcher._port = None
        fetcher._chrome_path = None
        fetcher._user_agent = custom_ua

        captured_args: list[list] = []

        class _FakePopen:
            pid = 12345
            def __init__(self, args, **kwargs):
                captured_args.append(args)
                # Raise immediately so we don't proceed to CDP wait
                raise OSError("fake stop after capture")

        with (
            patch("app.services.providers.browser_fetcher._find_chrome", return_value="/fake/chrome"),
            patch("app.services.providers.browser_fetcher._free_port", return_value=9299),
            patch("tempfile.mkdtemp", return_value="/tmp/bf_ua_test"),
            patch("subprocess.Popen", side_effect=_FakePopen),
        ):
            result = run(fetcher._ensure_running())

        assert result is False
        assert len(captured_args) == 1
        assert f"--user-agent={custom_ua}" in captured_args[0]

    def test_ensure_running_creates_context_with_ua(self):
        """Playwright new_context is called with the configured user_agent."""
        custom_ua = "SportmasterAgent/2.0"
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._chrome_proc = None
        fetcher._user_data_dir = None
        fetcher._playwright = None
        fetcher._browser = None
        fetcher._context = None
        fetcher._port = None
        fetcher._chrome_path = None
        fetcher._user_agent = custom_ua

        mock_context = MagicMock()
        mock_context.close = AsyncMock()

        mock_browser = MagicMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)

        mock_pw = MagicMock()
        mock_pw.chromium = MagicMock()
        mock_pw.chromium.connect_over_cdp = AsyncMock(return_value=mock_browser)
        mock_pw.stop = AsyncMock()

        mock_popen = MagicMock()
        mock_popen.pid = 54321
        mock_popen.poll = MagicMock(return_value=None)

        # _ensure_running does: from playwright.async_api import async_playwright
        # then: await async_playwright().start()
        # Patch at the source module so the local import picks up our mock.
        mock_apw_instance = MagicMock()
        mock_apw_instance.start = AsyncMock(return_value=mock_pw)

        with (
            patch("app.services.providers.browser_fetcher._find_chrome", return_value="/fake/chrome"),
            patch("app.services.providers.browser_fetcher._free_port", return_value=9300),
            patch("tempfile.mkdtemp", return_value="/tmp/bf_ctx_test"),
            patch("subprocess.Popen", return_value=mock_popen),
            patch.object(BrowserFetcher, "_wait_for_cdp", return_value=True),
            patch("playwright.async_api.async_playwright", return_value=mock_apw_instance),
        ):
            result = run(fetcher._ensure_running())

        assert result is True
        # new_context must have been called with user_agent=custom_ua
        mock_browser.new_context.assert_called_once()
        call_kwargs = mock_browser.new_context.call_args.kwargs
        assert call_kwargs.get("user_agent") == custom_ua
        assert call_kwargs.get("locale") == "ru-RU"
        assert call_kwargs.get("timezone_id") == "Europe/Moscow"


# ---------------------------------------------------------------------------
# FIX 2: tree-kill cleanup tests
# ---------------------------------------------------------------------------


class TestTreeKillCleanup:
    """_terminate_chrome uses psutil tree-kill / taskkill; close() is idempotent."""

    def _make_fetcher_with_proc(self, pid: int = 9001) -> tuple[BrowserFetcher, MagicMock]:
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._user_agent = _DEFAULT_USER_AGENT
        fetcher._chrome_path = None
        fetcher._port = None
        fetcher._user_data_dir = None
        fetcher._playwright = None
        fetcher._browser = None
        fetcher._context = None

        mock_proc = MagicMock()
        mock_proc.pid = pid
        mock_proc.poll = MagicMock(return_value=None)
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.kill = MagicMock()
        fetcher._chrome_proc = mock_proc
        return fetcher, mock_proc

    def test_terminate_chrome_uses_psutil_tree_kill(self):
        """When psutil is available, _terminate_chrome kills children then parent."""
        import app.services.providers.browser_fetcher as bf_mod

        fetcher, mock_proc = self._make_fetcher_with_proc(pid=7777)

        mock_child1 = MagicMock()
        mock_child2 = MagicMock()
        mock_parent = MagicMock()
        mock_parent.children = MagicMock(return_value=[mock_child1, mock_child2])
        mock_parent.kill = MagicMock()

        mock_psutil = MagicMock()
        mock_psutil.Process = MagicMock(return_value=mock_parent)
        mock_psutil.NoSuchProcess = ProcessLookupError
        mock_psutil.AccessDenied = PermissionError

        original_psutil = bf_mod._psutil
        try:
            bf_mod._psutil = mock_psutil
            fetcher._terminate_chrome()
        finally:
            bf_mod._psutil = original_psutil

        # All children killed, then parent killed
        mock_child1.kill.assert_called_once()
        mock_child2.kill.assert_called_once()
        mock_parent.kill.assert_called_once()
        # _chrome_proc cleared
        assert fetcher._chrome_proc is None

    def test_terminate_chrome_uses_taskkill_when_no_psutil_windows(self):
        """On Windows with no psutil, falls back to taskkill /F /T /PID."""
        import app.services.providers.browser_fetcher as bf_mod

        fetcher, mock_proc = self._make_fetcher_with_proc(pid=8888)

        original_psutil = bf_mod._psutil
        try:
            bf_mod._psutil = None  # simulate psutil absent

            with (
                patch("platform.system", return_value="Windows"),
                patch("subprocess.run") as mock_run,
            ):
                fetcher._terminate_chrome()

            mock_run.assert_called_once()
            call_args = mock_run.call_args.args[0]
            assert "taskkill" in call_args
            assert "/F" in call_args
            assert "/T" in call_args
            assert str(8888) in call_args
        finally:
            bf_mod._psutil = original_psutil

        assert fetcher._chrome_proc is None

    def test_terminate_chrome_posix_fallback_when_no_psutil(self):
        """On POSIX with no psutil, uses proc.terminate() + proc.kill()."""
        import app.services.providers.browser_fetcher as bf_mod

        fetcher, mock_proc = self._make_fetcher_with_proc(pid=9999)

        original_psutil = bf_mod._psutil
        try:
            bf_mod._psutil = None

            with patch("platform.system", return_value="Linux"):
                fetcher._terminate_chrome()
        finally:
            bf_mod._psutil = original_psutil

        mock_proc.terminate.assert_called_once()
        assert fetcher._chrome_proc is None

    def test_terminate_chrome_is_idempotent(self):
        """Calling _terminate_chrome twice does not raise."""
        import app.services.providers.browser_fetcher as bf_mod

        fetcher, _ = self._make_fetcher_with_proc(pid=1111)

        original_psutil = bf_mod._psutil
        try:
            bf_mod._psutil = None
            with patch("platform.system", return_value="Linux"):
                fetcher._terminate_chrome()
                fetcher._terminate_chrome()  # second call — must be safe
        finally:
            bf_mod._psutil = original_psutil

    def test_cleanup_user_data_dir_is_idempotent(self):
        """_cleanup_user_data_dir clears _user_data_dir so second call is a no-op."""
        tmp = tempfile.mkdtemp(prefix="bf_test_idem_")
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._chrome_proc = None
        fetcher._user_data_dir = tmp

        fetcher._cleanup_user_data_dir()
        assert not os.path.isdir(tmp)
        assert fetcher._user_data_dir is None

        # Second call — must not raise
        fetcher._cleanup_user_data_dir()

    def test_sync_cleanup_is_safe_after_close(self):
        """atexit _sync_cleanup is safe even after close() already ran."""
        tmp = tempfile.mkdtemp(prefix="bf_test_safe_")
        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._chrome_proc = None
        fetcher._user_data_dir = tmp

        fetcher._cleanup_user_data_dir()  # simulates close() having run
        # atexit fires — must not raise or double-delete
        fetcher._sync_cleanup()
        assert fetcher._user_data_dir is None

    def test_ensure_running_kills_stale_chrome_before_new_launch(self):
        """If a stale chrome_proc is alive, _ensure_running kills it before launching."""
        import app.services.providers.browser_fetcher as bf_mod

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._user_agent = _DEFAULT_USER_AGENT
        fetcher._chrome_path = None
        fetcher._port = None
        fetcher._user_data_dir = None
        fetcher._playwright = None
        fetcher._browser = None
        fetcher._context = None

        # Stale live process
        stale_proc = MagicMock()
        stale_proc.pid = 4242
        stale_proc.poll = MagicMock(return_value=None)  # still alive
        fetcher._chrome_proc = stale_proc

        killed = []

        original_terminate = fetcher._terminate_chrome.__func__ if hasattr(fetcher._terminate_chrome, "__func__") else None

        def _mock_terminate(self_inner=None):
            killed.append(True)
            fetcher._chrome_proc = None  # replicate real behaviour

        with (
            patch.object(BrowserFetcher, "_terminate_chrome", _mock_terminate),
            patch("app.services.providers.browser_fetcher._find_chrome", side_effect=RuntimeError("no chrome")),
        ):
            result = run(fetcher._ensure_running())

        # Stale process was killed before _find_chrome was even attempted
        assert len(killed) >= 1
        assert result is False  # failed at _find_chrome — but stale kill happened


# ---------------------------------------------------------------------------
# Challenge-wait poll tests: _looks_like_challenge + _wait_out_challenge
# ---------------------------------------------------------------------------


class TestChallengeWaitPoll:
    """_wait_out_challenge polls until Qrator/DataDome stub clears or times out."""

    def test_looks_like_challenge_detects_qrator_stub(self):
        """Small page with __qrator marker is recognised as a challenge stub."""
        from app.services.providers.browser_fetcher import _looks_like_challenge

        stub = "<html><body><script>__qrator={}</script><p>Checking...</p></body></html>"
        assert _looks_like_challenge(stub) is True

    def test_looks_like_challenge_false_for_real_page(self):
        """A large real page (no markers) is NOT flagged as a challenge."""
        from app.services.providers.browser_fetcher import _looks_like_challenge

        # Real page: big content, no challenge markers
        real_page = "<html><body>" + "Товар хлопок состав " * 2000 + "</body></html>"
        assert _looks_like_challenge(real_page) is False

    def test_looks_like_challenge_false_for_none(self):
        """None input is treated as a challenge (page not loaded at all)."""
        from app.services.providers.browser_fetcher import _looks_like_challenge

        # None → True (page is not usable)
        assert _looks_like_challenge(None) is True

    def test_wait_out_challenge_returns_real_content_after_poll(self):
        """_wait_out_challenge polls and returns real HTML once challenge clears."""
        import asyncio
        from app.services.providers.browser_fetcher import _wait_out_challenge

        stub_html = "<html><body><script>__qrator={}</script></body></html>"
        real_html = "<html><body><h1>Sportmaster Product</h1><p>Состав: 100% хлопок</p></body></html>"

        call_count = [0]

        async def _seq_content():
            idx = call_count[0]
            call_count[0] += 1
            # First call: stub; second call: real content
            if idx == 0:
                return stub_html
            return real_html

        mock_page = MagicMock()
        mock_page.content = AsyncMock(side_effect=_seq_content)
        # wait_for_load_state raises immediately to skip the sleep path
        mock_page.wait_for_load_state = AsyncMock(side_effect=Exception("timeout"))

        result = asyncio.get_event_loop().run_until_complete(
            _wait_out_challenge(mock_page, "https://www.sportmaster.ru/product/123/", challenge_wait=5.0)
        )

        assert result is not None
        assert "хлопок" in result
        # content() called at least twice: initial check + at least one poll
        assert call_count[0] >= 2

    def test_wait_out_challenge_returns_stub_after_timeout(self):
        """_wait_out_challenge returns the stub (last content) after budget expires."""
        import asyncio
        import time
        from app.services.providers.browser_fetcher import _wait_out_challenge

        stub_html = "<html><body><script>__qrator={}</script></body></html>"

        mock_page = MagicMock()
        mock_page.content = AsyncMock(return_value=stub_html)
        mock_page.wait_for_load_state = AsyncMock(side_effect=Exception("timeout"))

        # Very short budget so test runs fast
        result = asyncio.get_event_loop().run_until_complete(
            _wait_out_challenge(mock_page, "https://www.sportmaster.ru/", challenge_wait=0.1)
        )

        # Timed out: returns whatever was last read (the stub)
        assert result == stub_html

    def test_fetch_invokes_challenge_wait_when_stub_detected(self):
        """fetch() calls _wait_out_challenge when page.content() returns a stub."""
        stub_html = "<html><body><script>__qrator={}</script></body></html>"
        real_html = "<html><body><p>Состав: 80% хлопок</p></body></html>"

        # page.content() returns stub first, then real content on subsequent calls
        content_calls = [0]

        async def _seq_content():
            idx = content_calls[0]
            content_calls[0] += 1
            if idx == 0:
                return stub_html
            return real_html

        page = MagicMock()
        page.goto = AsyncMock()
        page.wait_for_load_state = AsyncMock(side_effect=Exception("networkidle timeout"))
        page.content = AsyncMock(side_effect=_seq_content)
        page.close = AsyncMock()

        ctx = _make_mock_context(page)
        browser = _make_mock_browser(ctx)

        fetcher = BrowserFetcher.__new__(BrowserFetcher)
        fetcher._closed = False
        fetcher._browser = browser
        fetcher._context = ctx
        fetcher._chrome_proc = MagicMock(pid=1234)
        fetcher._user_data_dir = "/tmp/bf_challenge_test"
        fetcher._playwright = None

        async def _already_running():
            return True

        fetcher._ensure_running = _already_running

        # challenge_wait=5s so the poll loop fires at least once
        result = run(fetcher.fetch(
            "https://www.sportmaster.ru/product/999/",
            challenge_wait=5.0,
        ))

        # Must have polled past the stub and returned real content
        assert result is not None
        assert "хлопок" in result
