"""Real-Chrome headless CDP fetcher that defeats DataDome / Qrator.

Key insight (proven in session): launching the REAL system chrome.exe with
--headless=new and connecting via Playwright connect_over_cdp yields real TLS
fingerprints, window.chrome, authentic plugin/Navigator signals — all the cues
that DataDome checks.  A bundled Playwright Chromium lacks these and is blocked.

Design
------
* Lazy startup — Chrome is NOT launched at import time, only on first `fetch()`.
* Session reuse — one Chrome process + one Playwright context per BrowserFetcher
  instance.  DataDome session cookie persists across product-page requests, so
  the challenge is solved once per session.
* Port selection — picks a random free TCP port to avoid conflicts when multiple
  instances run (future pool path: instantiate N BrowserFetchers on N ports).
* Graceful failure — `fetch()` returns None on any error; never raises.
* atexit cleanup — Chrome process is terminated and the temp user-data dir is
  removed when the Python process exits.

Where a pool would go
---------------------
For high-throughput use, create a list[BrowserFetcher] with distinct ports and
round-robin across them.  Each instance owns its own Chrome process + session.
The single-instance prototype here is a drop-in: just wrap it in a pool manager.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chrome auto-detect: Windows paths first, then Linux (for prod portability)
# ---------------------------------------------------------------------------

_CHROME_CANDIDATES: list[str] = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chromium-browser",
    "/snap/bin/chromium",
]


def _find_chrome() -> str:
    """Return chrome path from env override or first existing candidate.

    Raises RuntimeError when no Chrome is found so the caller gets a clear
    error at startup rather than a cryptic subprocess failure later.
    """
    override = os.environ.get("BROWSER_FETCHER_CHROME_PATH")
    if override:
        if os.path.isfile(override):
            return override
        raise RuntimeError(
            f"BROWSER_FETCHER_CHROME_PATH={override!r} does not exist"
        )
    for path in _CHROME_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise RuntimeError(
        "No Chrome installation found. "
        "Set BROWSER_FETCHER_CHROME_PATH env var to the chrome.exe path."
    )


def _free_port() -> int:
    """Bind to port 0 and let the OS assign a free port, then release it."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# BrowserFetcher
# ---------------------------------------------------------------------------


class BrowserFetcher:
    """Headless real-Chrome fetcher connected via Playwright CDP.

    Usage::

        fetcher = BrowserFetcher()
        html = await fetcher.fetch("https://www.lamoda.ru/p/...")
        await fetcher.close()

    Or use as an async context manager::

        async with BrowserFetcher() as fetcher:
            html = await fetcher.fetch(url)
    """

    def __init__(self) -> None:
        self._chrome_path: Optional[str] = None
        self._port: Optional[int] = None
        self._user_data_dir: Optional[str] = None
        self._chrome_proc: Optional[subprocess.Popen] = None

        # Playwright objects — initialised on first fetch()
        self._playwright = None   # playwright.async_api.Playwright
        self._browser = None      # playwright BrowserContext connection
        self._context = None      # default context (reused for session cookies)

        self._closed = False
        atexit.register(self._sync_cleanup)

    # ------------------------------------------------------------------
    # Lazy launch
    # ------------------------------------------------------------------

    async def _ensure_running(self) -> bool:
        """Launch Chrome and connect Playwright if not already running.

        Returns True on success, False on any failure (caller treats as
        unavailable and returns None from fetch()).
        """
        if self._browser is not None:
            return True

        try:
            self._chrome_path = _find_chrome()
        except RuntimeError as exc:
            logger.error("BrowserFetcher: %s", exc)
            return False

        self._port = _free_port()
        self._user_data_dir = tempfile.mkdtemp(prefix="bf_chrome_")

        launch_args = [
            self._chrome_path,
            "--headless=new",
            f"--remote-debugging-port={self._port}",
            f"--user-data-dir={self._user_data_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            "--disable-gpu",
            "--window-size=1920,1080",
            "--disable-blink-features=AutomationControlled",
        ]

        logger.info(
            "BrowserFetcher: launching Chrome on port %d (headless=new)", self._port
        )
        try:
            self._chrome_proc = subprocess.Popen(
                launch_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            logger.error("BrowserFetcher: Chrome launch failed: %s", exc)
            self._cleanup_user_data_dir()
            return False

        # Wait for Chrome's DevTools protocol to be ready
        if not self._wait_for_cdp(timeout=15):
            logger.error(
                "BrowserFetcher: Chrome did not open CDP port %d within 15s", self._port
            )
            self._terminate_chrome()
            return False

        # Connect Playwright
        try:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            cdp_url = f"http://localhost:{self._port}"
            self._browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
            # Reuse the existing default context (has the real Chrome session / cookies)
            contexts = self._browser.contexts
            self._context = contexts[0] if contexts else await self._browser.new_context()
            logger.info(
                "BrowserFetcher: Playwright connected to Chrome CDP at %s", cdp_url
            )
            return True
        except Exception as exc:
            logger.error("BrowserFetcher: Playwright CDP connect failed: %s", exc)
            self._terminate_chrome()
            return False

    def _wait_for_cdp(self, timeout: float = 15.0) -> bool:
        """Poll until Chrome's CDP port is open (TCP SYN accepted)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self._port), timeout=0.3):
                    return True
            except OSError:
                time.sleep(0.2)
        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(
        self,
        url: str,
        wait_selector: Optional[str] = None,
        timeout: float = 45.0,
    ) -> Optional[str]:
        """Navigate to *url* with real Chrome and return the fully-rendered HTML.

        The rendered DOM includes all XHR/SPA-injected content (spec tables,
        composition labels, etc.) that a plain httpx GET would never see.

        Args:
            url: HTTPS URL to fetch.
            wait_selector: Optional CSS selector to wait for (e.g. a spec-block
                           class).  When None, waits for networkidle.
            timeout: Per-fetch timeout in seconds.

        Returns:
            Full page HTML string, or None on any failure (fail-closed).
        """
        if self._closed:
            logger.warning("BrowserFetcher.fetch called on a closed instance")
            return None

        if not await self._ensure_running():
            return None

        page = None
        try:
            page = await self._context.new_page()
            timeout_ms = int(timeout * 1000)

            await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)

            if wait_selector:
                try:
                    await page.wait_for_selector(
                        wait_selector, timeout=min(timeout_ms, 20_000)
                    )
                except Exception:
                    # Selector not found — page may still be usable (spec might
                    # render differently); continue and return whatever we have.
                    logger.debug(
                        "BrowserFetcher: wait_selector %r not found on %s", wait_selector, url
                    )
            else:
                try:
                    await page.wait_for_load_state(
                        "networkidle", timeout=min(timeout_ms, 30_000)
                    )
                except Exception:
                    # networkidle timeout is common on analytics-heavy pages;
                    # the content is usually already there.
                    logger.debug(
                        "BrowserFetcher: networkidle timeout for %s — using current DOM", url
                    )

            html = await page.content()
            logger.info(
                "BrowserFetcher: fetched %s (%d chars)", url, len(html)
            )
            return html

        except Exception as exc:
            logger.warning("BrowserFetcher.fetch error for %r: %s", url, exc)
            return None
        finally:
            if page is not None:
                try:
                    await page.close()
                except Exception:
                    pass

    async def close(self) -> None:
        """Gracefully close Playwright and terminate Chrome."""
        if self._closed:
            return
        self._closed = True
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
        self._terminate_chrome()
        self._cleanup_user_data_dir()
        logger.info("BrowserFetcher: closed cleanly")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "BrowserFetcher":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal cleanup helpers (sync — safe for atexit)
    # ------------------------------------------------------------------

    def _terminate_chrome(self) -> None:
        if self._chrome_proc is not None:
            try:
                self._chrome_proc.terminate()
                self._chrome_proc.wait(timeout=5)
            except Exception:
                try:
                    self._chrome_proc.kill()
                except Exception:
                    pass
            self._chrome_proc = None

    def _cleanup_user_data_dir(self) -> None:
        if self._user_data_dir and os.path.isdir(self._user_data_dir):
            try:
                shutil.rmtree(self._user_data_dir, ignore_errors=True)
                logger.debug(
                    "BrowserFetcher: removed temp dir %s", self._user_data_dir
                )
            except Exception:
                pass

    def _sync_cleanup(self) -> None:
        """atexit handler — synchronous teardown of the Chrome process."""
        self._terminate_chrome()
        self._cleanup_user_data_dir()
