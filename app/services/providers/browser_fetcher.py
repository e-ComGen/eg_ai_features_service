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
  removed when the Python process exits.  Uses psutil for process-tree kill
  (Chrome spawns child renderers); falls back to taskkill /F /T on Windows or
  proc.kill() on POSIX when psutil is absent.
* fetch_with_warmup() — navigates to a warmup URL first (e.g. homepage),
  waits patiently for DataDome cookie to settle, then navigates to the real
  product URL.  Mirrors the "homepage first, then product" manual flow that
  DataDome accepts.
* user_agent — pass a real desktop Chrome UA so Qrator/Sportmaster never see
  "HeadlessChrome" in the User-Agent string.  Threaded into both --user-agent
  launch arg and Playwright new_context(user_agent=...).

Where a pool would go
---------------------
For high-throughput use, create a list[BrowserFetcher] with distinct ports and
round-robin across them.  Each instance owns its own Chrome process + session.
The single-instance prototype here is a drop-in: just wrap it in a pool manager.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from typing import Optional

try:
    import psutil as _psutil  # optional; used for process-tree kill
except ImportError:  # pragma: no cover
    _psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default User-Agent — desktop Chrome, never exposes "HeadlessChrome".
# Sportmaster (Qrator) and similar anti-bot systems reject the default
# headless UA; injecting a real desktop UA resolves the compat stub.
# ---------------------------------------------------------------------------

_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# ---------------------------------------------------------------------------
# Anti-bot JS-challenge handling (Qrator / DataDome interstitials)
# ---------------------------------------------------------------------------
#
# Qrator and similar systems first serve a tiny JS-challenge "stub" page that
# reloads itself once the browser solves a proof-of-work / sets a cookie.  Plain
# `page.goto(wait_until="domcontentloaded")` returns on this stub BEFORE the
# challenge resolves, so we'd grab the interstitial instead of the real page.
#
# The fix is to POLL the rendered text after navigation: while it still looks
# like a challenge stub, wait (tolerating Qrator's automatic reload) up to
# `BROWSER_FETCHER_CHALLENGE_WAIT` seconds before giving up.

def _challenge_wait_seconds() -> float:
    """Read the challenge-wait budget from env (default 30s)."""
    raw = os.environ.get("BROWSER_FETCHER_CHALLENGE_WAIT", "30")
    try:
        val = float(raw)
        return val if val >= 0 else 30.0
    except (TypeError, ValueError):
        return 30.0


# Markers that identify an unresolved JS-challenge / access-block stub.
# Case-insensitive substring match against the rendered page text/HTML.
_CHALLENGE_MARKERS: tuple[str, ...] = (
    "__qrator",
    "qauth",
    'id="request-id"',
    'id="request-ip"',
    "доступ ограничен",
    "checking your browser",
    "проверка браузера",
    "ddos protection",
)


def _looks_like_challenge(html: Optional[str]) -> bool:
    """True if *html* looks like an unresolved Qrator/DataDome challenge stub.

    Heuristic: a challenge marker is present AND the page is small (stubs are
    tiny — the real page is much larger).  We keep the size guard generous so a
    real page that merely *mentions* a marker word isn't misclassified.
    """
    if not html:
        return True
    low = html.lower()
    has_marker = any(m in low for m in _CHALLENGE_MARKERS)
    if not has_marker:
        return False
    # Real pages are large; challenge stubs are tiny shells (~few KB).
    return len(html) < 30_000


async def _wait_out_challenge(
    page,
    url: str,
    *,
    challenge_wait: float,
) -> Optional[str]:
    """Poll *page* until it is no longer a challenge stub or the budget expires.

    Returns the final page HTML (challenge-cleared if we got lucky, otherwise
    whatever the last poll produced).  Never raises.
    """
    try:
        html = await page.content()
    except Exception:
        return None

    if not _looks_like_challenge(html):
        return html

    deadline = time.monotonic() + max(challenge_wait, 0.0)
    poll_interval = 1.5
    logger.info(
        "BrowserFetcher: challenge stub detected on %s — waiting up to %.0fs for it to resolve",
        url, challenge_wait,
    )
    while time.monotonic() < deadline:
        # Qrator auto-reloads; give the network a chance to settle, tolerating
        # the reload navigation that may be mid-flight.
        try:
            await page.wait_for_load_state("networkidle", timeout=4_000)
        except Exception:
            await asyncio.sleep(poll_interval)
        try:
            html = await page.content()
        except Exception:
            await asyncio.sleep(poll_interval)
            continue
        if not _looks_like_challenge(html):
            logger.info("BrowserFetcher: challenge CLEARED on %s", url)
            return html
        await asyncio.sleep(poll_interval)

    logger.warning(
        "BrowserFetcher: challenge NOT cleared on %s within %.0fs — returning stub",
        url, challenge_wait,
    )
    return html

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

    def __init__(self, user_agent: Optional[str] = None) -> None:
        # None → use the module-level default desktop UA (never "HeadlessChrome")
        self._user_agent: str = user_agent if user_agent is not None else _DEFAULT_USER_AGENT

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

        Belt-and-suspenders UA injection strategy:
          1. --user-agent=<ua> on the Chrome process level (affects all
             contexts that inherit the process-level UA, including default ctx).
          2. browser.new_context(user_agent=...) at the Playwright level so the
             overriding UA is reliably set even if the default CDP context is
             reused (connect_over_cdp hands back the existing default context
             whose UA is already overridden at the process level, but we create
             a fresh context to be explicit and avoid CDP context surprises).
        """
        if self._browser is not None:
            return True

        # Safety: if a stale chrome proc is still alive from a previous partial
        # init (e.g. _ensure_running failed mid-way), kill it before starting a
        # new one so we don't orphan it.
        if self._chrome_proc is not None and self._chrome_proc.poll() is None:
            logger.warning(
                "BrowserFetcher: stale Chrome process found (pid=%d), cleaning up",
                self._chrome_proc.pid,
            )
            self._terminate_chrome()

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
            # Belt: override UA at the process level so all Chrome internals see it
            f"--user-agent={self._user_agent}",
        ]

        logger.info(
            "BrowserFetcher: launching Chrome on port %d (headless=new, ua=%r)",
            self._port,
            self._user_agent[:40] + "..." if len(self._user_agent) > 40 else self._user_agent,
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
            # Suspenders: create a fresh context with explicit UA, locale, timezone
            # so navigator.userAgent is guaranteed to show the desktop UA string.
            self._context = await self._browser.new_context(
                user_agent=self._user_agent,
                locale="ru-RU",
                timezone_id="Europe/Moscow",
            )
            logger.info(
                "BrowserFetcher: Playwright connected to Chrome CDP at %s (ua injected)",
                cdp_url,
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
        challenge_wait: Optional[float] = None,
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

            # Patient anti-bot challenge wait: if Chrome returned the Qrator/
            # DataDome JS-challenge stub, poll until it resolves (or budget ends)
            # instead of handing back the interstitial.
            cw = challenge_wait if challenge_wait is not None else _challenge_wait_seconds()
            if cw > 0 and _looks_like_challenge(html):
                resolved = await _wait_out_challenge(page, url, challenge_wait=cw)
                if resolved is not None:
                    html = resolved

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

    # ------------------------------------------------------------------
    # DataDome warmup path
    # ------------------------------------------------------------------

    async def fetch_with_warmup(
        self,
        url: str,
        warmup_url: str,
        *,
        warmup_timeout: float = 60.0,
        product_timeout: float = 60.0,
        warmup_real_selector: Optional[str] = None,
        product_real_selector: Optional[str] = None,
        warmup_block_marker: str = "Доступ ограничен",
        product_block_marker: str = "Доступ ограничен",
        warmup_retries: int = 3,
        warmup_retry_delay: float = 5.0,
        challenge_wait: Optional[float] = None,
    ) -> Optional[str]:
        """Navigate to *warmup_url* first to let DataDome set its cookie,
        then navigate to *url* with the warmed session.

        DataDome challenge flow:
          1. Hit the warmup URL (homepage).  DataDome may serve a JS/redirect
             challenge page ("Доступ ограничен") on the first hit — wait and
             retry up to *warmup_retries* times.  The cookie `datadome` is set
             as challenges resolve (automatic in headless Chrome with JS enabled).
          2. Once the warmup page shows real content (no block marker, or
             *warmup_real_selector* found), navigate to the product URL.
          3. Wait patiently for product content (selector or absence of block
             marker) up to *product_timeout* seconds.

        Args:
            url:                    Product URL to fetch after warmup.
            warmup_url:             Warmup URL (e.g. site homepage).
            warmup_timeout:         Total seconds to wait for warmup to clear
                                    DataDome (spread across retries).
            product_timeout:        Seconds to wait for the product page to load.
            warmup_real_selector:   Optional CSS selector that, when found,
                                    confirms the warmup page is real content.
            product_real_selector:  Optional CSS selector to wait for on the
                                    product page.
            warmup_block_marker:    Text that indicates the warmup page is still
                                    a DataDome block stub.
            product_block_marker:   Text that indicates the product page is still
                                    a DataDome block stub.
            warmup_retries:         Max retry attempts if warmup page stays blocked.
            warmup_retry_delay:     Seconds to wait between warmup retries.

        Returns:
            Full product page HTML, or None on failure.  Never raises.
        """
        if self._closed:
            logger.warning("BrowserFetcher.fetch_with_warmup called on a closed instance")
            return None

        if not await self._ensure_running():
            return None

        # ------------------------------------------------------------------
        # Phase 1: warmup navigation with retries
        # ------------------------------------------------------------------
        warmup_cleared = False
        warmup_html_len = 0
        warmup_attempts = 0
        per_attempt_timeout = max(warmup_timeout / max(warmup_retries, 1), 15.0)

        page = None
        try:
            page = await self._context.new_page()
        except Exception as exc:
            logger.warning("BrowserFetcher.fetch_with_warmup: new_page failed: %s", exc)
            return None

        try:
            for attempt in range(warmup_retries):
                warmup_attempts = attempt + 1
                timeout_ms = int(per_attempt_timeout * 1000)

                try:
                    logger.info(
                        "BrowserFetcher.warmup: attempt %d/%d → %s",
                        attempt + 1, warmup_retries, warmup_url,
                    )
                    await page.goto(warmup_url, wait_until="domcontentloaded", timeout=timeout_ms)
                except Exception as exc:
                    logger.warning(
                        "BrowserFetcher.warmup: goto failed on attempt %d: %s", attempt + 1, exc
                    )
                    if attempt < warmup_retries - 1:
                        await asyncio.sleep(warmup_retry_delay)
                    continue

                # Wait for either real selector or networkidle, tolerating timeouts
                if warmup_real_selector:
                    try:
                        await page.wait_for_selector(
                            warmup_real_selector,
                            timeout=min(timeout_ms, int(per_attempt_timeout * 800)),
                        )
                    except Exception:
                        pass
                else:
                    try:
                        await page.wait_for_load_state(
                            "networkidle", timeout=min(timeout_ms, int(per_attempt_timeout * 800))
                        )
                    except Exception:
                        pass

                warmup_html = await page.content()

                # Patient JS-challenge wait on the warmup page too: Qrator may
                # serve its self-reloading stub here before the real homepage.
                cw = challenge_wait if challenge_wait is not None else _challenge_wait_seconds()
                if cw > 0 and _looks_like_challenge(warmup_html):
                    resolved = await _wait_out_challenge(page, warmup_url, challenge_wait=cw)
                    if resolved is not None:
                        warmup_html = resolved

                warmup_html_len = len(warmup_html)

                if warmup_block_marker in warmup_html:
                    logger.info(
                        "BrowserFetcher.warmup: still blocked (%d chars) on attempt %d — retrying in %.1fs",
                        warmup_html_len, attempt + 1, warmup_retry_delay,
                    )
                    if attempt < warmup_retries - 1:
                        await asyncio.sleep(warmup_retry_delay)
                    continue

                # Real content detected
                warmup_cleared = True
                logger.info(
                    "BrowserFetcher.warmup: CLEARED on attempt %d (%d chars)",
                    attempt + 1, warmup_html_len,
                )
                break

            if not warmup_cleared:
                logger.warning(
                    "BrowserFetcher.warmup: DataDome NOT cleared after %d attempts "
                    "(last html len=%d) — still trying product URL with accumulated cookies",
                    warmup_attempts, warmup_html_len,
                )

            # ------------------------------------------------------------------
            # Phase 2: product URL fetch with accumulated DataDome cookies
            # ------------------------------------------------------------------
            product_timeout_ms = int(product_timeout * 1000)

            logger.info("BrowserFetcher.warmup: navigating to product URL %s", url)
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=product_timeout_ms)
            except Exception as exc:
                logger.warning("BrowserFetcher.warmup: product goto failed: %s", exc)
                return None

            # Wait patiently for product content
            if product_real_selector:
                try:
                    await page.wait_for_selector(
                        product_real_selector,
                        timeout=min(product_timeout_ms, 40_000),
                    )
                except Exception:
                    logger.debug(
                        "BrowserFetcher.warmup: product selector %r not found — using DOM as-is",
                        product_real_selector,
                    )
            else:
                try:
                    await page.wait_for_load_state(
                        "networkidle", timeout=min(product_timeout_ms, 40_000)
                    )
                except Exception:
                    pass

            product_html = await page.content()

            # Patient JS-challenge wait on the product page: don't return the
            # Qrator stub — poll until the real product DOM appears.
            cw = challenge_wait if challenge_wait is not None else _challenge_wait_seconds()
            if cw > 0 and _looks_like_challenge(product_html):
                resolved = await _wait_out_challenge(page, url, challenge_wait=cw)
                if resolved is not None:
                    product_html = resolved

            product_html_len = len(product_html)

            if product_block_marker in product_html:
                logger.warning(
                    "BrowserFetcher.warmup: product page still BLOCKED after warmup "
                    "(%d chars) — IP may be rate-limited; returning None",
                    product_html_len,
                )
                return None

            logger.info(
                "BrowserFetcher.warmup: product page OK (%d chars) warmup_cleared=%s",
                product_html_len, warmup_cleared,
            )
            return product_html

        except Exception as exc:
            logger.warning("BrowserFetcher.fetch_with_warmup error: %s", exc)
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
        """Kill the Chrome process tree (parent + all child renderers).

        Strategy (in priority order):
          1. psutil — walks the full process tree recursively, kills each child
             then the parent.  Most reliable on both Windows and POSIX.
          2. taskkill /F /T /PID — Windows fallback when psutil is absent; /T
             kills the entire process tree rooted at the given PID.
          3. proc.terminate() + proc.kill() — last-resort POSIX fallback.

        The method is idempotent: sets _chrome_proc to None after the first call.
        All exceptions are swallowed so callers (atexit, close(), context exit)
        are never interrupted.
        """
        proc = self._chrome_proc
        if proc is None:
            return
        self._chrome_proc = None  # clear early so re-entrancy is safe

        pid = proc.pid

        if _psutil is not None:
            # psutil path: kill whole tree
            try:
                parent = _psutil.Process(pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.kill()
                    except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                        pass
                try:
                    parent.kill()
                except (_psutil.NoSuchProcess, _psutil.AccessDenied):
                    pass
            except Exception:
                pass
        elif platform.system() == "Windows":
            # taskkill /F /T /PID kills the process tree on Windows
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            except Exception:
                pass
        else:
            # POSIX fallback: terminate + kill
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    def _cleanup_user_data_dir(self) -> None:
        path = self._user_data_dir
        if not path:
            return
        self._user_data_dir = None  # clear early — idempotent
        if os.path.isdir(path):
            try:
                shutil.rmtree(path, ignore_errors=True)
                logger.debug("BrowserFetcher: removed temp dir %s", path)
            except Exception:
                pass

    def _sync_cleanup(self) -> None:
        """atexit handler — synchronous teardown of the Chrome process.

        Idempotent: safe to call even after close() already ran (all state
        fields are set to None by _terminate_chrome / _cleanup_user_data_dir).
        """
        self._terminate_chrome()
        self._cleanup_user_data_dir()
