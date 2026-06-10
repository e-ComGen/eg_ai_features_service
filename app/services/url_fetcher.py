"""
app/services/url_fetcher.py

MVP Stage 2 — web fetch by URL.
Fetches supplier/competitor product pages and returns aggregated text
that is then injected into the existing extraction pipeline.

Supported sources with dedicated extractors:
  - Wildberries (public card.wb.ru API)
  - AliExpress (JSON-LD extraction)
  - Ozon (__NEXT_DATA__ / __INITIAL_STATE__ extraction)
  - Generic pages (trafilatura main-content extraction, fallback to raw text)

All fetches are:
  - HTTPS-only
  - Capped at 50 KB per URL
  - Limited to 10 s timeout per URL
  - Run in parallel via asyncio.gather
  - Non-fatal: individual URL failures are logged and skipped
  - Retried on transient errors (429/5xx/timeout/connect) with exponential backoff
  - Cached on disk (SHA-256 key, only successful non-empty results)

Scrappey browser-bypass fallback
---------------------------------
When enabled (URL_FETCHER_SCRAPPEY_FALLBACK=1) the fallback fires whenever the
plain-httpx attempt yields UNUSABLE content, i.e. any of:

  a. ban / anti-bot status: 401, 403, 418
  b. transient 429/503/5xx — only AFTER all backoff retries are exhausted
  c. a 3xx redirect that lands on a block/captcha page (detected by
     _looks_like_block on the final body)
  d. a 200 whose extracted body is too short (< SCRAPPEY_MIN_TEXT_LEN chars) or
     contains anti-bot markers (_looks_like_block)
  e. connect-error / timeout after all retries on a resolvable host

Cost bounds (critical — must not explode):
  - At most ONE Scrappey attempt per URL.
  - Per-process cap URL_FETCHER_SCRAPPEY_MAX (default 200); logged when hit.
  - Non-content hosts (CDNs, trackers, social) are never sent to Scrappey
    (_is_nontext_host denylist).
  - URL_FETCHER_SCRAPPEY_DOMAINS is kept as an optional "always-eligible" fast
    set; the restriction that ONLY those domains could trigger Scrappey is
    REMOVED — any host not in the denylist is now eligible.

If Scrappey returns nothing or its result still looks like a block, the
function returns None gracefully (fail-closed, never surfaces captcha pages).
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Literal, Optional

import httpx

logger = logging.getLogger(__name__)

MAX_CONTENT_BYTES = 50_000  # 50 KB per URL
DEFAULT_TIMEOUT = 10        # seconds

# Realistic Chrome user-agent — WB/Ozon block the default python-httpx UA
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_RETRY_ATTEMPTS = 3                    # total attempts (1 original + 2 retries)
_RETRY_BACKOFF = [1.0, 2.0, 4.0]      # seconds between retries
# HTTP status codes that are transient and worth retrying
_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}
# Errors that indicate a hard block / missing page — do NOT retry
_HARD_FAIL_STATUSES = {403, 404}
# Statuses that mean "domain banned us" (WAF / anti-bot) — immediately eligible
# for the Scrappey browser-bypass (no retry needed for these).
_BAN_STATUSES = {401, 403, 418}

# ---------------------------------------------------------------------------
# Content-quality thresholds
# ---------------------------------------------------------------------------

# Minimum meaningful text length to consider a page usable
SCRAPPEY_MIN_TEXT_LEN = 500

# Anti-bot / block-page markers (case-insensitive substring search)
_BLOCK_MARKERS = [
    "captcha",
    "smartcaptcha",
    "datadome",
    "access denied",
    "just a moment",
    "cf-challenge",
    "cloudflare",
    "проверка",
    "подтвердите, что вы не робот",
    "are you a robot",
]


def _looks_like_block(text: str) -> bool:
    """Return True when text looks like an anti-bot / captcha wall.

    Checks case-insensitively against known block-page markers.  Used both to
    decide whether to fire Scrappey and to validate its result.
    """
    if not text:
        return True
    lower = text.lower()
    return any(marker in lower for marker in _BLOCK_MARKERS)


# ---------------------------------------------------------------------------
# Scrappey paid fallback — cost-bound configuration
# ---------------------------------------------------------------------------
# OFF by default so production cost does not change unless explicitly enabled.

# Per-process Scrappey call counter — shared across all coroutines via this
# module-level int.  asyncio is single-threaded so no lock is needed.
_scrappey_call_count: int = 0


def _scrappey_fallback_enabled() -> bool:
    return os.environ.get("URL_FETCHER_SCRAPPEY_FALLBACK", "0") == "1"


def _scrappey_max() -> int:
    """Per-process cap on total Scrappey fallback calls."""
    try:
        return int(os.environ.get("URL_FETCHER_SCRAPPEY_MAX", "200"))
    except ValueError:
        return 200


def _scrappey_timeout_cap() -> float:
    """Hard timeout cap (seconds) for a single Scrappey fallback call.

    Configured via URL_FETCHER_SCRAPPEY_TIMEOUT (default 25s).
    Rationale: the 4 known wall domains (lamoda/sportmaster/dns-shop/citilink)
    NEVER succeed even with residential proxies, so killing them at 25s is pure
    win; soft sites that Scrappey CAN beat usually respond well under 25s.

    Note: for browser mode use _scrappey_browser_timeout_cap() instead — JS
    rendering legitimately needs more time.
    """
    try:
        return float(os.environ.get("URL_FETCHER_SCRAPPEY_TIMEOUT", "25"))
    except ValueError:
        return 25.0


def _scrappey_browser_timeout_cap() -> float:
    """Hard timeout cap (seconds) for a Scrappey browser-mode fallback call.

    Configured via URL_FETCHER_SCRAPPEY_BROWSER_TIMEOUT (default 40s).
    Browser mode spins up a full Chromium instance and executes JS (including
    Qrator/Cloudflare JS challenges), which takes 15–35 s on typical retail
    pages.  40 s gives a comfortable margin while still bounding runaway calls.
    """
    try:
        return float(os.environ.get("URL_FETCHER_SCRAPPEY_BROWSER_TIMEOUT", "40"))
    except ValueError:
        return 40.0


def _scrappey_fast_domains() -> set:
    """Optional set of domains that are always eligible for Scrappey fallback.

    Previously this was the ONLY allowlist — now it is an *optional* fast set.
    Any host not in the denylist (_is_nontext_host) is eligible regardless.
    Kept for backward compatibility / explicit opt-in list.
    """
    raw = os.environ.get(
        "URL_FETCHER_SCRAPPEY_DOMAINS",
        "dns-shop.ru,ozon.ru,wildberries.ru,citilink.ru",
    )
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


# Denylist: non-content hosts — CDNs, trackers, social, ad networks.
# Never send these to Scrappey (no real text content to gain, wastes credits).
_NONTEXT_HOST_PATTERNS = re.compile(
    r"""
    ^cdn[.\-]                       |   # cdn.* subdomains
    \.googletagmanager\.com$        |
    \.doubleclick\.net$             |
    \.google-analytics\.com$        |
    \.facebook\.com$                |
    \.fbcdn\.net$                   |
    \.instagram\.com$               |
    \.twitter\.com$                 |
    \.t\.co$                        |
    \.vk\.com$                      |
    \.mc\.yandex\.ru$               |
    \.yandex-team\.ru$              |
    \.adnxs\.com$                   |
    \.criteo\.com$                  |
    \.akamaized\.net$               |
    \.cloudfront\.net$              |
    \.fastly\.net$                  |
    \.gstatic\.com$                 |
    \.googleapis\.com$              |
    \.ajax\.googleapis\.com$
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _extract_host(url: str) -> str:
    """Extract the lower-case hostname from a URL."""
    host = re.sub(r"^https?://", "", url.lower())
    return host.split("/", 1)[0].split(":", 1)[0]


def _is_nontext_host(url: str) -> bool:
    """Return True for CDN/tracker/social hosts that carry no product text."""
    host = _extract_host(url)
    return bool(_NONTEXT_HOST_PATTERNS.search(host))


def _host_is_fast_domain(url: str) -> bool:
    """Return True if the URL's host is in the configured fast-eligible set."""
    host = _extract_host(url)
    for dom in _scrappey_fast_domains():
        if host == dom or host.endswith("." + dom):
            return True
    return False


def _eligible_for_scrappey(url: str, force: bool = False) -> bool:
    """Return True if this URL should be attempted via Scrappey fallback.

    Rules:
      1. Global flag must be ON *or* ``force=True`` (per-call override).
      2. Per-process cap must not be exceeded.
      3. Host must NOT be in the non-content denylist.
      4. Any host that passes rules 1-3 is eligible (no allowlist restriction).

    The ``force`` flag lets callers (e.g. the composition harvester) enable
    Scrappey for a specific request without flipping the global env default,
    keeping other callers' behaviour unchanged.
    """
    global _scrappey_call_count
    if not force and not _scrappey_fallback_enabled():
        return False
    cap = _scrappey_max()
    if _scrappey_call_count >= cap:
        logger.warning(
            "Scrappey per-process cap (%d) reached — not firing for %r", cap, url
        )
        return False
    if _is_nontext_host(url):
        logger.debug("Scrappey skipped for non-text host: %r", url)
        return False
    return True


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

# Cache directory lives beside this file's project root; override via env var.
_CACHE_DIR = os.environ.get(
    "FETCH_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".fetch_cache"),
)


def _cache_key(url: str) -> str:
    """SHA-256 hex digest of the URL, used as cache file name."""
    return hashlib.sha256(url.encode()).hexdigest()


def _cache_path(url: str) -> str:
    return os.path.join(_CACHE_DIR, _cache_key(url) + ".txt")


def _cache_read(url: str) -> Optional[str]:
    """Return cached text for URL, or None on miss."""
    path = _cache_path(url)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if text:
            logger.debug("cache hit: %s", url)
            return text
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.debug("cache read error for %r: %s", url, exc)
    return None


def _cache_write(url: str, text: str) -> None:
    """Persist text for URL only when text is non-empty."""
    if not text:
        return
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = _cache_path(url)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        logger.debug("cache write: %s", url)
    except Exception as exc:
        logger.debug("cache write error for %r: %s", url, exc)


# ---------------------------------------------------------------------------
# Scrappey fallback (content-aware, general)
# ---------------------------------------------------------------------------

async def _try_scrappey_fallback(
    url: str,
    reason: str,
    timeout: int,
    force: bool = False,
) -> Optional[httpx.Response]:
    """Attempt to bypass a block/unusable result via the Scrappey browser.

    Returns a synthetic httpx.Response(200) wrapping the bypassed HTML, or
    None if Scrappey is disabled, the cap is hit, the host is denylisted,
    the domain is in the dead-domain store, or Scrappey itself fails /
    returns a block page.

    The ``reason`` string is logged for observability (e.g. "HTTP 403",
    "short body (120 chars)", "block marker in 200").

    ``force=True`` bypasses the global ``URL_FETCHER_SCRAPPEY_FALLBACK`` env
    flag so per-call callers (e.g. the composition harvester) can opt-in
    without changing the global default.
    """
    global _scrappey_call_count

    if not _eligible_for_scrappey(url, force=force):
        return None

    # Dead-domain check: skip paid Scrappey call if this domain is known dead.
    host = _extract_host(url)
    try:
        from app.services.providers.domain_health import (
            should_skip_scrappey,
            record_scrappey_outcome,
        )
        if should_skip_scrappey(host):
            import time as _time
            from app.services.providers import domain_health as _dh
            _dh._load_store()
            from app.services.providers.domain_health import _registrable_domain, _store
            domain = _registrable_domain(host)
            rec = _store.get(domain)
            dead_until_ts = rec.dead_until if rec else None
            logger.info(
                "skip Scrappey: %s marked dead until %s",
                host,
                dead_until_ts,
            )
            return None
    except Exception as _dh_exc:
        logger.debug("domain_health check failed for %r: %s", host, _dh_exc)
        # degrade gracefully — proceed with Scrappey normally
        record_scrappey_outcome = None  # type: ignore[assignment]

    logger.info(
        "Scrappey fallback triggered for %r — reason: %s (call #%d)",
        url, reason, _scrappey_call_count + 1,
    )

    _scrappey_call_count += 1

    # Always use browser mode in the fallback: the fallback only fires after
    # plain httpx already failed (anti-bot block / bad content), so bare
    # Scrappey is pointless.  Browser mode (full Chromium + JS rendering) beats
    # Qrator WAF and Cloudflare JS challenges that the bare mode cannot handle.
    #
    # Hard timeout cap: use the browser-mode cap (URL_FETCHER_SCRAPPEY_BROWSER_TIMEOUT,
    # default 40s) because JS rendering legitimately needs more time than bare
    # mode.  asyncio.wait_for enforces the cap even if the underlying httpx
    # client stalls (e.g. stalled TLS handshake on a wall domain).
    cap = _scrappey_browser_timeout_cap()

    html: Optional[str] = None
    scrappey_exception: Optional[Exception] = None
    timed_out = False
    try:
        from app.services.providers.scrappey_client import scrappey_fetch
        html = await asyncio.wait_for(
            scrappey_fetch(url, timeout=cap, browser=True),
            timeout=cap,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Scrappey browser fallback TIMED OUT after %.0fs for %r — recording TRANSIENT",
            cap, url,
        )
        timed_out = True
        scrappey_exception = asyncio.TimeoutError(f"Scrappey cap {cap}s exceeded")
    except Exception as exc:
        logger.warning("Scrappey fallback error for %r: %s", url, exc)
        scrappey_exception = exc

    # Classify outcome and record into domain-health store.
    try:
        from app.services.providers.domain_health import record_scrappey_outcome as _record
        if timed_out:
            # Timeout ≠ confirmed block: the server may just be slow right now.
            # Record TRANSIENT so the death-counter is NOT incremented.
            _record(host, url, "TRANSIENT")
        elif scrappey_exception is not None:
            # Network/SSL/timeout from scrappey_fetch itself → TRANSIENT
            _record(host, url, "TRANSIENT")
        elif not html:
            # Scrappey returned None (upstream non-200, DataDome, empty) → BLOCKED
            _record(host, url, "BLOCKED")
        elif _looks_like_block(html):
            # Scrappey returned a captcha shell → BLOCKED
            _record(host, url, "BLOCKED")
        else:
            # Content looks real → USABLE
            _record(host, url, "USABLE")
    except Exception as _rec_exc:
        logger.debug("domain_health record failed for %r: %s", host, _rec_exc)

    if scrappey_exception is not None:
        return None

    if not html:
        logger.info("Scrappey returned nothing for %r", url)
        return None

    # Fail-closed: never surface a captcha page as if it were content
    if _looks_like_block(html):
        logger.info(
            "Scrappey result still looks like a block page for %r — discarding", url
        )
        return None

    logger.info("Scrappey fallback succeeded for %r (%d chars)", url, len(html))
    return httpx.Response(
        status_code=200,
        text=html,
        request=httpx.Request("GET", url),
    )


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _get_with_retry(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    extra_headers: Optional[dict] = None,
    force_scrappey: bool = False,
) -> Optional[httpx.Response]:
    """
    Perform an HTTP GET with retry + exponential backoff on transient errors.

    Transient (retried):  429, 5xx, TimeoutException, ConnectError,
                          RemoteProtocolError.
    Ban statuses (401, 403, 418): immediately try Scrappey; no HTTP retry.
    Hard failure (skipped): 404 — return None immediately, no retry.
    Other HTTP errors:    return None after logging.

    After all retries are exhausted for transient errors or connect failures,
    Scrappey is attempted as the last resort.

    Returns the Response on success, None on final failure.
    Respects Retry-After header on 429.

    ``force_scrappey=True`` activates the Scrappey tier even when the global
    ``URL_FETCHER_SCRAPPEY_FALLBACK`` env flag is OFF.  This allows per-call
    opt-in (e.g. the composition harvester) without affecting other callers.
    """
    headers = dict(_HEADERS)
    if extra_headers:
        headers.update(extra_headers)

    last_exc: Optional[Exception] = None
    exhausted_transient = False  # set True when we give up on transient errors

    for attempt in range(_RETRY_ATTEMPTS):
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                headers=headers,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)

            # a. Ban / anti-bot statuses — immediate Scrappey, no HTTP retry
            if resp.status_code in _BAN_STATUSES:
                logger.warning(
                    "fetch HTTP %s for %r — anti-bot block, trying Scrappey",
                    resp.status_code, url,
                )
                fb = await _try_scrappey_fallback(
                    url, f"HTTP {resp.status_code}", timeout, force=force_scrappey
                )
                if fb is not None:
                    return fb
                return None

            # 404 — genuinely missing, don't retry
            if resp.status_code == 404:
                logger.warning("fetch 404 for %r — not retrying", url)
                return None

            # b. Transient — retry with backoff; Scrappey only after exhaustion
            if resp.status_code in _TRANSIENT_STATUSES:
                wait = _retry_wait(resp, attempt)
                logger.warning(
                    "fetch HTTP %s for %r (attempt %d/%d) — retrying in %.1fs",
                    resp.status_code, url, attempt + 1, _RETRY_ATTEMPTS, wait,
                )
                await asyncio.sleep(wait)
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
                if attempt == _RETRY_ATTEMPTS - 1:
                    exhausted_transient = True
                continue

            resp.raise_for_status()   # raise for any remaining 4xx/5xx

            # c/d. Successful HTTP but content may still be unusable — check
            # body quality BEFORE returning.
            body = resp.text
            if _looks_like_block(body):
                logger.warning(
                    "fetch 200 with block marker for %r — trying Scrappey", url
                )
                fb = await _try_scrappey_fallback(
                    url, "block marker in 200", timeout, force=force_scrappey
                )
                return fb  # None is fine (fail-closed)

            if len(body) < SCRAPPEY_MIN_TEXT_LEN:
                logger.warning(
                    "fetch 200 too-short body (%d chars) for %r — trying Scrappey",
                    len(body), url,
                )
                fb = await _try_scrappey_fallback(
                    url, f"short body ({len(body)} chars)", timeout, force=force_scrappey
                )
                if fb is not None:
                    return fb
                # Short body from Scrappey too — return original short resp
                # rather than None so callers can attempt extraction.
                return resp

            return resp

        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            logger.warning(
                "fetch transient error for %r (attempt %d/%d, %s) — retrying in %.1fs",
                url, attempt + 1, _RETRY_ATTEMPTS, type(exc).__name__, wait,
            )
            last_exc = exc
            await asyncio.sleep(wait)
            if attempt == _RETRY_ATTEMPTS - 1:
                exhausted_transient = True

        except httpx.HTTPStatusError as exc:
            # Non-transient, non-handled HTTP error (e.g. 405) — give up
            logger.warning("fetch HTTP error for %r: %s", url, exc)
            return None

        except Exception as exc:
            logger.warning("fetch unexpected error for %r: %s", url, exc)
            return None

    # e. After exhausting all retries (transient or connect errors) — last resort
    if exhausted_transient:
        err_desc = type(last_exc).__name__ if last_exc else "unknown"
        logger.warning(
            "fetch gave up after %d attempts for %r (%s) — trying Scrappey",
            _RETRY_ATTEMPTS, url, err_desc,
        )
        fb = await _try_scrappey_fallback(
            url, f"exhausted retries ({err_desc})", timeout, force=force_scrappey
        )
        if fb is not None:
            return fb

    logger.warning("fetch gave up after %d attempts for %r: %s", _RETRY_ATTEMPTS, url, last_exc)
    return None


def _retry_wait(resp: httpx.Response, attempt: int) -> float:
    """Return seconds to wait before the next retry, honouring Retry-After."""
    ra = resp.headers.get("Retry-After")
    if ra:
        try:
            return max(0.0, float(ra))
        except ValueError:
            pass
        # Retry-After may also be an HTTP-date — skip parsing, use backoff
    return _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]


# ---------------------------------------------------------------------------

@dataclass
class FetchResult:
    url: str
    content: str          # extracted text, already trimmed to MAX_CONTENT_BYTES
    source_type: str      # "wb" | "ali" | "ozon" | "generic"
    raw_html: Optional[str] = None  # full raw HTML before cap/trafilatura (generic pages only)
                                    # used by composition_extractor which needs the full page


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------

def detect_source(url: str) -> Literal["ali", "wb", "ozon", "generic"]:
    """Detect the product source by domain."""
    url_lower = url.lower()
    if "aliexpress.com" in url_lower or "aliexpress.ru" in url_lower:
        return "ali"
    if "wildberries.ru" in url_lower or "wb.ru" in url_lower:
        return "wb"
    if "ozon.ru" in url_lower:
        return "ozon"
    return "generic"


# ---------------------------------------------------------------------------
# Wildberries
# ---------------------------------------------------------------------------

_WB_NM_RE = re.compile(r"/catalog/(\d+)/")


def _extract_wb_nm_id(url: str) -> Optional[str]:
    """Extract nm_id from a Wildberries product URL."""
    m = _WB_NM_RE.search(url)
    if m:
        return m.group(1)
    # Also handle numeric-only input (article passed directly)
    if url.strip().isdigit():
        return url.strip()
    return None


async def fetch_wildberries(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[FetchResult]:
    """
    Fetch product data from the public Wildberries card API.
    Falls back to generic fetch if nm_id cannot be extracted.
    """
    nm_id = _extract_wb_nm_id(url)
    if not nm_id:
        logger.warning("fetch_wildberries: cannot extract nm_id from %r, falling back to generic", url)
        return await fetch_url_content(url, timeout=timeout)

    api_url = (
        f"https://card.wb.ru/cards/v2/detail"
        f"?appType=1&curr=rub&dest=-1257786&spp=27&nm={nm_id}"
    )

    # Check cache first (cache key on the canonical api_url)
    cached = _cache_read(api_url)
    if cached is not None:
        return FetchResult(url=url, content=cached, source_type="wb")

    try:
        resp = await _get_with_retry(api_url, timeout=timeout)
        if resp is None:
            return None
        data = resp.json()

        # Extract useful text fields from the WB API response
        parts: list[str] = []
        products = data.get("data", {}).get("products", [])
        if products:
            p = products[0]
            if p.get("name"):
                parts.append(f"Name: {p['name']}")
            if p.get("brand"):
                parts.append(f"Brand: {p['brand']}")
            if p.get("description"):
                parts.append(f"Description: {p['description']}")
            # Flatten characteristics
            for char in p.get("options", []):
                name = char.get("name", "")
                value = char.get("value", "")
                if name and value:
                    parts.append(f"{name}: {value}")
        content = "\n".join(parts)
        if not content:
            content = json.dumps(data, ensure_ascii=False)
        content = content[:MAX_CONTENT_BYTES]

        _cache_write(api_url, content)
        return FetchResult(url=url, content=content, source_type="wb")

    except Exception as exc:
        logger.warning("fetch_wildberries failed for %r: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# AliExpress
# ---------------------------------------------------------------------------

async def fetch_aliexpress(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[FetchResult]:
    """
    Attempt to extract JSON-LD from AliExpress product page.
    Falls back to generic extraction.
    """
    cached = _cache_read(url)
    if cached is not None:
        return FetchResult(url=url, content=cached, source_type="ali")

    try:
        resp = await _get_with_retry(url, timeout=timeout)
        if resp is None:
            return None
        html = resp.text

        # Try JSON-LD first
        ld_matches = re.findall(
            r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
            html, re.DOTALL | re.IGNORECASE
        )
        parts: list[str] = []
        for raw in ld_matches:
            try:
                obj = json.loads(raw.strip())
                if isinstance(obj, dict):
                    for field in ("name", "description", "brand", "model"):
                        val = obj.get(field)
                        if val:
                            if isinstance(val, dict):
                                val = val.get("name", "")
                            parts.append(f"{field.capitalize()}: {val}")
            except (json.JSONDecodeError, TypeError):
                continue

        if parts:
            content = "\n".join(parts)[:MAX_CONTENT_BYTES]
            _cache_write(url, content)
            return FetchResult(url=url, content=content, source_type="ali")

        # Fallback to generic
        result = await _extract_generic_from_html(url, html)
        if result:
            result.source_type = "ali"
            _cache_write(url, result.content)
        return result

    except Exception as exc:
        logger.warning("fetch_aliexpress failed for %r: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Ozon
# ---------------------------------------------------------------------------

async def fetch_ozon(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[FetchResult]:
    """
    Attempt to extract __NEXT_DATA__ or __INITIAL_STATE__ from Ozon product page.
    Falls back to generic extraction.
    """
    cached = _cache_read(url)
    if cached is not None:
        return FetchResult(url=url, content=cached, source_type="ozon")

    try:
        resp = await _get_with_retry(url, timeout=timeout)
        if resp is None:
            return None
        html = resp.text

        # Try __NEXT_DATA__
        m = re.search(r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>', html, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                # Walk props.pageProps for product info
                page_props = data.get("props", {}).get("pageProps", {})
                parts: list[str] = []
                _extract_ozon_props(page_props, parts)
                if parts:
                    content = "\n".join(parts)[:MAX_CONTENT_BYTES]
                    _cache_write(url, content)
                    return FetchResult(url=url, content=content, source_type="ozon")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        # Try __INITIAL_STATE__
        m2 = re.search(r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});\s*</script>', html, re.DOTALL)
        if m2:
            try:
                data = json.loads(m2.group(1))
                text_dump = json.dumps(data, ensure_ascii=False)[:MAX_CONTENT_BYTES]
                _cache_write(url, text_dump)
                return FetchResult(url=url, content=text_dump, source_type="ozon")
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback to generic
        result = await _extract_generic_from_html(url, html)
        if result:
            result.source_type = "ozon"
            _cache_write(url, result.content)
        return result

    except Exception as exc:
        logger.warning("fetch_ozon failed for %r: %s", url, exc)
        return None


def _extract_ozon_props(obj: dict, parts: list[str], depth: int = 0) -> None:
    """Recursively extract name/description/title fields from Ozon page props."""
    if depth > 5 or not isinstance(obj, dict):
        return
    for key, val in obj.items():
        if key.lower() in ("name", "title", "description", "brand") and isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")
        elif isinstance(val, dict):
            _extract_ozon_props(val, parts, depth + 1)
        elif isinstance(val, list) and depth < 3:
            for item in val[:10]:
                if isinstance(item, dict):
                    _extract_ozon_props(item, parts, depth + 1)


# ---------------------------------------------------------------------------
# Generic
# ---------------------------------------------------------------------------

async def fetch_url_content(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    force_scrappey: bool = False,
) -> Optional[FetchResult]:
    """
    Generic HTTPS-only fetch with main-content extraction.
    Uses trafilatura if available, otherwise falls back to basic HTML stripping.
    Retries on transient errors; returns cached result on repeated calls.

    ``force_scrappey=True`` enables the Scrappey proxy tier for this specific
    call even when the global ``URL_FETCHER_SCRAPPEY_FALLBACK`` env flag is
    OFF.  Other callers are unaffected.  Intended for the composition harvester
    which needs Scrappey as an IP-shielding layer on open (non-walled) sites.
    """
    if not url.startswith("https://"):
        logger.warning("fetch_url_content: rejected non-HTTPS URL %r", url)
        return None

    # Cache check
    cached = _cache_read(url)
    if cached is not None:
        return FetchResult(url=url, content=cached, source_type="generic")

    try:
        resp = await _get_with_retry(url, timeout=timeout, force_scrappey=force_scrappey)
        if resp is None:
            return None
        html = resp.text

        result = await _extract_generic_from_html(url, html)
        if result:
            _cache_write(url, result.content)
        return result

    except Exception as exc:
        logger.warning("fetch_url_content failed for %r: %s", url, exc)
        return None


async def _extract_generic_from_html(url: str, html: str) -> Optional[FetchResult]:
    """Extract main content from HTML. Tries trafilatura, then strips tags.

    Stores the FULL raw HTML in FetchResult.raw_html (capped at 1 MB) so that
    composition_extractor can mine fabric composition from the original page
    before trafilatura/content-cap discard the spec block.
    """
    content: Optional[str] = None

    # Try trafilatura (best quality)
    try:
        import trafilatura  # type: ignore
        # favor_recall=True pulls MORE body text (spec tables/lists that the
        # default precision mode drops). The _looks_like_boilerplate guard
        # downstream (websearch_producer, commit 325573d) still filters noise.
        content = trafilatura.extract(
            html, include_comments=False, include_tables=True, favor_recall=True
        )
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("trafilatura extraction failed: %s", exc)

    # Fallback: strip HTML tags naively
    if not content:
        content = re.sub(r'<[^>]+>', ' ', html)
        content = re.sub(r'\s+', ' ', content).strip()

    if not content:
        return None

    # Keep raw_html for composition mining (cap at 1 MB to avoid OOM on huge pages)
    raw_html_for_mining = html[:1_000_000] if html else None

    content = content[:MAX_CONTENT_BYTES]
    return FetchResult(url=url, content=content, source_type="generic", raw_html=raw_html_for_mining)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def _fetch_one_result(url: str) -> Optional[FetchResult]:
    """Fetch a single URL and return a FetchResult (with raw_html for generic pages)."""
    if not url.startswith("https://"):
        logger.warning("fetch_all: skipping non-HTTPS URL %r", url)
        return None
    source = detect_source(url)
    try:
        if source == "wb":
            return await fetch_wildberries(url)
        elif source == "ali":
            return await fetch_aliexpress(url)
        elif source == "ozon":
            return await fetch_ozon(url)
        else:
            return await fetch_url_content(url)
    except Exception as exc:
        logger.warning("fetch_all: unhandled error for %r: %s", url, exc)
        return None


async def fetch_all_results(urls: list[str]) -> list[FetchResult]:
    """Parallel fetch all URLs. Returns list of FetchResult (preserves raw_html).

    Use this when you need raw HTML for composition mining.
    Individual failures are silently skipped (non-fatal).
    Only HTTPS URLs are accepted.
    """
    if not urls:
        return []

    gathered = await asyncio.gather(
        *[_fetch_one_result(u) for u in urls], return_exceptions=True
    )

    out: list[FetchResult] = []
    for url, result in zip(urls, gathered):
        if isinstance(result, Exception):
            logger.warning("fetch_all_results: exception for %r: %s", url, result)
            continue
        if result is None:
            continue
        out.append(result)

    return out


async def fetch_all(urls: list[str]) -> str:
    """
    Parallel fetch all URLs. Aggregates results with source markers.
    Individual failures are silently skipped (non-fatal).
    Only HTTPS URLs are accepted.
    """
    if not urls:
        return ""

    results = await fetch_all_results(urls)

    parts: list[str] = []
    for result in results:
        parts.append(f"=== Source: {result.url} ===\n{result.content}")

    return "\n\n".join(parts)
