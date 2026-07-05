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

scrape.do anti-bot fallback
---------------------------
When plain-httpx yields UNUSABLE content, the fetch retries once through
scrape.do (the sole anti-bot tier since Scrappey was retired 2026-06-14),
firing on a ban status (401/403/418), an exhausted transient (429/5xx), a
block/captcha page (_looks_like_block), a too-short 200 body
(< FALLBACK_MIN_TEXT_LEN), or a connect-error/timeout after retries.

scrape.do runs only when SCRAPEDO_TOKEN is set; otherwise the fallback is a
no-op and the fetch returns None gracefully (fail-closed, never surfaces
captcha pages).
"""

import asyncio
import hashlib
import json
import logging
import os
import re
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
# for the scrape.do anti-bot fallback (no retry needed for these).
_BAN_STATUSES = {401, 403, 418}

# ---------------------------------------------------------------------------
# Content-quality thresholds
# ---------------------------------------------------------------------------

# Minimum meaningful text length to consider a page usable
FALLBACK_MIN_TEXT_LEN = 500

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
    decide whether to fire the scrape.do fallback and to validate its result.
    """
    if not text:
        return True
    lower = text.lower()
    return any(marker in lower for marker in _BLOCK_MARKERS)


# ---------------------------------------------------------------------------
# Non-content host denylist (CDNs / trackers / social) — never worth fetching
# ---------------------------------------------------------------------------
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
# Anti-bot fallback (scrape.do - the sole anti-bot tier since Scrappey retired)
# ---------------------------------------------------------------------------

async def _try_antibot_fallback(
    url: str,
    reason: str,
    timeout: int,
) -> Optional[httpx.Response]:
    """Attempt to bypass a block/unusable result via scrape.do.

    scrape.do is the PRIMARY (and only) anti-bot tier since Scrappey was retired
    2026-06-14. It pierces RU anti-bot the same way the Yandex.Market / Lamoda
    card sources do. Returns a synthetic httpx.Response(200) wrapping the bypassed
    HTML, or None if scrape.do is unconfigured, fails, or returns no content.

    The ``reason`` string is logged for observability (e.g. "HTTP 403").
    """
    if not os.environ.get("SCRAPEDO_TOKEN"):
        return None
    try:
        from app.services.providers.scrapedo_client import scrapedo_fetch
        _sd = await scrapedo_fetch(url, render=True, super_proxy=True, geo="ru")
        # Trust scrapedo's own success gate (HTTP 200 AND len >= 50k) - same as
        # the YM / Lamoda card sources. Do NOT re-run _looks_like_block here: it is
        # a substring marker scan tuned for raw httpx bodies and false-positives on
        # full JS-rendered RU-retail pages (a real 390KB citilink page trips it).
        if _sd.success and _sd.content:
            logger.info(
                "scrape.do fallback OK for %r (%d chars, reason: %s)",
                url, len(_sd.content), reason,
            )
            return httpx.Response(
                status_code=200, text=_sd.content,
                request=httpx.Request("GET", url),
            )
        logger.info(
            "scrape.do fallback no content for %r (success=%s)",
            url, _sd.success,
        )
    except Exception as _sd_exc:
        logger.warning("scrape.do fallback error for %r: %s", url, _sd_exc)
    return None


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

async def _get_with_retry(
    url: str,
    timeout: int = DEFAULT_TIMEOUT,
    extra_headers: Optional[dict] = None,
) -> Optional[httpx.Response]:
    """
    Perform an HTTP GET with retry + exponential backoff on transient errors.

    Transient (retried):  429, 5xx, TimeoutException, ConnectError,
                          RemoteProtocolError.
    Ban statuses (401, 403, 418): immediately try scrape.do; no HTTP retry.
    Hard failure (skipped): 404 — return None immediately, no retry.
    Other HTTP errors:    return None after logging.

    After all retries are exhausted for transient errors or connect failures,
    the scrape.do fallback is attempted as the last resort.

    Returns the Response on success, None on final failure.
    Respects Retry-After header on 429.

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

            # a. Ban / anti-bot statuses — immediate scrape.do, no HTTP retry
            if resp.status_code in _BAN_STATUSES:
                logger.warning(
                    "fetch HTTP %s for %r — anti-bot block, trying scrape.do",
                    resp.status_code, url,
                )
                fb = await _try_antibot_fallback(
                    url, f"HTTP {resp.status_code}", timeout
                )
                if fb is not None:
                    return fb
                return None

            # 404 — genuinely missing, don't retry
            if resp.status_code == 404:
                logger.warning("fetch 404 for %r — not retrying", url)
                return None

            # b. Transient — retry with backoff; scrape.do only after exhaustion
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
                    "fetch 200 with block marker for %r — trying scrape.do", url
                )
                fb = await _try_antibot_fallback(
                    url, "block marker in 200", timeout
                )
                return fb  # None is fine (fail-closed)

            if len(body) < FALLBACK_MIN_TEXT_LEN:
                logger.warning(
                    "fetch 200 too-short body (%d chars) for %r — trying scrape.do",
                    len(body), url,
                )
                fb = await _try_antibot_fallback(
                    url, f"short body ({len(body)} chars)", timeout
                )
                if fb is not None:
                    return fb
                # Short body from scrape.do too — return original short resp
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
            "fetch gave up after %d attempts for %r (%s) — trying scrape.do",
            _RETRY_ATTEMPTS, url, err_desc,
        )
        fb = await _try_antibot_fallback(
            url, f"exhausted retries ({err_desc})", timeout
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
) -> Optional[FetchResult]:
    """
    Generic HTTPS-only fetch with main-content extraction.
    Uses trafilatura if available, otherwise falls back to basic HTML stripping.
    Retries on transient errors; returns cached result on repeated calls.

    """
    if not url.startswith("https://"):
        logger.warning("fetch_url_content: rejected non-HTTPS URL %r", url)
        return None

    # Cache check
    cached = _cache_read(url)
    if cached is not None:
        return FetchResult(url=url, content=cached, source_type="generic")

    try:
        resp = await _get_with_retry(url, timeout=timeout)
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
