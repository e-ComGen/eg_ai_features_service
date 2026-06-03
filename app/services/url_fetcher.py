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
    Hard failure (skipped): 403, 404 — return None immediately, no retry.
    Other HTTP errors:    return None after logging.

    Returns the Response on success, None on final failure.
    Respects Retry-After header on 429.
    """
    headers = dict(_HEADERS)
    if extra_headers:
        headers.update(extra_headers)

    last_exc: Optional[Exception] = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                headers=headers,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)

            if resp.status_code in _HARD_FAIL_STATUSES:
                logger.warning(
                    "fetch hard-fail HTTP %s for %r — not retrying",
                    resp.status_code, url,
                )
                return None

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
                continue

            resp.raise_for_status()   # raise for any other 4xx/5xx
            return resp

        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            wait = _RETRY_BACKOFF[min(attempt, len(_RETRY_BACKOFF) - 1)]
            logger.warning(
                "fetch transient error for %r (attempt %d/%d, %s) — retrying in %.1fs",
                url, attempt + 1, _RETRY_ATTEMPTS, type(exc).__name__, wait,
            )
            last_exc = exc
            await asyncio.sleep(wait)

        except httpx.HTTPStatusError as exc:
            # Non-transient, non-hard HTTP error (e.g. 401, 405) — give up
            logger.warning("fetch HTTP error for %r: %s", url, exc)
            return None

        except Exception as exc:
            logger.warning("fetch unexpected error for %r: %s", url, exc)
            return None

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

async def fetch_url_content(url: str, timeout: int = DEFAULT_TIMEOUT) -> Optional[FetchResult]:
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
    """Extract main content from HTML. Tries trafilatura, then strips tags."""
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

    content = content[:MAX_CONTENT_BYTES]
    return FetchResult(url=url, content=content, source_type="generic")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

async def fetch_all(urls: list[str]) -> str:
    """
    Parallel fetch all URLs. Aggregates results with source markers.
    Individual failures are silently skipped (non-fatal).
    Only HTTPS URLs are accepted.
    """
    if not urls:
        return ""

    async def _fetch_one(url: str) -> Optional[FetchResult]:
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

    results = await asyncio.gather(*[_fetch_one(u) for u in urls], return_exceptions=True)

    parts: list[str] = []
    for url, result in zip(urls, results):
        if isinstance(result, Exception):
            logger.warning("fetch_all: exception for %r: %s", url, result)
            continue
        if result is None:
            continue
        parts.append(f"=== Source: {result.url} ===\n{result.content}")

    return "\n\n".join(parts)
