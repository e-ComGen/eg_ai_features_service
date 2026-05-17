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
"""

import asyncio
import json
import logging
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
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_HEADERS, follow_redirects=True) as client:
            resp = await client.get(api_url)
            resp.raise_for_status()
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
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
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
            return FetchResult(url=url, content=content, source_type="ali")

        # Fallback to generic
        result = await _extract_generic_from_html(url, html)
        if result:
            result.source_type = "ali"
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
    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
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
                    return FetchResult(url=url, content=content, source_type="ozon")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        # Try __INITIAL_STATE__
        m2 = re.search(r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});\s*</script>', html, re.DOTALL)
        if m2:
            try:
                data = json.loads(m2.group(1))
                text_dump = json.dumps(data, ensure_ascii=False)[:MAX_CONTENT_BYTES]
                return FetchResult(url=url, content=text_dump, source_type="ozon")
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback to generic
        result = await _extract_generic_from_html(url, html)
        if result:
            result.source_type = "ozon"
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
    """
    if not url.startswith("https://"):
        logger.warning("fetch_url_content: rejected non-HTTPS URL %r", url)
        return None

    try:
        async with httpx.AsyncClient(timeout=timeout, headers=_HEADERS, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        return await _extract_generic_from_html(url, html)

    except httpx.TimeoutException:
        logger.warning("fetch_url_content: timeout for %r", url)
        return None
    except Exception as exc:
        logger.warning("fetch_url_content failed for %r: %s", url, exc)
        return None


async def _extract_generic_from_html(url: str, html: str) -> Optional[FetchResult]:
    """Extract main content from HTML. Tries trafilatura, then strips tags."""
    content: Optional[str] = None

    # Try trafilatura (best quality)
    try:
        import trafilatura  # type: ignore
        content = trafilatura.extract(html, include_comments=False, include_tables=True)
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
