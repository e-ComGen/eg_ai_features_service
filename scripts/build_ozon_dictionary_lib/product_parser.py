"""Парсит __NEXT_DATA__ из Ozon product страниц.

Primary source: archive.org Wayback Machine (обходит DataDome/Cloudflare — архив
фетчит с собственных IP, антибот не срабатывает).
Fallback: Playwright headless Chromium.

Ozon отдаёт данные товара в JSON-блоке <script id="__NEXT_DATA__">, встроенном
в HTML страницы.

TODO: Структура __NEXT_DATA__ регулярно меняется — возможно потребуется
      адаптация walk-логики под актуальный формат Ozon.
      Рекомендуется проверить вручную, сохранив HTML страницы через:
          playwright codegen https://www.ozon.ru/product/<id>/
"""
import json
import re
import urllib.parse
import urllib.request
from typing import Any, Optional

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


_DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Regex to extract __NEXT_DATA__ JSON block
_NEXT_DATA_RE = re.compile(
    r'<script\s+id=["\']__NEXT_DATA__["\'][^>]*>\s*(\{.+?})\s*</script>',
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Wayback Machine helpers
# ---------------------------------------------------------------------------

def _fetch_via_wayback(product_url: str) -> Optional[str]:
    """Fetch an Ozon product page HTML from archive.org Wayback Machine.

    Steps:
    1. Query the availability API to get the latest snapshot URL.
    2. Download the snapshot via curl_cffi (impersonate=chrome120) or stdlib urllib.

    Returns raw HTML string, or None if no snapshot is found / any error occurs.
    """
    # Step 1: resolve snapshot URL
    api = (
        "https://archive.org/wayback/available"
        f"?url={urllib.parse.quote(product_url, safe=':/')}"
    )
    snapshot_url: Optional[str] = None
    try:
        try:
            from curl_cffi import requests as _cffi
            resp = _cffi.get(api, timeout=12)
            data = resp.json()
        except ImportError:
            with urllib.request.urlopen(api, timeout=12) as r:
                data = json.loads(r.read())

        snapshot = data.get("archived_snapshots", {}).get("closest", {})
        if snapshot.get("available"):
            snapshot_url = snapshot["url"]
    except Exception as exc:
        print(f"[product_parser] Wayback availability check failed for {product_url}: {exc}")

    # Fall back to CDX API when availability API returns nothing
    if not snapshot_url:
        cdx = (
            "https://web.archive.org/cdx/search/cdx"
            f"?url={urllib.parse.quote(product_url, safe=':/')}"
            "&output=json&limit=1&fl=timestamp,original&filter=statuscode:200&fastLatest=true"
        )
        try:
            try:
                from curl_cffi import requests as _cffi
                resp = _cffi.get(cdx, timeout=15)
                rows = resp.json()
            except ImportError:
                with urllib.request.urlopen(cdx, timeout=15) as r:
                    rows = json.loads(r.read())

            if len(rows) > 1:
                ts, orig = rows[1][0], rows[1][1]
                snapshot_url = f"https://web.archive.org/web/{ts}/{orig}"
        except Exception as exc:
            print(f"[product_parser] Wayback CDX lookup failed for {product_url}: {exc}")

    if not snapshot_url:
        return None

    # Step 2: download the snapshot
    try:
        try:
            from curl_cffi import requests as _cffi
            resp = _cffi.get(snapshot_url, timeout=30, impersonate="chrome120")
            return resp.text
        except ImportError:
            req = urllib.request.Request(
                snapshot_url,
                headers={"User-Agent": _DEFAULT_USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"[product_parser] Wayback snapshot download failed ({snapshot_url}): {exc}")
        return None


async def _fetch_via_playwright(product_url: str) -> Optional[str]:
    """Fetch an Ozon product page HTML via headless Playwright (Cloudflare fallback).

    Returns raw HTML string, or None on any failure.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError(
            "playwright is not installed.  Run: pip install playwright && playwright install chromium"
        )

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = await browser.new_context(
            user_agent=_DEFAULT_USER_AGENT,
            viewport={"width": 1280, "height": 720},
            locale="ru-RU",
        )
        page = await context.new_page()
        try:
            await page.goto(product_url, wait_until="networkidle", timeout=30_000)
            return await page.content()
        except Exception as exc:
            print(f"[product_parser] Playwright navigation failed for {product_url}: {exc}")
            return None
        finally:
            await browser.close()


async def parse_product_characteristics(product_url: str) -> list[dict]:
    """Load an Ozon product page and extract characteristics from __NEXT_DATA__.

    Strategy (DataDome/Cloudflare bypass):
    - Primary:  archive.org Wayback Machine via curl_cffi (no antibot exposure).
    - Fallback: headless Playwright (may be blocked on datacenter IPs).

    Args:
        product_url: Full URL of the Ozon product page.

    Returns:
        List of dicts with keys: key, display_name, value.
        Returns [] on any failure (network error, CF block, missing data).
    """
    # Try Wayback first
    html = _fetch_via_wayback(product_url)
    if html:
        print(f"[product_parser] Wayback: fetched snapshot for {product_url}")
    else:
        print(f"[product_parser] Wayback returned nothing for {product_url}, falling back to Playwright")
        # Fallback to Playwright
        html = await _fetch_via_playwright(product_url)

    if not html:
        return []

    return _extract_from_html(html)


def _extract_from_html(html: str) -> list[dict]:
    """Parse __NEXT_DATA__ JSON from raw HTML and extract characteristics."""
    match = _NEXT_DATA_RE.search(html)
    if not match:
        return []

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    return _extract_chars_from_nextdata(data)


def _extract_chars_from_nextdata(data: Any) -> list[dict]:
    """Walk __NEXT_DATA__ recursively and collect product characteristics.

    Ozon's __NEXT_DATA__ is deeply nested and changes between releases.
    Known paths (as of 2024):
      - props.pageProps.layoutTrackingInfo.characteristics  (older)
      - props.pageProps.initialState.{stateKey}.characteristics  (newer)
      - widget data with type "webCharacteristics"

    The walker below searches for any dict containing "characteristics" or
    "attributes" keys and yields entries that look like Ozon char objects.

    TODO: Validate against real Ozon pages; adapt field names if schema changed.
    """
    chars: list[dict] = []
    seen_keys: set[str] = set()

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            # Pattern 1: {"characteristics": [...]}
            if "characteristics" in obj and isinstance(obj["characteristics"], list):
                for c in obj["characteristics"]:
                    _try_add_char(c, chars, seen_keys)

            # Pattern 2: {"attributes": [...]} — Ozon Content API style
            if "attributes" in obj and isinstance(obj["attributes"], list):
                for c in obj["attributes"]:
                    _try_add_char(c, chars, seen_keys)

            # Pattern 3: Ozon widget "webCharacteristics"
            if obj.get("type") == "webCharacteristics":
                items = obj.get("items") or obj.get("characteristics") or []
                for c in items:
                    _try_add_char(c, chars, seen_keys)

            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return chars


def _try_add_char(c: Any, chars: list[dict], seen: set[str]) -> None:
    """Attempt to parse a single characteristic object from __NEXT_DATA__."""
    if not isinstance(c, dict):
        return

    # Field name candidates from different Ozon API versions
    key = (
        c.get("key")
        or c.get("id")
        or c.get("attribute_id")
        or c.get("name")
        or ""
    )
    display_name = (
        c.get("name")
        or c.get("display_name")
        or c.get("title")
        or str(key)
    )

    if not key:
        return

    str_key = str(key)
    if str_key in seen:
        return
    seen.add(str_key)

    # Extract value (may be a list of {text: "..."} objects or a plain string)
    value: str | None = None
    raw_values = c.get("values") or c.get("value") or []
    if isinstance(raw_values, list) and raw_values:
        first = raw_values[0]
        value = first.get("text") or first.get("value") if isinstance(first, dict) else str(first)
    elif isinstance(raw_values, str):
        value = raw_values

    chars.append(
        {
            "key": str_key,
            "display_name": display_name,
            "value": value,
        }
    )
