"""Парсит __NEXT_DATA__ из Ozon product страниц через Playwright.

Ozon отдаёт данные товара в JSON-блоке <script id="__NEXT_DATA__">, встроенном
в HTML страницы.  Playwright нужен для обхода Cloudflare и JS-рендеринга.

TODO: Структура __NEXT_DATA__ регулярно меняется — возможно потребуется
      адаптация walk-логики под актуальный формат Ozon.
      Рекомендуется проверить вручную, сохранив HTML страницы через:
          playwright codegen https://www.ozon.ru/product/<id>/
"""
import json
import re
from typing import Any

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


async def parse_product_characteristics(product_url: str) -> list[dict]:
    """Load an Ozon product page and extract characteristics from __NEXT_DATA__.

    Args:
        product_url: Full URL of the Ozon product page.

    Returns:
        List of dicts with keys: key, display_name, value.
        Returns [] on any failure (network error, CF block, missing data).
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
            html = await page.content()
        except Exception as exc:
            print(f"[product_parser] Navigation failed for {product_url}: {exc}")
            return []
        finally:
            await browser.close()

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
