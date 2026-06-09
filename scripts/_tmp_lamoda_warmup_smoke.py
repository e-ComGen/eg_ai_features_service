"""Live smoke test: Lamoda homepage warmup + product page fetch.

Run with:
    ./venv/Scripts/python.exe scripts/_tmp_lamoda_warmup_smoke.py

Prints:
  - Whether homepage warmup cleared DataDome (real content length)
  - How many retries / seconds it took
  - Whether the product page yielded real composition (or still blocked)
  - Extracted composition string + Lamoda URL
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("lamoda_smoke")

# ── products to test ────────────────────────────────────────────────────────
PRODUCTS = [
    {
        "product_name": "Толстовка худи Champion Reverse Weave",
        "brand": "Champion",
    },
    {
        "product_name": "Платье Zara миди",
        "brand": "Zara",
    },
]

# ── direct warmup smoke (bypass mine_lamoda_composition to get raw metrics) ──
BLOCK_MARKER = "Доступ ограничен"
HOMEPAGE = "https://www.lamoda.ru/"

SPEC_SELECTOR = (
    "[class*='x-product-characteristics'], [class*='product-characteristics']"
)
WARMUP_REAL_SELECTOR = (
    "[class*='header'], [class*='catalog'], [class*='navigation'], nav, header"
)


async def _direct_warmup_test(fetcher, warmup_url: str, product_url: str) -> dict:
    """Run fetch_with_warmup and return diagnostics dict."""
    t0 = time.monotonic()
    html = await fetcher.fetch_with_warmup(
        product_url,
        warmup_url=warmup_url,
        warmup_timeout=60.0,
        product_timeout=60.0,
        warmup_real_selector=WARMUP_REAL_SELECTOR,
        product_real_selector=SPEC_SELECTOR,
        warmup_block_marker=BLOCK_MARKER,
        product_block_marker=BLOCK_MARKER,
        warmup_retries=3,
        warmup_retry_delay=6.0,
    )
    elapsed = time.monotonic() - t0
    return {
        "html": html,
        "html_len": len(html) if html else 0,
        "elapsed_s": round(elapsed, 1),
        "blocked": html is None or BLOCK_MARKER in html if html else True,
    }


async def main() -> None:
    from app.services.providers.browser_fetcher import BrowserFetcher
    from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition
    from app.services.providers.serper_client import SerperClient

    print("\n" + "=" * 70)
    print("LAMODA WARMUP SMOKE TEST")
    print("=" * 70)

    async with BrowserFetcher() as fetcher:
        # ── Phase 1: raw warmup metrics ──────────────────────────────────────
        print("\n[PHASE 1] Direct warmup + product fetch (Champion hoodie)")
        champion_url = None

        # First get a URL from Serper
        serper = SerperClient()
        try:
            results = await serper.search(
                "site:lamoda.ru Champion Reverse Weave худи", num_results=5
            )
            for r in results.organic_results:
                link = getattr(r, "link", "") or ""
                if "/p/" in link and "lamoda.ru" in link:
                    champion_url = link
                    break
        except Exception as exc:
            print(f"  Serper error: {exc}")

        if champion_url:
            print(f"  Product URL: {champion_url}")
            t_start = time.monotonic()
            diag = await _direct_warmup_test(fetcher, HOMEPAGE, champion_url)
            print(f"  Elapsed: {diag['elapsed_s']}s")
            print(f"  HTML length: {diag['html_len']} chars")
            print(f"  Blocked: {diag['blocked']}")

            if not diag["blocked"] and diag["html"]:
                from app.services.enrichment.composition_extractor import extract_composition
                comps = extract_composition(diag["html"])
                if comps:
                    print(f"  Composition extracted: {comps[0]!r}")
                else:
                    print("  Composition: NOT found in page (page loaded but no composition block?)")
                    # Show a snippet of the html for debugging
                    snippet = diag["html"][:500].replace("\n", " ")
                    print(f"  HTML snippet: {snippet!r}")
            else:
                print("  Still BLOCKED after warmup+retries.")
        else:
            print("  No Champion URL from Serper — skipping direct test")

        print()

        # ── Phase 2: full mine_lamoda_composition for both products ──────────
        print("[PHASE 2] mine_lamoda_composition (uses warmup internally)")
        for prod in PRODUCTS:
            print(f"\n  Product: {prod['product_name']!r} | Brand: {prod['brand']!r}")
            t0 = time.monotonic()
            try:
                result = await mine_lamoda_composition(
                    prod["product_name"],
                    prod["brand"],
                    fetch_timeout=60.0,
                    fetcher=fetcher,
                )
            except Exception as exc:
                result = None
                print(f"    ERROR: {exc}")

            elapsed = round(time.monotonic() - t0, 1)
            if result:
                print(f"    FOUND in {elapsed}s:")
                print(f"      composition: {result['composition']!r}")
                print(f"      source_url:  {result['source_url']}")
                print(f"      html_length: {result.get('html_length', '?')} chars")
            else:
                print(f"    NONE in {elapsed}s (blocked or no composition found)")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
