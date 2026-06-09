"""Live end-to-end verification: mine_lamoda_composition via real Chrome headless.

Usage:
    python scripts/verify_lamoda_browser_fetch.py

What this proves:
  1. Real chrome.exe launched with --headless=new → NO visible window.
  2. Playwright connect_over_cdp succeeds.
  3. Lamoda product page fetched (DataDome bypassed or not — we report honestly).
  4. composition_extractor finds composition string from rendered DOM.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys

# Make sure the project root is on PYTHONPATH when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("smoke")

from app.services.enrichment.sources.lamoda_composition import mine_lamoda_composition
from app.services.providers.browser_fetcher import BrowserFetcher


PRODUCTS = [
    ("Толстовка худи Champion Reverse Weave", "Champion"),
    ("Худи Nike Club Fleece", "Nike"),
    ("Спортивная куртка Adidas Tiro", "Adidas"),
]


async def run_smoke() -> None:
    # Use a single shared BrowserFetcher so DataDome session is reused
    async with BrowserFetcher() as fetcher:
        for product_name, brand in PRODUCTS:
            print(f"\n{'='*60}")
            print(f"Product: {product_name}")
            print(f"Brand:   {brand}")
            print(f"{'='*60}")

            result = await mine_lamoda_composition(
                product_name,
                brand,
                fetcher=fetcher,
                fetch_timeout=55.0,
            )

            if result is None:
                print("RESULT: None — no composition found (Lamoda blocked or no matching page)")
            else:
                print(f"URL:         {result['source_url']}")
                print(f"HTML length: {result.get('html_length', 'n/a')}")
                print(f"Composition: {result['composition']}")
                print(f"Evidence:    {result['evidence']}")
                if len(result.get("all_compositions", [])) > 1:
                    print(f"All found:   {result['all_compositions']}")

    print("\n\nSMOKE COMPLETE")
    print("Headless=new: no window should have appeared.")
    print("If compositions were found → DataDome beaten by real chrome.exe CDP.")


if __name__ == "__main__":
    asyncio.run(run_smoke())
