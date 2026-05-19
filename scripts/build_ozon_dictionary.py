"""Build Ozon Dictionary — standalone utility.

Usage:
    python scripts/build_ozon_dictionary.py [--samples N] [--output PATH] [--resume]

Without arguments:
    - --samples=20   (sample product pages per category via Playwright)
    - --output=app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json
    - --resume       disabled by default

Requires Playwright (headless Chrome) to bypass Cloudflare protection.
Install once:
    pip install playwright
    playwright install chromium  # ~130 MB

Runtime: 8-15 hours for all categories (Playwright is slower than plain httpx).
Script saves intermediate progress to {output}.partial every 50 processed URLs.
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Allow running as `python scripts/build_ozon_dictionary.py` from project root
sys.path.insert(0, str(Path(__file__).parent))

from build_ozon_dictionary_lib.seed_loader import load_seed_categories
from build_ozon_dictionary_lib.product_parser import parse_product_characteristics
from build_ozon_dictionary_lib.aggregator import aggregate_by_category


async def main(args: argparse.Namespace) -> None:
    print("[1/3] Loading category seed...")
    categories = load_seed_categories()
    print(f"      {len(categories)} categories loaded")

    # Resume support — read partial progress if flag is set
    partial_path = Path(str(args.output) + ".partial")
    chars_by_category: dict[int, set[tuple[str, str]]] = defaultdict(set)

    if args.resume and partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            chars_by_category = defaultdict(
                set,
                {int(k): set(tuple(c) for c in v) for k, v in partial.items()},
            )
            print(f"      [RESUME] {len(chars_by_category)} categories already done")
        except Exception as exc:
            print(f"      [WARN] Could not read partial file: {exc}, starting fresh")

    print(f"[2/3] Parsing product pages via Playwright ({args.samples} samples/category)...")

    # Count total work units (skipping already-done categories)
    pending_categories = [
        cat for cat in categories
        if cat.get("id") is not None and cat["id"] not in chars_by_category
    ]
    total = len(pending_categories) * args.samples
    done = 0

    for cat in pending_categories:
        cat_id: int = cat["id"]
        sample_urls: list[str] = (cat.get("sample_urls") or [])[:args.samples]

        for url in sample_urls:
            try:
                chars = await parse_product_characteristics(url)
                if chars:
                    for c in chars:
                        key = str(c.get("key", "")).strip()
                        display_name = str(c.get("display_name", key)).strip()
                        if key:
                            chars_by_category[cat_id].add((key, display_name))
            except Exception as exc:
                print(f"      [WARN] {url}: {exc}")

            done += 1
            await asyncio.sleep(2)  # Playwright-friendly rate limit

            if done % 50 == 0:
                _save_partial(partial_path, chars_by_category)
                print(f"      Progress: {done}/{total} | categories with data: {len(chars_by_category)}")

    print("[3/3] Saving final dictionary...")
    final = aggregate_by_category(chars_by_category, categories)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"      Saved {len(final)} categories to {args.output}")

    # Clean up partial file on success
    if partial_path.exists():
        partial_path.unlink()


def _save_partial(
    path: Path,
    chars_by_category: dict[int, set[tuple[str, str]]],
) -> None:
    """Persist current progress to a .partial file for resume support."""
    try:
        path.write_text(
            json.dumps(
                {k: list(v) for k, v in chars_by_category.items()},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"      [WARN] Could not save partial: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build Ozon category characteristics dictionary "
            "via Playwright (bypasses Cloudflare)."
        )
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of sample product pages per category (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=(
            "app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json"
        ),
        help="Output JSON path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from .partial file if it exists",
    )
    asyncio.run(main(parser.parse_args()))
