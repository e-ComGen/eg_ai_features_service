"""Build WB Dictionary — standalone utility.

Usage:
    python scripts/build_wb_dictionary.py [--samples N] [--output PATH] [--resume]

Without arguments:
    - --samples=30 (sample cards per category)
    - --output=app/services/enrichment/strategies/dictionaries/data/wb_dictionary.json
    - --resume — if partial file exists, continue from where it left off

Runtime: ~6-12 hours for all ~1200 categories (rate limit ~3 req/sec).
Script saves intermediate progress to `{output}.partial` every 100 requests.
"""
import asyncio
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

# Allow running as `python scripts/build_wb_dictionary.py` from project root
sys.path.insert(0, str(Path(__file__).parent))

from build_wb_dictionary_lib.menu_fetcher import fetch_main_menu, extract_all_subjects
from build_wb_dictionary_lib.hf_seed import build_seed_from_hf
from build_wb_dictionary_lib.card_parser import fetch_characteristics
from build_wb_dictionary_lib.aggregator import aggregate_characteristics_by_subject


async def main(args):
    print(f"[1/4] Fetching main menu...")
    menu = await fetch_main_menu()
    subjects = extract_all_subjects(menu)
    print(f"      Found {len(subjects)} subjects")

    # Resume support
    partial_path = Path(str(args.output) + ".partial")
    completed_subjects = set()
    characteristics_by_subj = defaultdict(set)
    if args.resume and partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        characteristics_by_subj = defaultdict(
            set, {int(k): set(map(tuple, v)) for k, v in partial.items()}
        )
        completed_subjects = set(characteristics_by_subj.keys())
        print(f"      [RESUME] Already processed {len(completed_subjects)} subjects")

    print(f"[2/4] Loading sample nm_ids from HuggingFace dataset (offline)...")
    hf_seed = build_seed_from_hf(samples_per_subj=args.samples)
    print(f"      Got {len(hf_seed)} subjects from HF dataset")
    # Filter to only subjects present in the menu, skip already-completed ones
    samples_per_subj = {
        subj["id"]: hf_seed[subj["id"]]
        for subj in subjects
        if subj["id"] in hf_seed and subj["id"] not in completed_subjects
    }

    print(f"[3/4] Parsing card characteristics...")
    total = sum(len(v) for v in samples_per_subj.values())
    done = 0
    for subj_id, nm_ids in samples_per_subj.items():
        for nm_id in nm_ids:
            try:
                charcs = await fetch_characteristics(nm_id)
                if charcs:
                    for c in charcs:
                        # c = {"id": int, "name": str, "value": str|list}
                        characteristics_by_subj[subj_id].add(
                            (c.get("id", 0), c["name"])
                        )
            except Exception:
                pass  # skip failed cards
            done += 1
            if done % 100 == 0:
                # Save partial progress
                partial_path.write_text(
                    json.dumps(
                        {k: list(v) for k, v in characteristics_by_subj.items()},
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                print(f"      Progress: {done}/{total} cards | saved partial")
            await asyncio.sleep(0.3)  # rate limit

    print(f"[4/4] Aggregating + saving final dictionary...")
    final = aggregate_characteristics_by_subject(characteristics_by_subj, subjects)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"      Saved {len(final)} subjects to {args.output}")

    # Clean partial
    if partial_path.exists():
        partial_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build WB category characteristics dictionary from public WB endpoints."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help="Number of sample cards per category (default: 30)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="app/services/enrichment/strategies/dictionaries/data/wb_dictionary.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from partial file if it exists",
    )
    asyncio.run(main(parser.parse_args()))
