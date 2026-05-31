"""Patch ozon_dictionary.json.gz with CORRECT per-(cat_id, type_id, attr_id) values
for one or more categories, fetched fresh from Ozon Seller API.

WHY:
build_ozon_values.py (v1, DEPRECATED) had a bug: it fetched values once per unique
attr_id using the FIRST category that contained that attr_id.  Because Ozon API
returns category-specific value lists, attr_id=8229 "Тип" ended up populated with
values from whichever category was first (e.g. "Корзина для белья", "Шкатулка")
across ALL categories — including clothing (Футболка, Джинсы, Куртка, …).
LLMs then cannot produce a valid "Тип" value → field is empty → −14% required score.

This script is the GENERIC fix: pass any (description_category_id, type_id) pair(s)
and it refetches every non-global dict-backed attribute for that pair, then patches
the live ozon_dictionary.json.gz in-place (atomic write).

Usage:
    # Single (cat_id, type_id) pair:
    python scripts/regen_ozon_category.py --cat 200000933 --type 91910

    # Multiple pairs (comma-separated):
    python scripts/regen_ozon_category.py --cat 200000933,17028612 --type 91910,91910

    # Dry run (fetch + log diffs, do NOT write to disk):
    python scripts/regen_ozon_category.py --cat 200000933 --type 91910 --dry-run

    # Only refetch specific attribute IDs (useful for targeted fixes):
    python scripts/regen_ozon_category.py --cat 200000933 --type 91910 --attrs 8229,22232

Rate limits:
    0.5 s/call (conservative).  ~20-50 dict-backed attrs per category → 10-25 s/category.
    No quota concerns for a handful of categories.

Output:
    Patches app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json.gz
    in-place via atomic tmp+replace.  Also invalidates the lru_cache in ozon_loader
    if run in-process (not relevant when run as a standalone script).

Note:
    build_ozon_values_v2.py does the same thing for ALL ~317k (cat,type,attr) tuples
    (ETA ~26 h).  Use THIS script for targeted, fast patches of specific categories.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CLIENT_ID = os.getenv("OZON_CLIENT_ID")
API_KEY = os.getenv("OZON_API_KEY")
BASE_URL = "https://api-seller.ozon.ru"

DATA_DIR = PROJECT_ROOT / "app" / "services" / "enrichment" / "strategies" / "dictionaries" / "data"
DICT_GZ = DATA_DIR / "ozon_dictionary.json.gz"
DICT_JSON = DATA_DIR / "ozon_dictionary.json"

# Ozon-documented truly global attrs — same values across all categories.
# Values for these were correct in the v1 run; skip to avoid redundant fetches.
CONFIRMED_GLOBAL_ATTR_IDS: frozenset[int] = frozenset({
    85,     # Бренд
    4389,   # Страна-изготовитель
    10096,  # Цвет товара
})

RATE_LIMIT_SLEEP = 0.5   # seconds between API calls (conservative)
VALUES_PAGE_LIMIT = 5000  # Ozon single-page cap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regen_category")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_headers() -> dict[str, str]:
    if not CLIENT_ID or not API_KEY:
        raise RuntimeError("OZON_CLIENT_ID or OZON_API_KEY not set in .env")
    return {
        "Client-Id": CLIENT_ID,
        "Api-Key": API_KEY,
        "Content-Type": "application/json",
    }


def fetch_values(
    client: httpx.Client,
    cat_id: int,
    type_id: int,
    attr_id: int,
) -> tuple[list[dict], bool] | None:
    """Fetch first page of allowed values for (cat_id, type_id, attr_id).

    Returns:
        (values_list, has_next) — may be ([], False) for dict-backed attr with 0 values
        None                    — not dict-backed for this (cat, type) pair (404/400)
    """
    payload = {
        "description_category_id": cat_id,
        "type_id": type_id,
        "attribute_id": attr_id,
        "language": "DEFAULT",
        "last_value_id": 0,
        "limit": VALUES_PAGE_LIMIT,
    }

    for attempt in range(2):
        try:
            r = client.post(
                f"{BASE_URL}/v1/description-category/attribute/values",
                headers=api_headers(),
                json=payload,
                timeout=30,
            )
        except Exception as exc:
            log.warning("  net error (%d,%d,%d): %s", cat_id, type_id, attr_id, exc)
            return None

        if r.status_code == 200:
            body = r.json()
            vals_raw = body.get("result") or []
            has_next = bool(body.get("has_next", False))
            vals = [{"id": v["id"], "value": v["value"]} for v in vals_raw]
            return vals, has_next

        if r.status_code in (400, 404):
            return None  # not dict-backed for this (cat, type)

        if r.status_code == 429:
            if attempt == 0:
                log.warning("  429 for (%d,%d,%d), sleep 5s...", cat_id, type_id, attr_id)
                time.sleep(5)
                continue
            log.error("  429 after retry for (%d,%d,%d), skip", cat_id, type_id, attr_id)
            return None

        log.warning("  HTTP %d for (%d,%d,%d): %s", r.status_code, cat_id, type_id, attr_id, r.text[:200])
        return None

    return None


# ---------------------------------------------------------------------------
# Dictionary I/O
# ---------------------------------------------------------------------------

def load_dict() -> tuple[dict, bool]:
    """Load ozon_dictionary from .json.gz (preferred) or .json.

    Returns (full_data_dict, is_gz).
    full_data_dict always has a top-level "categories" key.
    """
    if DICT_GZ.exists():
        with gzip.open(DICT_GZ, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        if "categories" not in data:
            data = {"categories": data}
        return data, True
    if DICT_JSON.exists():
        data = json.loads(DICT_JSON.read_text(encoding="utf-8"))
        if "categories" not in data:
            data = {"categories": data}
        return data, False
    raise FileNotFoundError(f"Neither {DICT_GZ} nor {DICT_JSON} found")


def save_dict(data: dict, is_gz: bool) -> None:
    """Atomic write back to the same format (gz or plain json)."""
    if is_gz:
        tmp = DICT_GZ.with_suffix(".json.tmp.gz")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as gz:
            json.dump(data, gz, ensure_ascii=False)
        tmp.replace(DICT_GZ)
        log.info("Atomically replaced %s (%d bytes)", DICT_GZ, DICT_GZ.stat().st_size)
    else:
        tmp = DICT_JSON.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(DICT_JSON)
        log.info("Atomically replaced %s (%d bytes)", DICT_JSON, DICT_JSON.stat().st_size)


# ---------------------------------------------------------------------------
# Core patch logic
# ---------------------------------------------------------------------------

def patch_category(
    client: httpx.Client,
    cats: dict,
    cat_id: int,
    type_id: int,
    only_attr_ids: frozenset[int] | None,
    dry_run: bool,
) -> list[str]:
    """Refetch values for all dict-backed attributes of (cat_id, type_id).

    Returns list of human-readable diff strings for logging.
    Mutates cats[key]["characteristics"] in-place unless dry_run=True.
    """
    key = f"{cat_id}:{type_id}"
    entry = cats.get(key)
    if entry is None:
        log.warning("Key %s not found in dictionary — skipping", key)
        return []

    cat_name = entry.get("name", key)
    chars = entry.get("characteristics", [])
    log.info(
        "Category %s (%s): %d characteristics total",
        key, cat_name, len(chars),
    )

    # Select characteristics to refetch
    to_refetch: list[dict] = []
    for ch in chars:
        attr_id = ch["id"]
        if attr_id in CONFIRMED_GLOBAL_ATTR_IDS:
            continue
        if only_attr_ids is not None and attr_id not in only_attr_ids:
            continue
        # Refetch if: (a) has existing values (possibly wrong), OR (b) type suggests dict
        # We refetch ALL non-global attrs to catch those that were missed by v1.
        to_refetch.append(ch)

    log.info(
        "  Will refetch %d/%d attributes (skipping %d global)",
        len(to_refetch),
        len(chars),
        sum(1 for ch in chars if ch["id"] in CONFIRMED_GLOBAL_ATTR_IDS),
    )

    diffs: list[str] = []
    refreshed = cleared = unchanged = 0

    for ch in to_refetch:
        attr_id = ch["id"]
        name = ch.get("name", f"attr_{attr_id}")
        old_vals = ch.get("values") or []
        old_set = {v["id"] for v in old_vals}
        is_required = ch.get("is_required", False)

        log.info("    fetch attr_id=%d (%s)%s ...", attr_id, name, " [REQUIRED]" if is_required else "")
        result = fetch_values(client, cat_id, type_id, attr_id)
        time.sleep(RATE_LIMIT_SLEEP)

        if result is None:
            # API says not dict-backed → drop values
            if not dry_run:
                ch.pop("values", None)
                ch.pop("values_truncated", None)
            cleared += 1
            diffs.append(f"  CLEARED  attr_id={attr_id} ({name}): not dict-backed for this (cat,type)")
            continue

        new_vals, has_next = result
        new_set = {v["id"] for v in new_vals}

        if not dry_run:
            ch["values"] = new_vals
            if has_next:
                ch["values_truncated"] = True
            else:
                ch.pop("values_truncated", None)

        if old_set != new_set:
            added = len(new_set - old_set)
            removed = len(old_set - new_set)
            preview = ", ".join(v["value"][:40] for v in new_vals[:8])
            diffs.append(
                f"  CHANGED  attr_id={attr_id} ({name}): "
                f"{len(old_vals)} → {len(new_vals)} values (+{added}/-{removed}) "
                f"| preview: {preview}"
            )
            refreshed += 1
        else:
            unchanged += 1

    log.info(
        "  Done: refreshed=%d unchanged=%d cleared=%d%s",
        refreshed, unchanged, cleared,
        " [DRY RUN — no writes]" if dry_run else "",
    )
    return diffs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Patch ozon_dictionary.json.gz with correct values for given (cat,type) pairs."
    )
    p.add_argument(
        "--cat", required=True,
        help="Comma-separated description_category_id(s), e.g. 200000933 or 200000933,17028612",
    )
    p.add_argument(
        "--type", required=True, dest="type_id",
        help="Comma-separated type_id(s) — must match length of --cat",
    )
    p.add_argument(
        "--attrs", default=None,
        help="Optional comma-separated attr_id(s) to refetch (default: all non-global)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch and log diffs but do NOT write changes to disk",
    )
    return p.parse_args()


def main() -> None:
    if not CLIENT_ID or not API_KEY:
        raise SystemExit("ERROR: OZON_CLIENT_ID or OZON_API_KEY not set in .env")

    args = parse_args()

    cat_ids = [int(x.strip()) for x in args.cat.split(",")]
    type_ids = [int(x.strip()) for x in args.type_id.split(",")]
    if len(cat_ids) != len(type_ids):
        raise SystemExit(
            f"ERROR: --cat has {len(cat_ids)} entries but --type has {len(type_ids)}"
        )
    pairs = list(zip(cat_ids, type_ids))

    only_attr_ids: frozenset[int] | None = None
    if args.attrs:
        only_attr_ids = frozenset(int(x.strip()) for x in args.attrs.split(","))
        log.info("Only refetching specific attr_ids: %s", only_attr_ids)

    if args.dry_run:
        log.info("DRY RUN — no changes will be written to disk")

    log.info("Loading dictionary...")
    data, is_gz = load_dict()
    cats = data["categories"]
    log.info("Loaded %d category entries", len(cats))

    all_diffs: list[str] = []

    with httpx.Client() as client:
        for cat_id, type_id in pairs:
            log.info("=== Processing cat=%d type=%d ===", cat_id, type_id)
            diffs = patch_category(client, cats, cat_id, type_id, only_attr_ids, args.dry_run)
            all_diffs.extend([f"[{cat_id}:{type_id}] {d}" for d in diffs])

    if all_diffs:
        log.info("=== DIFF SUMMARY ===")
        for d in all_diffs:
            log.info(d)
    else:
        log.info("No diffs detected (all values were already correct or no dict-backed attrs found)")

    if not args.dry_run:
        log.info("Writing patched dictionary back to disk...")
        save_dict(data, is_gz)
        log.info("Done. Run your pipeline to pick up the new values.")
        log.info(
            "IMPORTANT: If ozon_loader is running in the same process, call "
            "ozon_loader.load_ozon_dictionary.cache_clear() to invalidate the lru_cache."
        )
    else:
        log.info("DRY RUN complete — dictionary NOT modified.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8")
    main()
