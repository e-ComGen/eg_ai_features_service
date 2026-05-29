"""Re-fetch Ozon dictionary values CORRECTLY per (cat_id, type_id, attribute_id).

WHY v2:
The original build_ozon_values.py had a bug: it fetched values once per unique
attribute_id using an arbitrary representative (cat, type).  But Ozon API returns
category-specific value lists — attr_id=8229 "Тип" returns completely different
values for "Блок питания компьютера" vs household categories.  v2 fixes this by
fetching per (cat_id, type_id, attr_id) tuple.

Strategy:
- Iterate over every (cat_id, type_id, attr_id) in ozon_dictionary.json.gz.
- Skip CONFIRMED_GLOBAL attr IDs that Ozon documents as truly global (Бренд=85,
  Страна-изготовитель=4389, Цвет товара=10096).  These share values across all
  categories and were correctly fetched in the old run.
- 317 697 tuples × 0.3 s ≈ 26.5 hours.
- Partial save keyed by "cat_id:type_id:attr_id" every PARTIAL_SAVE_EVERY calls.
- Resume: on restart, already-processed keys are skipped.
- 404/400 → store None (not dict-backed for this category).
- 200 + empty result → store [].
- 429 → sleep 5 s, retry once; on second 429 → skip.
- Cap 5000 values/attr (single page), mark values_truncated=true if has_next.

Output: app/services/enrichment/strategies/dictionaries/data/ozon_dictionary_v3.json.gz
        (do NOT overwrite ozon_dictionary.json.gz — atomic swap is manual after review)

Partial file: same dir, ozon_values_v3.partial.json  (flat dict, key="cat:type:attr")
"""

import gzip
import json
import logging
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("OZON_CLIENT_ID")
API_KEY = os.getenv("OZON_API_KEY")
BASE_URL = "https://api-seller.ozon.ru"

DATA_DIR = Path("app/services/enrichment/strategies/dictionaries/data")
INPUT_PATH = DATA_DIR / "ozon_dictionary.json.gz"
OUTPUT_PATH = DATA_DIR / "ozon_dictionary_v3.json.gz"
PARTIAL_PATH = DATA_DIR / "ozon_values_v3.partial.json"

RATE_LIMIT_SLEEP = 0.3      # seconds between API calls
PARTIAL_SAVE_EVERY = 200    # flush partial file every N successful fetches
VALUES_PAGE_LIMIT = 5000    # Ozon single-page cap

# Attr IDs confirmed by Ozon docs to be truly global (same values for all categories).
# Values for these were already correctly captured by the old per-attr_id run.
CONFIRMED_GLOBAL_ATTR_IDS: frozenset[int] = frozenset({
    85,     # Бренд
    4389,   # Страна-изготовитель
    10096,  # Цвет товара
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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
        (values_list, has_next)  — may be ([], False) for dict-backed attr with 0 values
        None                     — not dict-backed for this (cat, type) pair
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
            log.warning("  Network error for (%d,%d,%d): %s", cat_id, type_id, attr_id, exc)
            return None

        if r.status_code == 200:
            body = r.json()
            vals_raw = body.get("result") or []
            has_next = bool(body.get("has_next", False))
            # Drop 'info' and 'picture' fields to keep output size manageable
            vals = [{"id": v["id"], "value": v["value"]} for v in vals_raw]
            return vals, has_next

        if r.status_code in (400, 404):
            # Attr is not dict-backed for this specific (cat, type) pair
            return None

        if r.status_code == 429:
            if attempt == 0:
                log.warning(
                    "  429 rate-limit for (%d,%d,%d), sleeping 5 s...",
                    cat_id, type_id, attr_id,
                )
                time.sleep(5)
                continue
            else:
                log.error(
                    "  429 after retry for (%d,%d,%d), skipping",
                    cat_id, type_id, attr_id,
                )
                return None

        log.warning(
            "  HTTP %d for (%d,%d,%d): %s",
            r.status_code, cat_id, type_id, attr_id, r.text[:200],
        )
        return None

    return None


# ---------------------------------------------------------------------------
# Partial file I/O
# ---------------------------------------------------------------------------

def load_partial() -> dict[str, list | None]:
    """Load partial results keyed by 'cat_id:type_id:attr_id'."""
    if PARTIAL_PATH.exists():
        try:
            return json.loads(PARTIAL_PATH.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not load partial file (%s) — starting fresh", exc)
    return {}


def save_partial(results: dict[str, list | None]) -> None:
    PARTIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PARTIAL_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    tmp.replace(PARTIAL_PATH)


# ---------------------------------------------------------------------------
# Dictionary build helpers
# ---------------------------------------------------------------------------

def collect_tuples(cats: dict) -> list[tuple[int, int, int]]:
    """Return list of (cat_id, type_id, attr_id) skipping CONFIRMED_GLOBAL attrs."""
    tuples = []
    for v in cats.values():
        cat_id = v["description_category_id"]
        type_id = v["type_id"]
        for ch in v.get("characteristics", []):
            attr_id = ch["id"]
            if attr_id not in CONFIRMED_GLOBAL_ATTR_IDS:
                tuples.append((cat_id, type_id, attr_id))
    return tuples


def build_output(data: dict, results: dict[str, list | None]) -> dict:
    """Return enriched copy of data with per-(cat,type) values injected."""
    cats = data.get("categories", {})

    enriched_count = 0
    truncated_count = 0
    none_count = 0

    for key, cat_val in cats.items():
        cat_id = cat_val["description_category_id"]
        type_id = cat_val["type_id"]
        for ch in cat_val.get("characteristics", []):
            attr_id = ch["id"]
            lookup = f"{cat_id}:{type_id}:{attr_id}"
            entry = results.get(lookup)
            if entry is None:
                # Not in results: either global (keep existing values), or not fetched
                # If it's a global attr, values are already in ch from the original dict
                none_count += 1
                continue
            if entry == "__none__":
                # Explicitly not dict-backed for this (cat, type)
                ch.pop("values", None)
                ch.pop("values_truncated", None)
                continue
            # entry is [values_list, has_next]
            vals, has_next = entry
            ch["values"] = vals
            if has_next:
                ch["values_truncated"] = True
                truncated_count += 1
            else:
                ch.pop("values_truncated", None)
            enriched_count += 1

    log.info(
        "Injected values: %d enriched, %d truncated, %d skipped (global/not-fetched)",
        enriched_count, truncated_count, none_count,
    )
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CLIENT_ID or not API_KEY:
        raise SystemExit("ERROR: OZON_CLIENT_ID or OZON_API_KEY not set in .env")

    log.info("Loading dictionary from %s ...", INPUT_PATH)
    with gzip.open(INPUT_PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)

    cats = data.get("categories", {})
    log.info(
        "Loaded %d categories, %d total characteristics",
        len(cats),
        sum(len(v.get("characteristics", [])) for v in cats.values()),
    )

    all_tuples = collect_tuples(cats)
    total = len(all_tuples)
    est_hours = total * RATE_LIMIT_SLEEP / 3600
    log.info(
        "Tuples to fetch: %d (skipping %d confirmed-global attr IDs)",
        total,
        len(CONFIRMED_GLOBAL_ATTR_IDS),
    )
    log.info(
        "ETA at %.1fs/call: %.1f hours  (~%.0f min)",
        RATE_LIMIT_SLEEP, est_hours, est_hours * 60,
    )

    # Resume support
    results: dict[str, list | None] = load_partial()
    already_done = len(results)
    if already_done:
        log.info("Resuming: %d tuples already processed, %d remaining", already_done, total - already_done)

    dict_backed = 0
    not_backed = 0
    processed_this_run = 0
    start_time = time.time()

    with httpx.Client() as client:
        for i, (cat_id, type_id, attr_id) in enumerate(all_tuples):
            key = f"{cat_id}:{type_id}:{attr_id}"
            if key in results:
                continue  # Already done in a previous run

            result = fetch_values(client, cat_id, type_id, attr_id)

            if result is None:
                results[key] = "__none__"
                not_backed += 1
            else:
                vals, has_next = result
                results[key] = [vals, has_next]
                dict_backed += 1
                if has_next:
                    log.debug(
                        "  (%d,%d,%d): %d values (truncated)", cat_id, type_id, attr_id, len(vals)
                    )

            processed_this_run += 1

            if processed_this_run % PARTIAL_SAVE_EVERY == 0:
                save_partial(results)
                elapsed = time.time() - start_time
                rate_actual = processed_this_run / elapsed if elapsed > 0 else RATE_LIMIT_SLEEP
                remaining = total - (already_done + processed_this_run)
                eta_h = remaining / rate_actual / 3600 if rate_actual > 0 else 0
                pct = (already_done + processed_this_run) / total * 100
                log.info(
                    "Progress %d/%d (%.1f%%) | dict-backed=%d not=%d | ETA %.1f h",
                    already_done + processed_this_run, total, pct,
                    dict_backed, not_backed, eta_h,
                )

            time.sleep(RATE_LIMIT_SLEEP)

    # Final partial save
    save_partial(results)
    log.info(
        "Fetch complete: %d dict-backed, %d not-dict-backed (of %d tuples this run)",
        dict_backed, not_backed, processed_this_run,
    )

    # Build enriched output
    log.info("Building enriched dictionary...")
    enriched_data = build_output(data, results)

    # Update metadata
    enriched_data["schema_version"] = 3
    enriched_data["values_v3_note"] = (
        "Values fetched per (cat_id, type_id, attr_id) — category-specific. "
        "Global attrs (85, 4389, 10096) use values from original build."
    )
    enriched_data["values_v3_enriched_at"] = time.strftime("%Y-%m-%d")

    # Streaming gzip write — avoids building 400-500 MB string in RAM
    log.info("Writing gzip output to %s ...", OUTPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = OUTPUT_PATH.with_suffix(".tmp.gz")
    with gzip.open(tmp_output, "wt", encoding="utf-8", compresslevel=6) as gz:
        json.dump(enriched_data, gz, ensure_ascii=False)

    tmp_output.replace(OUTPUT_PATH)
    log.info("Output written atomically to %s", OUTPUT_PATH)

    # Clean up partial file
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
        log.info("Removed partial file %s", PARTIAL_PATH)

    log.info(
        "DONE. %d tuples processed. Output: %s",
        total,
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
