# DEPRECATED — bug: values copied across categories (fetched once per attr_id, not per
# (cat_id, type_id, attr_id)).  Use build_ozon_values_v2.py instead.

"""Enrich ozon_dictionary.json with allowed_values for dictionary-backed attributes.

Strategy:
- Values for the same attribute_id are SHARED across all (cat, type) pairs.
- So we fetch once per unique attribute_id using a representative (cat_id, type_id).
- 7288 unique attrs → ~36-45 min at 0.3s/call rate limit.
- 404 = not dict-backed → skip.  200 + empty result = no values → skip.
- 429 = sleep 5s and retry once.
- Cap: 5000 values max per attribute (1 page at limit=5000).  If has_next, store values_truncated=true.
- Partial save every 50 attributes to ozon_dictionary.json.values_partial
- Final output written atomically to ozon_dictionary.json.with_values.json, then renamed over original.

Output schema per characteristic:
  "values": [{"id": int, "value": str}, ...]   (info/picture dropped)
  "values_truncated": true   (only if has_next=true after 5000)
"""
import os
import json
import time
import logging
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("OZON_CLIENT_ID")
API_KEY = os.getenv("OZON_API_KEY")
BASE_URL = "https://api-seller.ozon.ru"

DICT_PATH = Path("app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json")
PARTIAL_PATH = DICT_PATH.parent / "ozon_dictionary.json.values_partial"
OUTPUT_PATH = DICT_PATH.parent / "ozon_dictionary.json.with_values.json"

RATE_LIMIT_SLEEP = 0.3  # seconds between requests
PARTIAL_SAVE_EVERY = 50  # save progress every N attribute fetches
VALUES_PAGE_LIMIT = 5000  # Ozon max per page; we do single page then cap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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
    """Fetch first page of values (up to 5000).

    Returns (values, has_next) or None if not dict-backed.
    On 404/400: returns None.
    On 429: sleeps 5s and retries once.
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
        r = client.post(
            f"{BASE_URL}/v1/description-category/attribute/values",
            headers=api_headers(),
            json=payload,
            timeout=30,
        )

        if r.status_code == 200:
            body = r.json()
            vals_raw = body.get("result") or []
            has_next = bool(body.get("has_next", False))
            # Drop info + picture to keep file size manageable
            vals = [{"id": v["id"], "value": v["value"]} for v in vals_raw]
            return vals, has_next

        if r.status_code in (400, 404):
            # Attribute is not dict-backed or doesn't exist
            return None

        if r.status_code == 429:
            if attempt == 0:
                log.warning("  429 rate limit for attr_id=%d, sleeping 5s...", attr_id)
                time.sleep(5)
                continue
            else:
                log.error("  429 after retry for attr_id=%d, skipping", attr_id)
                return None

        # Other error
        log.warning(
            "  HTTP %d for attr_id=%d: %s", r.status_code, attr_id, r.text[:200]
        )
        return None

    return None


def build_attr_index(cats: dict) -> dict[int, tuple[int, int]]:
    """Build {attr_id: (cat_id, type_id)} using first occurrence as representative."""
    index: dict[int, tuple[int, int]] = {}
    for cat_val in cats.values():
        cat_id = cat_val["description_category_id"]
        type_id = cat_val["type_id"]
        for char in cat_val.get("characteristics", []):
            attr_id = char["id"]
            if attr_id not in index:
                index[attr_id] = (cat_id, type_id)
    return index


def load_partial() -> dict[int, list | None]:
    """Load partial results: {attr_id: values_or_None}. None = not dict-backed."""
    if PARTIAL_PATH.exists():
        try:
            data = json.loads(PARTIAL_PATH.read_text(encoding="utf-8"))
            # Keys stored as strings in JSON → convert back to int
            return {int(k): v for k, v in data.items()}
        except Exception as e:
            log.warning("Could not load partial file: %s", e)
    return {}


def save_partial(results: dict[int, list | None]) -> None:
    PARTIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARTIAL_PATH.write_text(
        json.dumps(results, ensure_ascii=False),
        encoding="utf-8",
    )


def apply_values_to_dict(cats: dict, attr_values: dict[int, list | None]) -> dict:
    """Inject values into each characteristic in the categories dict."""
    enriched_count = 0
    truncated_count = 0

    for cat_val in cats.values():
        for char in cat_val.get("characteristics", []):
            attr_id = char["id"]
            if attr_id not in attr_values:
                continue
            entry = attr_values[attr_id]
            if entry is None:
                # Not dict-backed — no values field
                continue
            vals, has_next = entry
            char["values"] = vals
            if has_next:
                char["values_truncated"] = True
                truncated_count += 1
            enriched_count += 1

    log.info("Applied values to %d characteristics (%d truncated)", enriched_count, truncated_count)
    return cats


def main():
    if not CLIENT_ID or not API_KEY:
        raise SystemExit("ERROR: OZON_CLIENT_ID or OZON_API_KEY not set in .env")

    log.info("Loading dictionary from %s ...", DICT_PATH)
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cats = data.get("categories", {})
    total_cats = len(cats)
    total_chars = sum(len(v.get("characteristics", [])) for v in cats.values())
    log.info("Loaded %d categories, %d total characteristics", total_cats, total_chars)

    # Build unique attr index
    attr_index = build_attr_index(cats)
    unique_attrs = list(attr_index.items())
    log.info("Unique attribute IDs: %d", len(unique_attrs))

    # Load partial results for resume
    attr_values: dict[int, tuple | None] = load_partial()
    already_done = len(attr_values)
    if already_done:
        log.info("Resuming: %d attrs already processed", already_done)

    log.info(
        "Starting fetch. Estimated time: ~%.0f min (%.0f attrs × 0.3s)",
        len(unique_attrs) * RATE_LIMIT_SLEEP / 60,
        len(unique_attrs),
    )

    with httpx.Client() as client:
        processed_this_run = 0
        dict_backed = 0
        not_backed = 0

        for i, (attr_id, (cat_id, type_id)) in enumerate(unique_attrs):
            if attr_id in attr_values:
                continue  # Already fetched in a previous run

            result = fetch_values(client, cat_id, type_id, attr_id)

            if result is None:
                attr_values[attr_id] = None  # Marks as "tried, not dict-backed"
                not_backed += 1
            else:
                vals, has_next = result
                attr_values[attr_id] = (vals, has_next)
                dict_backed += 1
                if has_next:
                    log.info(
                        "  attr_id=%d: %d values (truncated at 5000)", attr_id, len(vals)
                    )

            processed_this_run += 1

            # Partial save every 50 fetches
            if processed_this_run % PARTIAL_SAVE_EVERY == 0:
                save_partial(attr_values)
                pct = (i + 1) / len(unique_attrs) * 100
                log.info(
                    "Progress: %d/%d attrs (%.1f%%) — %d dict-backed, %d not",
                    i + 1, len(unique_attrs), pct, dict_backed, not_backed,
                )

            time.sleep(RATE_LIMIT_SLEEP)

    log.info(
        "Fetch complete: %d dict-backed, %d not dict-backed (of %d unique attrs)",
        dict_backed, not_backed, len(unique_attrs),
    )

    # Apply values to dict
    log.info("Applying values to dictionary...")
    enriched_cats = apply_values_to_dict(cats, attr_values)

    # Write output atomically
    enriched_data = {
        "schema_version": 2,
        "source": data.get("source", "ozon_seller_api"),
        "generated_at": data.get("generated_at"),
        "values_enriched_at": time.strftime("%Y-%m-%d"),
        "categories": enriched_cats,
    }

    log.info("Writing enriched dictionary to %s ...", OUTPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Streaming write без indent — индент удваивает память и размер.
    # json.dump пишет напрямую в файл без построения 500MB строки в памяти.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_data, f, ensure_ascii=False)

    # Atomic rename over original
    import shutil
    shutil.move(str(OUTPUT_PATH), str(DICT_PATH))
    log.info("Renamed %s → %s", OUTPUT_PATH.name, DICT_PATH.name)

    # Clean up partial
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()
        log.info("Removed partial file")

    log.info("DONE. Enriched dictionary saved to %s", DICT_PATH)


if __name__ == "__main__":
    main()
