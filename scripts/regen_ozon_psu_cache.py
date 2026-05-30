"""Regenerate scripts/eval_results/ozon_power_supply_cache.json with CORRECT
per-(cat_id, type_id, attr_id) values for PSU category (17028612, 91910).

WHY:
The current cache had values copied from another category for at least
attr_id=8229 (Тип) and attr_id=22232 (ТН ВЭД) — they contained "Корзина для
белья"/"Шкатулка" and 3924/6912/7010 (plastic dishes/glass) instead of PSU
values like "Модульный"/"Полумодульный" and 8504*/8505* codes.

This script refetches values for every characteristic of (17028612, 91910)
that has a `values` field (i.e. is dict-backed), using the same Ozon Seller
API endpoint `/v1/description-category/attribute/values` that build_ozon_values_v2
uses, but only for that one (cat, type) pair.

Confirmed-global attrs (85=Бренд, 4389=Страна-изготовитель, 10096=Цвет товара)
are left untouched — their values were correct in the old cache.

Writes back to: scripts/eval_results/ozon_power_supply_cache.json (in-place,
atomic via tmp+replace).
"""
from __future__ import annotations

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

CACHE_PATH = PROJECT_ROOT / "scripts" / "eval_results" / "ozon_power_supply_cache.json"

CAT_ID = 17028612
TYPE_ID = 91910

# Same as build_ozon_values_v2
CONFIRMED_GLOBAL_ATTR_IDS: frozenset[int] = frozenset({85, 4389, 10096})

RATE_LIMIT_SLEEP = 0.5
VALUES_PAGE_LIMIT = 5000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("regen_psu")


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
            log.warning("  net error attr_id=%d: %s", attr_id, exc)
            return None

        if r.status_code == 200:
            body = r.json()
            vals_raw = body.get("result") or []
            has_next = bool(body.get("has_next", False))
            vals = [{"id": v["id"], "value": v["value"]} for v in vals_raw]
            return vals, has_next

        if r.status_code in (400, 404):
            return None

        if r.status_code == 429:
            if attempt == 0:
                log.warning("  429 for attr_id=%d, sleep 5s...", attr_id)
                time.sleep(5)
                continue
            log.error("  429 after retry for attr_id=%d, skip", attr_id)
            return None

        log.warning("  HTTP %d for attr_id=%d: %s", r.status_code, attr_id, r.text[:200])
        return None
    return None


def main() -> None:
    if not CLIENT_ID or not API_KEY:
        raise SystemExit("ERROR: OZON_CLIENT_ID or OZON_API_KEY not set in .env")
    if not CACHE_PATH.exists():
        raise SystemExit(f"ERROR: cache file not found at {CACHE_PATH}")

    log.info("Loading cache from %s", CACHE_PATH)
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    cats = data.get("categories", {})
    key = f"{CAT_ID}:{TYPE_ID}"
    if key not in cats:
        raise SystemExit(f"ERROR: category {key} not in cache")

    cat_val = cats[key]
    chars = cat_val.get("characteristics", [])
    log.info("Found %d characteristics for cat=%d type=%d", len(chars), CAT_ID, TYPE_ID)

    # Characteristics worth refetching: those with `values` field present
    # (dict-backed). Skip confirmed-global attrs.
    to_refetch: list[dict] = []
    for ch in chars:
        attr_id = ch["id"]
        if attr_id in CONFIRMED_GLOBAL_ATTR_IDS:
            continue
        if "values" in ch or ch.get("values_truncated"):
            to_refetch.append(ch)

    log.info(
        "Will refetch %d characteristics (skipping %d global attrs)",
        len(to_refetch),
        len(CONFIRMED_GLOBAL_ATTR_IDS),
    )

    refreshed = 0
    cleared = 0
    unchanged = 0
    diffs: list[str] = []

    with httpx.Client() as client:
        for ch in to_refetch:
            attr_id = ch["id"]
            name = ch.get("name", "")
            old_vals = ch.get("values") or []
            old_set = {v["id"] for v in old_vals}

            log.info("  fetch attr_id=%d (%s) ...", attr_id, name)
            result = fetch_values(client, CAT_ID, TYPE_ID, attr_id)
            time.sleep(RATE_LIMIT_SLEEP)

            if result is None:
                # Ozon API says not dict-backed for this (cat, type) — drop values
                ch.pop("values", None)
                ch.pop("values_truncated", None)
                cleared += 1
                diffs.append(f"  CLEARED attr_id={attr_id} ({name}): API returned not-dict-backed")
                continue

            new_vals, has_next = result
            new_set = {v["id"] for v in new_vals}

            ch["values"] = new_vals
            if has_next:
                ch["values_truncated"] = True
            else:
                ch.pop("values_truncated", None)

            refreshed += 1
            if old_set != new_set:
                added = len(new_set - old_set)
                removed = len(old_set - new_set)
                # Build a short preview of new values for required attrs
                preview = ", ".join(v["value"][:40] for v in new_vals[:6])
                diffs.append(
                    f"  CHANGED attr_id={attr_id} ({name}): "
                    f"{len(old_vals)} -> {len(new_vals)} "
                    f"(+{added}/-{removed}) | preview: {preview}"
                )
            else:
                unchanged += 1

    log.info(
        "Done: refreshed=%d (unchanged=%d) cleared=%d", refreshed, unchanged, cleared
    )

    if diffs:
        log.info("=== DIFF SUMMARY ===")
        for d in diffs:
            log.info(d)

    # Atomic write
    tmp = CACHE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(CACHE_PATH)
    log.info("Cache rewritten atomically to %s (%d bytes)", CACHE_PATH, CACHE_PATH.stat().st_size)

    # Verify critical required attrs
    cat_val_after = json.loads(CACHE_PATH.read_text(encoding="utf-8"))["categories"][key]
    for attr_id in (8229, 22232):
        ch = next((c for c in cat_val_after["characteristics"] if c["id"] == attr_id), None)
        if ch is None:
            log.warning("VERIFY: attr_id=%d not found", attr_id)
            continue
        vals = ch.get("values") or []
        sample = [v["value"] for v in vals[:8]]
        log.info("VERIFY attr_id=%d (%s): %d values | sample=%s",
                 attr_id, ch.get("name"), len(vals), sample)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8")
    main()
