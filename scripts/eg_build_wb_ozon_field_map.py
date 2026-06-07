"""One-time consolidation of eg-importer WB→Ozon field maps.

Reads ~6.5k per-category map files from eg-importer
(data/field_maps/{wb_subject}__{ozon_catid}_{ozon_typeid}.json), each a flat
JSON object {WB field name: Ozon attr ID or null}, and consolidates them into a
single file keyed by basename-without-extension → {WB name: Ozon ID}.

Null-valued entries (fields with no verified Ozon mapping) are dropped; only
real integer IDs are kept. Output is written to the dictionaries data dir as
eg_wb_ozon_field_map.json (UTF-8, ensure_ascii=False).

Run once:
    python scripts/eg_build_wb_ozon_field_map.py
"""
import json
import os
import glob

SRC_DIR = r"C:/Users/Venya/PycharmProjects/eg-importer/data/field_maps"
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "app", "services", "enrichment", "strategies", "dictionaries", "data",
)
OUT_PATH = os.path.join(OUT_DIR, "eg_wb_ozon_field_map.json")


def main() -> None:
    files = glob.glob(os.path.join(SRC_DIR, "*.json"))
    consolidated: dict[str, dict[str, int]] = {}
    total_pairs = 0
    for fp in files:
        key = os.path.splitext(os.path.basename(fp))[0]
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                inner = json.load(fh)
        except Exception:
            continue
        if not isinstance(inner, dict):
            continue
        real = {k: v for k, v in inner.items() if v is not None}
        if not real:
            continue
        consolidated[key] = real
        total_pairs += len(real)

    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            consolidated, fh, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        )

    size = os.path.getsize(OUT_PATH)
    print(f"source files:            {len(files)}")
    print(f"category keys (kept):    {len(consolidated)}")
    print(f"total name->ID pairs:    {total_pairs}")
    print(f"file size:               {size} bytes ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
