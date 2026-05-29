"""excel_diff_corrections.py — diff customer's corrected Excel vs original AI output.

Сравнивает AI-output Excel (тот что выдал ``excel_importer.py``) с правленым файлом
который вернул клиент. Для каждой клетки которая изменилась — записывает новое
значение в SQLite таблицу ``corrections`` (поля ``corrected_value`` + ``corrected_at``)
для последующего сбора датасета human-corrections.

Match по (product_name_hash, attribute_name). product_name берётся из колонки
``product_name``; attribute_name — из заголовка колонки (точное совпадение,
плюс убираем суффикс ' *' для обязательных).

Usage:
    python scripts/excel_diff_corrections.py ORIGINAL.xlsx CORRECTED.xlsx
    python scripts/excel_diff_corrections.py ORIGINAL.xlsx CORRECTED.xlsx \\
        --run-id <uuid>   # ограничить апдейт одним run-id, иначе обновляются все
        --db data/corrections.db
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("excel_diff")


REQUIRED_REQUIRED_SUFFIX = " *"  # excel_importer marks required attrs in header


def hash_product_name(name: str) -> str:
    return hashlib.sha1(str(name).strip().lower().encode("utf-8")).hexdigest()[:16]


def _norm_header(h: str) -> str:
    s = str(h).strip()
    if s.endswith(REQUIRED_REQUIRED_SUFFIX):
        s = s[: -len(REQUIRED_REQUIRED_SUFFIX)].rstrip()
    return s


def _norm_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def diff_excels(original: Path, corrected: Path) -> list[dict]:
    """Return list of {product_name_hash, attribute_name, corrected_value} for changed cells."""
    df_o = pd.read_excel(original, sheet_name=0)
    df_c = pd.read_excel(corrected, sheet_name=0)

    if "product_name" not in df_o.columns or "product_name" not in df_c.columns:
        raise ValueError("Both files must contain 'product_name' column on sheet 1")

    # Drop the score column for the diff comparison
    drop_cols = [c for c in ("Score, %",) if c in df_o.columns]
    df_o = df_o.drop(columns=drop_cols, errors="ignore")
    df_c = df_c.drop(columns=drop_cols, errors="ignore")

    # Map normalized header -> original header
    headers_o = {_norm_header(c): c for c in df_o.columns}
    headers_c = {_norm_header(c): c for c in df_c.columns}

    # Index original by product_name_hash
    o_by_hash = {}
    for _, row in df_o.iterrows():
        h = hash_product_name(row["product_name"])
        o_by_hash[h] = row

    # Skip columns considered "input" (not attributes). We exclude any column that
    # also exists in the customer's original input set (product_name, brand, etc).
    input_cols = {"product_name", "brand", "category_path", "image_urls",
                  "ozon_category_id", "ozon_type_id"}

    diffs = []
    for _, c_row in df_c.iterrows():
        pname = c_row["product_name"]
        h = hash_product_name(pname)
        o_row = o_by_hash.get(h)
        if o_row is None:
            logger.warning("No match in original for product: %s", str(pname)[:60])
            continue
        for norm_name, orig_col in headers_o.items():
            if norm_name in input_cols:
                continue
            c_col = headers_c.get(norm_name)
            if c_col is None:
                continue
            o_val = _norm_value(o_row[orig_col])
            c_val = _norm_value(c_row[c_col])
            if o_val != c_val:
                diffs.append({
                    "product_name": pname,
                    "product_name_hash": h,
                    "attribute_name": norm_name,
                    "ai_value": o_val,
                    "corrected_value": c_val,
                })
    return diffs


def apply_corrections_to_db(
    db_path: Path,
    diffs: list[dict],
    run_id: Optional[str] = None,
) -> int:
    """Update corrections table. Returns # rows updated."""
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        now = datetime.now(timezone.utc).isoformat()
        n_updated = 0
        for d in diffs:
            if run_id:
                cur = conn.execute(
                    """
                    UPDATE corrections
                       SET corrected_value = ?, corrected_at = ?
                     WHERE product_name_hash = ? AND attribute_name = ? AND run_id = ?
                    """,
                    (d["corrected_value"], now, d["product_name_hash"],
                     d["attribute_name"], run_id),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE corrections
                       SET corrected_value = ?, corrected_at = ?
                     WHERE product_name_hash = ? AND attribute_name = ?
                    """,
                    (d["corrected_value"], now, d["product_name_hash"],
                     d["attribute_name"]),
                )
            n_updated += cur.rowcount
        conn.commit()
        return n_updated
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(description="Diff corrected Excel vs AI output, log to SQLite")
    ap.add_argument("original", type=Path, help="Original AI-output .xlsx (from excel_importer.py)")
    ap.add_argument("corrected", type=Path, help="Customer's corrected .xlsx")
    ap.add_argument("--db", type=Path,
                    default=PROJECT_ROOT / "data" / "corrections.db",
                    help="SQLite path with corrections table")
    ap.add_argument("--run-id", default=None, help="Limit update to one run_id")
    ap.add_argument("--dry-run", action="store_true", help="Print diffs without writing DB")
    args = ap.parse_args()

    if not args.original.exists():
        ap.error(f"Original not found: {args.original}")
    if not args.corrected.exists():
        ap.error(f"Corrected not found: {args.corrected}")

    diffs = diff_excels(args.original, args.corrected)
    logger.info("Found %d cell-level diffs", len(diffs))
    for d in diffs[:10]:
        logger.info("  %s | %s : %r -> %r",
                    str(d["product_name"])[:40], d["attribute_name"],
                    d["ai_value"][:40], d["corrected_value"][:40])
    if args.dry_run:
        print(f"DRY-RUN: would update {len(diffs)} rows")
        return

    if not diffs:
        print("No diffs to apply.")
        return

    n = apply_corrections_to_db(args.db, diffs, run_id=args.run_id)
    print(f"Updated {n} rows in {args.db}")


if __name__ == "__main__":
    main()
