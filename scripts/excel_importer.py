"""excel_importer.py — customer-facing Excel/CSV → AI-enriched Excel.

Читает customer Excel/CSV (минимум колонка ``product_name``), прогоняет каждую
строку через ``PipelineOrchestrator``, и пишет новый Excel с:
  * добавленными колонками для каждого заполненного атрибута,
  * conditional formatting (зелёный/жёлтый/красный по confidence),
  * Sheet 2 «Чек-лист» — строки/поля требующие ручной проверки,
  * Sheet 3 «Сводка» — агрегированная статистика.

Параллельно AI-output логируется в SQLite (``data/corrections.db``) таблицу
``corrections`` для последующего сбора правок (см. ``excel_diff_corrections.py``).

Usage:
    python scripts/excel_importer.py INPUT.xlsx -o OUT.xlsx
    python scripts/excel_importer.py INPUT.csv  --marketplace ozon \\
        --ozon-category-id 17028612 --ozon-type-id 91910 \\
        --concurrency 5
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(line_buffering=True)

os.environ["LLM_CACHE_ENABLED"] = "1"
os.environ.setdefault("LLM_CACHE_DB", str(PROJECT_ROOT / ".llm_cache.sqlite"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.formatting.rule import CellIsRule, FormulaRule  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402

from app.services.enrichment.base import (  # noqa: E402
    ExtractionContext,
    TargetAttribute,
)
from app.services.enrichment.pipeline import PipelineOrchestrator  # noqa: E402
from app.services.enrichment.sources.competitor_rag_source import CompetitorRagSource  # noqa: E402
from app.services.enrichment.sources.icecat_source import IceCatSource  # noqa: E402
from app.services.enrichment.sources.pdf_datasheet_source import PdfDatasheetSource  # noqa: E402
from app.services.enrichment.strategies.dictionaries.ozon_loader import (  # noqa: E402
    get_ozon_characteristics_for_type,
)
from app.services.enrichment.strategies.factory import get_strategy  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("excel_importer")
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Дефолтная пара (PSU) — используется если в CLI/Excel не указано иное.
# Для MVP customer задаёт одну пару на весь файл или per-row через колонки
# ``ozon_category_id`` / ``ozon_type_id``.
DEFAULT_OZON_CATEGORY_ID = 17028612
DEFAULT_OZON_TYPE_ID = 91910

CONFIDENCE_HIGH = 0.85
CONFIDENCE_MED = 0.60

# Conditional-format fill colors
FILL_GREEN = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
FILL_YELLOW = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
FILL_RED = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
FILL_HEADER = PatternFill(start_color="305496", end_color="305496", fill_type="solid")
FONT_HEADER = Font(bold=True, color="FFFFFF")


# ---------------------------------------------------------------------------
# SQLite layer for `corrections` table
# ---------------------------------------------------------------------------

CORRECTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    product_name_hash TEXT NOT NULL,
    product_name TEXT NOT NULL,
    brand TEXT,
    category_path TEXT,
    attribute_id INTEGER NOT NULL,
    attribute_name TEXT,
    ai_value TEXT,
    ai_confidence REAL,
    ai_source TEXT,
    corrected_value TEXT,
    corrected_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_corrections_run ON corrections(run_id);
CREATE INDEX IF NOT EXISTS idx_corrections_hash ON corrections(product_name_hash);
"""


def open_corrections_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(CORRECTIONS_SCHEMA)
    conn.commit()
    return conn


def hash_product_name(name: str) -> str:
    return hashlib.sha1(name.strip().lower().encode("utf-8")).hexdigest()[:16]


def insert_corrections(
    conn: sqlite3.Connection,
    run_id: str,
    rows: list[dict],
) -> None:
    if not rows:
        return
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO corrections (
            run_id, created_at, product_name_hash, product_name,
            brand, category_path, attribute_id, attribute_name,
            ai_value, ai_confidence, ai_source,
            corrected_value, corrected_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
        """,
        [
            (
                run_id,
                r.get("created_at") or now,
                r["product_name_hash"],
                r["product_name"],
                r.get("brand"),
                r.get("category_path"),
                int(r["attribute_id"]),
                r.get("attribute_name"),
                _stringify(r.get("ai_value")),
                float(r.get("ai_confidence") or 0.0),
                r.get("ai_source"),
            )
            for r in rows
        ],
    )
    conn.commit()


def _stringify(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return ", ".join(_stringify(x) or "" for x in v)
    return str(v)


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------

REQUIRED_COL = "product_name"
OPTIONAL_COLS = ("brand", "category_path", "image_urls", "ozon_category_id", "ozon_type_id")


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in (".xlsx", ".xlsm"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported input format: {suffix}")
    if REQUIRED_COL not in df.columns:
        raise ValueError(
            f"Input must contain column '{REQUIRED_COL}'. Found: {list(df.columns)}"
        )
    # Normalize NaN → None for optional cols, keep them present
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = None
    return df


def parse_image_urls(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    return [u.strip() for u in s.split(",") if u.strip()]


def parse_category_path(raw: Any) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(">") if p.strip()]


def _coerce_int(val: Any, default: int) -> int:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def build_targets(chars: list[dict]) -> list[TargetAttribute]:
    out: list[TargetAttribute] = []
    for c in chars:
        allowed = None
        if c.get("values"):
            allowed = [v["value"] for v in c["values"][:50]]
        out.append(TargetAttribute(
            id=c["id"],
            name=c["name"],
            type="text",
            allowed_values=allowed,
            is_collection=c.get("is_collection", False),
            is_required=c.get("is_required", False),
        ))
    return out


async def enrich_row(
    orchestrator: PipelineOrchestrator,
    targets: list[TargetAttribute],
    char_by_id: dict[int, dict],
    row_idx: int,
    row: dict,
    marketplace: str,
    default_cat_id: int,
    default_type_id: int,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run pipeline for one row. Returns dict with row_idx, filled, context info."""
    async with semaphore:
        product_name = str(row[REQUIRED_COL]).strip()
        brand = row.get("brand")
        if isinstance(brand, float) and pd.isna(brand):
            brand = None
        elif brand is not None:
            brand = str(brand).strip() or None

        cat_path = parse_category_path(row.get("category_path"))
        image_urls = parse_image_urls(row.get("image_urls"))
        cat_id = _coerce_int(row.get("ozon_category_id"), default_cat_id)
        type_id = _coerce_int(row.get("ozon_type_id"), default_type_id)

        ctx = ExtractionContext(
            product_id=row_idx + 1,
            product_name=product_name,
            product_description="",
            brand=brand,
            category_id=cat_id,
            category_path=cat_path,
            image_urls=image_urls,
            marketplace=marketplace if marketplace != "default" else None,
            ozon_type_id=type_id if marketplace == "ozon" else None,
        )

        t0 = time.time()
        avs: list = []
        err: Optional[str] = None
        try:
            avs = await orchestrator.enrich(ctx, targets)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            logger.warning("[row %d] pipeline error: %s", row_idx, err)
        elapsed = time.time() - t0

        filled = [
            {
                "attribute_id": av.attribute_id,
                "attribute_name": char_by_id.get(av.attribute_id, {}).get("name")
                                  or next((t.name for t in targets if t.id == av.attribute_id), None),
                "value": av.value,
                "confidence": float(av.confidence),
                "source": str(av.source),
                "is_required": char_by_id.get(av.attribute_id, {}).get("is_required", False),
            }
            for av in avs
        ]
        logger.info(
            "[row %d/%s] %s -> %d attrs in %.1fs%s",
            row_idx + 1, product_name[:50], "ok" if not err else "err",
            len(filled), elapsed, f" ({err})" if err else "",
        )
        return {
            "row_idx": row_idx,
            "product_name": product_name,
            "brand": brand,
            "category_path": " > ".join(cat_path) if cat_path else None,
            "filled": filled,
            "error": err,
            "elapsed_sec": elapsed,
        }


# ---------------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------------

def write_excel(
    out_path: Path,
    input_df: pd.DataFrame,
    enriched: list[dict],
    targets: list[TargetAttribute],
) -> None:
    """Write the 3-sheet Excel: Results, Чек-лист, Сводка."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Результаты"

    # Determine attribute columns: union of all filled attribute_ids + required targets
    target_by_id = {t.id: t for t in targets}
    attr_ids: list[int] = []
    seen: set[int] = set()
    # Required targets first
    for t in targets:
        if t.is_required and t.id not in seen:
            attr_ids.append(t.id)
            seen.add(t.id)
    # Then any other actually-filled IDs
    for e in enriched:
        for f in e["filled"]:
            if f["attribute_id"] not in seen:
                attr_ids.append(f["attribute_id"])
                seen.add(f["attribute_id"])

    # Header row
    input_cols = list(input_df.columns)
    headers = list(input_cols)
    for aid in attr_ids:
        t = target_by_id.get(aid)
        nm = (t.name if t else f"id={aid}")
        suffix = " *" if t and t.is_required else ""
        headers.append(f"{nm}{suffix}")
    headers.append("Score, %")
    for col_idx, name in enumerate(headers, 1):
        c = ws.cell(row=1, column=col_idx, value=name)
        c.font = FONT_HEADER
        c.fill = FILL_HEADER
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Track per-cell confidence in a parallel structure for formatting
    n_inputs = len(input_cols)
    first_attr_col = n_inputs + 1  # 1-based
    last_attr_col = n_inputs + len(attr_ids)
    score_col = last_attr_col + 1

    enriched_by_row = {e["row_idx"]: e for e in enriched}
    total_required = sum(1 for t in targets if t.is_required)
    total_optional = len(targets) - total_required

    # Confidence map per cell — used for conditional fills (we directly set fills below
    # because conditional formatting in openpyxl across many cells with per-cell rules
    # is cumbersome; per-cell PatternFill is simpler).
    for row_pos, (idx, src_row) in enumerate(input_df.iterrows(), start=2):
        # Input columns
        for col_idx, col_name in enumerate(input_cols, 1):
            v = src_row[col_name]
            if isinstance(v, float) and pd.isna(v):
                v = None
            ws.cell(row=row_pos, column=col_idx, value=v)

        e = enriched_by_row.get(idx)
        filled_by_attr: dict[int, dict] = (
            {f["attribute_id"]: f for f in e["filled"]} if e else {}
        )

        # Attribute columns
        filled_required = 0
        filled_optional = 0
        for j, aid in enumerate(attr_ids):
            col = first_attr_col + j
            f = filled_by_attr.get(aid)
            target = target_by_id.get(aid)
            if f is None:
                # Empty — red
                cell = ws.cell(row=row_pos, column=col, value=None)
                cell.fill = FILL_RED
                continue
            val = f["value"]
            if isinstance(val, list):
                val = ", ".join(_stringify(x) or "" for x in val)
            cell = ws.cell(row=row_pos, column=col, value=val)
            conf = float(f["confidence"])
            if conf >= CONFIDENCE_HIGH:
                cell.fill = FILL_GREEN
            elif conf >= CONFIDENCE_MED:
                cell.fill = FILL_YELLOW
            else:
                cell.fill = FILL_RED
            # Tooltip-ish: comment with source+confidence
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            if target and target.is_required:
                filled_required += 1
            else:
                filled_optional += 1

        # Score: % coverage of all targets
        denom_req = total_required or 1
        denom_opt = total_optional or 1
        # Coverage = avg of required-coverage and optional-coverage (weighted equally
        # if both exist, else whichever).
        if total_required and total_optional:
            score = 0.5 * (filled_required / denom_req) + 0.5 * (filled_optional / denom_opt)
        elif total_required:
            score = filled_required / denom_req
        else:
            score = filled_optional / denom_opt
        score_cell = ws.cell(row=row_pos, column=score_col, value=round(score * 100, 1))
        if score >= CONFIDENCE_HIGH:
            score_cell.fill = FILL_GREEN
        elif score >= CONFIDENCE_MED:
            score_cell.fill = FILL_YELLOW
        else:
            score_cell.fill = FILL_RED

    # Column widths
    for col_idx, hdr in enumerate(headers, 1):
        letter = get_column_letter(col_idx)
        ws.column_dimensions[letter].width = max(14, min(45, len(str(hdr)) + 4))
    ws.freeze_panes = "A2"

    # Conditional-formatting rules at the worksheet level (additionally — for tools
    # that respect rules over static fills, e.g. when a user edits a cell).
    last_row = 1 + len(input_df)
    if last_row >= 2 and last_attr_col >= first_attr_col:
        rng = f"{get_column_letter(first_attr_col)}2:{get_column_letter(last_attr_col)}{last_row}"
        # NOTE: rules apply to text content presence — we already coloured by confidence;
        # but adding "blanks → red" rule keeps red on cells the user later wipes.
        ws.conditional_formatting.add(
            rng,
            FormulaRule(
                formula=[f'LEN(TRIM({get_column_letter(first_attr_col)}2))=0'],
                fill=FILL_RED,
                stopIfTrue=False,
            ),
        )

    # ----------------- Sheet 2: Чек-лист -----------------
    ws2 = wb.create_sheet("Чек-лист")
    ws2.append(["#", "Товар", "Поле", "AI значение", "Confidence", "Источник", "Статус"])
    for c in ws2[1]:
        c.font = FONT_HEADER
        c.fill = FILL_HEADER
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    review_row_idx = 2
    for e in enriched:
        prod = e["product_name"]
        filled_by_attr = {f["attribute_id"]: f for f in e["filled"]}
        for aid in attr_ids:
            target = target_by_id.get(aid)
            attr_name = target.name if target else f"id={aid}"
            f = filled_by_attr.get(aid)
            if f is None:
                # Missing field — needs filling. Only required missing → critical.
                status = "ПУСТО" + (" (обязательно)" if target and target.is_required else "")
                ws2.append([review_row_idx - 1, prod, attr_name, "", "", "", status])
                _color_status_cell(ws2.cell(row=review_row_idx, column=7),
                                   "red" if target and target.is_required else "red")
                review_row_idx += 1
            elif float(f["confidence"]) < CONFIDENCE_HIGH:
                conf = float(f["confidence"])
                status = "Проверить"
                val = f["value"]
                if isinstance(val, list):
                    val = ", ".join(_stringify(x) or "" for x in val)
                ws2.append([
                    review_row_idx - 1, prod, attr_name, val,
                    round(conf, 2), f["source"], status,
                ])
                _color_status_cell(ws2.cell(row=review_row_idx, column=7),
                                   "yellow" if conf >= CONFIDENCE_MED else "red")
                review_row_idx += 1
    for col_idx, width in enumerate([5, 45, 30, 30, 12, 18, 22], 1):
        ws2.column_dimensions[get_column_letter(col_idx)].width = width
    ws2.freeze_panes = "A2"

    # ----------------- Sheet 3: Сводка -----------------
    ws3 = wb.create_sheet("Сводка")
    n = len(enriched)
    total_required_filled = sum(
        sum(1 for f in e["filled"] if target_by_id.get(f["attribute_id"]) and target_by_id[f["attribute_id"]].is_required)
        for e in enriched
    )
    total_optional_filled = sum(
        sum(1 for f in e["filled"] if target_by_id.get(f["attribute_id"]) and not target_by_id[f["attribute_id"]].is_required)
        for e in enriched
    )
    avg_req = (total_required_filled / n / total_required) if n and total_required else 0.0
    avg_opt = (total_optional_filled / n / total_optional) if n and total_optional else 0.0

    # Missing-field stats
    miss_counter: dict[int, int] = {aid: 0 for aid in attr_ids}
    for e in enriched:
        ids = {f["attribute_id"] for f in e["filled"]}
        for aid in attr_ids:
            if aid not in ids:
                miss_counter[aid] += 1
    top_missing = sorted(miss_counter.items(), key=lambda kv: -kv[1])[:15]

    ws3.append(["Метрика", "Значение"])
    ws3.append(["Товаров обработано", n])
    ws3.append(["Покрытие обязательных (avg)", f"{avg_req * 100:.1f}%"])
    ws3.append(["Покрытие опциональных (avg)", f"{avg_opt * 100:.1f}%"])
    ws3.append(["Ошибок pipeline", sum(1 for e in enriched if e.get("error"))])
    ws3.append([])
    ws3.append(["Топ пустых полей (#товаров без значения)", ""])
    for aid, cnt in top_missing:
        t = target_by_id.get(aid)
        nm = (t.name if t else f"id={aid}")
        ws3.append([nm + (" *" if t and t.is_required else ""), cnt])

    for c in ws3[1]:
        c.font = FONT_HEADER
        c.fill = FILL_HEADER
    ws3.column_dimensions["A"].width = 45
    ws3.column_dimensions["B"].width = 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out_path))


def _color_status_cell(cell, kind: str) -> None:
    if kind == "red":
        cell.fill = FILL_RED
    elif kind == "yellow":
        cell.fill = FILL_YELLOW
    else:
        cell.fill = FILL_GREEN


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _run(
    input_path: Path,
    output_path: Path,
    marketplace: str,
    ozon_cat_id: int,
    ozon_type_id: int,
    concurrency: int,
    skip_rag: bool,
    db_path: Path,
    no_icecat: bool,
    no_pdf: bool,
) -> dict:
    df = read_input(input_path)
    logger.info("Loaded %d rows from %s", len(df), input_path.name)

    chars = get_ozon_characteristics_for_type(ozon_cat_id, ozon_type_id)
    if not chars:
        logger.warning(
            "No Ozon chars for (cat=%s, type=%s) — running with empty target schema",
            ozon_cat_id, ozon_type_id,
        )
    char_by_id = {c["id"]: c for c in chars}
    targets = build_targets(chars)
    logger.info("Target schema: %d attrs (%d required)",
                len(targets), sum(1 for t in targets if t.is_required))

    strategy = get_strategy(marketplace)
    competitor_rag = None if skip_rag else CompetitorRagSource()
    icecat = None if no_icecat else IceCatSource()
    pdf_datasheet = None if no_pdf else PdfDatasheetSource()
    orchestrator = PipelineOrchestrator(
        strategy=strategy,
        competitor_rag_source=competitor_rag,
        icecat_source=icecat,
        pdf_datasheet_source=pdf_datasheet,
    )

    semaphore = asyncio.Semaphore(max(1, concurrency))
    t_start = time.time()
    tasks = [
        enrich_row(
            orchestrator, targets, char_by_id, int(idx), row.to_dict(),
            marketplace, ozon_cat_id, ozon_type_id, semaphore,
        )
        for idx, row in df.iterrows()
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    elapsed = time.time() - t_start
    logger.info("Pipeline done in %.1fs for %d rows", elapsed, len(results))

    # ---- Persist corrections to SQLite ----
    run_id = uuid.uuid4().hex
    now_iso = datetime.now(timezone.utc).isoformat()
    corrections_rows = []
    for r in results:
        h = hash_product_name(r["product_name"])
        for f in r["filled"]:
            corrections_rows.append({
                "created_at": now_iso,
                "product_name_hash": h,
                "product_name": r["product_name"],
                "brand": r.get("brand"),
                "category_path": r.get("category_path"),
                "attribute_id": f["attribute_id"],
                "attribute_name": f["attribute_name"],
                "ai_value": f["value"],
                "ai_confidence": f["confidence"],
                "ai_source": f["source"],
            })
    conn = open_corrections_db(db_path)
    try:
        insert_corrections(conn, run_id, corrections_rows)
        logger.info("Inserted %d rows into %s (run_id=%s)",
                    len(corrections_rows), db_path, run_id)
    finally:
        conn.close()

    # ---- Write Excel ----
    write_excel(output_path, df, results, targets)
    logger.info("Wrote Excel: %s", output_path)

    return {
        "run_id": run_id,
        "rows": len(results),
        "elapsed_sec": elapsed,
        "corrections_inserted": len(corrections_rows),
        "output": str(output_path),
    }


def main():
    ap = argparse.ArgumentParser(description="Customer Excel/CSV → AI-enriched Excel")
    ap.add_argument("input", type=Path, help="Input .xlsx/.csv file")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output .xlsx (default: <input>_enriched.xlsx)")
    ap.add_argument("--marketplace", choices=["ozon", "wb", "default"], default="ozon")
    ap.add_argument("--ozon-category-id", type=int, default=DEFAULT_OZON_CATEGORY_ID)
    ap.add_argument("--ozon-type-id", type=int, default=DEFAULT_OZON_TYPE_ID)
    ap.add_argument("--concurrency", type=int, default=5,
                    help="Max parallel pipeline calls (default 5)")
    ap.add_argument("--skip-rag", action="store_true",
                    help="Disable CompetitorRagSource (no Qdrant)")
    ap.add_argument("--no-icecat", action="store_true",
                    help="Disable IceCatSource")
    ap.add_argument("--no-pdf", action="store_true",
                    help="Disable PdfDatasheetSource")
    ap.add_argument("--db", type=Path,
                    default=PROJECT_ROOT / "data" / "corrections.db",
                    help="SQLite path for corrections log")
    args = ap.parse_args()

    if not args.input.exists():
        ap.error(f"Input not found: {args.input}")

    output = args.output or args.input.with_name(args.input.stem + "_enriched.xlsx")
    summary = asyncio.run(_run(
        input_path=args.input,
        output_path=output,
        marketplace=args.marketplace,
        ozon_cat_id=args.ozon_category_id,
        ozon_type_id=args.ozon_type_id,
        concurrency=args.concurrency,
        skip_rag=args.skip_rag,
        db_path=args.db,
        no_icecat=args.no_icecat,
        no_pdf=args.no_pdf,
    ))
    print(f"\nDONE: {summary}")


if __name__ == "__main__":
    main()
