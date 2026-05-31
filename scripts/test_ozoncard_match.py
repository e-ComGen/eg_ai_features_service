"""Быстрый тест-индикатор матчинга OzonCard — БЕЗ LLM, только Scrappey search+parse.

Запуск:
    .\\venv\\Scripts\\python.exe scripts\\test_ozoncard_match.py

Для каждого из 40 товаров в PRODUCTS вызывает ozon_card.probe(ctx) — search Ozon +
fetch /features/ + parse характеристик. LLM не вызывается.

Печатает таблицу + агрегат (hit-rate, разбивка по stage).
Сохраняет JSON scripts/eval_results/ozoncard_probe_<ts>.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# UTF-8 stdout — иначе print() с кириллицей ломается на Windows (cp1251)
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app.services.enrichment.base import ExtractionContext
from app.services.enrichment.sources.ozon_card_source import OzonCardSource
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    load_ozon_dictionary,
)

# Импортируем список товаров из eval_multi_category
from scripts.eval_multi_category import PRODUCTS  # type: ignore

CONCURRENCY = 6
RESULTS_DIR = PROJECT_ROOT / "scripts" / "eval_results"


def _get_category_path(cat_id: int, type_id: int) -> list[str]:
    """Извлечь path из словаря для (cat_id, type_id), или вернуть []."""
    d = load_ozon_dictionary()
    key = f"{cat_id}:{type_id}"
    entry = d.get(key)
    if entry and entry.get("path"):
        return entry["path"]
    return []


def _category_label(cat_id: int, type_id: int, path: list[str]) -> str:
    """Короткое название категории для таблицы."""
    if path:
        return path[-1][:30]
    return f"cat:{cat_id}"


async def run_probe(
    idx: int,
    cat_id: int,
    type_id: int,
    prod_name: str,
    ozon_card: OzonCardSource,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        words = prod_name.split()
        brand_guess = words[1] if len(words) > 1 else words[0]
        category_path = _get_category_path(cat_id, type_id)

        ctx = ExtractionContext(
            product_id=idx,
            product_name=prod_name,
            product_description="",
            brand=brand_guess,
            category_id=cat_id,
            category_path=category_path,
            image_urls=[],
            marketplace="ozon",
            ozon_type_id=type_id,
        )

        t0 = time.monotonic()
        result = await ozon_card.probe(ctx)
        elapsed = time.monotonic() - t0

        cat_label = _category_label(cat_id, type_id, category_path)
        result["idx"] = idx
        result["product_name"] = prod_name
        result["cat_id"] = cat_id
        result["type_id"] = type_id
        result["category_label"] = cat_label
        result["elapsed_s"] = round(elapsed, 1)

        found_mark = "✓" if result.get("found") else "✗"
        score_str = f"{result['match_score']:.1f}" if result.get("match_score") is not None else "  -  "
        print(
            f"  [{idx:2d}/40] {found_mark} stage={result['stage']:<12} "
            f"tiles={result['tiles_count']:2d} score={score_str} "
            f"chars={result['raw_chars']:3d} "
            f"q={result['query'][:40]:<40} | {cat_label}",
            flush=True,
        )
        # Для low_match — показать заголовок лучшего tile чтобы понять причину
        if result.get("stage") == "low_match" and result.get("best_tile_title"):
            print(
                f"           best_tile: {result['best_tile_title'][:90]}",
                flush=True,
            )
        return result


async def main() -> None:
    print(f"[Probe] Loading ozon_dictionary...", flush=True)
    d = load_ozon_dictionary()
    print(f"[Probe] {len(d)} entries. SCRAPPEY_KEY={'SET' if os.environ.get('SCRAPPEY_KEY') else 'MISSING'}", flush=True)

    ozon_card = OzonCardSource()
    sem = asyncio.Semaphore(CONCURRENCY)

    print(
        f"\n[Probe] Running probe on {len(PRODUCTS)} products (concurrency={CONCURRENCY}) ...\n",
        flush=True,
    )
    print(
        f"{'':>6} {'F'} {'stage':<12} {'til':>3} {'score':>6} {'chr':>4} "
        f"{'query':<42} category",
        flush=True,
    )
    print("-" * 110, flush=True)

    tasks = [
        run_probe(i + 1, cat_id, type_id, name, ozon_card, sem)
        for i, (cat_id, type_id, name) in enumerate(PRODUCTS)
    ]
    results = await asyncio.gather(*tasks)

    # ---- Aggregate ----
    total = len(results)
    found = sum(1 for r in results if r.get("found"))

    stage_counts: dict[str, int] = {}
    for r in results:
        s = r.get("stage", "unknown")
        stage_counts[s] = stage_counts.get(s, 0) + 1

    print("\n" + "=" * 110, flush=True)
    print(f"HIT-RATE: {found}/{total} found ({100 * found / total:.0f}%)", flush=True)
    print("Stage breakdown:", flush=True)
    for stage, cnt in sorted(stage_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {stage:<15} {cnt:3d}", flush=True)

    # Failed examples — разбивка по low_match и остальным
    failed = [r for r in results if not r.get("found")]
    low_match_rows = [r for r in failed if r.get("stage") == "low_match"]
    other_failed = [r for r in failed if r.get("stage") != "low_match"]

    if other_failed:
        print(f"\nFailed (not low_match) ({len(other_failed)}):", flush=True)
        for r in other_failed:
            score_str = f"{r['match_score']:.1f}" if r.get("match_score") is not None else "None"
            print(
                f"  [{r['idx']:2d}] {r['stage']:<12} score={score_str:<6} "
                f"q={r['query'][:50]:<50} | {r['product_name'][:50]}",
                flush=True,
            )

    if low_match_rows:
        print(f"\nlow_match ({len(low_match_rows)}) — query → best_tile (score):", flush=True)
        for r in low_match_rows:
            score_str = f"{r['match_score']:.1f}" if r.get("match_score") is not None else "None"
            best_title = r.get("best_tile_title") or "(нет тайтла)"
            print(
                f"  [{r['idx']:2d}] q={r['query'][:45]:<45} score={score_str:<6}"
                f"\n       best: {best_title[:90]}"
                f"\n       prod: {r['product_name'][:70]}",
                flush=True,
            )

    # ---- Save JSON ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ozoncard_probe_{ts}.json"

    payload = {
        "timestamp": ts,
        "total": total,
        "found": found,
        "hit_rate": round(found / total, 3),
        "stage_breakdown": stage_counts,
        "products": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Probe] Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
