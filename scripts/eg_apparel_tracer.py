"""eg_apparel_tracer — reusable per-attribute drain diagnostic for apparel products.

Goal
----
For each apparel product (a NAME + its REAL Ozon category_id/type_id), run the
REAL enrichment pipeline and, for EVERY Ozon target attribute of that category,
record the outcome and — if the attribute ends up EMPTY — classify WHY it drained.

CRITICAL: this runs the pipeline with the REAL (category_id, type_id) of the
product so that value_id resolution is ACTIVE. We DO NOT pass category_id=0 — a
zero category silently disables resolve_value_ids (every enum value_id would be
None) and the trace would be garbage. The categories below are real leaf
type_ids pulled from the live ozon_dictionary (Одежда / Обувь branches).

There is NO name->category classifier in this codebase: category (cat_id,type_id)
is always supplied externally (see pipeline_adapter — addon/job passes ozon_type_id).
So "resolving the real category" == supplying the real (cat_id,type_id) tuple,
exactly as eval_multi_category.py does. The tracer mirrors that exact setup
(strategy, all sources, ExtractionContext fields) so the enrichment path and
value_id resolution behave identically to the normal eval.

Drain codes (per empty target attribute)
-----------------------------------------
  A = no WB/Ozon card was found for the product at all (both card sources empty)
  B = a card WAS found, but this attribute is absent from it (and no other source
      produced a candidate for it either)
  C = a candidate value WAS produced but value_id resolution returned None
      (enum miss) — dropped at resolution / by _drop_unresolved_optional_enums
  D = a candidate existed but was removed by a guard (gender_guard / judge /
      strategy validate_value) — i.e. it vanished between sources and final, and
      it was NOT an enum-resolution drop
  E = no source ever produced any candidate for it

Usage
-----
  venv/Scripts/python.exe -m scripts.eg_apparel_tracer            # all 40
  TRACE_LIMIT=3 venv/Scripts/python.exe -m scripts.eg_apparel_tracer   # first 3 (validation)
  TRACE_FILTER=толстов,футбол venv/Scripts/python.exe -m scripts.eg_apparel_tracer

Writes JSON to scripts/eval_results/apparel_trace_<ts>.json and prints a summary.
The markdown report (eg_apparel_trace_report.md) is written by the caller from
the JSON, or pass WRITE_REPORT=1 to emit it directly.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

os.environ["LLM_CACHE_ENABLED"] = "1"
os.environ.setdefault("LLM_CACHE_DB", str(PROJECT_ROOT / ".llm_cache.sqlite"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

if os.environ.get("DEBUG_LOG") == "1":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    load_ozon_dictionary,
)
from app.services.enrichment.strategies.dictionaries.ozon_field_classifier import is_platform_field
from app.services.enrichment.base import ExtractionContext, TargetAttribute, Source
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.strategies.factory import get_strategy
from app.services.enrichment.sources.competitor_rag_source import CompetitorRagSource
from app.services.enrichment.sources.icecat_source import IceCatSource
from app.services.enrichment.sources.pdf_datasheet_source import PdfDatasheetSource
from app.services.enrichment.sources.ozon_card_source import OzonCardSource
from app.services.enrichment.sources.wb_card_source import WbCardSource
from app.services.enrichment.sources.tnved_source import TnvedSource

# ----------------------------------------------------------------------------
# PRODUCT SET — 40 apparel products, each tagged with its REAL (cat_id, type_id).
# Leaf type_ids verified against the live ozon_dictionary (Одежда / Обувь).
# Mix: men/women/kids, branded/generic, marketplace-style Russian names.
# The first 6 are the apparel products already present in eval_multi_category.py.
# ----------------------------------------------------------------------------
# Canonical leaf categories (path "Одежда / Одежда / X", "Обувь / Повседневная обувь / X"):
TOLSTOVKA = (200000933, 93232)
FUTBOLKA  = (200000933, 93244)
DZHINSY   = (200000933, 93080)
PLATYE    = (200000933, 93182)
KURTKA    = (200000933, 93137)
YUBKA     = (200000933, 93283)
RUBASHKA  = (200000933, 93209)
HUDI      = (200000933, 93253)
SVITSHOT  = (200000933, 93216)
BRYUKI    = (200000933, 93055)
SHORTY    = (200000933, 93272)
KROSSOVKI = (15621048,  91248)
BOTINKI   = (15621048,  91239)
PALTO     = (200000933, 93166)
SVITER    = (200000933, 93214)
KOFTA_BABY = (200001519, 970744798)  # «Кофточка для новорожденного»

PRODUCTS: list[tuple[int, int, str]] = [
    # --- already in eval_multi_category.py (apparel ones) ---
    (*FUTBOLKA,  "Футболка мужская Nike Sportswear Club"),
    (*DZHINSY,   "Джинсы мужские Levis 501 Original"),
    (*KURTKA,    "Куртка мужская The North Face Resolve 2"),
    (*KROSSOVKI, "Кроссовки Adidas Ultraboost 22"),
    (*PLATYE,    "Платье женское befree летнее"),
    (*TOLSTOVKA, "Толстовка худи Champion Reverse Weave"),
    # --- толстовка ---
    (*TOLSTOVKA, "Толстовка мужская оверсайз на флисе с капюшоном"),
    (*TOLSTOVKA, "Толстовка женская Adidas Originals Trefoil"),
    # --- футболка ---
    (*FUTBOLKA,  "Футболка хлопковая базовая белая унисекс"),
    (*FUTBOLKA,  "Футболка детская с принтом Человек-паук"),
    # --- джинсы ---
    (*DZHINSY,   "Джинсы женские скинни с высокой посадкой"),
    (*DZHINSY,   "Джинсы мужские прямые Wrangler Texas"),
    # --- платье ---
    (*PLATYE,    "Платье вечернее длинное в пол с разрезом"),
    (*PLATYE,    "Платье трикотажное миди Zarina осеннее"),
    # --- куртка ---
    (*KURTKA,    "Куртка зимняя женская пуховик с мехом"),
    (*KURTKA,    "Куртка детская демисезонная для мальчика"),
    # --- юбка ---
    (*YUBKA,     "Юбка женская джинсовая мини"),
    (*YUBKA,     "Юбка плиссированная миди черная"),
    # --- рубашка ---
    (*RUBASHKA,  "Рубашка мужская классическая белая приталенная"),
    (*RUBASHKA,  "Рубашка женская оверсайз в клетку фланель"),
    # --- худи ---
    (*HUDI,      "Худи мужское черное на молнии Nike"),
    (*HUDI,      "Худи женское укороченное с капюшоном"),
    # --- свитшот ---
    (*SVITSHOT,  "Свитшот мужской флисовый утепленный"),
    (*SVITSHOT,  "Свитшот женский oversize молочный"),
    # --- брюки ---
    (*BRYUKI,    "Брюки женские палаццо широкие"),
    (*BRYUKI,    "Брюки мужские карго хлопковые"),
    # --- шорты ---
    (*SHORTY,    "Шорты мужские спортивные Nike Dri-FIT"),
    (*SHORTY,    "Шорты женские джинсовые с высокой талией"),
    # --- кроссовки ---
    (*KROSSOVKI, "Кроссовки New Balance 574 серые"),
    (*KROSSOVKI, "Кроссовки женские Puma RS-X"),
    # --- ботинки ---
    (*BOTINKI,   "Ботинки мужские зимние кожаные на меху"),
    (*BOTINKI,   "Ботинки женские демисезонные Timberland"),
    # --- пальто ---
    (*PALTO,     "Пальто женское шерстяное прямое бежевое"),
    (*PALTO,     "Пальто мужское кашемировое классическое"),
    # --- свитер ---
    (*SVITER,    "Свитер мужской вязаный с горлом шерсть"),
    (*SVITER,    "Свитер женский крупной вязки оверсайз"),
    # --- кофта (baby) ---
    (*KOFTA_BABY, "Кофточка для новорожденного хлопок с начесом"),
    (*KOFTA_BABY, "Кофточка детская на кнопках для малыша"),
    # --- extra branded fillers to reach 40 ---
    (*FUTBOLKA,  "Футболка Tommy Hilfiger мужская с логотипом"),
    (*DZHINSY,   "Джинсы Calvin Klein женские mom fit"),
]


def build_targets(chars: list[dict]) -> list[TargetAttribute]:
    out = []
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


def _get_category_path(cat_id: int, type_id: int) -> list[str]:
    d = load_ozon_dictionary()
    entry = d.get(f"{cat_id}:{type_id}")
    if entry and entry.get("path"):
        return entry["path"]
    return []


def _vid_present(av) -> bool:
    """True if this AttributeValue carries any resolved value_id / value_ids."""
    vid = getattr(av, "value_id", None)
    vids = getattr(av, "value_ids", None)
    if vid:
        return True
    if vids and any(x is not None for x in vids):
        return True
    return False


async def trace_one(
    orch: PipelineOrchestrator,
    idx: int,
    n_total: int,
    cat_id: int,
    type_id: int,
    name: str,
) -> dict:
    """Run pipeline for one product and classify drain code for every target."""
    chars = get_ozon_characteristics_for_type(cat_id, type_id)
    if not chars:
        return {
            "idx": idx, "cat_id": cat_id, "type_id": type_id, "name": name,
            "skipped": True, "skip_reason": f"no chars for {cat_id}:{type_id}",
        }

    char_by_id = {c["id"]: c for c in chars}
    targets = build_targets(chars)
    category_path = _get_category_path(cat_id, type_id)
    words = name.split()
    brand_guess = words[1] if len(words) > 1 else words[0]

    ctx = ExtractionContext(
        product_id=idx,
        product_name=name,
        product_description="",
        brand=brand_guess,
        category_id=cat_id,           # REAL category id — value_id resolution ON
        category_path=category_path,
        image_urls=[],
        marketplace="ozon",
        ozon_type_id=type_id,         # REAL type id — value_id resolution ON
    )

    # --- per-run capture state (closure, no global, safe under concurrency=1) ---
    cap: dict = {
        "pre_finalize": None,     # list of raw candidate AVs entering _finalize_async
        "wb_card_avs": 0,         # # candidates wb_card stage produced (card-found probe)
        "ozon_card_avs": 0,       # # candidates ozon_card stage produced
        "ozon_card_ran": False,   # whether ozon_card stage executed at all
    }

    # Hook the bound methods on THIS orchestrator instance only.
    orig_fin = orch._finalize_async
    orig_wb = orch._run_wb_card_stage
    orig_oz = orch._run_ozon_card_stage

    async def fin_hook(all_values, tgts, context):
        cap["pre_finalize"] = [
            {
                "attr": v.attribute_id,
                "value": v.value,
                "source": str(v.source),
                "conf": v.confidence,
                "vid": getattr(v, "value_id", None),
                "vids": getattr(v, "value_ids", None),
                "vid_present": _vid_present(v),
            }
            for v in all_values
        ]
        return await orig_fin(all_values, tgts, context)

    async def wb_hook(context, tgts, already_filled=None):
        res = await orig_wb(context, tgts, already_filled=already_filled)
        cap["wb_card_avs"] += len(res)
        return res

    async def oz_hook(context, tgts, already_filled=None):
        cap["ozon_card_ran"] = True
        res = await orig_oz(context, tgts, already_filled=already_filled)
        cap["ozon_card_avs"] += len(res)
        return res

    orch._finalize_async = fin_hook
    orch._run_wb_card_stage = wb_hook
    orch._run_ozon_card_stage = oz_hook

    err = None
    try:
        avs = await orch.enrich(ctx, targets)
    except Exception as e:  # pragma: no cover
        err = f"{type(e).__name__}: {e}"
        avs = []
    finally:
        orch._finalize_async = orig_fin
        orch._run_wb_card_stage = orig_wb
        orch._run_ozon_card_stage = orig_oz

    final_by_id = {av.attribute_id: av for av in avs}

    # pre-finalize candidates grouped by attribute id
    pre = cap["pre_finalize"] or []
    pre_by_id: dict[int, list[dict]] = defaultdict(list)
    for c in pre:
        pre_by_id[c["attr"]].append(c)

    card_found = (cap["wb_card_avs"] > 0) or (cap["ozon_card_avs"] > 0)

    print(
        f"  [{idx:2d}/{n_total}] {name[:48]:<48} -> filled {len(avs)}/{len(chars)} "
        f"| wb={cap['wb_card_avs']} oz={cap['ozon_card_avs']} card={'Y' if card_found else 'N'}"
        + (f" | ERR {err}" if err else ""),
        flush=True,
    )

    # --- classify each target ---
    attr_rows = []
    drain_counter: Counter[str] = Counter()
    for c in chars:
        aid = c["id"]
        is_req = bool(c.get("is_required"))
        is_enum = bool(c.get("values"))
        is_platform = is_platform_field(c)  # media/platform slot (Rich-контент, Видео, Хештеги…)
        final_av = final_by_id.get(aid)
        cands = pre_by_id.get(aid, [])

        if final_av is not None:
            # FILLED
            row = {
                "attr_id": aid,
                "name": c.get("name"),
                "is_required": is_req,
                "is_enum": is_enum,
                "is_platform": is_platform,
                "filled": True,
                "value": final_av.value,
                "source": str(final_av.source),
                "value_id": getattr(final_av, "value_id", None),
                "value_ids": getattr(final_av, "value_ids", None),
                "drain": None,
            }
            attr_rows.append(row)
            continue

        # EMPTY → classify drain
        had_candidate = len(cands) > 0
        cand_had_vid = any(c2["vid_present"] for c2 in cands)

        if not had_candidate:
            # No source produced anything for this attr.
            # Distinguish "no card at all" (A) vs "card found but attr absent" (B)
            # vs "no source ever produced any candidate" (E).
            if not card_found:
                drain = "A"
            else:
                # A card existed but did not carry this attr, and nothing else did.
                # If it's an enum/required attr a card normally fills, that's a B
                # (card-coverage gap). For attrs no source ever touches → E.
                # Heuristic: if ANY card source ran and produced candidates for the
                # product, an attr it skipped is a card-coverage gap (B); else E.
                drain = "B"
        else:
            # A candidate existed pre-finalize but the attr is empty in the final.
            if is_enum and not cand_had_vid:
                # Candidate(s) produced raw text but none resolved to a value_id.
                # Optional enums get dropped by _drop_unresolved_optional_enums;
                # required enums can also vanish via validate_value. Either way the
                # death cause is enum-resolution miss.
                drain = "C"
            else:
                # Candidate had a value_id (or is free-text) yet still removed →
                # a guard killed it (gender_guard / judge / strategy validate_value).
                drain = "D"

        drain_counter[drain] += 1
        attr_rows.append({
            "attr_id": aid,
            "name": c.get("name"),
            "is_required": is_req,
            "is_enum": is_enum,
            "is_platform": is_platform,
            "filled": False,
            "n_candidates": len(cands),
            "cand_sources": sorted({c2["source"] for c2 in cands}),
            "cand_had_vid": cand_had_vid,
            "drain": drain,
        })

    # metrics
    n_total_attrs = len(chars)
    n_filled = sum(1 for r in attr_rows if r["filled"])
    n_req = sum(1 for c in chars if c.get("is_required"))
    n_filled_req = sum(1 for r in attr_rows if r["filled"] and r["is_required"])
    # realistically-fillable bucket: required + non-platform optionals
    fillable_ids = {
        c["id"] for c in chars
        if c.get("is_required") or not is_platform_field(c)
    }
    n_fillable = len(fillable_ids)
    n_filled_fillable = sum(
        1 for r in attr_rows if r["filled"] and r["attr_id"] in fillable_ids
    )

    dominant = drain_counter.most_common(1)[0][0] if drain_counter else None

    return {
        "idx": idx, "cat_id": cat_id, "type_id": type_id, "name": name,
        "leaf_category": category_path[-1] if category_path else f"cat{cat_id}",
        "category_path": category_path,
        "skipped": False,
        "error": err,
        "card_found": card_found,
        "wb_card_avs": cap["wb_card_avs"],
        "ozon_card_avs": cap["ozon_card_avs"],
        "ozon_card_ran": cap["ozon_card_ran"],
        "n_total": n_total_attrs,
        "n_filled": n_filled,
        "n_required": n_req,
        "n_filled_required": n_filled_req,
        "n_fillable": n_fillable,
        "n_filled_fillable": n_filled_fillable,
        "drain_counts": dict(drain_counter),
        "dominant_drain": dominant,
        "attrs": attr_rows,
    }


async def main():
    d = load_ozon_dictionary()
    print(f"[Dict] {len(d)} category:type entries loaded", flush=True)

    strategy = get_strategy("ozon")
    skip_rag = os.environ.get("SKIP_RAG", "0") == "1"
    competitor_rag = None if skip_rag else CompetitorRagSource()
    if skip_rag:
        print("[Trace] SKIP_RAG=1 -> CompetitorRagSource disabled", flush=True)

    orch = PipelineOrchestrator(
        strategy=strategy,
        competitor_rag_source=competitor_rag,
        icecat_source=IceCatSource(),
        pdf_datasheet_source=PdfDatasheetSource(),
        ozon_card_source=OzonCardSource(),
        wb_card_source=WbCardSource(),
        tnved_source=TnvedSource(),
    )

    limit = int(os.environ.get("TRACE_LIMIT", str(len(PRODUCTS))))
    products = PRODUCTS[:limit]
    _filter = os.environ.get("TRACE_FILTER", "").strip()
    if _filter:
        subs = [s.strip().lower() for s in _filter.split(",") if s.strip()]
        products = [p for p in products if any(s in p[2].lower() for s in subs)]

    # NB: concurrency=1 — the per-run monkey-patch hooks on the shared orchestrator
    # are not safe under parallelism (they'd cross-capture). Sequential is fine for
    # a diagnostic and keeps Serper/Scrappey load gentle.
    print(f"[Trace] Running {len(products)} products sequentially", flush=True)
    t0 = time.time()
    results = []
    for i, (cat_id, type_id, name) in enumerate(products, 1):
        r = await trace_one(orch, i, len(products), cat_id, type_id, name)
        results.append(r)
    elapsed = time.time() - t0

    active = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]

    # aggregate drain distribution — total AND "real" (excluding platform/media
    # slots like Rich-контент JSON / Ozon.Видео / #Хештеги, which no card or LLM
    # can fill and which would otherwise drown the signal in drain-B noise).
    agg_drain: Counter[str] = Counter()
    agg_drain_real: Counter[str] = Counter()
    for r in active:
        for a in r["attrs"]:
            if a["filled"] or not a.get("drain"):
                continue
            agg_drain[a["drain"]] += 1
            if not a.get("is_platform"):
                agg_drain_real[a["drain"]] += 1

    # which attributes are most often empty + their typical drain (real only,
    # so platform slots don't dominate the "top empty" lever list).
    empty_attr_counter: Counter[str] = Counter()
    empty_attr_drains: dict[str, Counter] = defaultdict(Counter)
    empty_attr_real_counter: Counter[str] = Counter()
    for r in active:
        for a in r["attrs"]:
            if a["filled"]:
                continue
            empty_attr_counter[a["name"]] += 1
            empty_attr_drains[a["name"]][a["drain"]] += 1
            if not a.get("is_platform"):
                empty_attr_real_counter[a["name"]] += 1

    n_card_found = sum(1 for r in active if r.get("card_found"))
    n_ozon_ran = sum(1 for r in active if r.get("ozon_card_ran"))
    n_ozon_hit = sum(1 for r in active if r.get("ozon_card_avs", 0) > 0)
    n_err = sum(1 for r in active if r.get("error"))

    out = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed, 1),
        "n_products": len(results),
        "n_active": len(active),
        "n_skipped": len(skipped),
        "n_errors": n_err,
        "scrappey_ozon_card_ran": n_ozon_ran,
        "scrappey_ozon_card_hit": n_ozon_hit,
        "n_card_found": n_card_found,
        "aggregate_drain": dict(agg_drain),
        "top_empty_attrs": [
            {
                "name": nm,
                "empty_count": cnt,
                "drains": dict(empty_attr_drains[nm]),
            }
            for nm, cnt in empty_attr_counter.most_common(25)
        ],
        "per_product": results,
    }

    out_dir = PROJECT_ROOT / "scripts" / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"apparel_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- console summary ---
    print("\n" + "=" * 90, flush=True)
    print(f"AGGREGATE  ({len(active)} active, {len(skipped)} skipped, "
          f"{n_err} errors, {elapsed:.0f}s)", flush=True)
    print(f"  Drain distribution: {dict(agg_drain)}", flush=True)
    print(f"  Cards found: {n_card_found}/{len(active)}  | "
          f"Ozon-card ran: {n_ozon_ran} hit: {n_ozon_hit} (Scrappey gate)", flush=True)
    print("  Top empty attributes:", flush=True)
    for nm, cnt in empty_attr_counter.most_common(10):
        print(f"    {cnt:>3}x  {nm[:40]:<40}  {dict(empty_attr_drains[nm])}", flush=True)
    print(f"\nSaved JSON: {out_path}", flush=True)

    if os.environ.get("WRITE_REPORT") == "1":
        _write_report(out, PROJECT_ROOT / "eg_apparel_trace_report.md")
        print(f"Wrote report: {PROJECT_ROOT / 'eg_apparel_trace_report.md'}", flush=True)

    return out_path


DRAIN_LEGEND = {
    "A": "no WB/Ozon card found for the product at all",
    "B": "card found but this attribute absent from it (no other source had it)",
    "C": "candidate produced but value_id resolution returned None (enum miss)",
    "D": "candidate existed but removed by a guard (gender_guard / judge / validate)",
    "E": "no source ever produced any candidate for it",
}


def _write_report(data: dict, path: Path) -> None:
    L = []
    L.append("# Apparel enrichment drain trace\n")
    L.append(f"_Generated {data['generated']} · {data['n_active']} products · "
             f"{data['elapsed_sec']}s_\n")
    L.append("")
    L.append("## Drain codes\n")
    for k, v in DRAIN_LEGEND.items():
        L.append(f"- **{k}** — {v}")
    L.append("")
    L.append("## Per-product\n")
    L.append("| # | Product | Category | filled/total | fillable | dominant | empty attrs (code) |")
    L.append("|---|---------|----------|--------------|----------|----------|--------------------|")
    for r in data["per_product"]:
        if r.get("skipped"):
            L.append(f"| {r['idx']} | {r['name']} | — | SKIPPED | — | — | {r.get('skip_reason','')} |")
            continue
        empties = [a for a in r["attrs"] if not a["filled"]]
        empties_str = ", ".join(
            f"{a['name']}({a['drain']})" for a in empties
        ) or "—"
        L.append(
            f"| {r['idx']} | {r['name']} | {r.get('leaf_category','')} | "
            f"{r['n_filled']}/{r['n_total']} | "
            f"{r['n_filled_fillable']}/{r['n_fillable']} | "
            f"{r.get('dominant_drain') or '—'} | {empties_str} |"
        )
    L.append("")
    L.append("## Aggregate drain distribution\n")
    agg = data["aggregate_drain"]
    total = sum(agg.values()) or 1
    L.append("| Code | Meaning | Count | % of empties |")
    L.append("|------|---------|-------|--------------|")
    for k in ["A", "B", "C", "D", "E"]:
        c = agg.get(k, 0)
        L.append(f"| {k} | {DRAIN_LEGEND[k]} | {c} | {c/total*100:.1f}% |")
    L.append(f"\n_Total empty attribute-slots across all products: {sum(agg.values())}_\n")
    L.append(f"_Cards found: {data['n_card_found']}/{data['n_active']} products. "
             f"Ozon-card (Scrappey) ran on {data['scrappey_ozon_card_ran']}, "
             f"produced candidates on {data['scrappey_ozon_card_hit']}._\n")
    L.append("## Attributes most often empty\n")
    L.append("| Attribute | Times empty | Typical drain |")
    L.append("|-----------|-------------|---------------|")
    for e in data["top_empty_attrs"]:
        drains = e["drains"]
        drain_str = ", ".join(f"{k}:{v}" for k, v in sorted(drains.items(), key=lambda x: -x[1]))
        L.append(f"| {e['name']} | {e['empty_count']} | {drain_str} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
