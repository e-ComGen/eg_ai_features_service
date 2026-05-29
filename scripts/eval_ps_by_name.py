"""Eval: 20 БП по одному названию (без description) → pipeline.

Цель: понять что pipeline вытянет имея только product_name.
DescriptionSource будет skip (нет описания), но LlmKnowledgeSource + WebSearchSource отработают.
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
sys.stdout.reconfigure(line_buffering=True)

os.environ["LLM_CACHE_ENABLED"] = "1"
os.environ.setdefault("LLM_CACHE_DB", str(PROJECT_ROOT / ".llm_cache.sqlite"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Мини-словарь patch
_CACHE_FILE = PROJECT_ROOT / "scripts" / "eval_results" / "ozon_power_supply_cache.json"
import app.services.enrichment.strategies.dictionaries.ozon_loader as _loader
if _CACHE_FILE.exists():
    _mini = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    _cats = _mini.get("categories", _mini)
    _loader.load_ozon_dictionary = lambda: _cats
    print(f"[Dict] mini-cache: {len(_cats)} categories", flush=True)

from app.services.enrichment.strategies.dictionaries.ozon_loader import get_ozon_characteristics_for_type
from app.services.enrichment.base import ExtractionContext, TargetAttribute
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.strategies.factory import get_strategy
from app.services.enrichment.sources.competitor_rag_source import CompetitorRagSource
from app.services.enrichment.sources.icecat_source import IceCatSource
from app.services.enrichment.sources.pdf_datasheet_source import PdfDatasheetSource
from app.services.enrichment.sources.ozon_card_source import OzonCardSource

DESCRIPTION_CATEGORY_ID = 17028612
TYPE_ID = 91910

# 20 реальных БП — известные модели разных брендов
PRODUCTS = [
    "Блок питания Cooler Master MWE Gold 750 V2 Full Modular 750W ATX",
    "Блок питания Deepcool PM650D 650W 80+ Bronze",
    "Блок питания Zalman ZM600-XEII 600W 80+ Bronze",
    "Блок питания ASUS ROG STRIX 850G 850W 80+ Gold",
    "Блок питания AeroCool Cylon 600W ATX",
    "Блок питания be quiet! Pure Power 11 600W 80+ Gold",
    "Блок питания Corsair RM750x 750W 80+ Gold Fully Modular",
    "Блок питания Seasonic FOCUS GX-650 650W 80+ Gold",
    "Блок питания EVGA SuperNOVA 750 G6 750W 80+ Gold",
    "Блок питания Thermaltake Toughpower GF1 850W 80+ Gold",
    "Блок питания Cooler Master V850 SFX Gold 850W",
    "Блок питания FSP Hyper K 600W 80+ Bronze",
    "Блок питания Chieftec PolarFrost 650W 80+ Gold",
    "Блок питания NZXT C850 Gold 850W ATX 3.0",
    "Блок питания MSI MPG A850G PCIE5 850W 80+ Gold",
    "Блок питания Gigabyte UD850GM 850W 80+ Gold Modular",
    "Блок питания Lian Li SP750 SFX 750W 80+ Gold",
    "Блок питания SilverStone DA850 GOLD 850W ATX 3.0",
    "Блок питания XPG CYBERCORE 1000W 80+ Platinum",
    "Блок питания Deepcool PX1000G 1000W 80+ Gold Gen.5",
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
        ))
    return out


async def main():
    chars = get_ozon_characteristics_for_type(DESCRIPTION_CATEGORY_ID, TYPE_ID)
    if not chars:
        print("[ERR] no chars from dict", flush=True)
        return
    print(f"[Ozon] {len(chars)} characteristics ({sum(1 for c in chars if c.get('is_required'))} required)", flush=True)
    char_by_id = {c["id"]: c for c in chars}
    targets = build_targets(chars)

    strategy = get_strategy("ozon")
    skip_rag = os.environ.get("SKIP_RAG", "0") == "1"
    competitor_rag = None if skip_rag else CompetitorRagSource()
    if skip_rag:
        print("[Eval] SKIP_RAG=1 -> CompetitorRagSource disabled", flush=True)
    icecat = IceCatSource()  # читает ICECAT_EMAIL, ICECAT_TOKEN из .env
    pdf_datasheet = PdfDatasheetSource()  # Serper + Gemini 2.5 Flash PDF native
    ozon_card = OzonCardSource(apify_token=os.environ.get("APIFY_TOKEN"))  # Apify ozon-scraper-pro
    orchestrator = PipelineOrchestrator(
        strategy=strategy,
        competitor_rag_source=competitor_rag,
        icecat_source=icecat,
        pdf_datasheet_source=pdf_datasheet,
        ozon_card_source=ozon_card,
    )

    results = []
    t_start = time.time()
    eval_limit = int(os.environ.get("EVAL_LIMIT", str(len(PRODUCTS))))
    products_to_run = PRODUCTS[:eval_limit]
    print(f"[Eval] Running {len(products_to_run)} of {len(PRODUCTS)} products (EVAL_LIMIT={eval_limit})", flush=True)
    for i, name in enumerate(products_to_run, 1):
        # Бренд можно извлечь грубой эвристикой (второе слово)
        brand_guess = name.replace("Блок питания ", "").split()[0]
        print(f"  [{i:2d}/{len(PRODUCTS)}] {name[:70]}", flush=True)
        ctx = ExtractionContext(
            product_id=i,
            product_name=name,
            product_description="",
            brand=brand_guess,
            category_id=str(DESCRIPTION_CATEGORY_ID),
            category_path=["Электроника", "Блоки питания ПК", "Блок питания компьютера"],
            image_urls=[],
            marketplace="ozon",
            ozon_type_id=TYPE_ID,
        )
        try:
            avs = await orchestrator.enrich(ctx, targets)
        except Exception as e:
            print(f"    ! pipeline error: {type(e).__name__}: {e}", flush=True)
            avs = []
        print(f"    -> {len(avs)} attrs filled", flush=True)
        results.append({
            "name": name,
            "filled": [{
                "attribute_id": av.attribute_id,
                "name": char_by_id.get(av.attribute_id, {}).get("name"),
                "value": av.value,
                "value_id": av.value_id,
                "source": str(av.source),
                "confidence": av.confidence,
                "evidence": (av.evidence or "")[:120],
            } for av in avs],
        })

    t_elapsed = time.time() - t_start

    # Aggregate
    from collections import Counter, defaultdict
    n = len(results)
    attr_fill = Counter()
    source_counts = defaultdict(int)
    value_id_resolved = 0
    value_id_total = 0
    for r in results:
        for f in r["filled"]:
            attr_fill[f["attribute_id"]] += 1
            source_counts[f["source"]] += 1
            if char_by_id.get(f["attribute_id"], {}).get("values"):
                value_id_total += 1
                if f["value_id"]:
                    value_id_resolved += 1

    total_req = sum(1 for c in chars if c.get("is_required"))
    total_opt = len(chars) - total_req
    avg_req = sum(sum(1 for f in r["filled"] if char_by_id.get(f["attribute_id"], {}).get("is_required")) for r in results) / n / total_req if total_req else 0
    avg_opt = sum(sum(1 for f in r["filled"] if not char_by_id.get(f["attribute_id"], {}).get("is_required")) for r in results) / n / total_opt if total_opt else 0

    print("\n" + "=" * 70, flush=True)
    print(f"AGGREGATE  ({n} products, {t_elapsed:.1f}s)", flush=True)
    print("=" * 70, flush=True)
    print(f"Coverage required: {avg_req*100:.1f}%  optional: {avg_opt*100:.1f}%", flush=True)
    print(f"Sources: {dict(source_counts)}", flush=True)
    print(f"value_id resolution: {value_id_resolved}/{value_id_total} ({value_id_resolved/max(value_id_total,1)*100:.1f}%)", flush=True)
    print(f"\nTop attrs filled (fill rate across {n} products):", flush=True)
    for aid, count in attr_fill.most_common(15):
        nm = char_by_id.get(aid, {}).get("name", f"id={aid}")
        req = " [REQ]" if char_by_id.get(aid, {}).get("is_required") else ""
        print(f"  {nm[:45]:<45} {count}/{n} ({count/n*100:.0f}%){req}", flush=True)
    never = [c for c in chars if c["id"] not in attr_fill]
    print(f"\nNever filled ({len(never)}/{len(chars)}):", flush=True)
    for c in never[:15]:
        req = " [REQ]" if c.get("is_required") else ""
        print(f"  - {c['name']}{req}", flush=True)

    # IceCat-specific stats
    from app.services.enrichment.sources.icecat_source import closed_brands as icecat_closed, open_brands as icecat_open
    icecat_keys = [k for k in source_counts if "icecat" in k.lower()]
    icecat_total = sum(source_counts[k] for k in icecat_keys)
    products_with_icecat = [
        r for r in results
        if any("icecat" in (f["source"] or "").lower() for f in r["filled"])
    ]
    print(f"\nIceCat stats:", flush=True)
    print(f"  Total IceCat fills: {icecat_total}  avg/product: {icecat_total/n:.2f}", flush=True)
    print(f"  Products with >=1 IceCat fill: {len(products_with_icecat)}/{n}", flush=True)
    print(f"  Brands hit (200): {dict(icecat_open)}", flush=True)
    print(f"  Brands blocked (403 Full IceCat only): {dict(icecat_closed)}", flush=True)
    if products_with_icecat:
        print(f"\nSample 3 products where IceCat fired:", flush=True)
        for r in products_with_icecat[:3]:
            icecat_fills = [f for f in r["filled"] if "icecat" in (f["source"] or "").lower()]
            print(f"  Product: {r['name'][:70]}", flush=True)
            for f in icecat_fills[:5]:
                print(f"    {f['name']}: {f['value']}  (conf={f['confidence']:.2f}, {f['evidence'][:60]})", flush=True)

    # RAG-specific stats
    rag_source_key = "Source.COMPETITOR_RAG"
    # normalize key — may be 'competitor_rag' or 'Source.COMPETITOR_RAG'
    rag_keys = [k for k in source_counts if "competitor_rag" in k.lower()]
    rag_total = sum(source_counts[k] for k in rag_keys)
    products_with_rag = [
        r for r in results
        if any("competitor_rag" in (f["source"] or "").lower() for f in r["filled"])
    ]
    avg_rag_per_product = rag_total / n
    print(f"\nRAG stats:", flush=True)
    print(f"  Total RAG fills: {rag_total}  avg/product: {avg_rag_per_product:.2f}", flush=True)
    print(f"  Products with >=1 RAG fill: {len(products_with_rag)}/{n}", flush=True)
    if products_with_rag:
        print(f"\nSample 3 products where RAG fired:", flush=True)
        for r in products_with_rag[:3]:
            rag_fills = [f for f in r["filled"] if "competitor_rag" in (f["source"] or "").lower()]
            print(f"  Product: {r['name'][:70]}", flush=True)
            for f in rag_fills[:5]:
                print(f"    {f['name']}: {f['value']}  (conf={f['confidence']:.2f}, {f['evidence'][:60]})", flush=True)

    out_path = PROJECT_ROOT / "scripts" / "eval_results" / f"ps_by_name_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps({
        "n": n,
        "elapsed_sec": t_elapsed,
        "coverage_required": avg_req,
        "coverage_optional": avg_opt,
        "sources": dict(source_counts),
        "value_id_resolved": value_id_resolved,
        "value_id_total": value_id_total,
        "products": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
