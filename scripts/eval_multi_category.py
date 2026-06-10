"""Eval: 40 товаров из разных категорий Ozon → проверка генерализации пайплайна.

Цель: убедиться, что pipeline работает не только на БП, но и на любой категории
(смартфоны, ноутбуки, одежда, игрушки, инструменты и т.д.).
Для каждого товара грузим СВОИ chars из живого ozon_dictionary.json.gz.
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

os.environ["LLM_CACHE_ENABLED"] = "1"
os.environ.setdefault("LLM_CACHE_DB", str(PROJECT_ROOT / ".llm_cache.sqlite"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Per-source logging — enabled via DEBUG_LOG=1
if os.environ.get("DEBUG_LOG") == "1":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

# НЕТ monkey-patch мини-кэша — используем живой полный словарь
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    load_ozon_dictionary,
)
from app.services.enrichment.strategies.dictionaries.ozon_field_classifier import (
    is_platform_field,
    is_not_applicable,
    ProductContext,
)
from app.services.enrichment.base import ExtractionContext, TargetAttribute
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.strategies.factory import get_strategy
from app.services.enrichment.sources.competitor_rag_source import CompetitorRagSource
from app.services.enrichment.sources.icecat_source import IceCatSource
from app.services.enrichment.sources.pdf_datasheet_source import PdfDatasheetSource
from app.services.enrichment.sources.ozon_card_source import OzonCardSource
from app.services.enrichment.sources.wb_card_source import WbCardSource
from app.services.enrichment.sources.tnved_source import TnvedSource
from app.services.enrichment.sources.yandex_market_source import YandexMarketSource
from app.services.enrichment.pipeline import YANDEX_MARKET_ENABLED

# 40 товаров — каждый в своей категории Ozon (cat_id, type_id, product_name)
PRODUCTS = [
    (15621050,  95139, "Смартфон Samsung Galaxy A55 5G 8/256GB"),
    (17028619,  91477, "Ноутбук ASUS VivoBook 15 X1504VA Core i5"),
    (17028913,  91422, "Видеокарта MSI GeForce RTX 4060 Ventus 2X 8G"),
    (17028926,  91494, "Монитор LG UltraGear 27GP850-B 27 дюймов"),
    (17028644,  94698, "Смарт-часы Apple Watch Series 9 45mm"),
    (17028908, 447870437, "Умная колонка Яндекс Станция Мини 2"),
    (17028929, 504866264, "Наушники Sony WH-1000XM5"),
    (17028648,  91869, "Веб-камера Logitech C920 HD Pro"),
    (17028925,  95141, "Микрофон HyperX QuadCast S"),
    (48233792,  95525, "Квадрокоптер DJI Mini 4 Pro"),
    (17028618,  91870, "Электронная книга PocketBook 629 Verse"),
    (17039633,  93828, "Стиральная машина Bosch WGG2540MOE"),
    (17039634,  93826, "Холодильник ATLANT ХМ 4624-101"),
    (17039628, 504866213, "Кофеварка De'Longhi EC685.M"),
    (17039643,  94972, "Микроволновая печь Samsung MS23K3513AK"),
    (17039625,  91429, "Утюг Philips DST7050/20"),
    (17039624,  91687, "Электробритва Braun Series 7 70-N1200s"),
    (17039630,  96031, "Тостер Bosch TAT3A011"),
    (17039627,  94973, "Блендер Philips HR2543/00"),
    (17039629,  94731, "Мультиварка Redmond RMC-M90"),
    (200000933, 93244, "Футболка мужская Nike Sportswear Club"),
    (200000933, 93080, "Джинсы мужские Levis 501 Original"),
    (200000933, 93137, "Куртка мужская The North Face Resolve 2"),
    (15621048,  91248, "Кроссовки Adidas Ultraboost 22"),
    (200000933, 93182, "Платье женское befree летнее"),
    (200000933, 93232, "Толстовка худи Champion Reverse Weave"),
    (17028988,  93397, "Духи Dior Sauvage Eau de Parfum 100ml"),
    (17028990,  93453, "Тушь для ресниц Maybelline Lash Sensational"),
    (17028695,  98396, "Велосипед Stels Navigator 500 V 26"),
    (17029010,  93519, "Палатка Naturehike Cloud Up 2"),
    (17028701,  93623, "Самокат городской Novatrack Polis"),
    (17028701,  93624, "Скейтборд Penny Board 22 дюйма"),
    (17028973,  92851, "Мягкая игрушка Ty Beanie Boos сова"),
    (62573858,  92952, "Конструктор LEGO Technic 42154 Ford GT"),
    (17028945,  94769, "Дрель Bosch GSB 13 RE"),
    (17028945,  94773, "Перфоратор Makita HR2470"),
    (17028732,  92462, "Сковорода Tefal Unlimited 28 см"),
    (17028622,  95859, "Акустическая гитара Yamaha F310"),
    (17028701,  96958, "Электросамокат Ninebot KickScooter E2"),
    (17028612,  91910, "Блок питания Cooler Master MWE Gold 750 V2"),
]


def build_targets(chars: list[dict]) -> list[TargetAttribute]:
    """Строим targets из chars конкретной категории.

    is_platform_field НЕ используется как фильтр — пайплайн получает ВСЕ поля,
    чтобы не срезать случайно required-поле. Классификатор — только для
    honest-метрики в отчёте.
    """
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
    """Извлечь path из словаря для (cat_id, type_id), или вернуть пустой список."""
    d = load_ozon_dictionary()
    key = f"{cat_id}:{type_id}"
    entry = d.get(key)
    if entry and entry.get("path"):
        return entry["path"]
    return []


async def main():
    # Проверяем что живой словарь загружается
    d = load_ozon_dictionary()
    print(f"[Dict] Loaded {len(d)} category:type entries from live ozon_dictionary", flush=True)

    strategy = get_strategy("ozon")
    skip_rag = os.environ.get("SKIP_RAG", "0") == "1"
    competitor_rag = None if skip_rag else CompetitorRagSource()
    if skip_rag:
        print("[Eval] SKIP_RAG=1 -> CompetitorRagSource disabled", flush=True)
    icecat = IceCatSource()
    pdf_datasheet = PdfDatasheetSource()
    ozon_card = OzonCardSource()
    wb_card = WbCardSource()
    tnved = TnvedSource()
    # YandexMarket is disabled (dead via Scrappey captcha/301) — honour the pipeline flag.
    yandex_market = YandexMarketSource() if YANDEX_MARKET_ENABLED else None
    if not YANDEX_MARKET_ENABLED:
        print("[Eval] YANDEX_MARKET_ENABLED=False -> YandexMarketSource disabled", flush=True)

    orchestrator = PipelineOrchestrator(
        strategy=strategy,
        competitor_rag_source=competitor_rag,
        icecat_source=icecat,
        pdf_datasheet_source=pdf_datasheet,
        ozon_card_source=ozon_card,
        wb_card_source=wb_card,
        yandex_market_source=yandex_market,
        tnved_source=tnved,
    )

    t_start = time.time()
    eval_limit = int(os.environ.get("EVAL_LIMIT", str(len(PRODUCTS))))
    products_to_run = PRODUCTS[:eval_limit]
    # EVAL_FILTER=подстрока1,подстрока2 — оставить только товары, чьё имя содержит
    # любую из подстрок (для точечных прогонов по подмножеству, напр. одежде).
    _filter = os.environ.get("EVAL_FILTER", "").strip()
    if _filter:
        subs = [s.strip().lower() for s in _filter.split(",") if s.strip()]
        products_to_run = [p for p in products_to_run if any(s in p[2].lower() for s in subs)]

    concurrency = int(os.environ.get("EVAL_CONCURRENCY", "8"))
    sem = asyncio.Semaphore(concurrency)
    print(
        f"[Eval] Running {len(products_to_run)} of {len(PRODUCTS)} products "
        f"(EVAL_LIMIT={eval_limit}, concurrency={concurrency})",
        flush=True,
    )

    async def _process_one(idx: int, cat_id: int, type_id: int, prod_name: str) -> dict:
        async with sem:
            print(f"  [{idx:2d}/{len(products_to_run)}] start: {prod_name[:70]}", flush=True)

            # Загружаем chars для ЭТОЙ конкретной категории из живого словаря
            chars = get_ozon_characteristics_for_type(cat_id, type_id)
            if not chars:
                print(
                    f"  [{idx:2d}/{len(products_to_run)}] SKIP — no chars for "
                    f"cat_id={cat_id} type_id={type_id}",
                    flush=True,
                )
                return {
                    "idx": idx,
                    "cat_id": cat_id,
                    "type_id": type_id,
                    "name": prod_name,
                    "skipped": True,
                    "skip_reason": f"no chars for {cat_id}:{type_id}",
                    "chars_total": 0,
                    "chars_required": 0,
                    "filled": [],
                }

            char_by_id = {c["id"]: c for c in chars}
            targets = build_targets(chars)
            category_path = _get_category_path(cat_id, type_id)

            # Берём первое слово после пробела как guess для brand (эвристика)
            words = prod_name.split()
            brand_guess = words[1] if len(words) > 1 else words[0]

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

            try:
                avs = await orchestrator.enrich(ctx, targets)
            except Exception as e:
                print(
                    f"  [{idx:2d}/{len(products_to_run)}] ! pipeline error: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
                avs = []

            print(
                f"  [{idx:2d}/{len(products_to_run)}] -> {len(avs)} attrs filled "
                f"({prod_name[:50]})",
                flush=True,
            )

            # Per-product метрики для ЭТОЙ категории
            n_req = sum(1 for c in chars if c.get("is_required"))
            n_opt = len(chars) - n_req
            filled_req = sum(
                1 for av in avs
                if char_by_id.get(av.attribute_id, {}).get("is_required")
            )
            filled_opt = sum(
                1 for av in avs
                if not char_by_id.get(av.attribute_id, {}).get("is_required")
            )

            # Build product context for N/A detection: filled attr name→value mapping
            filled_attr_map: dict[str, str] = {}
            for av in avs:
                c_name = char_by_id.get(av.attribute_id, {}).get("name", "")
                if c_name:
                    val = av.value
                    filled_attr_map[c_name.strip().lower()] = str(val) if val is not None else ""

            prod_ctx = ProductContext(
                product_name=prod_name,
                category_path=category_path,
                filled_attrs=filled_attr_map,
            )

            # Extractable optional — без платформенных полей и N/A полей
            na_excluded: list[str] = []
            extractable_opt_ids = set()
            for c in chars:
                if c.get("is_required"):
                    continue
                if is_platform_field(c):
                    continue
                if is_not_applicable(c, prod_ctx):
                    na_excluded.append(c["name"])
                    continue
                extractable_opt_ids.add(c["id"])

            if na_excluded:
                print(
                    f"  [{idx:2d}/{len(products_to_run)}] N/A excluded ({len(na_excluded)}): "
                    + ", ".join(na_excluded),
                    flush=True,
                )

            n_opt_extractable = len(extractable_opt_ids)
            filled_opt_extractable = sum(
                1 for av in avs
                if av.attribute_id in extractable_opt_ids
            )

            # value_id resolved для этой категории
            vid_total = 0
            vid_resolved = 0
            for av in avs:
                char = char_by_id.get(av.attribute_id, {})
                if char.get("values"):
                    vid_total += 1
                    if getattr(av, "value_id", None) or getattr(av, "value_ids", None):
                        vid_resolved += 1

            leaf_name = category_path[-1] if category_path else f"cat{cat_id}"

            return {
                "idx": idx,
                "cat_id": cat_id,
                "type_id": type_id,
                "name": prod_name,
                "leaf_category": leaf_name,
                "category_path": category_path,
                "skipped": False,
                "chars_total": len(chars),
                "chars_required": n_req,
                "chars_opt_extractable": n_opt_extractable,
                "filled_required": filled_req,
                "filled_optional": filled_opt,
                "filled_opt_extractable": filled_opt_extractable,
                "value_id_resolved": vid_resolved,
                "value_id_total": vid_total,
                "coverage_required": filled_req / n_req if n_req else None,
                "coverage_optional_raw": filled_opt / n_opt if n_opt else None,
                "coverage_optional_honest": (
                    filled_opt_extractable / n_opt_extractable
                    if n_opt_extractable else None
                ),
                "na_excluded_attrs": na_excluded,
                "filled": [{
                    "attribute_id": av.attribute_id,
                    "name": char_by_id.get(av.attribute_id, {}).get("name"),
                    "value": av.value,
                    "value_id": getattr(av, "value_id", None),
                    "value_ids": getattr(av, "value_ids", None),
                    "source": str(av.source),
                    "confidence": av.confidence,
                    "evidence": (av.evidence or "")[:120],
                } for av in avs],
            }

    results = await asyncio.gather(
        *(
            _process_one(i, cat_id, type_id, name)
            for i, (cat_id, type_id, name) in enumerate(products_to_run, 1)
        )
    )

    t_elapsed = time.time() - t_start

    # ---- Агрегация по всем категориям ----
    from collections import Counter, defaultdict

    active = [r for r in results if not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    n = len(active)

    if n == 0:
        print("[ERR] All products skipped — no chars found in dictionary", flush=True)
        return

    source_counts = defaultdict(int)
    for r in active:
        for f in r["filled"]:
            source_counts[f["source"]] += 1

    def safe_avg(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    avg_req = safe_avg([r["coverage_required"] for r in active])
    avg_opt_raw = safe_avg([r["coverage_optional_raw"] for r in active])
    avg_opt_honest = safe_avg([r["coverage_optional_honest"] for r in active])
    total_vid_resolved = sum(r["value_id_resolved"] for r in active)
    total_vid_total = sum(r["value_id_total"] for r in active)

    # Should-fill: доля товаров у которых required coverage >= 80%
    should_fill_threshold = 0.8
    should_fill_count = sum(
        1 for r in active
        if r["coverage_required"] is not None and r["coverage_required"] >= should_fill_threshold
    )
    should_fill_rate = should_fill_count / n if n else 0.0

    # ---- Вывод таблицы per-product ----
    print("\n" + "=" * 110, flush=True)
    print(
        f"{'#':>3}  {'Категория':<28}  {'Товар':<35}  "
        f"{'chars':>5}  {'req':>8}  {'opt':>8}  {'opt_h':>8}  {'vid':>10}",
        flush=True,
    )
    print("-" * 110, flush=True)

    for r in results:
        if r.get("skipped"):
            print(
                f"{r['idx']:>3}  {'--SKIPPED--':<28}  {r['name'][:35]:<35}  "
                f"  skip: {r['skip_reason']}",
                flush=True,
            )
            continue
        req_str = (
            f"{r['filled_required']}/{r['chars_required']}"
            if r['chars_required'] else "n/a"
        )
        opt_str = (
            f"{r['filled_optional']}/{r['chars_total'] - r['chars_required']}"
            if (r['chars_total'] - r['chars_required']) else "n/a"
        )
        opt_h_str = (
            f"{r['filled_opt_extractable']}/{r['chars_opt_extractable']}"
            if r['chars_opt_extractable'] else "n/a"
        )
        vid_str = f"{r['value_id_resolved']}/{r['value_id_total']}"
        print(
            f"{r['idx']:>3}  {r['leaf_category'][:28]:<28}  {r['name'][:35]:<35}  "
            f"{r['chars_total']:>5}  {req_str:>8}  {opt_str:>8}  {opt_h_str:>8}  {vid_str:>10}",
            flush=True,
        )

    print("=" * 110, flush=True)
    print(f"\nAGGREGATE  ({n} active + {len(skipped)} skipped, {t_elapsed:.1f}s)", flush=True)
    print(f"  Coverage required:         {avg_req*100:.1f}%", flush=True)
    print(f"  Coverage optional_raw:     {avg_opt_raw*100:.1f}%", flush=True)
    print(f"  Coverage optional_honest:  {avg_opt_honest*100:.1f}%", flush=True)
    print(f"  value_id resolution:       {total_vid_resolved}/{total_vid_total} ({total_vid_resolved/max(total_vid_total,1)*100:.1f}%)", flush=True)
    print(f"  Should-fill (req>=80%):    {should_fill_count}/{n} ({should_fill_rate*100:.1f}%)", flush=True)
    print(f"  Sources: {dict(source_counts)}", flush=True)

    if skipped:
        print(f"\nSkipped products ({len(skipped)}):", flush=True)
        for r in skipped:
            print(f"  - [{r['idx']}] {r['name']}: {r['skip_reason']}", flush=True)

    # ---- Сохранение JSON ----
    out_dir = PROJECT_ROOT / "scripts" / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"multi_cat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps({
        "n_active": n,
        "n_skipped": len(skipped),
        "elapsed_sec": t_elapsed,
        "coverage_required": avg_req,
        "coverage_optional_raw": avg_opt_raw,
        "coverage_optional_honest": avg_opt_honest,
        "value_id_resolved": total_vid_resolved,
        "value_id_total": total_vid_total,
        "should_fill_rate": should_fill_rate,
        "should_fill_count": should_fill_count,
        "should_fill_n": n,
        "sources": dict(source_counts),
        "per_product": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
