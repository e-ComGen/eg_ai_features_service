"""Eval script: LLM-backed quality evaluation of the enrichment pipeline
on 100 real power supply products from HuggingFace WB dataset.

Ozon target: description_category_id=17028612, type_id=91910 ("Блок питания компьютера").
Sources used: DescriptionSource + LlmKnowledgeSource (no vision, no web search).
LLM backend: configured via .env (default: deepseek via PROVIDER_MAIN).

Run: python scripts/eval_ozon_power_supply.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Добавляем корень проекта в sys.path для импорта app.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Enable LLM response cache for this eval run
os.environ["LLM_CACHE_ENABLED"] = "1"
os.environ.setdefault("LLM_CACHE_DB", str(PROJECT_ROOT / ".llm_cache.sqlite"))

# Загружаем .env до первого import app.*
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app import config

# Проверяем наличие ключа
if not config.OPENROUTER_API_KEY and config.PROVIDER_MAIN == "openrouter":
    print("[ERROR] OPENROUTER_API_KEY not set. Please add it to .env")
    sys.exit(1)

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    TargetAttribute,
)
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.sources import DescriptionSource, LlmKnowledgeSource
from app.services.enrichment.sources.vision_source import VisionSource
from app.services.enrichment.sources.web_search_source import WebSearchSource
from app.services.enrichment.intelligence import LlmClassifier, CostPredictor
from app.services.enrichment.strategies.factory import get_strategy
from app.services.providers.factory import get_main_manager

# ---------------------------------------------------------------------------
# Обход MemoryError при загрузке большого ozon_dictionary.json.gz
# Используем предварительно извлечённый кэш только для категории блоков питания.
# Файл ozon_power_supply_cache.json создаётся один раз extract-скриптом выше.
# ---------------------------------------------------------------------------
_CACHE_FILE = Path(__file__).parent / "eval_results" / "ozon_power_supply_cache.json"


def _load_mini_dict() -> dict:
    """Загрузить мини-словарь только для блоков питания. Падает gracefully если файла нет."""
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


# Патчим ozon_loader чтобы он использовал мини-кэш, а не полный gz-файл
import app.services.enrichment.strategies.dictionaries.ozon_loader as _ozon_loader
_mini_dict = _load_mini_dict()
if _mini_dict:
    # load_ozon_dictionary возвращает inner "categories" dict, не wrapper
    _categories = _mini_dict.get("categories", _mini_dict)
    _ozon_loader.load_ozon_dictionary = lambda: _categories
    print(f"[Dict] Using mini-cache ({len(_categories)} categories, file: {_CACHE_FILE.name})")
else:
    print(f"[Dict] WARNING: mini-cache not found at {_CACHE_FILE}, falling back to full gz (may fail)")

from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("eval_power_supply")

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
DESCRIPTION_CATEGORY_ID = 17028612
TYPE_ID = 91910
HF_DATASET = "nyuuzyou/wb-products"
TARGET_COUNT = 20
MIN_DESCRIPTION_LEN = 50
SUBJECT_FILTER = "блок питания"  # case-insensitive prefix match


# ---------------------------------------------------------------------------
# Сборка TargetAttribute[] из Ozon dictionary
# ---------------------------------------------------------------------------

def build_targets_from_ozon_dict(
    description_category_id: int,
    type_id: int,
) -> list[TargetAttribute]:
    """Построить список TargetAttribute из Ozon characteristics dictionary."""
    chars = get_ozon_characteristics_for_type(description_category_id, type_id)
    targets = []
    for c in chars:
        attr_type = {
            "String": "text",
            "Integer": "numeric",
            "Decimal": "numeric",
            "Boolean": "bool",
            "URL": "text",
            "url": "text",
        }.get(c.get("type", "String"), "text")

        # Allowed values — берём строковые значения из словаря (первые 50, чтобы не раздувать промпт)
        values_raw = c.get("values") or []
        allowed: Optional[list[str]] = None
        if values_raw:
            allowed = [str(v.get("value", "")) for v in values_raw[:50] if v.get("value")]

        targets.append(TargetAttribute(
            id=c["id"],
            name=c["name"],
            type=attr_type,
            allowed_values=allowed,
            is_collection=bool(c.get("is_collection", False)),
            description=c.get("description"),
        ))
    return targets


# ---------------------------------------------------------------------------
# Загрузка товаров из HuggingFace
# ---------------------------------------------------------------------------

def load_power_supply_products(target_count: int = TARGET_COUNT) -> list[dict]:
    """Стримить HF dataset, собрать первые target_count уникальных (по nm_id) блоков питания с описанием."""
    import sys as _sys
    _sys.stdout.reconfigure(line_buffering=True)
    print(f"[HF] Loading dataset {HF_DATASET} (streaming, dedup by nm_id)...", flush=True)
    from datasets import load_dataset
    import time as _time
    _t0 = _time.time()
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    print(f"[HF] dataset object ready in {_time.time()-_t0:.1f}s, starting iteration...", flush=True)

    products = []
    seen_nm_ids: set = set()
    scanned = 0
    duplicates_skipped = 0
    try:
        for row in ds:
            if scanned == 0:
                print(f"[HF] first row received at {_time.time()-_t0:.1f}s", flush=True)
            scanned += 1
            subj = (row.get("subj_name") or "").strip()
            desc = (row.get("description") or "").strip()

            # Фильтруем по subj_name (регистронезависимо, подстрочный матч "блок питания")
            if SUBJECT_FILTER not in subj.lower():
                continue

            # Нужно непустое описание
            if len(desc) < MIN_DESCRIPTION_LEN:
                continue

            # Дедупликация по nm_id — пропускаем повторяющиеся товары
            nm_id = row.get("nm_id")
            if nm_id is not None and nm_id in seen_nm_ids:
                duplicates_skipped += 1
                continue
            if nm_id is not None:
                seen_nm_ids.add(nm_id)

            products.append({
                "imt_id": row.get("imt_id"),
                "nm_id": nm_id,
                "imt_name": row.get("imt_name") or "",
                "subj_name": subj,
                "subj_root_name": row.get("subj_root_name") or "",
                "nm_colors_names": row.get("nm_colors_names") or "",
                "vendor_code": row.get("vendor_code") or "",
                "description": desc,
                "brand_name": row.get("brand_name") or "",
            })

            if len(products) >= target_count:
                break

            if scanned % 50_000 == 0:
                print(f"[HF]   scanned {scanned} rows, found {len(products)} unique power supplies (skipped {duplicates_skipped} duplicates)...", flush=True)

    except Exception as e:
        # HF streaming может падать с 504/timeout после того как данные уже получены
        print(f"[HF] Stream interrupted: {e} (collected {len(products)} products so far)")

    print(f"[HF] Done: scanned {scanned} rows, collected {len(products)} unique products "
          f"(skipped {duplicates_skipped} duplicates, target={target_count})")
    if len(products) < target_count:
        print(f"[HF] WARNING: only {len(products)} unique power supplies found in dataset (< {target_count})")
    return products


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _value_as_str(value) -> str:
    """Convert AttributeValue.value to searchable string for substring check."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _check_hallucination_signal(av: AttributeValue, product: dict) -> bool:
    """Heuristic: высокая уверенность, но значение нет ни в описании ни в бренде.

    Returns True если похоже на потенциальную галлюцинацию:
    - confidence > 0.85
    - само значение (как подстрока) не встречается в description + brand_name
    """
    if av.confidence <= 0.85:
        return False

    val_str = _value_as_str(av.value).strip().lower()
    if not val_str or len(val_str) < 2:
        return False

    haystack = (
        (product.get("description") or "").lower()
        + " "
        + (product.get("brand_name") or "").lower()
        + " "
        + (product.get("imt_name") or "").lower()
    )

    # Числа и короткие технические значения (типа "220В") часто не упоминаются дословно — пропускаем
    if val_str.isdigit() or len(val_str) <= 3:
        return False

    return val_str not in haystack


def _has_dict_values(char_meta: dict) -> bool:
    """Есть ли у характеристики словарные значения."""
    return bool(char_meta.get("values"))


# ---------------------------------------------------------------------------
# Основной async eval
# ---------------------------------------------------------------------------

async def run_eval():
    """Запустить полный eval pipeline."""
    start_time = time.time()

    # 1. Загрузить Ozon dictionary и построить targets
    print("[Ozon] Loading characteristics for category 17028612, type 91910...")
    all_chars = get_ozon_characteristics_for_type(DESCRIPTION_CATEGORY_ID, TYPE_ID)
    if not all_chars:
        import sys as _sys
        print("[ERROR] No Ozon characteristics found. Check ozon_dictionary.json.gz")
        _sys.exit(1)

    char_index = {c["id"]: c for c in all_chars}
    print(f"[Ozon] {len(all_chars)} characteristics loaded")

    # Строим TargetAttribute[] — все 48
    all_targets = build_targets_from_ozon_dict(DESCRIPTION_CATEGORY_ID, TYPE_ID)
    required_ids = {c["id"] for c in all_chars if c.get("is_required")}
    print(f"[Ozon] Required attrs: {len(required_ids)}, Optional: {len(all_targets) - len(required_ids)}")

    # 2. Загрузить продукты
    products = load_power_supply_products(TARGET_COUNT)
    if not products:
        print("[ERROR] No products loaded from HuggingFace")
        sys.exit(1)

    # 3. Инициализировать pipeline
    print(f"[Pipeline] Initializing (PROVIDER_MAIN={config.PROVIDER_MAIN}, MAIN_MODEL={config.MAIN_MODEL})...")
    llm_manager = get_main_manager()

    # Инициализируем sources — только Description + LlmKnowledge (без vision/web_search)
    desc_source = DescriptionSource(llm_manager=llm_manager)
    knowledge_source = LlmKnowledgeSource(llm_manager=llm_manager)

    # Строим "заглушки" для неиспользуемых sources (они не будут вызываться — no image_urls, no web config)
    # PipelineOrchestrator сам обработает is_applicable=False для vision/websearch
    strategy = get_strategy("ozon")

    orchestrator = PipelineOrchestrator(
        description_source=desc_source,
        knowledge_source=knowledge_source,
        strategy=strategy,
    )

    # 4. Обработка каждого товара
    print(f"\n[Eval] Processing {len(products)} products...")
    print(f"[Cache] LLM_CACHE_ENABLED={os.environ.get('LLM_CACHE_ENABLED','0')}, "
          f"DB={os.environ.get('LLM_CACHE_DB', '.llm_cache.sqlite')}")

    results = []
    total_llm_calls = 0

    # Track cache stats by monitoring token counts:
    # cache hit → tokens=0; real call → tokens>0
    cache_hits = 0
    cache_misses = 0

    PILOT_SIZE = 5  # process first N products and print interim summary

    for idx, product in enumerate(products):
        nm_id = product.get("nm_id", idx)
        imt_name = product.get("imt_name", "")
        print(f"  [{idx+1:3d}/{len(products)}] nm_id={nm_id} | {imt_name[:60]}")

        try:
            context = ExtractionContext(
                product_id=int(nm_id) if nm_id else idx,
                product_name=imt_name,
                product_description=product.get("description"),
                category_id=DESCRIPTION_CATEGORY_ID,
                ozon_type_id=TYPE_ID,
                brand=product.get("brand_name") or None,
                marketplace="ozon",
                image_urls=[],   # нет изображений
                source_urls=[],  # нет веб-ссылок
            )

            # Запускаем pipeline
            extracted: list[AttributeValue] = await orchestrator.enrich(context, list(all_targets))
            calls_this_product = context.llm_calls_so_far
            total_llm_calls += calls_this_product

            # Cache hit detection: pipeline increments llm_calls_so_far only on real calls.
            # We compare with expected calls per product (typically 2: description + knowledge).
            # Simpler approach: track via token totals is not available here, so we use
            # context.llm_calls_so_far — 0 means full cache hit, partial means partial hit.
            if calls_this_product == 0:
                cache_hits += 1
            else:
                cache_misses += 1

            # Собираем per-product eval
            attr_results = []
            for av in extracted:
                char_meta = char_index.get(av.attribute_id, {})
                has_dict = _has_dict_values(char_meta)

                # value_id resolution: для словарных атрибутов проверяем что value_id заполнен
                if has_dict:
                    if av.is_collection:
                        value_id_resolved = bool(av.value_ids)
                    else:
                        value_id_resolved = av.value_id is not None
                else:
                    value_id_resolved = None  # N/A для бессловарных

                # value_in_dict: для словарных проверяем case-insensitive матч
                value_in_dict = None
                if has_dict:
                    dict_values_lower = {
                        str(v.get("value", "")).lower()
                        for v in char_meta.get("values", [])
                    }
                    if av.is_collection and isinstance(av.value, list):
                        value_in_dict = all(str(v).lower() in dict_values_lower for v in av.value)
                    else:
                        value_in_dict = str(av.value).lower() in dict_values_lower

                # Hallucination signal
                hallucination_signal = _check_hallucination_signal(av, product)

                attr_results.append({
                    "attribute_id": av.attribute_id,
                    "attribute_name": char_meta.get("name", str(av.attribute_id)),
                    "value": _value_as_str(av.value),
                    "confidence": round(av.confidence, 3),
                    "source": av.source,
                    "is_required": av.attribute_id in required_ids,
                    "has_dict_values": has_dict,
                    "value_id_resolved": value_id_resolved,
                    "value_in_dict": value_in_dict,
                    "hallucination_signal": hallucination_signal,
                    "evidence": av.evidence,
                })

            # Считаем coverage
            filled_ids = {av.attribute_id for av in extracted}
            required_filled = len(required_ids & filled_ids)
            optional_ids = {t.id for t in all_targets if t.id not in required_ids}
            optional_filled = len(optional_ids & filled_ids)

            results.append({
                "product": {
                    "nm_id": nm_id,
                    "imt_name": imt_name,
                    "brand_name": product.get("brand_name", ""),
                    "subj_name": product.get("subj_name", ""),
                    "description_len": len(product.get("description", "")),
                },
                "coverage": {
                    "required_total": len(required_ids),
                    "required_filled": required_filled,
                    "required_pct": round(100 * required_filled / len(required_ids), 1) if required_ids else 0,
                    "optional_total": len(optional_ids),
                    "optional_filled": optional_filled,
                    "optional_pct": round(100 * optional_filled / len(optional_ids), 1) if optional_ids else 0,
                },
                "attributes": attr_results,
                "llm_calls": calls_this_product,
            })

        except Exception as e:
            logger.exception(f"Product {nm_id} failed: {e}")
            results.append({
                "product": {"nm_id": nm_id, "imt_name": imt_name},
                "error": str(e),
                "attributes": [],
                "llm_calls": 0,
            })
            cache_misses += 1  # count as miss since we attempted a real call

        # Rate limit pause
        await asyncio.sleep(0.1)

        # Smart pilot: after first PILOT_SIZE products, print interim summary
        if idx + 1 == PILOT_SIZE:
            print(f"\n{'─'*50}")
            print(f"PILOT SUMMARY (first {PILOT_SIZE} products)")
            pilot_ok = [r for r in results if "error" not in r]
            if pilot_ok:
                p_req = sum(r["coverage"]["required_pct"] for r in pilot_ok) / len(pilot_ok)
                p_opt = sum(r["coverage"]["optional_pct"] for r in pilot_ok) / len(pilot_ok)
                print(f"  Required coverage: {p_req:.1f}%, Optional: {p_opt:.1f}%")
            print(f"  LLM calls so far: {total_llm_calls}")
            print(f"  Cache: {cache_hits} hits / {cache_hits + cache_misses} products")
            # Cost check: ~2K tokens × $0.27/M input blended → abort if approaching $1
            est_cost_so_far = (total_llm_calls * 2000 / 1_000_000) * 0.50
            print(f"  Est. cost so far: ~${est_cost_so_far:.4f}")
            remaining = len(products) - PILOT_SIZE
            est_cost_remaining = (remaining * (total_llm_calls / max(len(pilot_ok), 1)) * 2000 / 1_000_000) * 0.50
            print(f"  Projected total cost: ~${est_cost_so_far + est_cost_remaining:.4f}")
            if est_cost_so_far + est_cost_remaining > 1.0:
                print(f"\n[COST GUARD] Projected spend exceeds $1.00! Stopping after pilot.")
                print(f"[COST GUARD] To continue anyway, set COST_LIMIT_USD env var > 1.0")
                cost_limit = float(os.environ.get("COST_LIMIT_USD", "1.0"))
                if est_cost_so_far + est_cost_remaining > cost_limit:
                    break
            print(f"{'─'*50}\n")

    elapsed = time.time() - start_time

    # 5. Aggregate report
    print("\n" + "=" * 70)
    print("AGGREGATE REPORT")
    print("=" * 70)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    total_products_seen = cache_hits + cache_misses
    cache_hit_rate = (100.0 * cache_hits / total_products_seen) if total_products_seen else 0.0
    print(f"Products processed: {len(results)} total, {len(successful)} successful, {len(failed)} failed")
    print(f"  Unique products (dedup by nm_id): {len(products)}")
    print(f"Total LLM calls (cache misses only): {total_llm_calls}")
    print(f"Cache: {cache_hits} hits / {total_products_seen} products = {cache_hit_rate:.1f}% hit rate")
    print(f"Elapsed: {elapsed:.1f}s")

    if not successful:
        print("[ERROR] No successful results to aggregate!")
        return

    # Coverage avg
    avg_req_pct = sum(r["coverage"]["required_pct"] for r in successful) / len(successful)
    avg_opt_pct = sum(r["coverage"]["optional_pct"] for r in successful) / len(successful)
    print(f"\nCoverage (avg across {len(successful)} products):")
    print(f"  Required attrs:  {avg_req_pct:.1f}%")
    print(f"  Optional attrs:  {avg_opt_pct:.1f}%")

    # Per-attribute fill rate
    attr_fill_count: dict[int, int] = {}
    attr_names: dict[int, str] = {c["id"]: c["name"] for c in all_chars}
    for r in successful:
        for av_r in r["attributes"]:
            aid = av_r["attribute_id"]
            attr_fill_count[aid] = attr_fill_count.get(aid, 0) + 1

    total_prods = len(successful)
    fill_rates = {
        aid: (count / total_prods * 100)
        for aid, count in attr_fill_count.items()
    }

    # Always-filled (fill rate >= 90%)
    always_filled = sorted(
        [(aid, rate) for aid, rate in fill_rates.items() if rate >= 90],
        key=lambda x: -x[1],
    )[:5]
    # Never-filled (fill rate == 0%)
    never_filled = [
        (t.id, attr_names.get(t.id, str(t.id)))
        for t in all_targets
        if fill_rates.get(t.id, 0) == 0
    ]

    print(f"\nTop 5 always-filled attrs (fill rate >= 90%):")
    for aid, rate in always_filled:
        print(f"  {attr_names.get(aid, aid):45s}  {rate:.0f}%")

    print(f"\nNever-filled attrs ({len(never_filled)} total):")
    for aid, name in never_filled[:10]:
        print(f"  {name[:45]:45s}  (id={aid})")
    if len(never_filled) > 10:
        print(f"  ... and {len(never_filled) - 10} more")

    # Value ID resolution rate
    all_attr_vals = [av for r in successful for av in r["attributes"]]
    dict_backed = [av for av in all_attr_vals if av["has_dict_values"]]
    if dict_backed:
        resolved = [av for av in dict_backed if av["value_id_resolved"] is True]
        resolution_rate = 100 * len(resolved) / len(dict_backed)
        print(f"\nValue ID resolution rate: {len(resolved)}/{len(dict_backed)} = {resolution_rate:.1f}%")
    else:
        print("\nValue ID resolution: no dict-backed attributes found")
        resolution_rate = 0

    # value_in_dict check
    in_dict = [av for av in dict_backed if av["value_in_dict"] is True]
    if dict_backed:
        in_dict_rate = 100 * len(in_dict) / len(dict_backed)
        print(f"Value-in-dict rate:     {len(in_dict)}/{len(dict_backed)} = {in_dict_rate:.1f}%")

    # Hallucination signals
    hallucinations = [av for av in all_attr_vals if av["hallucination_signal"]]
    print(f"\nHallucination signals: {len(hallucinations)} (high-confidence values not found in description/brand)")
    if hallucinations[:5]:
        print("  Examples:")
        for h in hallucinations[:5]:
            print(f"    attr={h['attribute_name'][:30]:30s}  val={h['value'][:30]:30s}  conf={h['confidence']:.2f}")

    # Cost estimate (DeepSeek deepseek-chat: ~$0.27/M input, ~$1.10/M output — rough)
    # Assume ~2000 tokens per LLM call (1500 input + 500 output)
    avg_tokens_per_call = 2000
    estimated_tokens = total_llm_calls * avg_tokens_per_call
    # DeepSeek V3 (chat) pricing: $0.27/M input tokens, $1.10/M output tokens
    # Blended estimate: ~$0.50/M tokens
    estimated_cost_usd = (estimated_tokens / 1_000_000) * 0.50
    # Tokens saved by cache
    cache_saved_tokens = cache_hits * avg_tokens_per_call * 2  # rough: 2 calls per product
    cache_saved_usd = (cache_saved_tokens / 1_000_000) * 0.50
    print(f"\nCost estimate (ACTUAL SPEND = cache misses only):")
    print(f"  Real LLM calls (cache misses): {total_llm_calls}")
    print(f"  Cache hits (free): {cache_hits}")
    print(f"  Estimated tokens spent: ~{estimated_tokens:,}")
    print(f"  Estimated cost SPENT: ~${estimated_cost_usd:.4f} (DeepSeek blended $0.50/M)")
    print(f"  Estimated cost SAVED by cache: ~${cache_saved_usd:.4f} (~{cache_saved_tokens:,} tokens)")

    # 6. Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "eval_results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"power_supply_{timestamp}.json"

    report = {
        "meta": {
            "timestamp": timestamp,
            "dataset": HF_DATASET,
            "description_category_id": DESCRIPTION_CATEGORY_ID,
            "type_id": TYPE_ID,
            "provider": config.PROVIDER_MAIN,
            "model": config.MAIN_MODEL,
            "products_requested": TARGET_COUNT,
            "products_evaluated": len(results),
            "products_successful": len(successful),
            "products_failed": len(failed),
            "products_unique_by_nm_id": len(products),
            "total_llm_calls": total_llm_calls,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate_pct": round(cache_hit_rate, 1),
            "elapsed_seconds": round(elapsed, 1),
        },
        "aggregate": {
            "avg_required_coverage_pct": round(avg_req_pct, 1),
            "avg_optional_coverage_pct": round(avg_opt_pct, 1),
            "value_id_resolution_rate_pct": round(resolution_rate, 1) if dict_backed else None,
            "hallucination_signal_count": len(hallucinations),
            "estimated_cost_usd": round(estimated_cost_usd, 4),
            "estimated_tokens": estimated_tokens,
            "cache_saved_usd": round(cache_saved_usd, 4),
            "cache_saved_tokens": cache_saved_tokens,
            "always_filled_attrs": [
                {"id": aid, "name": attr_names.get(aid, str(aid)), "fill_pct": round(rate, 1)}
                for aid, rate in always_filled
            ],
            "never_filled_attrs": [
                {"id": aid, "name": name}
                for aid, name in never_filled
            ],
        },
        "per_product": results,
    }

    try:
        json_str = json.dumps(report, ensure_ascii=False, indent=2)
        output_path.write_text(json_str, encoding="utf-8")
        print(f"\nResults saved to: {output_path}")
        import sys
        sys.stdout.flush()
    except Exception as e:
        print(f"\n[ERROR] Failed to write results: {e}", flush=True)
        # Fallback: write to a temp location
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', prefix='power_supply_',
            delete=False, encoding='utf-8',
        )
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        print(f"[FALLBACK] Results written to: {tmp.name}", flush=True)
        output_path = tmp.name
    return output_path


def main():
    result_path = asyncio.run(run_eval())
    # Записываем путь к результату в маркер-файл для отслеживания
    marker = Path(__file__).parent / "eval_results" / "last_run.txt"
    try:
        marker.write_text(str(result_path) + "\n", encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
