"""Eval script: WebSearch-enabled evaluation of the enrichment pipeline.

Tests all three sources: DescriptionSource + LlmKnowledgeSource + WebSearchSource
on the first 20 unique power supply products from HuggingFace WB dataset.

Key insight: shows coverage DELTA — how much WebSearch adds on top of Desc+Knowledge.

Ozon target: description_category_id=17028612, type_id=91910 ("Блок питания компьютера").
Run: python scripts/eval_ozon_power_supply_websearch.py

Cost guard: stop at 10 products if projected spend > $6.
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

# Добавляем корень проекта в sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Cache: Description + Knowledge calls должны быть бесплатны (cache hit)
os.environ["LLM_CACHE_ENABLED"] = "1"
os.environ.setdefault("LLM_CACHE_DB", str(PROJECT_ROOT / ".llm_cache.sqlite"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from app import config

# ---------------------------------------------------------------------------
# Проверка необходимых ключей
# ---------------------------------------------------------------------------
if not config.SERPER_API_KEY:
    print("[ERROR] SERPER_API_KEY not set. Please add it to .env")
    sys.exit(1)
if not config.OPENROUTER_API_KEY and not config.DEEPSEEK_API_KEY:
    print("[ERROR] Neither OPEN_ROUTER_API_KEY nor DEEP_SEEK_API_KEY set. Please add one to .env")
    sys.exit(1)

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    TargetAttribute,
    Source,
)
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.sources import DescriptionSource, LlmKnowledgeSource
from app.services.enrichment.sources.web_search_source import WebSearchSource
from app.services.enrichment.intelligence import LlmClassifier, CostPredictor
from app.services.enrichment.strategies.factory import get_strategy
from app.services.providers.factory import get_main_manager

# ---------------------------------------------------------------------------
# Патч мини-словаря (такой же как в оригинальном eval-скрипте)
# ---------------------------------------------------------------------------
_CACHE_FILE = Path(__file__).parent / "eval_results" / "ozon_power_supply_cache.json"

def _load_mini_dict() -> dict:
    """Загрузить мини-словарь только для блоков питания."""
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    return {}

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
logger = logging.getLogger("eval_ps_websearch")

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
DESCRIPTION_CATEGORY_ID = 17028612
TYPE_ID = 91910
HF_DATASET = "nyuuzyou/wb-products"
TARGET_COUNT = 20          # первые 20 уникальных товаров
MIN_DESCRIPTION_LEN = 50
SUBJECT_FILTER = "блок питания"
COST_LIMIT_USD = float(os.environ.get("COST_LIMIT_USD", "6.0"))
STOP_AT_ON_LIMIT = 10      # если projected cost > $6, остановиться после 10 товаров

# Pricing constants (USD per 1M tokens)
# DeepSeek V3 (chat): input $0.27, output $1.10 → blended ~$0.50/M
# Gemini Flash via OpenRouter: input $0.075, output $0.30 → blended ~$0.12/M
# Serper: $0.001 per search
PRICE_DEEPSEEK_BLENDED_PER_M = 0.50
PRICE_GEMINI_FLASH_BLENDED_PER_M = 0.12
PRICE_SERPER_PER_SEARCH = 0.001

# Approximate token counts per call type
TOKENS_DESCRIPTION_CALL = 2000    # описание → атрибуты
TOKENS_KNOWLEDGE_CALL = 1800      # knowledge call
TOKENS_WS_SUMMARY_CALL = 1200     # Serper snippets → summary (gemini flash)
TOKENS_WS_EXTRACTION_CALL = 2000  # summary → attributes (deepseek)


# ---------------------------------------------------------------------------
# Построение TargetAttribute[] из Ozon dictionary
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
    """Стримить HF dataset, собрать первые target_count уникальных блоков питания."""
    print(f"[HF] Loading dataset {HF_DATASET} (streaming, dedup by nm_id, target={target_count})...")
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    products = []
    seen_nm_ids: set = set()
    scanned = 0
    duplicates_skipped = 0
    try:
        for row in ds:
            scanned += 1
            subj = (row.get("subj_name") or "").strip()
            desc = (row.get("description") or "").strip()

            if SUBJECT_FILTER not in subj.lower():
                continue
            if len(desc) < MIN_DESCRIPTION_LEN:
                continue

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
                "brand_name": row.get("brand_name") or "",
                "description": desc,
            })

            if len(products) >= target_count:
                break

            if scanned % 50_000 == 0:
                print(f"[HF]   scanned {scanned}, found {len(products)} unique power supplies...")

    except Exception as e:
        print(f"[HF] Stream interrupted: {e} (collected {len(products)} products)")

    print(f"[HF] Done: scanned {scanned} rows, collected {len(products)} unique products "
          f"(skipped {duplicates_skipped} duplicates)")
    return products


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------
def _value_as_str(value) -> str:
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def _check_hallucination_signal(av: AttributeValue, product: dict, web_summary: Optional[str] = None) -> bool:
    """Heuristic: высокая уверенность, значение нет ни в описании ни в web summary."""
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
    if web_summary:
        haystack += " " + web_summary.lower()

    if val_str.isdigit() or len(val_str) <= 3:
        return False
    return val_str not in haystack


def _has_dict_values(char_meta: dict) -> bool:
    return bool(char_meta.get("values"))


# ---------------------------------------------------------------------------
# Подсчёт coverage по набору заполненных attributes
# ---------------------------------------------------------------------------
def _coverage(filled_ids: set[int], required_ids: set[int], all_targets: list[TargetAttribute]) -> dict:
    optional_ids = {t.id for t in all_targets if t.id not in required_ids}
    req_filled = len(required_ids & filled_ids)
    opt_filled = len(optional_ids & filled_ids)
    return {
        "required_total": len(required_ids),
        "required_filled": req_filled,
        "required_pct": round(100 * req_filled / len(required_ids), 1) if required_ids else 0.0,
        "optional_total": len(optional_ids),
        "optional_filled": opt_filled,
        "optional_pct": round(100 * opt_filled / len(optional_ids), 1) if optional_ids else 0.0,
    }


# ---------------------------------------------------------------------------
# Класс-обёртка для принудительного запуска WebSearch без CostPredictor
# ---------------------------------------------------------------------------
class ForcedWebSearchPredictor(CostPredictor):
    """Всегда возвращает True — обходим cost gate для eval."""

    async def is_web_search_worth(self, context, targets) -> bool:
        return True


# ---------------------------------------------------------------------------
# Eval одного продукта: два прохода
# ---------------------------------------------------------------------------
async def eval_product(
    product: dict,
    idx: int,
    total: int,
    all_targets: list[TargetAttribute],
    required_ids: set[int],
    char_index: dict,
    llm_manager,
    web_source: WebSearchSource,
    strategy,
) -> dict:
    """
    Два прохода для одного товара:
    1. Только Description + Knowledge (pipeline без web)
    2. WebSearch для оставшихся unfilled атрибутов

    Возвращает структурированный результат с before/after coverage и cost breakdown.
    """
    nm_id = product.get("nm_id", idx)
    imt_name = product.get("imt_name", "")
    print(f"  [{idx+1:3d}/{total}] nm_id={nm_id} | {imt_name[:60]}")

    base_context = ExtractionContext(
        product_id=int(nm_id) if nm_id else idx,
        product_name=imt_name,
        product_description=product.get("description"),
        category_id=DESCRIPTION_CATEGORY_ID,
        ozon_type_id=TYPE_ID,
        brand=product.get("brand_name") or None,
        marketplace="ozon",
        image_urls=[],
        source_urls=[],
        max_cost_usd=99.0,   # не ограничиваем — cost guard на уровне eval
    )

    # --- Проход 1: Description + Knowledge (должны хитить кэш) ---
    desc_source = DescriptionSource(llm_manager=llm_manager)
    knowledge_source = LlmKnowledgeSource(llm_manager=llm_manager)

    # Строим pipeline без web search (через ForcedWebSearchPredictor с пустым websearch_source)
    # Для pass-1 websearch_source в orchestrator не нужен — используем strategy чтобы получить
    # нормализованные targets, потом вызываем sources напрямую
    norm_strategy = strategy

    # Создаём orchestrator только для Description + Knowledge (без WebSearch)
    # Передаём None-заглушку websearch_source — pipeline сам не дойдёт до него
    # поскольку routing не вернёт WEB_SEARCH для этого orchestrator
    ctx1 = ExtractionContext(
        product_id=base_context.product_id,
        product_name=base_context.product_name,
        product_description=base_context.product_description,
        category_id=base_context.category_id,
        ozon_type_id=base_context.ozon_type_id,
        brand=base_context.brand,
        marketplace=base_context.marketplace,
        image_urls=[],
        source_urls=[],
        max_cost_usd=99.0,
    )

    # Специальный orchestrator без WebSearch — нет websearch_source в конструкторе
    # (используется дефолтный WebSearchSource, но он не пройдёт cost gate если не force)
    # Проще: создаём PipelineOrchestrator с принудительным is_web_search_worth=False
    class _NeverWebPredictor(CostPredictor):
        async def is_web_search_worth(self, ctx, tgts) -> bool:
            return False  # Никогда не запускаем web search в pass-1

    orchestrator_no_web = PipelineOrchestrator(
        description_source=DescriptionSource(llm_manager=llm_manager),
        knowledge_source=LlmKnowledgeSource(llm_manager=llm_manager),
        strategy=norm_strategy,
        cost_predictor=_NeverWebPredictor(),
    )

    pass1_values: list[AttributeValue] = []
    pass1_llm_calls = 0
    try:
        pass1_values = await orchestrator_no_web.enrich(ctx1, list(all_targets))
        pass1_llm_calls = ctx1.llm_calls_so_far
    except Exception as e:
        logger.warning(f"Product {nm_id} pass-1 failed: {e}")

    pass1_filled_ids = {av.attribute_id for av in pass1_values}
    cov_before = _coverage(pass1_filled_ids, required_ids, all_targets)
    print(f"       Pass-1 (Desc+Knowledge): req={cov_before['required_pct']:.0f}% "
          f"opt={cov_before['optional_pct']:.0f}%, llm_calls={pass1_llm_calls}")

    # --- Проход 2: WebSearch для оставшихся атрибутов ---
    # Remaining = targets не заполненные с уверенностью в pass1
    confident_after_pass1 = {av.attribute_id for av in pass1_values if av.is_confident()}
    remaining_targets = [t for t in all_targets if t.id not in confident_after_pass1]

    pass2_values: list[AttributeValue] = []
    pass2_llm_calls = 0
    web_summary: Optional[str] = None
    web_search_ran = False

    if remaining_targets:
        ctx2 = ExtractionContext(
            product_id=base_context.product_id,
            product_name=base_context.product_name,
            product_description=base_context.product_description,
            category_id=base_context.category_id,
            ozon_type_id=base_context.ozon_type_id,
            brand=base_context.brand,
            marketplace=base_context.marketplace,
            image_urls=[],
            source_urls=[],
            max_cost_usd=99.0,
        )
        try:
            # Напрямую вызываем WebSearchSource.extract() — без CostPredictor gate
            pass2_values = await web_source.extract(ctx2, remaining_targets)
            pass2_llm_calls = ctx2.llm_calls_so_far
            web_search_ran = True

            # Сохраняем web summary для hallucination check
            web_summary = web_source._summary_cache.get(ctx2.product_id)

            # Применяем strategy post-processing для web values
            pass2_values = [norm_strategy.resolve_value_ids(v, ctx2) for v in pass2_values]
            pass2_values = norm_strategy.post_process_values(pass2_values, remaining_targets, ctx2)

            print(f"       Pass-2 (WebSearch): +{len(pass2_values)} attrs, llm_calls={pass2_llm_calls}")

        except Exception as e:
            logger.warning(f"Product {nm_id} WebSearch failed: {e}")
            print(f"       Pass-2 (WebSearch): FAILED — {e}")

    # Объединяем: pass1 wins, pass2 fills gaps
    all_filled: dict[int, AttributeValue] = {av.attribute_id: av for av in pass1_values}
    for av in pass2_values:
        if av.attribute_id not in all_filled:
            all_filled[av.attribute_id] = av
        elif av.confidence > all_filled[av.attribute_id].confidence:
            # Заменяем если web нашёл более уверенное значение
            all_filled[av.attribute_id] = av

    merged_values = list(all_filled.values())
    merged_filled_ids = set(all_filled.keys())
    cov_after = _coverage(merged_filled_ids, required_ids, all_targets)

    # --- Per-attribute results ---
    attr_results = []
    for av in merged_values:
        char_meta = char_index.get(av.attribute_id, {})
        has_dict = _has_dict_values(char_meta)

        if has_dict:
            if av.is_collection:
                value_id_resolved = bool(av.value_ids)
            else:
                value_id_resolved = av.value_id is not None
        else:
            value_id_resolved = None

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

        hallucination_signal = _check_hallucination_signal(av, product, web_summary)

        attr_results.append({
            "attribute_id": av.attribute_id,
            "attribute_name": char_meta.get("name", str(av.attribute_id)),
            "value": _value_as_str(av.value),
            "confidence": round(av.confidence, 3),
            "source": av.source.value if hasattr(av.source, "value") else str(av.source),
            "is_required": av.attribute_id in required_ids,
            "has_dict_values": has_dict,
            "value_id_resolved": value_id_resolved,
            "value_in_dict": value_in_dict,
            "hallucination_signal": hallucination_signal,
            "evidence": av.evidence,
        })

    return {
        "product": {
            "nm_id": nm_id,
            "imt_name": imt_name,
            "brand_name": product.get("brand_name", ""),
            "description_len": len(product.get("description", "")),
        },
        "coverage_before_web": cov_before,
        "coverage_after_web": cov_after,
        "coverage_delta": {
            "required_delta": cov_after["required_filled"] - cov_before["required_filled"],
            "optional_delta": cov_after["optional_filled"] - cov_before["optional_filled"],
        },
        "attributes": attr_results,
        "llm_calls_pass1": pass1_llm_calls,
        "llm_calls_pass2": pass2_llm_calls,
        "web_search_ran": web_search_ran,
    }


# ---------------------------------------------------------------------------
# Основной async eval
# ---------------------------------------------------------------------------
async def run_eval():
    start_time = time.time()

    # 1. Ozon dictionary
    print("[Ozon] Loading characteristics for category 17028612, type 91910...")
    all_chars = get_ozon_characteristics_for_type(DESCRIPTION_CATEGORY_ID, TYPE_ID)
    if not all_chars:
        print("[ERROR] No Ozon characteristics found. Check ozon_dictionary or mini-cache.")
        sys.exit(1)

    char_index = {c["id"]: c for c in all_chars}
    all_targets = build_targets_from_ozon_dict(DESCRIPTION_CATEGORY_ID, TYPE_ID)
    required_ids = {c["id"] for c in all_chars if c.get("is_required")}
    print(f"[Ozon] {len(all_chars)} characteristics loaded. "
          f"Required: {len(required_ids)}, Optional: {len(all_targets) - len(required_ids)}")

    # 2. Продукты
    products = load_power_supply_products(TARGET_COUNT)
    if not products:
        print("[ERROR] No products loaded from HuggingFace")
        sys.exit(1)

    # 3. Инициализация sources
    print(f"[Pipeline] Provider={config.PROVIDER_MAIN}, model={config.MAIN_MODEL}")
    print(f"[Pipeline] WebSearch provider={config.PROVIDER_WEB_SEARCH}, extraction_model={config.EXTRACTION_FROM_TEXT_MODEL}")
    llm_manager = get_main_manager()
    strategy = get_strategy("ozon")

    # WebSearchSource используется глобально (кэш summary per product_id)
    web_source = WebSearchSource()

    # 4. Обработка товаров
    print(f"\n[Eval] Processing {len(products)} products (all 3 sources enabled)...")
    print(f"[Cache] LLM_CACHE_ENABLED=1, DB={os.environ.get('LLM_CACHE_DB', '.llm_cache.sqlite')}")
    print(f"[Cost] Limit=${COST_LIMIT_USD:.2f}, stop at {STOP_AT_ON_LIMIT} products if exceeded\n")

    results = []
    total_llm_calls_pass1 = 0
    total_llm_calls_pass2 = 0
    ws_products_ran = 0
    ws_products_failed = 0
    cost_stop_triggered = False

    # Cost tracking
    cost_desc_knowledge_usd = 0.0   # будут кэш-хиты = 0
    cost_ws_summary_usd = 0.0       # Serper + Gemini Flash summary
    cost_ws_extraction_usd = 0.0    # DeepSeek extraction
    cost_serper_usd = 0.0

    PILOT_SIZE = 5

    for idx, product in enumerate(products):
        try:
            res = await eval_product(
                product=product,
                idx=idx,
                total=len(products),
                all_targets=all_targets,
                required_ids=required_ids,
                char_index=char_index,
                llm_manager=llm_manager,
                web_source=web_source,
                strategy=strategy,
            )
            results.append(res)

            total_llm_calls_pass1 += res["llm_calls_pass1"]
            total_llm_calls_pass2 += res["llm_calls_pass2"]

            if res["web_search_ran"]:
                ws_products_ran += 1
                # Cost: 1 Serper search + 1 summary call (EXTRACTION_FROM_TEXT_MODEL) + 1 extraction call
                cost_serper_usd += PRICE_SERPER_PER_SEARCH
                cost_ws_summary_usd += (TOKENS_WS_SUMMARY_CALL / 1_000_000) * PRICE_GEMINI_FLASH_BLENDED_PER_M
                cost_ws_extraction_usd += (TOKENS_WS_EXTRACTION_CALL / 1_000_000) * PRICE_DEEPSEEK_BLENDED_PER_M

        except Exception as e:
            logger.exception(f"Product {product.get('nm_id')} outer error: {e}")
            results.append({
                "product": {
                    "nm_id": product.get("nm_id", idx),
                    "imt_name": product.get("imt_name", ""),
                },
                "error": str(e),
                "attributes": [],
                "llm_calls_pass1": 0,
                "llm_calls_pass2": 0,
                "web_search_ran": False,
            })
            ws_products_failed += 1

        await asyncio.sleep(0.2)

        # Cost check after pilot
        if idx + 1 == PILOT_SIZE:
            print(f"\n{'─'*55}")
            print(f"PILOT SUMMARY (first {PILOT_SIZE} products)")
            pilot_ok = [r for r in results if "error" not in r]
            if pilot_ok:
                def _avg(key_path):
                    parts = key_path.split(".")
                    vals = []
                    for r in pilot_ok:
                        v = r
                        for p in parts:
                            v = v.get(p, {})
                        if isinstance(v, (int, float)):
                            vals.append(v)
                    return sum(vals) / len(vals) if vals else 0
                p_req_before = _avg("coverage_before_web.required_pct")
                p_opt_before = _avg("coverage_before_web.optional_pct")
                p_req_after = _avg("coverage_after_web.required_pct")
                p_opt_after = _avg("coverage_after_web.optional_pct")
                print(f"  Before WebSearch: req={p_req_before:.1f}%, opt={p_opt_before:.1f}%")
                print(f"  After  WebSearch: req={p_req_after:.1f}%, opt={p_opt_after:.1f}%")
                print(f"  Delta: req={p_req_after - p_req_before:+.1f}pp, opt={p_opt_after - p_opt_before:+.1f}pp")
            total_ws_cost_so_far = cost_serper_usd + cost_ws_summary_usd + cost_ws_extraction_usd
            print(f"  WebSearch cost so far: ${total_ws_cost_so_far:.4f}")
            projected_total = total_ws_cost_so_far * (len(products) / max(idx + 1, 1))
            print(f"  Projected total (linear): ${projected_total:.4f}")
            if projected_total > COST_LIMIT_USD:
                print(f"\n[COST GUARD] Projected ${projected_total:.2f} > limit ${COST_LIMIT_USD:.2f}")
                print(f"[COST GUARD] Stopping after {STOP_AT_ON_LIMIT} products.")
                cost_stop_triggered = True
            print(f"{'─'*55}\n")

            if cost_stop_triggered and idx + 1 >= STOP_AT_ON_LIMIT:
                break

        # Дополнительная проверка cost guard после STOP_AT_ON_LIMIT товаров
        if cost_stop_triggered and idx + 1 >= STOP_AT_ON_LIMIT:
            break

    elapsed = time.time() - start_time

    # ---------------------------------------------------------------------------
    # Агрегированный отчёт
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AGGREGATE REPORT — WebSearch Eval")
    print("=" * 70)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"Products processed: {len(results)} ({len(successful)} ok, {len(failed)} failed)")
    if cost_stop_triggered:
        print(f"[NOTE] Stopped early at {len(results)} products (cost guard triggered)")
    print(f"WebSearch ran: {ws_products_ran}/{len(results)} products")
    print(f"Pass-1 LLM calls (Desc+Knowledge): {total_llm_calls_pass1}")
    print(f"Pass-2 LLM calls (WebSearch): {total_llm_calls_pass2}")
    print(f"Elapsed: {elapsed:.1f}s")

    if not successful:
        print("[ERROR] No successful results to aggregate!")
        sys.exit(1)

    # Coverage before/after WebSearch
    avg_req_before = sum(r["coverage_before_web"]["required_pct"] for r in successful) / len(successful)
    avg_opt_before = sum(r["coverage_before_web"]["optional_pct"] for r in successful) / len(successful)
    avg_req_after  = sum(r["coverage_after_web"]["required_pct"]  for r in successful) / len(successful)
    avg_opt_after  = sum(r["coverage_after_web"]["optional_pct"]  for r in successful) / len(successful)

    print(f"\nCoverage (avg across {len(successful)} products):")
    print(f"  Before WebSearch: required={avg_req_before:.1f}%, optional={avg_opt_before:.1f}%")
    print(f"  After  WebSearch: required={avg_req_after:.1f}%,  optional={avg_opt_after:.1f}%")
    print(f"  DELTA:            required={avg_req_after - avg_req_before:+.1f}pp, "
          f"optional={avg_opt_after - avg_opt_before:+.1f}pp")

    # Per-attribute fill rates
    attr_fill_count: dict[int, int] = {}
    attr_names = {c["id"]: c["name"] for c in all_chars}
    for r in successful:
        for av_r in r["attributes"]:
            aid = av_r["attribute_id"]
            attr_fill_count[aid] = attr_fill_count.get(aid, 0) + 1

    total_prods = len(successful)
    fill_rates = {
        aid: (count / total_prods * 100)
        for aid, count in attr_fill_count.items()
    }

    always_filled = sorted(
        [(aid, rate) for aid, rate in fill_rates.items() if rate >= 90],
        key=lambda x: -x[1],
    )[:5]
    never_filled = [
        (t.id, attr_names.get(t.id, str(t.id)))
        for t in all_targets
        if fill_rates.get(t.id, 0) == 0
    ]

    print(f"\nTop 5 always-filled attrs (>=90%):")
    for aid, rate in always_filled:
        print(f"  {attr_names.get(aid, aid):45s}  {rate:.0f}%")
    print(f"\nNever-filled attrs ({len(never_filled)} total):")
    for aid, name in never_filled[:10]:
        print(f"  {name[:45]:45s}  (id={aid})")
    if len(never_filled) > 10:
        print(f"  ... and {len(never_filled) - 10} more")

    # Value ID resolution
    all_attr_vals = [av for r in successful for av in r["attributes"]]
    dict_backed = [av for av in all_attr_vals if av["has_dict_values"]]
    resolution_rate = 0.0
    if dict_backed:
        resolved = [av for av in dict_backed if av["value_id_resolved"] is True]
        resolution_rate = 100 * len(resolved) / len(dict_backed)
        print(f"\nValue ID resolution: {len(resolved)}/{len(dict_backed)} = {resolution_rate:.1f}%")
        in_dict = [av for av in dict_backed if av["value_in_dict"] is True]
        print(f"Value-in-dict rate:  {len(in_dict)}/{len(dict_backed)} = "
              f"{100 * len(in_dict) / len(dict_backed):.1f}%")

    # Hallucination signals
    hallucinations = [av for av in all_attr_vals if av["hallucination_signal"]]
    print(f"\nHallucination signals: {len(hallucinations)}")
    if hallucinations[:5]:
        print("  Examples:")
        for h in hallucinations[:5]:
            print(f"    attr={h['attribute_name'][:30]:30s}  val={h['value'][:30]:30s}  conf={h['confidence']:.2f}")

    # Cost breakdown
    total_ws_cost = cost_serper_usd + cost_ws_summary_usd + cost_ws_extraction_usd
    # Pass-1 cost: если кэш работает, это должно быть 0 (все cache hits)
    # Оцениваем как если бы не было кэша — для сравнения
    pass1_tokens_no_cache = (total_llm_calls_pass1 * TOKENS_DESCRIPTION_CALL)
    pass1_cost_no_cache = (pass1_tokens_no_cache / 1_000_000) * PRICE_DEEPSEEK_BLENDED_PER_M
    # Реальные call'ы = 0 если всё закэшировано
    pass1_cost_real = (total_llm_calls_pass1 * TOKENS_DESCRIPTION_CALL / 1_000_000) * PRICE_DEEPSEEK_BLENDED_PER_M

    print(f"\nCost breakdown:")
    print(f"  Description+Knowledge (cache={os.environ.get('LLM_CACHE_ENABLED','0')}):")
    print(f"    Real LLM calls: {total_llm_calls_pass1} → ~${pass1_cost_real:.4f}")
    print(f"  WebSearch ({ws_products_ran} products):")
    print(f"    Serper searches:     ${cost_serper_usd:.4f}")
    print(f"    Summary calls (LLM): ${cost_ws_summary_usd:.4f}")
    print(f"    Extraction (DeepSeek): ${cost_ws_extraction_usd:.4f}")
    print(f"  TOTAL WebSearch: ${total_ws_cost:.4f}")
    print(f"  TOTAL SPENT:     ~${pass1_cost_real + total_ws_cost:.4f}")
    print(f"  Cache saved:     ~${pass1_cost_no_cache:.4f} (estimated if no cache)")

    # ---------------------------------------------------------------------------
    # Сохранение результатов
    # ---------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent / "eval_results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"power_supply_websearch_{timestamp}.json"

    report = {
        "meta": {
            "timestamp": timestamp,
            "script": "eval_ozon_power_supply_websearch.py",
            "dataset": HF_DATASET,
            "description_category_id": DESCRIPTION_CATEGORY_ID,
            "type_id": TYPE_ID,
            "provider_main": config.PROVIDER_MAIN,
            "model_main": config.MAIN_MODEL,
            "provider_websearch": config.PROVIDER_WEB_SEARCH,
            "extraction_model": config.EXTRACTION_FROM_TEXT_MODEL,
            "products_requested": TARGET_COUNT,
            "products_processed": len(results),
            "products_successful": len(successful),
            "products_failed": len(failed),
            "ws_products_ran": ws_products_ran,
            "cost_stop_triggered": cost_stop_triggered,
            "llm_calls_pass1": total_llm_calls_pass1,
            "llm_calls_pass2": total_llm_calls_pass2,
            "elapsed_seconds": round(elapsed, 1),
        },
        "aggregate": {
            # Coverage before/after WebSearch — ключевая метрика
            "coverage_before_web": {
                "avg_required_pct": round(avg_req_before, 1),
                "avg_optional_pct": round(avg_opt_before, 1),
            },
            "coverage_after_web": {
                "avg_required_pct": round(avg_req_after, 1),
                "avg_optional_pct": round(avg_opt_after, 1),
            },
            "coverage_delta_pp": {
                "required": round(avg_req_after - avg_req_before, 1),
                "optional": round(avg_opt_after - avg_opt_before, 1),
            },
            "value_id_resolution_pct": round(resolution_rate, 1) if dict_backed else None,
            "hallucination_signals": len(hallucinations),
            "cost_breakdown_usd": {
                "desc_knowledge_real": round(pass1_cost_real, 4),
                "ws_serper": round(cost_serper_usd, 4),
                "ws_summary": round(cost_ws_summary_usd, 4),
                "ws_extraction": round(cost_ws_extraction_usd, 4),
                "ws_total": round(total_ws_cost, 4),
                "total_spent": round(pass1_cost_real + total_ws_cost, 4),
                "cache_saved_est": round(pass1_cost_no_cache, 4),
            },
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
        sys.stdout.flush()
    except Exception as e:
        print(f"\n[ERROR] Failed to write results: {e}")
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="power_supply_websearch_",
            delete=False, encoding="utf-8",
        )
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        print(f"[FALLBACK] Results written to: {tmp.name}")
        output_path = tmp.name

    return output_path


def main():
    result_path = asyncio.run(run_eval())
    marker = Path(__file__).parent / "eval_results" / "last_run_websearch.txt"
    try:
        marker.write_text(str(result_path) + "\n", encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
