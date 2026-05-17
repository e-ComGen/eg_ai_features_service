import asyncio
import json # <--- ДОБАВЛЕН ДЛЯ РАБОТЫ СО СЛОВАРЯМИ
import logging
import re
from .ai_pipeline import AiFeaturePipeline
from .db_cache import DatabaseCacheManager
from .matcher import MatcherService
from .url_fetcher import fetch_all
from ..models import ProductData, FeatureOption, ResearchMode
from ..database import AsyncSessionLocal
from ..config import VAGUE_FEATURE_PATTERNS

logger = logging.getLogger(__name__)

# Pre-compile vague-feature-name regexes once at import time. Each pattern
# is wrapped with fullmatch semantics (re.fullmatch) and IGNORECASE so the
# operator can add patterns in config.py without worrying about flags.
_VAGUE_FEATURE_REGEXES = [
    re.compile(p, re.IGNORECASE) for p in VAGUE_FEATURE_PATTERNS
]


def is_vague_feature_name(name: str) -> bool:
    """
    Returns True if the feature name is a placeholder / meaningless label
    (e.g. "NewFeature", "Field 1", "X", "123"). Such tuples must be
    short-circuited BEFORE any LLM call — otherwise the model is forced to
    invent a value, which is the operator's product-403 bug.
    """
    if not name:
        return True
    trimmed = name.strip()
    if len(trimmed) < 2:
        return True
    for rx in _VAGUE_FEATURE_REGEXES:
        if rx.fullmatch(trimmed):
            return True
    return False

class JobProcessor:
    def __init__(self, pipeline: AiFeaturePipeline, db_cache: DatabaseCacheManager, matcher: MatcherService,
                 global_semaphore: asyncio.Semaphore):
        self.pipeline = pipeline
        self.db_cache = db_cache
        self.matcher = matcher
        self.semaphore = global_semaphore

    async def process_product(self, product: ProductData, schema: dict[str, FeatureOption], client_id: int,
                              use_cache: bool = True,
                              research_mode: ResearchMode = ResearchMode.OFF) -> dict:
        result_features = {}
        total_tokens_used = 0
        is_fully_cached = True

        # --- Web-fetch enrichment (Stage 2) ---
        # Fetch supplier/competitor URLs in parallel and inject into description.
        # Failures are non-fatal: pipeline continues with whatever was fetched.
        fetched_content = ""
        if getattr(product, "source_urls", None):
            try:
                fetched_content = await fetch_all(product.source_urls)
            except Exception as _fetch_exc:
                logger.warning("fetch_all failed entirely, continuing without URL content: %s", _fetch_exc)

        description = product.description or ""
        if fetched_content:
            description = f"{description}\n\n=== Sourced from URLs ===\n{fetched_content}"

        info = f"Title: {product.name}\nDescription: {description[:50000]}"
        context_hash = f"{product.name} {product.description}"

        async def process_feature(f_name, f_schema):
            async with self.semaphore:
                async with AsyncSessionLocal() as session:
                    existing = product.context.existing_features
                    if isinstance(existing, dict) and f_name in existing:
                        return (f_name, None, 0, True, {}, "Skipped: Already exists", None, None, None)

                    # Early reject: vague/placeholder feature names ("NewFeature",
                    # "Field 1", "X", purely numeric, etc.) can't be answered
                    # without hallucinating. Skip the LLM entirely.
                    if is_vague_feature_name(f_name):
                        logger.info(f"Skipped vague feature name: {f_name!r}")
                        return (
                            f_name, None, 0, False, {},
                            f"⛔ EARLY REJECT: vague/placeholder feature name '{f_name}'.",
                            None, None, None,
                        )

                    if use_cache:
                        cached_val = await self.db_cache.get_cached_value(
                            session, client_id, product.id, f_name, context_hash
                        )
                        if cached_val:
                            debug_reason = "Cached" if cached_val != "__EMPTY__" else "Cached Empty"
                            val_to_return = None
                            if cached_val != "__EMPTY__":
                                # 👇 БЕЗОПАСНО ПАРСИМ JSON, ЕСЛИ ЭТО СЛОВАРЬ 👇
                                try:
                                    val_to_return = json.loads(cached_val)
                                except json.JSONDecodeError:
                                    val_to_return = cached_val
                            return (f_name, val_to_return, 0, True, {}, debug_reason, None, "cache", None)

                    suffix = getattr(f_schema, 'suffix', "")
                    opts = getattr(f_schema, 'options', [])

                    # If the operator configured a prompt_hint for this feature (stored in
                    # admin_rules.settings and forwarded in the BatchFillRequest schema),
                    # prepend it to the product text as a constraint block. The hint is
                    # injected BEFORE the main product info so the LLM sees it as a
                    # high-priority instruction before any extraction routing begins.
                    prompt_hint = getattr(f_schema, 'prompt_hint', None)
                    effective_product_text = info
                    if prompt_hint:
                        effective_product_text = (
                            f"--- OPERATOR CONSTRAINT ---\n{prompt_hint}\n\n"
                            f"--- PRODUCT INFO ---\n{info}"
                        )

                    # 👇 ПЕРЕДАЕМ product.languages + research_mode 👇
                    result_data = await self.pipeline.extract_feature(
                        product_text=effective_product_text,
                        feature_name=f_name,
                        suffix=suffix,
                        options=opts,
                        target_languages=product.languages,
                        research_mode=research_mode
                    )

                    raw_val = result_data["value"]
                    tokens = result_data["tokens"]
                    router_debug = result_data.get("router_debug", {})
                    extraction_reasoning = result_data.get("extraction_reasoning", "")
                    deduced_context = result_data.get("deduced_context", None)
                    source = result_data.get("source", None)
                    source_urls = result_data.get("source_urls", None)

                    # 👇 СЕРИАЛИЗУЕМ СЛОВАРЬ В СТРОКУ ПЕРЕД КЕШОМ 👇
                    if raw_val is not None:
                        val_to_save = json.dumps(raw_val, ensure_ascii=False) if isinstance(raw_val, dict) else str(raw_val)
                    else:
                        val_to_save = "__EMPTY__"

                    await self.db_cache.set_cached_value(
                        session, client_id, product.id, f_name, context_hash, val_to_save
                    )

                    return (f_name, raw_val, tokens, False, router_debug, extraction_reasoning, deduced_context, source, source_urls)

        tasks = []
        for f_name, f_schema in schema.items():
            tasks.append(process_feature(f_name, f_schema))

        results = await asyncio.gather(*tasks)
        debug_info = {}

        for res in results:
            if not res: continue

            f_name, val, tokens, from_cache, r_debug, e_reason, d_context, source, source_urls = res

            if not from_cache:
                total_tokens_used += tokens
                is_fully_cached = False

            if val and str(val).lower() == "none":
                val = None

            if val:
                result_features[f_name] = val

            debug_info[f_name] = {
                "router": r_debug,
                "extraction_reasoning": e_reason,
                "deduced_context": d_context,
                "source": source,
                "source_urls": source_urls,
            }

        final_result = {}
        for f_name, val in result_features.items():
            f_schema = schema.get(f_name)

            if f_schema and f_schema.type == 'select' and getattr(f_schema, 'options', None):
                matched = self.matcher.find_best_match(val, f_schema.options)
                if matched:
                    final_result[f_name] = matched
            else:
                final_result[f_name] = val

        return {
            "product_id": product.id,
            "filled_features": final_result,
            "debug_info": debug_info,
            "tokens_used": total_tokens_used,
            "is_cached": is_fully_cached
        }