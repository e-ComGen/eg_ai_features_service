"""BestBuySource — verbatim specs from Best Buy Developer API (free, official).

Best Buy API (https://developer.bestbuy.com/) is free with an API key. It provides
rich structured product data including specs, features, dimensions, and more.
Lookup by UPC when context.ean is present, otherwise by manufacturer+search query.

Architecture (mirror RegardSource):
  1. Lookup: UPC query ``products(upc=<ean>)`` when EAN present, else
     ``products(manufacturer=<brand>&search=<model>)``.
  2. Parse JSON: ``details[]`` name/value pairs, plus top-level fields
     (color, weight, shippingWeight, warrantyLabor, features[], etc.).
  3. Brand-gate: product name from API must contain brand + model tokens.
  4. Values are English → run through _translate_en_to_ru before enum-match.
  5. Enum attrs: resolve_value_id; None → drop (verbatim-safe, no guessing).

Position in pipeline: Stage 0.58 — after IceCat/Regard (0.55/0.57),
before CompetitorRAG (0.7). Fires ONLY when remaining > 0 (cost-aware).

Cost: 1 Best Buy API call (free, rate-limited to 5 req/s).
Source.WB_CARD: same semantics (verbatim spec copy, no LLM).

Graceful no-op: if BESTBUY_API_KEY env var is absent, extract() returns []
without crashing — the owner will add the key when ready.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import OrderedDict
from typing import Any, Optional, Union

import httpx

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.wb_card_judge import WbCardJudge
from app.services.enrichment.prompt_router import filter_already_filled_targets
from app.services.enrichment.sources.ozon_card_source import (
    _norm_char_name,
    _normalize_model,
    _split_multivalue,
    _extract_model_tokens,
)
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    resolve_value_id,
    _translate_en_to_ru,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_API_BASE = "https://api.bestbuy.com/v1"
_HTTP_TIMEOUT = 15.0
_MAX_RESULTS = 5          # max products returned from search
_CACHE_MAX = 256

# Confidence levels (mirror RegardSource)
_CONF_EXACT = 0.90
_CONF_BRAND_LINE = 0.82

# Match thresholds
_EXACT_THRESHOLD = 78.0
_BRAND_LINE_THRESHOLD = 60.0

# Skip-guard: if ≥80% targets already filled, don't bother.
_SKIP_FILL_RATIO = 0.80

# Brand-gate minimums (mirror RegardSource)
_BRAND_GATE_MIN_BRAND_TOKENS = 1
_BRAND_GATE_MIN_MODEL_TOKENS = 1

# Fuzzy name-match threshold
_FUZZY_THRESHOLD = 88

# Env var for API key
_API_KEY_ENV = "BESTBUY_API_KEY"

# Token RE for brand/model gate checks.
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Blacklist: model-specific identifiers to skip in brand_line mode.
_BRAND_LINE_BLACKLIST: frozenset[str] = frozenset(name.lower() for name in {
    "SKU", "UPC", "Model Number", "Manufacturer Part Number", "MPN",
    "Product ID", "Part Number", "Serial Number",
})

# Best Buy API top-level fields to include as spec pairs (field_name → display_name).
_TOP_LEVEL_FIELDS: dict[str, str] = {
    "color": "Цвет",
    "manufacturer": "Производитель",
    "modelNumber": "Номер модели",
    "warrantyLabor": "Гарантия (труд)",
    "warrantyParts": "Гарантия (детали)",
    "shippingWeight": "Вес (доставка)",
    "weight": "Вес",
    "depth": "Глубина",
    "height": "Высота",
    "width": "Ширина",
}


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower().replace("ё", "е"))


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def _parse_bestbuy_product(product: dict) -> tuple[str, list[dict]]:
    """Extract (product_name, specs) from a Best Buy API product dict.

    Sources (in order, deduped by normalized spec name):
      1. product["details"] — list of {name, value} — primary rich specs.
      2. Top-level fields (color, weight, dimensions, warranty, etc.).
      3. product["features"] — list of {feature} strings as free-text (name=Feature).

    Returns:
        product_name: str (for brand-gate).
        specs: list of {"name": ..., "value": ...} dicts.
    """
    product_name: str = product.get("name", "") or ""

    specs: list[dict] = []
    seen: set[str] = set()

    def _add(name: str, value: Any) -> None:
        if not name or value is None:
            return
        val_str = str(value).strip()
        if not val_str or val_str.lower() in ("none", "null", "", "0"):
            return
        key = name.lower().strip()
        if key in seen:
            return
        seen.add(key)
        specs.append({"name": name, "value": val_str})

    # 1. details[] — the richest source
    for detail in product.get("details") or []:
        if isinstance(detail, dict):
            _add(str(detail.get("name", "") or "").strip(),
                 detail.get("value"))

    # 2. Top-level fields
    for field, display_name in _TOP_LEVEL_FIELDS.items():
        _add(display_name, product.get(field))

    # 3. Features (free-text, lower value)
    for feat in product.get("features") or []:
        if isinstance(feat, dict):
            feat_text = str(feat.get("feature", "") or "").strip()
            if feat_text:
                _add("Feature", feat_text)

    return product_name, specs


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

class BestBuySource(AttributeSource):
    """Verbatim specs from Best Buy API (free, official).

    Lookup: UPC → products(upc=<ean>); else manufacturer+search query.
    Values are English → translated via _translate_en_to_ru before enum-match.
    Enum attributes: resolve_value_id; None → drop (verbatim-safe).
    Emits Source.WB_CARD (same verbatim semantics as RegardSource).

    Graceful no-op when BESTBUY_API_KEY is absent.
    """

    def __init__(self, **kwargs: Any) -> None:
        _ = kwargs
        self._api_key: Optional[str] = os.environ.get(_API_KEY_ENV)
        if not self._api_key:
            logger.info(
                "[BestBuy] %s not set — source dormant (no-op). "
                "Set the env var to activate.",
                _API_KEY_ENV,
            )
        self._judge = WbCardJudge()
        self._cache: OrderedDict[tuple[str, str], list[AttributeValue]] = OrderedDict()

    @property
    def source_type(self) -> Source:
        return Source.WB_CARD

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        return bool(
            self._api_key
            and context.product_name
            and len(context.product_name.strip()) >= 5
        )

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        # Graceful no-op if key is absent.
        if not self._api_key:
            return []
        if not targets or not context.product_name:
            return []

        already_filled = already_filled or []

        # Skip-guard: ≥80% targets already filled.
        filled_ids = {av.attribute_id for av in already_filled if av.confidence >= 0.85}
        if targets and len(filled_ids & {t.id for t in targets}) / len(targets) >= _SKIP_FILL_RATIO:
            logger.debug("[BestBuy] skip (≥%.0f%% targets filled)", _SKIP_FILL_RATIO * 100)
            return []

        effective = filter_already_filled_targets(targets, already_filled)
        if not effective:
            return []

        brand = (context.brand or "").strip()
        model_norm = _normalize_model(context.product_name, brand)
        cache_key = (brand.lower(), model_norm)

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._filter_for_targets(self._cache[cache_key], effective)

        try:
            all_values = await asyncio.wait_for(
                self._do_extract(context, targets),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[BestBuy] total timeout (30s) for '%s' — skipping",
                context.product_name[:60],
            )
            return []
        except Exception as exc:
            logger.warning(
                "[BestBuy] unexpected error for '%s': %s",
                context.product_name[:60], exc,
            )
            return []

        if all_values:
            self._cache_put(cache_key, all_values)
        return self._filter_for_targets(all_values, effective)

    def get_judge(self) -> LlmJudge:
        return self._judge

    # ------------------------------------------------------------------
    # Core flow
    # ------------------------------------------------------------------

    async def _do_extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        full_name = context.product_name.strip()
        brand = (context.brand or "").strip()

        # Build API query: UPC first if EAN present, else manufacturer+search.
        if context.ean:
            filter_str = f"upc={context.ean}"
            logger.info("[BestBuy] lookup by UPC: %s", context.ean)
        else:
            # Normalize model tokens for a clean search query.
            model_tokens = _extract_model_tokens(full_name)
            model_q = " ".join(model_tokens) if model_tokens else full_name
            filter_str = f"manufacturer={brand}&search={model_q}" if brand else f"search={model_q}"
            logger.info(
                "[BestBuy] lookup by manufacturer+search: manufacturer=%s search=%s",
                brand, model_q,
            )

        products = await self._api_fetch(filter_str)
        if not products:
            logger.info("[BestBuy] no products found")
            return []

        for product in products[:_MAX_RESULTS]:
            product_name, specs = _parse_bestbuy_product(product)
            if not specs:
                continue

            if not self._brand_gate(product_name, brand, full_name):
                logger.info(
                    "[BestBuy] brand-gate FAIL: product_name='%s' brand='%s' query='%s'",
                    product_name[:80], brand, full_name[:60],
                )
                continue

            score = self._score_title(product_name, full_name)
            mode = self._classify_match(score)
            if mode == "skip":
                logger.info(
                    "[BestBuy] score=%.1f < %.0f — skip product='%s'",
                    score, _BRAND_LINE_THRESHOLD, product_name[:60],
                )
                continue

            logger.info(
                "[BestBuy] match=%s score=%.1f product='%s'",
                mode, score, product_name[:60],
            )
            return self._map_characteristics(specs, targets, context, mode, product_name, score)

        logger.info("[BestBuy] no brand-gate-passing product found")
        return []

    # ------------------------------------------------------------------
    # API helper
    # ------------------------------------------------------------------

    async def _api_fetch(self, filter_str: str) -> list[dict]:
        """Call Best Buy products API, return list of product dicts."""
        show_fields = (
            "name,sku,upc,manufacturer,modelNumber,color,weight,shippingWeight,"
            "depth,height,width,warrantyLabor,warrantyParts,details,features"
        )
        url = (
            f"{_API_BASE}/products({filter_str})"
            f"?apiKey={self._api_key}"
            f"&format=json"
            f"&show={show_fields}"
            f"&pageSize={_MAX_RESULTS}"
        )
        try:
            async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
            ) as client:
                resp = await client.get(url)
                if resp.status_code == 403:
                    logger.warning(
                        "[BestBuy] HTTP 403 (invalid API key or rate limit) — no-op"
                    )
                    return []
                if resp.status_code == 429:
                    logger.warning("[BestBuy] HTTP 429 (rate limited) — no-op")
                    return []
                if resp.status_code != 200:
                    logger.info(
                        "[BestBuy] HTTP %s — skip", resp.status_code
                    )
                    return []
                data = resp.json()
                products = data.get("products") or []
                logger.info("[BestBuy] API → %d products", len(products))
                return products
        except Exception as exc:
            logger.info("[BestBuy] API fetch error: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Brand-gate (mirror RegardSource._brand_gate)
    # ------------------------------------------------------------------

    @staticmethod
    def _brand_gate(product_name: str, brand: Optional[str], query_name: str) -> bool:
        """Require brand tokens AND at least one model token in product_name."""
        title_tokens = set(_tokenize(product_name))
        if not title_tokens:
            return False

        brand_ok = True
        if brand and brand.strip():
            brand_tokens = _tokenize(brand)
            brand_ok = sum(1 for t in brand_tokens if t in title_tokens) >= _BRAND_GATE_MIN_BRAND_TOKENS

        model_tokens = _extract_model_tokens(query_name)
        if model_tokens:
            def _model_token_in_title(token: str) -> bool:
                t_low = token.lower()
                if t_low in title_tokens:
                    return True
                parts = _TOKEN_RE.findall(t_low)
                meaningful = [p for p in parts if len(p) >= 2]
                return bool(meaningful) and all(p in title_tokens for p in meaningful)
            model_ok = any(_model_token_in_title(t) for t in model_tokens)
        else:
            model_ok = True

        return brand_ok and model_ok

    # ------------------------------------------------------------------
    # Scoring (mirror RegardSource)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_title(title: str, query: str) -> float:
        try:
            from rapidfuzz import fuzz
            t = title.lower()
            q = query.lower()
            return (fuzz.partial_ratio(q, t) + fuzz.token_sort_ratio(q, t)) / 2.0
        except ImportError:
            t_toks = set(_tokenize(title))
            q_toks = set(_tokenize(query))
            if not q_toks:
                return 0.0
            return len(t_toks & q_toks) / len(q_toks) * 100.0

    @staticmethod
    def _classify_match(score: float) -> str:
        if score >= _EXACT_THRESHOLD:
            return "exact"
        if score >= _BRAND_LINE_THRESHOLD:
            return "brand_line"
        return "skip"

    # ------------------------------------------------------------------
    # Mapping: spec name → target → EN→RU translate → value_id
    # ------------------------------------------------------------------

    def _map_characteristics(
        self,
        specs: list[dict],
        targets: list[TargetAttribute],
        context: ExtractionContext,
        mode: str,
        product_name: str,
        score: float,
    ) -> list[AttributeValue]:
        """Map Best Buy spec pairs to target attributes.

        Key difference from RegardSource: values are English, so each value
        is run through _translate_en_to_ru before enum resolution. This converts
        "Black" → "чёрный", "Yes" → "да", etc. for enum matching.
        Non-translatable values pass through unchanged (e.g. "LGA1700", "18 МБ").
        """
        ozon_chars: list[dict] = []
        cat_id: Optional[int] = None
        type_id: Optional[int] = None
        try:
            cat_id = int(context.category_id) if context.category_id else None
            type_id = context.ozon_type_id
            if cat_id and type_id:
                ozon_chars = get_ozon_characteristics_for_type(cat_id, type_id)
        except (ValueError, TypeError):
            cat_id = None
            type_id = None

        attr_id_to_dict_name: dict[int, str] = {}
        for oc in ozon_chars:
            if isinstance(oc, dict) and "id" in oc and "name" in oc:
                attr_id_to_dict_name[int(oc["id"])] = str(oc["name"])

        target_names_low: dict[int, set[str]] = {}
        for t in targets:
            raw_low = t.name.lower()
            names = {raw_low, _norm_char_name(t.name)}
            dn = attr_id_to_dict_name.get(t.id)
            if dn:
                names.add(dn.lower())
                names.add(_norm_char_name(dn))
            target_names_low[t.id] = names

        name_to_target_id: dict[str, int] = {}
        for tid, names in target_names_low.items():
            for n in names:
                name_to_target_id.setdefault(n, tid)

        try:
            from rapidfuzz import process, fuzz
            all_target_names = list(name_to_target_id.keys())
        except ImportError:
            process = None
            fuzz = None
            all_target_names = []

        target_by_id: dict[int, TargetAttribute] = {t.id: t for t in targets}
        evidence = f"bestbuy:{product_name[:50]} | score={score:.1f}"
        conf = _CONF_EXACT if mode == "exact" else _CONF_BRAND_LINE

        results: list[AttributeValue] = []
        used_ids: set[int] = set()

        for spec in specs:
            char_name = spec["name"].strip()
            char_val_raw = spec["value"].strip()
            if not char_name or not char_val_raw:
                continue
            char_name_low = char_name.lower()
            char_name_norm = _norm_char_name(char_name)

            if mode == "brand_line" and char_name_low in _BRAND_LINE_BLACKLIST:
                continue

            # EN→RU translate BEFORE matching (whole-token, no-op if not in dict)
            char_val = _translate_en_to_ru(char_val_raw)

            # Name matching: exact → norm → substring → fuzzy
            target_id = name_to_target_id.get(char_name_low)
            if target_id is None and char_name_norm != char_name_low:
                target_id = name_to_target_id.get(char_name_norm)
            if target_id is None:
                for tn, tid in name_to_target_id.items():
                    if char_name_low in tn or tn in char_name_low:
                        target_id = tid
                        break
            if target_id is None and process is not None and all_target_names:
                q = char_name_norm if char_name_norm else char_name_low
                best = process.extractOne(q, all_target_names, scorer=fuzz.WRatio)
                if best is not None and best[1] >= _FUZZY_THRESHOLD:
                    target_id = name_to_target_id[best[0]]

            if target_id is None or target_id in used_ids:
                continue
            target = target_by_id.get(target_id)
            if target is None:
                continue
            if mode == "brand_line" and target.type == "numeric":
                continue
            used_ids.add(target_id)

            value_id: Optional[int] = None
            value_out: Union[str, list[str]]
            value_ids: Optional[list[int]] = None

            if target.is_collection:
                parts = _split_multivalue(char_val)
                # Also try EN→RU on each part for multi-value collections.
                parts = [_translate_en_to_ru(p) for p in parts]
                value_out = parts
                if cat_id and type_id:
                    try:
                        resolved = [
                            resolve_value_id(cat_id, type_id, target.id, p)
                            for p in parts
                        ]
                        if any(r is not None for r in resolved):
                            value_ids = resolved
                    except Exception as exc:
                        logger.debug("[BestBuy] resolve_value_id (list) failed: %s", exc)
            else:
                value_out = char_val
                if target.type == "enum" and cat_id and type_id:
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                    except Exception as exc:
                        logger.debug("[BestBuy] resolve_value_id failed: %s", exc)
                    if value_id is None:
                        logger.debug(
                            "[BestBuy] enum attr %s value '%s' (raw='%s') not in dict — drop",
                            target.id, char_val[:40], char_val_raw[:40],
                        )
                        used_ids.discard(target_id)
                        continue
                elif cat_id and type_id:
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                    except Exception as exc:
                        logger.debug("[BestBuy] resolve_value_id failed: %s", exc)

            results.append(AttributeValue(
                attribute_id=target.id,
                value=value_out,
                confidence=conf,
                source=Source.WB_CARD,
                evidence=evidence,
                semantic_type=target.semantic_type,
                is_collection=target.is_collection,
                value_id=value_id,
                value_ids=value_ids,
            ))

        logger.info(
            "[BestBuy] %s mode → %d attrs filled (from %d specs, %d targets)",
            mode, len(results), len(specs), len(targets),
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_for_targets(
        values: list[AttributeValue],
        effective: list[TargetAttribute],
    ) -> list[AttributeValue]:
        eff_ids = {t.id for t in effective}
        return [v for v in values if v.attribute_id in eff_ids]

    def _cache_put(
        self, key: tuple[str, str], value: list[AttributeValue]
    ) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)
