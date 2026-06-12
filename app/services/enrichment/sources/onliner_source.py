"""OnlinerSource — verbatim specs from Onliner.by (Belarus, public REST + JSON-LD).

Onliner.by is a Belarusian electronics/goods catalog with publicly accessible
product data in Russian. It has a public search API returning product slugs,
and product pages serve JSON-LD ``additionalProperty`` arrays (80+ clean RU
name/value pairs) in server-rendered HTML — accessible via plain httpx.

Architecture (mirror RegardSource):
  1. Search: ``catalog.api.onliner.by/search/products?query=<brand model>``
     → JSON with products[] each having a ``url`` and ``full_name``.
  2. Fetch product page (plain httpx, Chrome UA).
  3. Parse JSON-LD ``<script type="application/ld+json">`` → ``additionalProperty``
     array of {name, value} pairs. Falls back to HTML spec table if absent.
  4. Brand-gate: product URL/full_name must contain brand + model tokens.
  5. Map to targets via enum-matcher (same as RegardSource).
     Values are RU — no translation needed (pass-through harmless).
  6. Enum attrs: resolve_value_id; None → drop (verbatim-safe).

Position in pipeline: Stage 0.59 — after BestBuy (0.58), before CompetitorRAG (0.7).
Fires ONLY when remaining > 0 (cost-aware).

Cost: 1 httpx GET (search API, free) + 1 httpx GET (product page, free, ~200ms).
Source.WB_CARD: same semantics (verbatim table copy, no LLM).
"""
from __future__ import annotations

import asyncio
import json
import logging
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
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SEARCH_API = "https://catalog.api.onliner.by/search/products"
_HTTP_TIMEOUT = 15.0
_MAX_HTML_BYTES = 3_000_000   # 3MB cap
_MAX_CANDIDATES = 5           # max search results to try
_CACHE_MAX = 256

# Confidence levels (mirror RegardSource)
_CONF_EXACT = 0.90
_CONF_BRAND_LINE = 0.82

# Match thresholds
_EXACT_THRESHOLD = 78.0
_BRAND_LINE_THRESHOLD = 60.0

# Skip-guard
_SKIP_FILL_RATIO = 0.80

# Brand-gate minimums
_BRAND_GATE_MIN_BRAND_TOKENS = 1
_BRAND_GATE_MIN_MODEL_TOKENS = 1

# Fuzzy name-match threshold
_FUZZY_THRESHOLD = 88

# Chrome User-Agent
_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Token RE for brand/model gate checks.
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Blacklist: model-specific identifiers to skip in brand_line mode.
_BRAND_LINE_BLACKLIST: frozenset[str] = frozenset(name.lower() for name in {
    "Артикул", "Код производителя", "MPN", "Партномер", "Штрихкод",
    "EAN", "GTIN", "Модель", "Код товара", "Vendor Part Number",
    "Серийный номер", "ID товара",
})

# JSON-LD script tag (may contain multiple JSON objects).
_JSONLD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)

# HTML spec table fallback: Onliner renders dl/dt/dd or tr/td pairs.
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
_TAGS_RE = re.compile(r"<[^>]+>")


def _text(html_fragment: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    return re.sub(r"\s+", " ", _TAGS_RE.sub("", html_fragment)).strip()


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower().replace("ё", "е"))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_onliner_specs(html: str) -> tuple[str, list[dict]]:
    """Parse Onliner.by product page HTML.

    Returns:
        product_name: str (from JSON-LD ``name`` field or og:title for brand-gate).
        specs: list of {"name": ..., "value": ...} dicts from additionalProperty.

    Strategy:
      1. Extract all JSON-LD blocks — find the one containing ``additionalProperty``.
      2. Use additionalProperty[].{name, value} as the primary spec source.
         Onliner provides 80+ RU-language pairs.
      3. Fallback: if no additionalProperty found, try HTML <tr><td> pairs from
         the page's spec table (secondary).
    """
    product_name = ""

    # Try og:title for product name.
    om = re.search(
        r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    )
    if om:
        product_name = om.group(1).strip()

    # Extract JSON-LD blocks.
    specs: list[dict] = []
    seen: set[str] = set()

    for m in _JSONLD_RE.finditer(html):
        raw = m.group(1).strip()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue

        # JSON-LD can be a list or a single object.
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue

            # Update product name from ld+json if better.
            if not product_name and item.get("name"):
                product_name = str(item["name"]).strip()

            # Primary: additionalProperty array.
            for prop in item.get("additionalProperty") or []:
                if not isinstance(prop, dict):
                    continue
                name = str(prop.get("name", "") or "").strip()
                value = prop.get("value")
                if value is None:
                    value = prop.get("unitText") or prop.get("unitCode") or ""
                val_str = str(value).strip()
                if not name or not val_str:
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                specs.append({"name": name, "value": val_str})

    # Fallback: HTML table if JSON-LD yielded nothing.
    if not specs:
        for tr_m in _TR_RE.finditer(html):
            tds = _TD_RE.findall(tr_m.group(1))
            if len(tds) < 2:
                continue
            name = _text(tds[0])
            value = _text(tds[1])
            if not name or not value:
                continue
            if len(name) > 80 or not re.search(r"[а-яёa-zA-Z0-9]", name):
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            specs.append({"name": name, "value": value})

    return product_name, specs


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

class OnlinerSource(AttributeSource):
    """Verbatim specs from Onliner.by via public REST search + JSON-LD page parse.

    Russian-language values — no translation needed.
    Brand-gate: product name from API must contain brand + model tokens.
    Enum attributes: resolve_value_id; None → drop (verbatim-safe).
    Emits Source.WB_CARD (same verbatim semantics as RegardSource).
    """

    def __init__(self, **kwargs: Any) -> None:
        _ = kwargs
        self._judge = WbCardJudge()
        self._cache: OrderedDict[tuple[str, str], list[AttributeValue]] = OrderedDict()

    @property
    def source_type(self) -> Source:
        return Source.WB_CARD

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        return bool(
            context.product_name
            and len(context.product_name.strip()) >= 5
        )

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        if not targets or not context.product_name:
            return []

        already_filled = already_filled or []

        # Skip-guard: ≥80% targets already filled.
        filled_ids = {av.attribute_id for av in already_filled if av.confidence >= 0.85}
        if targets and len(filled_ids & {t.id for t in targets}) / len(targets) >= _SKIP_FILL_RATIO:
            logger.debug("[Onliner] skip (≥%.0f%% targets filled)", _SKIP_FILL_RATIO * 100)
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
                "[Onliner] total timeout (30s) for '%s' — skipping",
                context.product_name[:60],
            )
            return []
        except Exception as exc:
            logger.warning(
                "[Onliner] unexpected error for '%s': %s",
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

        # Build search query: brand + significant model tokens.
        model_tokens = _extract_model_tokens(full_name)
        if brand and model_tokens:
            query = f"{brand} {' '.join(model_tokens[:4])}"
        elif brand:
            query = brand + " " + full_name[:40]
        else:
            query = full_name[:60]

        logger.info("[Onliner] search query: '%s'", query)
        product_urls = await self._search(query)
        if not product_urls:
            logger.info("[Onliner] no product URLs found")
            return []

        for product_name_hint, url in product_urls[:_MAX_CANDIDATES]:
            html = await self._fetch_page(url)
            if not html:
                continue

            product_name, specs = _parse_onliner_specs(html)
            if not product_name:
                product_name = product_name_hint
            if not specs:
                continue

            if not self._brand_gate(product_name, brand, full_name):
                logger.info(
                    "[Onliner] brand-gate FAIL: product='%s' brand='%s' query='%s'",
                    product_name[:80], brand, full_name[:60],
                )
                continue

            score = self._score_title(product_name, full_name)
            mode = self._classify_match(score)
            if mode == "skip":
                logger.info(
                    "[Onliner] score=%.1f < %.0f — skip url=%s",
                    score, _BRAND_LINE_THRESHOLD, url[:80],
                )
                continue

            logger.info(
                "[Onliner] match=%s score=%.1f url=%s product='%s'",
                mode, score, url[:80], product_name[:60],
            )
            return self._map_characteristics(specs, targets, context, mode, product_name, score)

        logger.info("[Onliner] no brand-gate-passing product found")
        return []

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    async def _search(self, query: str) -> list[tuple[str, str]]:
        """Onliner catalog search API → list of (product_name, url) tuples."""
        try:
            async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _CHROME_UA},
            ) as client:
                resp = await client.get(
                    _SEARCH_API,
                    params={"query": query},
                )
                if resp.status_code != 200:
                    logger.info("[Onliner] search API HTTP %s — skip", resp.status_code)
                    return []
                data = resp.json()
        except Exception as exc:
            logger.info("[Onliner] search API error: %s", exc)
            return []

        products = data.get("products") or []
        results: list[tuple[str, str]] = []
        seen_urls: set[str] = set()
        for p in products:
            if not isinstance(p, dict):
                continue
            url = (p.get("url") or "").strip()
            name = str(p.get("full_name") or p.get("name") or "").strip()
            if not url:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)
            results.append((name, url))

        logger.info("[Onliner] search → %d product URLs", len(results))
        return results

    async def _fetch_page(self, url: str) -> Optional[str]:
        """Plain httpx GET — Onliner returns HTTP 200 without anti-bot."""
        try:
            async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _CHROME_UA},
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.info(
                        "[Onliner] HTTP %s for %s — skip", resp.status_code, url[:80]
                    )
                    return None
                raw = resp.content[:_MAX_HTML_BYTES]
                return raw.decode("utf-8", errors="replace")
        except Exception as exc:
            logger.info("[Onliner] fetch error for %s: %s", url[:80], exc)
            return None

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
    # Mapping (mirror RegardSource._map_characteristics — RU values, no translation)
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
        evidence = f"onliner:{product_name[:50]} | score={score:.1f}"
        conf = _CONF_EXACT if mode == "exact" else _CONF_BRAND_LINE

        results: list[AttributeValue] = []
        used_ids: set[int] = set()

        for spec in specs:
            char_name = spec["name"].strip()
            char_val = spec["value"].strip()
            if not char_name or not char_val:
                continue
            char_name_low = char_name.lower()
            char_name_norm = _norm_char_name(char_name)

            if mode == "brand_line" and char_name_low in _BRAND_LINE_BLACKLIST:
                continue

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
                        logger.debug("[Onliner] resolve_value_id (list) failed: %s", exc)
            else:
                value_out = char_val
                if target.type == "enum" and cat_id and type_id:
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                    except Exception as exc:
                        logger.debug("[Onliner] resolve_value_id failed: %s", exc)
                    if value_id is None:
                        logger.debug(
                            "[Onliner] enum attr %s value '%s' not in dict — drop",
                            target.id, char_val[:40],
                        )
                        used_ids.discard(target_id)
                        continue
                elif cat_id and type_id:
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                    except Exception as exc:
                        logger.debug("[Onliner] resolve_value_id failed: %s", exc)

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
            "[Onliner] %s mode → %d attrs filled (from %d specs, %d targets)",
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
