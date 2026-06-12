"""RegardSource — verbatim спеки из regard.ru (электроника, открытый сайт).

regard.ru возвращает HTTP 200 с полным server-rendered HTML (~3MB) на plain httpx
(без прокси / Scrappey). Несёт таблицы «Характеристики» для CPU/RAM/storage/
motherboard/GPU/peripherals — именно те категории, где IceCat неполный, а
Ozon/WB карточки тонкие.

Архитектура (зеркало WbCardSource / YandexMarketSource):
  1. Serper query: ``site:regard.ru <brand> <model>`` → URL карточки.
  2. Plain httpx GET → HTML.
  3. Парсинг таблицы «Характеристики» (tr > td пары name / value).
  4. Brand-gate: заголовок страницы должен содержать токены бренда + модели.
  5. Маппинг char name → target.name (exact / substring / rapidfuzz WRatio≥88).
  6. Enum attrs: resolve_value_id; при None — дроп (verbatim-safe, no guessing).

Position в pipeline: Stage 0.57 — после IceCat (0.55), до CompetitorRAG (0.7).
Fires ТОЛЬКО когда remaining > 0 (cost-aware: не тратим request если уже заполнено).

Cost: 1 Serper (~$0.001) + 1 httpx GET (бесплатно, ~0.3-1s).
Source.WB_CARD: same semantics (verbatim, pre-moderated spec table, no LLM).
"""
from __future__ import annotations

import asyncio
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
from app.services.enrichment.sources.wb_card_source import (
    _build_wb_query,
    _target_type_lemma,
)
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    resolve_value_id,
)
from app.services.providers.factory import get_web_search_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SERPER_NUM_RESULTS = 10
_MAX_CANDIDATES = 5          # max Serper URLs to try
_HTTP_TIMEOUT = 15.0         # seconds; regard.ru typically <1s
_MAX_HTML_BYTES = 4_000_000  # 4MB cap; regard pages ~3MB
_CACHE_MAX = 256

# Confidence levels (mirror WbCardSource)
_CONF_EXACT = 0.90
_CONF_BRAND_LINE = 0.82

# Match thresholds
_EXACT_THRESHOLD = 78.0
_BRAND_LINE_THRESHOLD = 60.0

# Skip-guard: if ≥80% targets already filled, don't bother fetching.
_SKIP_FILL_RATIO = 0.80

# Brand-gate: minimum shared token count between page title and (brand + model).
# At least 1 token from brand AND 1 from model must appear in title.
_BRAND_GATE_MIN_BRAND_TOKENS = 1
_BRAND_GATE_MIN_MODEL_TOKENS = 1

# Fuzzy name-match threshold for attribute name mapping.
_FUZZY_THRESHOLD = 88

# Chrome User-Agent to avoid trivial bot blocks.
_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

# Regard product URL pattern: /catalog/product/<digits> or /goods/<digits>
_REGARD_PRODUCT_RE = re.compile(
    r"regard\.ru/(?:catalog/product|goods)/(\d{3,})",
    re.IGNORECASE,
)

# Spec table row: regard.ru renders <tr> rows with two <td> cells.
# First <td> = name, second <td> = value (may contain nested <a>/<span>).
# We use a tight regex that captures the text of each <td> directly.
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_TD_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL | re.IGNORECASE)
# Strip inner HTML tags to extract text.
_TAGS_RE = re.compile(r"<[^>]+>")
# Spec section header on regard.ru: "Характеристики" (various wrapper patterns).
_SPEC_SECTION_RE = re.compile(
    r"[Хх]арактеристики",
    re.IGNORECASE,
)

# Blacklist of spec names that are model-specific identifiers — skip in
# brand_line mode (mirrors YandexMarketSource / WbCardSource).
_BRAND_LINE_BLACKLIST: frozenset[str] = frozenset(name.lower() for name in {
    "Артикул", "Код производителя", "MPN", "Партномер", "Штрихкод",
    "EAN", "GTIN", "Модель", "Код товара", "Vendor Part Number",
    "Серийный номер", "ID товара",
})

# Token RE for brand/model gate checks.
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _text(html_fragment: str) -> str:
    """Strip HTML tags and collapse whitespace."""
    return re.sub(r"\s+", " ", _TAGS_RE.sub("", html_fragment)).strip()


def _tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower().replace("ё", "е"))


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

def _parse_regard_specs(html: str) -> tuple[str, list[dict]]:
    """Parse regard.ru product page HTML.

    Returns:
        title: page/product title (for brand-gate).
        specs: list of {"name": ..., "value": ...} dicts.

    Parser strategy:
      1. Extract <title> for brand-gate.
      2. Find the spec section by locating "Характеристики" heading.
         regard.ru renders specs as a <table> inside a section with that label.
      3. Walk <tr> rows in the spec table: 2-cell rows → (name, value).
      4. Stop when we leave the spec table / hit a section that doesn't look
         like specs (empty rows, section dividers).
    """
    # -- Title --
    title = ""
    tm = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if tm:
        title = _text(tm.group(1))
    if not title:
        # og:title fallback
        om = re.search(
            r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE,
        )
        if om:
            title = om.group(1).strip()

    # -- Spec table --
    # Find the position of "Характеристики" section header.
    sec_m = _SPEC_SECTION_RE.search(html)
    if sec_m is None:
        return title, []

    # Slice HTML from spec-section start to limit search scope.
    # regard.ru pages are ~3MB; the spec table is within ~500KB after the header.
    slice_start = sec_m.start()
    slice_end = slice_start + 600_000
    html_slice = html[slice_start:slice_end]

    specs: list[dict] = []
    seen: set[str] = set()

    for tr_m in _TR_RE.finditer(html_slice):
        row_html = tr_m.group(1)
        tds = _TD_RE.findall(row_html)
        if len(tds) < 2:
            continue
        name = _text(tds[0])
        value = _text(tds[1])
        if not name or not value:
            continue
        # Skip rows where the first cell looks like a section header (very long
        # or contains no letter/digit content).
        if len(name) > 80 or not re.search(r"[а-яёa-zA-Z0-9]", name):
            continue
        name_low = name.lower()
        if name_low in seen:
            continue
        seen.add(name_low)
        specs.append({"name": name, "value": value})

    return title, specs


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

class RegardSource(AttributeSource):
    """Verbatim specs from regard.ru product pages via plain httpx.

    Covers electronics categories (CPU/RAM/storage/GPU/motherboard/peripherals)
    where IceCat Open API is weak and Ozon/WB cards are thin.

    Brand-gate: only accepts a page whose title contains tokens from both the
    product brand and model — prevents wrong-product spec injection.

    Enum attributes: value_id resolved via resolve_value_id; dropped (not
    guessed) when resolution returns None — verbatim-safe.

    Emits Source.WB_CARD: same semantics (verbatim table copy, no LLM).
    """

    def __init__(
        self,
        web_search_client: Any = None,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        self._search_client = web_search_client
        if self._search_client is None:
            try:
                self._search_client = get_web_search_client()
            except Exception as exc:
                logger.warning(
                    "[Regard] web search client unavailable (%s) — extract() → []", exc
                )
                self._search_client = None

        self._judge = WbCardJudge()
        self._cache: OrderedDict[tuple[str, str], list[AttributeValue]] = OrderedDict()

    @property
    def source_type(self) -> Source:
        # Same semantics as WB card: verbatim table copy, no LLM.
        return Source.WB_CARD

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        return bool(
            self._search_client
            and context.product_name
            and len(context.product_name.strip()) >= 5
        )

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        if not targets or not context.product_name or not self._search_client:
            return []

        already_filled = already_filled or []

        # Skip-guard: ≥80% targets already filled → don't spend the request.
        filled_ids = {av.attribute_id for av in already_filled if av.confidence >= 0.85}
        if targets and len(filled_ids & {t.id for t in targets}) / len(targets) >= _SKIP_FILL_RATIO:
            logger.debug(
                "[Regard] skip (≥%.0f%% targets filled)", _SKIP_FILL_RATIO * 100
            )
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
                "[Regard] total timeout (30s) for '%s' — skipping",
                context.product_name[:60],
            )
            return []
        except Exception as exc:
            logger.warning(
                "[Regard] unexpected error for '%s': %s",
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
        cat_leaf = context.category_path[-1] if context.category_path else None

        primary_query = _build_wb_query(full_name, context.brand, cat_leaf, max_tokens=5)
        fallback_query = _build_wb_query(full_name, context.brand, cat_leaf, max_tokens=3)
        queries = [primary_query]
        if fallback_query and fallback_query != primary_query:
            queries.append(fallback_query)

        urls: list[str] = []
        for q in queries:
            serper_q = f"site:regard.ru {q}".strip()
            logger.info("[Regard] Serper query: '%s'", serper_q)
            found = await self._search(serper_q)
            if found:
                urls = found
                break

        if not urls:
            logger.info("[Regard] no product URLs found")
            return []

        # Try URLs in order; use first one that passes brand-gate.
        for url in urls[:_MAX_CANDIDATES]:
            html = await self._fetch_page(url)
            if not html:
                continue
            title, specs = _parse_regard_specs(html)
            if not specs:
                continue

            if not self._brand_gate(title, context.brand, context.product_name):
                logger.info(
                    "[Regard] brand-gate FAIL: title='%s' brand='%s' product='%s'",
                    title[:80], context.brand, context.product_name[:60],
                )
                continue

            score = self._score_title(title, full_name)
            mode = self._classify_match(score)
            if mode == "skip":
                logger.info(
                    "[Regard] score=%.1f < %.0f — skip url=%s",
                    score, _BRAND_LINE_THRESHOLD, url[:80],
                )
                continue

            logger.info(
                "[Regard] match=%s score=%.1f url=%s title='%s'",
                mode, score, url[:80], title[:80],
            )
            return self._map_characteristics(specs, targets, context, mode, title, score)

        logger.info("[Regard] no brand-gate-passing page found")
        return []

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------

    async def _search(self, serper_query: str) -> list[str]:
        """Serper → regard.ru product URLs."""
        try:
            results = await self._search_client.search(
                serper_query, num_results=_SERPER_NUM_RESULTS
            )
        except Exception as exc:
            logger.info("[Regard] Serper error: %s", exc)
            return []

        organic = getattr(results, "organic_results", None) or []
        out: list[str] = []
        seen: set[str] = set()
        for item in organic:
            link = getattr(item, "link", "") or ""
            if not _REGARD_PRODUCT_RE.search(link):
                continue
            if link in seen:
                continue
            seen.add(link)
            out.append(link)

        logger.info("[Regard] Serper → %d product URLs", len(out))
        return out

    async def _fetch_page(self, url: str) -> Optional[str]:
        """Plain httpx GET — regard.ru returns HTTP 200 without anti-bot."""
        try:
            async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": _CHROME_UA},
            ) as client:
                resp = await client.get(url)
                if resp.status_code != 200:
                    logger.info(
                        "[Regard] HTTP %s for %s — skip",
                        resp.status_code, url[:80],
                    )
                    return None
                # Decode; limit to _MAX_HTML_BYTES to avoid processing huge pages.
                raw = resp.content[:_MAX_HTML_BYTES]
                return raw.decode("utf-8", errors="replace")
        except Exception as exc:
            logger.info("[Regard] fetch error for %s: %s", url[:80], exc)
            return None

    # ------------------------------------------------------------------
    # Brand-gate
    # ------------------------------------------------------------------

    @staticmethod
    def _brand_gate(title: str, brand: Optional[str], product_name: str) -> bool:
        """Require brand tokens AND at least one model token in page title.

        Prevents wrong-product spec injection: if the Serper result points to
        a different product (e.g. wrong model) the gate drops it.

        Logic:
          - Tokenize title (lowercase).
          - Brand tokens: at least _BRAND_GATE_MIN_BRAND_TOKENS from brand must
            appear in title tokens.
          - Model tokens: from _extract_model_tokens(product_name) — alphanumeric
            model identifiers (e.g. "i5-12400", "rtx3080"). At least
            _BRAND_GATE_MIN_MODEL_TOKENS must appear in title.
          - If brand is empty: skip brand-token check (only model tokens checked).
          - If no model tokens extractable: relax to brand-only check.
        """
        title_tokens = set(_tokenize(title))
        if not title_tokens:
            return False

        brand_ok = True
        if brand and brand.strip():
            brand_tokens = _tokenize(brand)
            brand_ok = sum(1 for t in brand_tokens if t in title_tokens) >= _BRAND_GATE_MIN_BRAND_TOKENS

        model_tokens = _extract_model_tokens(product_name)
        if model_tokens:
            # _extract_model_tokens may return hyphenated tokens like "i5-12400".
            # _tokenize splits on hyphens, so check either the whole token OR any
            # of its sub-parts appear in title_tokens (e.g. "12400" ∈ title_tokens).
            def _model_token_in_title(token: str) -> bool:
                t_low = token.lower()
                if t_low in title_tokens:
                    return True
                # Split hyphenated / slash-separated model identifiers.
                parts = _TOKEN_RE.findall(t_low)
                # Require all parts ≥2 chars to appear (avoids single-letter false positives).
                meaningful = [p for p in parts if len(p) >= 2]
                return bool(meaningful) and all(p in title_tokens for p in meaningful)
            model_ok = any(_model_token_in_title(t) for t in model_tokens)
        else:
            # No model tokens — fall back to brand-only check.
            model_ok = True

        return brand_ok and model_ok

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_title(title: str, query: str) -> float:
        """rapidfuzz partial_ratio + token_sort_ratio average."""
        try:
            from rapidfuzz import fuzz
            t = title.lower()
            q = query.lower()
            return (fuzz.partial_ratio(q, t) + fuzz.token_sort_ratio(q, t)) / 2.0
        except ImportError:
            # Fallback: simple token overlap ratio.
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
    # Mapping: char name → target attribute_id → value_id (Ozon dict)
    # ------------------------------------------------------------------

    def _map_characteristics(
        self,
        specs: list[dict],
        targets: list[TargetAttribute],
        context: ExtractionContext,
        mode: str,
        title: str,
        score: float,
    ) -> list[AttributeValue]:
        """Map regard.ru spec names to target attributes.

        Steps (mirror YandexMarketSource._map_characteristics):
          1. Build name lookup: target.name (raw + normalized) + Ozon dict name.
          2. For each spec: exact → norm → substring → rapidfuzz WRatio≥88 match.
          3. brand_line mode: skip blacklisted (model-id) and numeric attrs.
          4. Enum attrs: resolve_value_id; None → DROP (verbatim-safe).
          5. Collection attrs: per-element resolve.
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
        evidence = f"regard.ru:{title[:50]} | score={score:.1f}"
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
                        logger.debug("[Regard] resolve_value_id (list) failed: %s", exc)
            else:
                value_out = char_val
                # For enum attributes: require resolve_value_id to succeed.
                # If it returns None the value is not in the Ozon dictionary →
                # drop it rather than emit an unresolvable value (verbatim-safe).
                if target.type == "enum" and cat_id and type_id:
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                    except Exception as exc:
                        logger.debug("[Regard] resolve_value_id failed: %s", exc)
                    if value_id is None:
                        logger.debug(
                            "[Regard] enum attr %s value '%s' not in dict — drop",
                            target.id, char_val[:40],
                        )
                        used_ids.discard(target_id)
                        continue
                elif cat_id and type_id:
                    # Non-enum: try to resolve but don't drop on miss.
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                    except Exception as exc:
                        logger.debug("[Regard] resolve_value_id failed: %s", exc)

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
            "[Regard] %s mode → %d attrs filled (from %d specs, %d targets)",
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
