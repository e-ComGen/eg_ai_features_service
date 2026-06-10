"""ScrapflyOzonSource — last-resort Ozon card gap-filler via Scrapfly.

COST-CONTROL DESIGN (critical):
  This source fires ONLY when ALL of the following hold simultaneously:
    (a) The regular OzonCardSource (Scrappey) returned 0 results this run —
        meaning Scrappey timed out / got blocked → Ozon card was NOT obtained.
    (b) There are still-empty target attributes that an Ozon card would
        plausibly carry (gap_attrs > 0 after all previous sources).
    (c) The env flag SCRAPFLY_OZON_FALLBACK_ENABLED is set to "true" / "1".

  If Scrappey already delivered the Ozon card, this source is NEVER called.

ALGORITHM:
  1. Gate check: enabled flag + Scrappey-card-missing + remaining gaps.
  2. Search step: reuse OzonCardSource._compress_search_query to form a query,
     then Scrapfly-fetch https://www.ozon.ru/search/?text=<query> (30 credits).
     Parse product tiles via OzonCardSource._parse_search_tiles_html (reused).
  3. Match step: rapidfuzz scoring via OzonCardSource._pick_best_match (reused).
  4. Features fetch: Scrapfly-fetch https://www.ozon.ru/product/<slug>-<pid>/features/
     (30 credits). FIRST try the existing JSON data-state parser (OzonCardSource.
     _parse_characteristics_html). If that returns 0 results (Vue not hydrated in
     the Scrapfly snapshot), fall back to <dl><dt><dd> HTML parser that extracts
     {name: value} pairs from elements with class="webCharacteristics".
  5. Mapping: reuse OzonCardSource._map_characteristics verbatim.
  6. Fill only still-empty attributes (source emits with source=OZON_CARD so the
     existing merge / judge / confidence guards all apply as-is).

PARSER PATH (auto-detected):
  - JSON path (data-state): if OzonCardSource._parse_characteristics_html returns
    ≥1 chars → use it. Confidence same as regular OzonCard (0.93 exact / 0.85 brand_line).
  - DL/DT/DD path: fallback when JSON is empty. Extracts pairs from rendered
    <dl><dt>name</dt><dd>value</dd> ... </dl> blocks. Same confidence.

SOURCE TAG:
  emits Source.OZON_CARD with evidence f"ozon_card:scrapfly:{title[:40]}" so the
  merge layer treats it identically to regular OzonCard values.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.ozon_card_judge import OzonCardJudge
from app.services.enrichment.sources.ozon_card_source import (
    OzonCardSource,
    _compress_search_query,
)
from app.services.providers import scrapfly_client as _scrapfly_mod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OZON_SEARCH_URL = "https://www.ozon.ru/search/"
_OZON_PRODUCT_BASE = "https://www.ozon.ru/product/"

# Confidence mirrors OzonCardSource (exact=0.93, brand_line=0.85).
_CONF_EXACT = 0.93
_CONF_BRAND_LINE = 0.85

# Number of search tiles to score (mirrors OzonCardSource._MATCH_TOP_N).
_MATCH_TOP_N = 8

# Hard timeout for the entire Scrapfly gap-fill per product.
# 2x Scrapfly calls @ ~30s each = 60s max + 5s margin.
_TOTAL_TIMEOUT = 95.0

# Min HTML size to consider a valid Scrapfly response.
_MIN_VALID_HTML_LEN = 50_000

# Ozon renders characteristics as a `data-widget="webCharacteristics"` div containing
# individual `<dl>` blocks (one per attribute).  The DL class is an obfuscated hash
# (e.g. "pdp_ia9") — NOT "webCharacteristics".  We find the widget container first,
# then parse ALL <dl><dt>name</dt><dd>value</dd></dl> blocks inside it.
#
# Structure (from live Scrapfly HTML):
#   <div data-widget="webCharacteristics" class="pdp_a6e">
#     <dl class="pdp_ia9">
#       <dt class="pdp_ia8"><span class="pdp_ai9">Тип</span></dt>
#       <dd class="pdp_i8a">Наушники</dd>
#     </dl>
#     ...
#   </div>
# Find the start position of data-widget="webCharacteristics" div.
# We use this to locate where characteristics DLs begin in the page.
_WC_WIDGET_START_RE = re.compile(
    r'data-widget=["\']webCharacteristics["\']',
    re.IGNORECASE,
)
# All <dl>...</dl> blocks in HTML.
_DL_BLOCK_RE = re.compile(
    r'<dl[^>]*>(.*?)</dl>',
    re.DOTALL | re.IGNORECASE,
)
_DT_RE = re.compile(r'<dt[^>]*>(.*?)</dt>', re.DOTALL | re.IGNORECASE)
_DD_RE = re.compile(r'<dd[^>]*>(.*?)</dd>', re.DOTALL | re.IGNORECASE)
_TAG_STRIP_RE = re.compile(r'<[^>]+>')
_WHITESPACE_RE = re.compile(r'\s+')


def _strip_tags(html: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = _TAG_STRIP_RE.sub(' ', html)
    return _WHITESPACE_RE.sub(' ', text).strip()


def _parse_dl_pairs(html_corpus: str) -> list[tuple[str, str]]:
    """Extract (name, value) pairs from all <dl><dt>...</dt><dd>...</dd></dl> blocks.

    Ozon uses one <dl> per characteristic, each with one <dt> (name) and one <dd>
    (value). Handles cases where dt/dd contain nested HTML (strips tags).
    """
    pairs: list[tuple[str, str]] = []
    for dl_match in _DL_BLOCK_RE.finditer(html_corpus):
        dl_inner = dl_match.group(1)
        dt_matches = list(_DT_RE.finditer(dl_inner))
        dd_matches = list(_DD_RE.finditer(dl_inner))
        for i, dt_m in enumerate(dt_matches):
            if i >= len(dd_matches):
                break
            name = _strip_tags(dt_m.group(1))
            value = _strip_tags(dd_matches[i].group(1))
            if name and value:
                pairs.append((name, value))
    return pairs


def parse_dl_characteristics(html: str) -> list[dict]:
    """Extract {name, value, value_ids} pairs from Ozon's rendered characteristics HTML.

    Ozon renders characteristics inside a `data-widget="webCharacteristics"` div,
    with individual `<dl><dt>name</dt><dd>value</dd></dl>` blocks for each attribute.
    The DL class is an obfuscated hash — NOT "webCharacteristics".

    Strategy:
      1. Find the data-widget="webCharacteristics" marker and take the HTML slice
         starting there (captures all DL blocks in that section of the page).
      2. Parse all <dl> blocks from that slice.
      3. Fallback: if no marker or 0 results, parse all DL blocks in the full HTML.

    Returns a list of dicts compatible with OzonCardSource._map_characteristics:
      [{"name": str, "value": str, "value_ids": []}]
    """
    out: list[dict] = []
    seen: set[str] = set()

    def _add_pairs(pairs: list[tuple[str, str]]) -> None:
        for name, value in pairs:
            if len(name) > 60 or len(name.split()) > 6:
                continue
            name_low = name.lower()
            if name_low in seen:
                continue
            seen.add(name_low)
            out.append({"name": name, "value": value, "value_ids": []})

    # Primary: locate the webCharacteristics widget and scan DL blocks from there.
    wc_match = _WC_WIDGET_START_RE.search(html)
    if wc_match:
        # Take HTML from widget start position onward (up to a reasonable slice).
        # 50KB covers all characteristic DLs without pulling in unrelated sections.
        corpus = html[wc_match.start(): wc_match.start() + 50_000]
        _add_pairs(_parse_dl_pairs(corpus))

    # Fallback: if primary yielded nothing (widget marker missing or layout change),
    # scan the full HTML for DL blocks.
    if not out:
        _add_pairs(_parse_dl_pairs(html))

    return out


class ScrapflyOzonSource(AttributeSource):
    """Last-resort Ozon card gap-filler using Scrapfly.

    Only fires when Scrappey OzonCardSource returned 0 results AND
    SCRAPFLY_OZON_FALLBACK_ENABLED=true. Reuses all OzonCardSource
    parsing/matching/mapping logic — only the HTTP transport differs.

    Cost: 60 credits/product (30 search + 30 features).
    Latency: 40-90s end-to-end (2x Scrapfly JS render calls).
    """

    def __init__(
        self,
        scrapfly_key: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        from app import config as _cfg
        # Key: explicit arg > env > config
        self._key = (
            scrapfly_key
            or os.environ.get("SCRAPFLY_API_KEY")
            or os.environ.get("SCRAPFLY_KEY")
            or _cfg.SCRAPFLY_API_KEY
        )
        # Enabled: explicit arg > env > config
        if enabled is not None:
            self._enabled = enabled
        else:
            self._enabled = _cfg.SCRAPFLY_OZON_FALLBACK_ENABLED

        if not self._key:
            logger.warning(
                "[ScrapflyOzon] SCRAPFLY_API_KEY not configured — source will always return []."
            )

        # Reuse the same judge as OzonCardSource (it validates Ozon char quality).
        self._judge = OzonCardJudge()

        # We reuse OzonCardSource parsing/matching/mapping without its HTTP layer.
        # Instantiate with no scrappey_key so it NEVER makes network calls
        # (we only use its static/instance parsing methods).
        self._ozon_card_helper = OzonCardSource(scrappey_key=None)

    @property
    def source_type(self) -> Source:
        return Source.OZON_CARD

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Applicable only when enabled, key present, and product_name non-trivial."""
        return bool(
            self._enabled
            and self._key
            and context.product_name
            and len(context.product_name.strip()) >= 5
        )

    def get_judge(self) -> LlmJudge:
        return self._judge

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
        *,
        ozon_card_obtained: bool = False,
    ) -> list[AttributeValue]:
        """Extract attributes via Scrapfly Ozon card fetch.

        Parameters
        ----------
        ozon_card_obtained:
            Set to True by the pipeline when OzonCardSource (Scrappey) already
            returned ≥1 AttributeValues this run. When True, this method
            immediately returns [] — no Scrapfly call is made.
        """
        if not targets or not context.product_name:
            return []
        if not self._enabled or not self._key:
            return []

        # GATE (a): Scrappey already got the card → skip entirely.
        if ozon_card_obtained:
            logger.debug(
                "[ScrapflyOzon] skip — Scrappey already obtained Ozon card for '%s'",
                context.product_name[:60],
            )
            return []

        # GATE (b): no remaining gaps.
        already_filled = already_filled or []
        filled_attr_ids = {av.attribute_id for av in already_filled if av.confidence >= 0.80}
        gap_targets = [t for t in targets if t.id not in filled_attr_ids]
        if not gap_targets:
            logger.debug(
                "[ScrapflyOzon] skip — no gap attributes for '%s'",
                context.product_name[:60],
            )
            return []

        logger.info(
            "[ScrapflyOzon] gap-fill triggered for '%s' (%d gap attrs)",
            context.product_name[:60], len(gap_targets),
        )

        try:
            return await asyncio.wait_for(
                self._do_extract(context, gap_targets),
                timeout=_TOTAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[ScrapflyOzon] total timeout (%.0fs) for '%s' — giving up",
                _TOTAL_TIMEOUT, context.product_name[:60],
            )
            return []
        except Exception as exc:
            logger.warning(
                "[ScrapflyOzon] unexpected error for '%s': %s",
                context.product_name[:60], exc,
            )
            return []

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    async def _do_extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        scrapfly_fetch = _scrapfly_mod.scrapfly_fetch

        full_name = context.product_name.strip()
        cat_leaf = context.category_path[-1] if context.category_path else None
        query = _compress_search_query(
            full_name, context.brand, max_tokens=5, category_name=cat_leaf
        )

        logger.info("[ScrapflyOzon] search query: '%s' (product: '%s')", query, full_name[:60])

        # ---- Step 1: Search ----
        # Pass the raw query string (with spaces); scrapfly_client.py encodes the entire
        # target URL via urllib.parse.urlencode(..., quote_via=quote), producing %20 for
        # spaces in the url= param. Pre-encoding here would cause double-encoding (%2520).
        search_result = await scrapfly_fetch(
            f"{_OZON_SEARCH_URL}?text={query}",
            render_js=True,
        )
        if not search_result.success or not search_result.content:
            logger.info(
                "[ScrapflyOzon] search fetch failed: %s (credits_used=%d)",
                search_result.error, search_result.credits_used,
            )
            return []
        if len(search_result.content) < _MIN_VALID_HTML_LEN:
            logger.info(
                "[ScrapflyOzon] search HTML too short (%d chars) — likely empty SPA",
                len(search_result.content),
            )
            return []

        tiles = OzonCardSource._parse_search_tiles_html(search_result.content)
        if not tiles:
            logger.info("[ScrapflyOzon] no search tiles parsed")
            return []

        # ---- Step 2: Match ----
        top_tile, top_score = OzonCardSource._pick_best_match(
            query,
            tiles[:_MATCH_TOP_N],
            category_leaf=cat_leaf,
            query_brand=context.brand,
            query_name=full_name,
        )
        if top_tile is None:
            logger.info("[ScrapflyOzon] no tile passed matching threshold")
            return []

        mode = OzonCardSource._classify_match(top_score)
        if mode == "skip":
            logger.info(
                "[ScrapflyOzon] best score=%.1f below threshold — skip", top_score
            )
            return []

        slug = (top_tile.get("slug") or "").strip()
        pid = (top_tile.get("pid") or "").strip()
        title = (top_tile.get("title") or "").strip()

        if not slug or not pid:
            logger.info("[ScrapflyOzon] no slug/pid in best tile")
            return []

        logger.info(
            "[ScrapflyOzon] match=%s score=%.1f title='%s' pid=%s",
            mode, top_score, title[:80], pid,
        )

        # ---- Step 3: Features page fetch ----
        # wait_for_selector ensures Scrapfly captures the page AFTER the
        # webCharacteristics widget has fully rendered in the DOM. Without it,
        # Ozon's lazy-loading SPA may not have hydrated the characteristics section
        # yet when the snapshot is taken, causing both parsers to return 0 results.
        features_url = f"{_OZON_PRODUCT_BASE}{slug}-{pid}/features/"
        features_result = await scrapfly_fetch(
            features_url,
            render_js=True,
            wait_for_selector="[data-widget='webCharacteristics']",
        )
        if not features_result.success or not features_result.content:
            logger.info(
                "[ScrapflyOzon] features fetch failed: %s (credits_used=%d)",
                features_result.error, features_result.credits_used,
            )
            return []

        total_credits = search_result.credits_used + features_result.credits_used
        features_html = features_result.content

        # ---- Step 4: Parse characteristics ----
        # Try JSON data-state parser first (works when Vue SSR is intact).
        chars = OzonCardSource._parse_characteristics_html(features_html)
        parser_used = "json_data_state"

        if not chars:
            # Fallback: <dl><dt><dd> HTML parser (works when Vue is not hydrated).
            chars = parse_dl_characteristics(features_html)
            parser_used = "dl_dt_dd_html"

        logger.info(
            "[ScrapflyOzon] features: parser=%s, chars=%d, credits_total=%d",
            parser_used, len(chars), total_credits,
        )

        if not chars:
            logger.info("[ScrapflyOzon] 0 characteristics extracted from features page")
            return []

        # ---- Step 5: Map characteristics → AttributeValues ----
        evidence_suffix = f"ozon_card:scrapfly:{title[:40]}"
        return self._ozon_card_helper._map_characteristics(
            chars,
            targets,
            context,
            mode=mode,
            title=title,
            score=top_score,
            evidence_override=evidence_suffix,
        )
