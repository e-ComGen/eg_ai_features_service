"""YandexMarketSource — копирует характеристики из живой карточки market.yandex.ru.

PROTOTYPE / NOT WIRED INTO PIPELINE. Зеркалит структуру WbCardSource/OzonCardSource:
Serper находит URL карточки (`site:market.yandex.ru`), карточка скачивается, из неё
парсятся характеристики (Характеристики table), мапятся на Ozon-словарь, и
эмитятся как AttributeValue.

──────────────────────────────────────────────────────────────────────────────
ДОСТУПНОСТЬ (research-итог, см. report):
  • Официального ПУБЛИЧНОГО product-content API НЕТ. Yandex Market Partner API —
    seller-side (businessId + OAuth, управляет ТВОИМИ офферами, не читает чужие
    карточки). Для копирования характеристик чужого товара он непригоден.
  • Serper `site:market.yandex.ru <query>` НАХОДИТ URL карточек товара
    (/product--<slug>/<id> и /product/<id>). Подтверждено зеркалом wb_card.
  • Сама страница market.yandex.ru за Yandex SmartCaptcha: IP+fingerprint-гейт,
    режет datacenter-IP. Прямой httpx с серверного IP → captcha/redirect. Нужен
    Scrappey (как OzonCardSource) или аналогичный super-proxy. БЕЗ Scrappey-бюджета
    живой fetch не проверить → _fetch_card_html помечен TODO и по умолчанию
    отдаёт None (graceful []).
  • Что на странице ЕСТЬ для парсинга (когда HTML получен):
      1. `<script type="application/ld+json">` Product-разметка: name, brand,
         offers, category — НО обычно БЕЗ полной таблицы характеристик (только
         оффер-данные). Используем для type-gate / title.
      2. Встроенный state-блоб `window.__INITIAL_STATE__ = {...}` (или `__NEXT_DATA__`)
         в `<script>` — несёт specs/specifications с парами {name, value}. Это
         основной источник характеристик. Точная форма зависит от A/B-рендера Маркета,
         поэтому парсер устойчив к нескольким вариантам ключей (specs / specifications
         / parameterValues / characteristics; элементы {name,value} | {key,value} |
         {title, ...}).

ENUM: переиспользуем Source.OZON_CARD (НЕ добавляем Source.YANDEX_MARKET).
Причина — lower-risk для НЕ-подключённого источника: добавление enum-члена требует
правки shared base.py (Source + SOURCE_PRIORITY + SOURCE_CONFIDENCE_THRESHOLDS),
которую параллельно редактируют другие агенты. Семантически OZON_CARD = «копия с
pre-modered marketplace-карточки, смапленная на Ozon-словарь» — ровно наш случай.
При реальном подключении к pipeline enum можно выделить отдельно.

Cost: 1 Serper-запрос (~$0.001) + 1 Scrappey-запрос/товар (~$0.0002-0.001).
Latency: ~1-3s Serper + 8-20s Scrappey.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from typing import Any, Optional, Union

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.ozon_card_judge import OzonCardJudge
from app.services.enrichment.prompt_router import filter_already_filled_targets
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    resolve_value_id,
)
# Read-only reuse of helpers (these modules do NOT import this one → no cycle).
from app.services.enrichment.sources.ozon_card_source import (
    _norm_char_name,
    _normalize_model,
    _split_multivalue,
    _extract_model_tokens,
    _extract_alpha_model_tokens,
    _extract_gender_signal,
    _gender_conflict,
)
from app.services.enrichment.sources.wb_card_source import (
    _build_wb_query,        # type-word-preserving Serper query builder
    _target_type_lemma,     # target type lemma for type-gate
    _type_lemma,            # noun-aware lemma for card-type compatibility check
)
from app.services.providers.factory import get_web_search_client
from app.services.providers.scrappey_client import scrappey_fetch as _scrappey_fetch_page

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

_SERPER_NUM_RESULTS = 10
_MAX_CANDIDATES = 8

# Confidence — параллельно OzonCard/WbCard.
_CONF_EXACT = 0.93
_CONF_BRAND_LINE = 0.85

_EXACT_THRESHOLD = 78.0
_BRAND_LINE_THRESHOLD = 60.0

# Skip-guard: общий порог заполнения (generic). Если ≥80% всех targets уже
# заполнены high-confidence — не тратим Scrappey. НО: порог обходится когда
# высокоценные content-поля (Состав/Материал/Сезон) ещё пусты — это ключевые
# apparel-атрибуты, ради которых YM и запускается (data-desert после WB/Ozon).
_SKIP_FILL_RATIO = 0.80

# Ключевые слова content-таргетов, которые ЗАПРЕЩАЮТ skip по _SKIP_FILL_RATIO.
# Определяются по имени (lower, substring) — без хардкода attribute_id:
# "состав" → "Состав материала", "Состав"; "материал" → "Материал", "Материал верха";
# "сезон" → "Сезон". Если ХОТЬ ОДИН незаполненный target содержит любое из
# этих слов — skip отменяется, YM запускается.
_CONTENT_KEYWORDS: frozenset[str] = frozenset({"состав", "материал", "сезон"})

# LRU
_CACHE_MAX = 256

# Timeout constants — mirror OzonCardSource to prevent hangs.
# 60s matches the httpx-level timeout floor in OzonCard/_HTTP_TIMEOUT.
# 90s is the hard asyncio.wait_for cap (_OZON_CARD_TOTAL_TIMEOUT) — cloned here
# so YandexMarket can't stall the pipeline on N-retry × 60s Scrappey slowness.
_SCRAPPEY_HTTP_TIMEOUT = 60.0
_YM_TOTAL_TIMEOUT = 90.0

# Извлечение product-id/slug из URL карточки Маркета. Покрывает форматы:
#   https://market.yandex.ru/product--<slug>/<digits>
#   https://market.yandex.ru/product/<digits>
#   .../card/<digits>
_YM_PRODUCT_RE = re.compile(
    r"market\.yandex\.[a-z]+/(?:product(?:--[^/?#]+)?|card)/(\d{4,})",
    re.IGNORECASE,
)

# JSON-LD blocks.
_JSONLD_RE = re.compile(
    r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.DOTALL | re.IGNORECASE,
)
# Embedded state blob. Маркет рендерит начальное состояние в одном из вариантов.
# Greedy-to-</script> capture: state blobs contain nested braces, so a non-greedy
# {.*?} stops at the first inner "}" and yields invalid JSON. We capture the whole
# script body and let _balanced_json_slice() trim it back to a balanced object.
_INITIAL_STATE_RE = re.compile(
    r'(?:window\.)?__INITIAL_STATE__\s*=\s*(\{.*?)</script>',
    re.DOTALL,
)
_NEXT_DATA_RE = re.compile(
    r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>\s*(\{.*?)</script>',
    re.DOTALL,
)
# Live Yandex Market (2024-2026) does NOT use __INITIAL_STATE__ / __NEXT_DATA__.
# Instead it renders product specs into search-tile blobs of the form:
#   "specs":{"friendly":["Цвет: Чёрный","Пол: Мужской",...]}
# These are brief key-value strings that we can parse by splitting on ": ".
# They appear in search-result tiles embedded in the page alongside the target
# product, so we harvest ALL tiles and let the caller filter by score.
# Pattern uses [^\]]* (no-] char class) rather than .*? to reliably capture
# the list content without accidentally matching past the closing bracket when
# specs strings contain Unicode escape sequences. re.DOTALL not needed here.
_SPECS_FRIENDLY_RE = re.compile(
    r'"specs"\s*:\s*\{[^}]*"friendly"\s*:\s*(\[[^\]]*\])',
)

# YM 2024+ SSR: full spec table is rendered server-side into the product card HTML.
# Each spec row uses:
#   <span data-auto="product-spec" ...>SPEC NAME</span>
#   ... sibling div ...
#   EITHER: <... data-zone-name="specLink" data-zone-data="{...,\"text\":\"VALUE\"}">
#   OR:     <span>VALUE TEXT</span>
# We match spec-name spans and scope the value lookup to the region between
# consecutive spec-name spans (tight scoping prevents cross-row value leakage).
_SPEC_NAME_RE = re.compile(
    r'data-auto=["\']product-spec["\'][^>]*>([^<]{1,120})</span>',
    re.DOTALL,
)
# specLink zone carries the value as JSON text field
_SPEC_LINK_RE = re.compile(
    r'data-zone-name=["\']specLink["\'][^>]*data-zone-data=["\']([^"\']+)["\']',
)
# Fallback: plain text span within the value container (_1_zPW class is the value side)
_SPEC_VALUE_SPAN_RE = re.compile(r'<span[^>]*>([^<]{1,300})</span>')

# Brand-line BLACKLIST — model-specific атрибуты (зеркало других card-sources).
_BRAND_LINE_BLACKLIST: frozenset[str] = frozenset(name.lower() for name in {
    "Артикул", "Код производителя", "MPN", "Партномер", "Серийный номер",
    "EAN", "GTIN", "ASIN", "Штрихкод", "Дата производства", "Модель",
    "Название модели", "ID товара", "ID карточки",
})

# Ключи, под которыми в state-блобе обычно лежат характеристики.
_SPEC_CONTAINER_KEYS = (
    "specs", "specifications", "characteristics", "parameterValues",
    "specification", "params", "attributes",
)


def _build_ym_query(
    full_name: str,
    brand: Optional[str],
    cat_leaf: Optional[str],
    max_tokens: int,
) -> str:
    """Serper-запрос для Я.Маркета, сохраняя тип-слово товара.

    Полностью переиспользует логику WbCard (_build_wb_query): чистит спеки/
    стоп-слова/бренд-дубли, ПРЕПЕНДИТ тип-слово (cat_leaf / ведущее сущ.) если
    его нет. Для Serper site:market.yandex.ru тип-слово так же критично, как для
    WB — без него выдача уходит в чужой класс товара.
    """
    return _build_wb_query(full_name, brand, cat_leaf, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class YandexMarketSource(AttributeSource):
    """Копия характеристик с live market.yandex.ru-карточки похожего товара.

    PROTOTYPE: live HTML-fetch за SmartCaptcha — TODO (нужен Scrappey-бюджет).
    Query-building и парсинг характеристик из state/JSON-LD реализованы полностью
    и юнит-тестируемы. Эмитит Source.OZON_CARD (см. module docstring про enum).
    """

    def __init__(
        self,
        web_search_client: Any = None,
        scrappey_key: Optional[str] = None,
        **kwargs: Any,
    ):
        _ = kwargs

        self._search_client = web_search_client
        if self._search_client is None:
            try:
                self._search_client = get_web_search_client()
            except Exception as exc:  # SERPER_API_KEY не задан и т.п.
                logger.warning(
                    "[YandexMarket] web search client недоступен (%s) — extract() вернёт [].",
                    exc,
                )
                self._search_client = None

        import os
        self._scrappey_key = scrappey_key or os.environ.get("SCRAPPEY_KEY")

        self._judge = OzonCardJudge()  # reuse: same accept-on-confidence semantics
        self._cache: "OrderedDict[tuple[str, str], list[AttributeValue]]" = OrderedDict()

    @property
    def source_type(self) -> Source:
        # See module docstring: reuse OZON_CARD rather than editing shared base.py.
        return Source.OZON_CARD

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

        filled_ids = {av.attribute_id for av in already_filled if av.confidence >= 0.85}
        if targets and len(filled_ids & {t.id for t in targets}) / len(targets) >= _SKIP_FILL_RATIO:
            # Don't skip when high-value content targets (Состав/Материал/Сезон) are
            # still unfilled — these are the exact fields YM is meant to rescue in the
            # apparel data-desert. A generic ≥80%-filled ratio must not silence YM when
            # composition is still empty.
            unfilled_ids = {t.id for t in targets} - filled_ids
            content_unfilled = any(
                any(kw in t.name.lower() for kw in _CONTENT_KEYWORDS)
                for t in targets
                if t.id in unfilled_ids
            )
            if not content_unfilled:
                logger.debug("[YandexMarket] skip (≥%.0f%% targets filled)", _SKIP_FILL_RATIO * 100)
                return []
            logger.debug(
                "[YandexMarket] fill_ratio≥%.0f%% but content targets still empty — running YM",
                _SKIP_FILL_RATIO * 100,
            )

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
                timeout=_YM_TOTAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[YandexMarket] total timeout (%.0fs) for '%s' — skipping YandexMarket stage",
                _YM_TOTAL_TIMEOUT,
                context.product_name[:60],
            )
            return []
        except Exception as exc:
            logger.warning(
                "[YandexMarket] unexpected error для '%s': %s",
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
        """Полный flow: query → Serper(url) → fetch HTML → parse chars → map → AVs."""
        full_name = context.product_name.strip()
        cat_leaf = context.category_path[-1] if context.category_path else None
        target_type = _target_type_lemma(full_name, cat_leaf)

        primary_query = _build_ym_query(full_name, context.brand, cat_leaf, max_tokens=5)
        fallback_query = _build_ym_query(full_name, context.brand, cat_leaf, max_tokens=3)
        queries_to_try = [primary_query]
        if fallback_query and fallback_query != primary_query:
            queries_to_try.append(fallback_query)

        # ---- SEARCH (Serper → product URLs) ----
        urls: list[str] = []
        for q in queries_to_try:
            logger.info("[YandexMarket] search query: '%s' (was: '%s')", q, full_name[:80])
            found = await self._search(q)
            if found:
                urls = found
                break

        if not urls:
            logger.info("[YandexMarket] no product URL ни для primary ни для fallback")
            return []

        # ---- FETCH + PARSE candidates (best matched card wins) ----
        best: Optional[dict] = None
        for url in urls[:_MAX_CANDIDATES]:
            html = await self._fetch_card_html(url)
            if not html:
                continue
            parsed = self._parse_card(html)
            if not parsed["chars"]:
                continue
            title = parsed["title"] or full_name
            _, score = self._pick_best_match(
                primary_query, [{"title": title}], category_leaf=cat_leaf,
            )
            if best is None or score > best["score"]:
                best = {"url": url, "score": score, **parsed}

        if best is None:
            logger.info("[YandexMarket] no parseable card (fetch blocked / empty specs)")
            return []

        mode = self._classify_match(best["score"])
        if mode == "skip":
            logger.info("[YandexMarket] best score=%.1f < %.0f — skip", best["score"], _BRAND_LINE_THRESHOLD)
            return []

        # Type-gate: если у карточки есть категория/тип и он не совместим с целью — skip.
        if target_type and best.get("card_type"):
            card_type_lemma = _type_lemma(best["card_type"])
            if card_type_lemma and not self._type_compatible(target_type, card_type_lemma):
                logger.info(
                    "[YandexMarket] type-gate: цель='%s' vs карточка='%s' — skip",
                    target_type, card_type_lemma,
                )
                return []

        logger.info(
            "[YandexMarket] match=%s score=%.1f title='%s'",
            mode, best["score"], (best["title"] or "")[:80],
        )
        return self._map_characteristics(
            best["chars"], targets, context, mode, best["title"] or full_name, best["score"],
        )

    async def _search(self, query: str) -> list[str]:
        """Serper → список URL карточек market.yandex.ru (top-N уникальных)."""
        serper_query = f"site:market.yandex.ru {query}".strip()
        try:
            results = await self._search_client.search(
                serper_query, num_results=_SERPER_NUM_RESULTS
            )
        except Exception as exc:
            logger.info("[YandexMarket] Serper search err: %s", exc)
            return []

        organic = getattr(results, "organic_results", None) or []
        out: list[str] = []
        seen: set[str] = set()
        for item in organic:
            link = getattr(item, "link", "") or ""
            if not _YM_PRODUCT_RE.search(link):
                continue
            if link in seen:
                continue
            seen.add(link)
            out.append(link)
        logger.info("[YandexMarket] Serper → %d product URLs", len(out))
        return out[:_MAX_CANDIDATES]

    async def _fetch_card_html(self, url: str) -> Optional[str]:
        """Скачать HTML карточки market.yandex.ru через Scrappey browser-bypass.

        market.yandex.ru за Yandex SmartCaptcha (IP+fingerprint гейт, режет
        datacenter-IP). Прямой httpx с серверного IP вернёт captcha/redirect.
        Scrappey запускает реальный браузер и обходит этот гейт (тот же путь,
        что OzonCardSource для ozon.ru).

        Timeout: asyncio.wait_for(_SCRAPPEY_HTTP_TIMEOUT=60s) — один запрос.
        Hard cap на весь _do_extract (_YM_TOTAL_TIMEOUT=90s) задан выше в extract().

        Возвращает HTML response на success, None на любой ошибке (graceful).
        Парсинг (_parse_card) полностью готов к такому HTML.
        """
        if not self._scrappey_key:
            logger.info(
                "[YandexMarket] _fetch_card_html: SCRAPPEY_KEY не задан → None (graceful)"
            )
            return None
        try:
            html = await asyncio.wait_for(
                _scrappey_fetch_page(url, timeout=_SCRAPPEY_HTTP_TIMEOUT),
                timeout=_SCRAPPEY_HTTP_TIMEOUT + 5,  # slight buffer above httpx timeout
            )
        except asyncio.TimeoutError:
            logger.info("[YandexMarket] _fetch_card_html: asyncio timeout for %s → None", url[:80])
            return None
        except Exception as exc:
            logger.info("[YandexMarket] _fetch_card_html: error for %s: %s → None", url[:80], exc)
            return None
        if html:
            logger.info("[YandexMarket] _fetch_card_html: got %d chars for %s", len(html), url[:80])
        return html

    # ------------------------------------------------------------------
    # HTML parsing
    # ------------------------------------------------------------------

    @classmethod
    def _parse_card(cls, html: str) -> dict:
        """Распарсить карточку: title/brand/type (JSON-LD) + characteristics (state).

        Возвращает {"title", "brand", "card_type", "chars": [{name,value,value_ids}]}.
        Никогда не падает на битом JSON — отдаёт что смогло распарсить.
        """
        title = ""
        brand = ""
        card_type = ""

        # ---- JSON-LD: name / brand / category (для title и type-gate) ----
        for block in _JSONLD_RE.findall(html):
            data = cls._safe_json(block)
            if data is None:
                continue
            for node in cls._iter_jsonld_nodes(data):
                if not isinstance(node, dict):
                    continue
                ntype = node.get("@type")
                is_product = ntype == "Product" or (
                    isinstance(ntype, list) and "Product" in ntype
                )
                if not is_product:
                    continue
                if not title and isinstance(node.get("name"), str):
                    title = node["name"].strip()
                b = node.get("brand")
                if not brand:
                    if isinstance(b, dict) and isinstance(b.get("name"), str):
                        brand = b["name"].strip()
                    elif isinstance(b, str):
                        brand = b.strip()
                cat = node.get("category")
                if not card_type and isinstance(cat, str):
                    card_type = cat.strip()

        # ---- Embedded state: characteristics ----
        chars = cls._parse_state_specs(html)

        # title fallback: og:title (more reliable than <title> on YM product pages
        # which uses "Все товары" generic title) → then <title> as last resort.
        if not title:
            m = re.search(
                r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
                html, re.IGNORECASE | re.DOTALL,
            )
            if m:
                title = re.sub(r"\s+", " ", m.group(1)).strip()
        if not title:
            m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            if m:
                title = re.sub(r"\s+", " ", m.group(1)).strip()

        return {
            "title": title,
            "brand": brand,
            "card_type": card_type,
            "chars": chars,
        }

    @classmethod
    def _parse_state_specs(cls, html: str) -> list[dict]:
        """Извлечь характеристики из состояния страницы Яндекс Маркет.

        Поддерживаемые форматы (эволюция рендера Маркета):
          1. __INITIAL_STATE__ / __NEXT_DATA__ (старый Next.js рендер) — {name, value} пары.
          2. specs.friendly (актуальный рендер 2024-2026) — плоский список строк вида
             "Цвет: Чёрный", "Пол: Мужской". Парсим split(': ', 1).
          3. data-auto="product-spec" (основной рендер 2024-2026, SSR product card page) —
             полная таблица характеристик rendered server-side. Каждая строка:
               <span data-auto="product-spec">ИМЯ</span>
               ... (sibling div) ...
               ЛИБО: <... data-zone-name="specLink" data-zone-data='{"text":"ЗНАЧЕНИЕ"}'>
               ЛИБО: <span>ЗНАЧЕНИЕ</span>
             Сканируем все product-spec spans, ограничиваем поиск значения регионом до
             следующего product-spec (tight scoping, предотвращает cross-row leakage).
             Это даёт полную таблицу (~10-20 пар) включая Состав, Цвет, Пол, Сезон, Бренд,
             Страна-изготовитель — в отличие от specs.friendly (5-8 пар).

        Возвращает [{name, value, value_ids:[]}], deduped по lowercase name.
        """
        out: list[dict] = []
        seen: set[str] = set()

        def _add(nm: str, val: str) -> None:
            nm = nm.strip()
            val = val.strip()
            if not nm or not val:
                return
            low = nm.lower()
            if low in seen:
                return
            seen.add(low)
            out.append({"name": nm, "value": val, "value_ids": []})

        # --- Format 1: __INITIAL_STATE__ / __NEXT_DATA__ (legacy Next.js rendering) ---
        blobs: list[str] = []
        blobs += _INITIAL_STATE_RE.findall(html)
        blobs += _NEXT_DATA_RE.findall(html)
        for blob in blobs:
            sliced = cls._balanced_json_slice(blob)
            if sliced is None:
                continue
            data = cls._safe_json(sliced)
            if data is None:
                continue
            for container in cls._find_spec_containers(data):
                for name, value in cls._pairs_from_container(container):
                    _add(name, value)

        # --- Format 2: specs.friendly (current 2024-2026 YM rendering) ---
        # Each search-tile embeds a brief specs array as pre-formatted strings.
        # We collect ALL tiles and merge — _do_extract selects best by title score.
        for raw_list in _SPECS_FRIENDLY_RE.findall(html):
            items = cls._safe_json(raw_list)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, str):
                    continue
                # Format: "Ключ: Значение"
                if ": " in item:
                    name_part, _, val_part = item.partition(": ")
                    _add(name_part, val_part)

        # --- Format 3: data-auto="product-spec" (SSR full spec table, 2024-2026) ---
        # Harvest before format-2 (or even if format-2 already filled some) — dedupe handles it.
        # Collect all spec-name span positions first, then scope each value lookup.
        spec_matches = list(_SPEC_NAME_RE.finditer(html))
        for i, m in enumerate(spec_matches):
            spec_name = m.group(1).strip()
            if not spec_name:
                continue
            after_pos = m.end()
            # Scope: region between this span end and start of next spec-name span.
            # Cap at 1500 chars to stay well within a single spec row's HTML.
            next_pos = spec_matches[i + 1].start() if i + 1 < len(spec_matches) else after_pos + 1500
            region = html[after_pos:min(next_pos, after_pos + 1500)]

            spec_value: Optional[str] = None

            # First try: specLink zone (linked enum value)
            sl_m = _SPEC_LINK_RE.search(region)
            if sl_m:
                try:
                    zd = json.loads(sl_m.group(1).replace("&quot;", '"'))
                    v = str(zd.get("text") or "").strip()
                    if v:
                        spec_value = v
                except (ValueError, json.JSONDecodeError, KeyError):
                    pass

            if spec_value is None:
                # Second try: first plain text span in the value region.
                # The value container div (_1_zPW class) comes directly after the
                # name container. We allow any non-empty span content, INCLUDING
                # numeric-only values (article numbers, counts, etc.).
                for sv_m in _SPEC_VALUE_SPAN_RE.finditer(region[:800]):
                    sv = sv_m.group(1).strip()
                    # Skip empty, identical to spec name, purely whitespace, or HTML entities
                    if sv and sv != spec_name and "&" not in sv:
                        spec_value = sv
                        break

            if spec_value:
                _add(spec_name, spec_value)

        return out

    @classmethod
    def _find_spec_containers(cls, node: Any, _depth: int = 0) -> list:
        """Рекурсивно собрать все значения под ключами _SPEC_CONTAINER_KEYS."""
        if _depth > 12:
            return []
        found: list = []
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(k, str) and k.lower() in _SPEC_CONTAINER_KEYS:
                    found.append(v)
                found += cls._find_spec_containers(v, _depth + 1)
        elif isinstance(node, list):
            for item in node:
                found += cls._find_spec_containers(item, _depth + 1)
        return found

    @staticmethod
    def _pairs_from_container(container: Any) -> list[tuple[str, str]]:
        """Достать (name, value) пары из контейнера характеристик.

        Поддержанные формы:
          • list[{"name": "...", "value": "..."}]
          • list[{"key": "...", "value": "..."}]
          • list[{"name": "...", "values": ["...", ...]}]
          • list[{"title": "...", "value"/"values"/"text": ...}]
          • dict["Цвет": "Белый", ...]  (плоский словарь)
          • list[{"groupName": ..., "specs"/"values": [ ...nested... ]}] — nested groups
        """
        pairs: list[tuple[str, str]] = []

        def _value_to_str(v: Any) -> str:
            if isinstance(v, list):
                texts = [str(x).strip() for x in v if str(x).strip()]
                return ", ".join(texts)
            if isinstance(v, (str, int, float, bool)):
                return str(v).strip()
            return ""

        def _handle_item(item: Any) -> None:
            if not isinstance(item, dict):
                return
            # nested group → recurse into its spec list
            for nk in ("specs", "values", "params", "specifications"):
                nested = item.get(nk)
                if isinstance(nested, list) and nested and isinstance(nested[0], dict):
                    for sub in nested:
                        _handle_item(sub)
                    # don't also treat the group itself as a leaf
                    if any(isinstance(s, dict) and ("name" in s or "key" in s) for s in nested):
                        return
            name = None
            for nk in ("name", "key", "title", "label"):
                if isinstance(item.get(nk), str) and item[nk].strip():
                    name = item[nk].strip()
                    break
            if name is None:
                return
            value = ""
            for vk in ("value", "values", "text", "valueText"):
                if vk in item:
                    value = _value_to_str(item[vk])
                    if value:
                        break
            if name and value:
                pairs.append((name, value))

        if isinstance(container, list):
            for item in container:
                _handle_item(item)
        elif isinstance(container, dict):
            # Could be flat {name: value} OR {id: {name, value}} OR a single spec dict.
            if any(k in container for k in ("name", "key", "title")):
                _handle_item(container)
            else:
                for k, v in container.items():
                    if isinstance(v, (str, int, float)):
                        sval = str(v).strip()
                        if isinstance(k, str) and k.strip() and sval:
                            pairs.append((k.strip(), sval))
                    elif isinstance(v, dict):
                        _handle_item(v)
        return pairs

    @staticmethod
    def _balanced_json_slice(raw: str) -> Optional[str]:
        """Trim *raw* (starting at its first '{') to the matching closing brace.

        State blobs are captured greedily up to </script>, so they include
        trailing JS (`};`, code, ...) after the JSON object. We walk braces
        respecting string literals/escapes and return the balanced object text.
        Returns None if no balanced object is found.
        """
        start = raw.find("{")
        if start < 0:
            return None
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:i + 1]
        return None

    @staticmethod
    def _safe_json(raw: str) -> Any:
        """json.loads с HTML-entity fallback. None если не распарсилось."""
        raw = raw.strip()
        try:
            return json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            decoded = (
                raw.replace("&quot;", '"')
                .replace("&#39;", "'")
                .replace("&amp;", "&")
                .replace("&lt;", "<")
                .replace("&gt;", ">")
            )
            try:
                return json.loads(decoded)
            except (ValueError, json.JSONDecodeError):
                return None

    @staticmethod
    def _iter_jsonld_nodes(data: Any) -> list:
        """Развернуть JSON-LD: один объект, список, или @graph."""
        nodes: list = []
        if isinstance(data, list):
            for d in data:
                nodes += YandexMarketSource._iter_jsonld_nodes(d)
        elif isinstance(data, dict):
            if "@graph" in data and isinstance(data["@graph"], list):
                for d in data["@graph"]:
                    nodes += YandexMarketSource._iter_jsonld_nodes(d)
            else:
                nodes.append(data)
        return nodes

    # ------------------------------------------------------------------
    # Match scoring (mirror of WbCard/OzonCard logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_best_match(
        query: str,
        tiles: list[dict],
        category_leaf: Optional[str] = None,
    ) -> tuple[Optional[dict], float]:
        """rapidfuzz match с model-token бонусом, type/gender штрафами."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return (tiles[0], 100.0) if tiles else (None, 0.0)

        _MODEL_BONUS = 5.0
        _MODEL_BONUS_ALPHA = 3.0
        _TYPE_MISMATCH_PENALTY = 30.0
        _GENDER_MISMATCH_PENALTY = 30.0
        q_models = _extract_model_tokens(query)
        q_alpha = _extract_alpha_model_tokens(query)
        q_gender = _extract_gender_signal(query)
        best_tile: Optional[dict] = None
        best_score = 0.0
        q = query.lower()
        cat_leaf_low = category_leaf.strip().lower() if category_leaf else None
        for tile in tiles:
            title = (tile.get("title") or "").strip()
            if not title:
                continue
            t = title.lower()
            score = (fuzz.partial_ratio(q, t) + fuzz.token_sort_ratio(q, t)) / 2.0
            if q_models and q_models & _extract_model_tokens(title):
                score += _MODEL_BONUS
            elif q_alpha and q_alpha & _extract_alpha_model_tokens(title):
                score += _MODEL_BONUS_ALPHA
            if cat_leaf_low and not q_models and cat_leaf_low not in t:
                score -= _TYPE_MISMATCH_PENALTY
            if q_gender is not None and _gender_conflict(query, title):
                score -= _GENDER_MISMATCH_PENALTY
            if score > best_score:
                best_score = score
                best_tile = tile
        return best_tile, best_score

    @staticmethod
    def _classify_match(score: float) -> str:
        if score >= _EXACT_THRESHOLD:
            return "exact"
        if score >= _BRAND_LINE_THRESHOLD:
            return "brand_line"
        return "skip"

    @staticmethod
    def _type_compatible(target_type: str, card_type_lemma: str) -> bool:
        """Совместим ли тип карточки с целевым (зеркало WbCard._type_compatible)."""
        if target_type == card_type_lemma:
            return True
        common = 0
        for x, y in zip(target_type, card_type_lemma):
            if x == y:
                common += 1
            else:
                break
        shorter = min(len(target_type), len(card_type_lemma))
        return bool(shorter and common >= 5 and common / shorter >= 0.7)

    # ------------------------------------------------------------------
    # Mapping: char name → target attribute_id → value_id (Ozon dict)
    # ------------------------------------------------------------------

    def _map_characteristics(
        self,
        chars: list[dict],
        targets: list[TargetAttribute],
        context: ExtractionContext,
        mode: str,
        title: str,
        score: float,
    ) -> list[AttributeValue]:
        """Сопоставить Я.Маркет-char names с target.name через Ozon dictionary.

        Логика зеркалит WbCardSource._map_characteristics: exact/substring/fuzzy
        матч имени, resolve_value_id для enum-полей, brand_line BLACKLIST.
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
        evidence_short = f"yandex_market:{title[:50]} | match={score:.1f}"
        conf = _CONF_EXACT if mode == "exact" else _CONF_BRAND_LINE

        results: list[AttributeValue] = []
        used_ids: set[int] = set()

        for c in chars:
            char_name = c["name"].strip()
            char_val = c["value"].strip()
            char_name_low = char_name.lower()
            char_name_norm = _norm_char_name(char_name)

            if mode == "brand_line" and char_name_low in _BRAND_LINE_BLACKLIST:
                continue

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
                if best is not None and best[1] >= 88:
                    target_id = name_to_target_id[best[0]]

            if target_id is None or target_id in used_ids:
                continue
            target = target_by_id.get(target_id)
            if target is None:
                continue
            # brand_line: skip numeric — модель-специфичны.
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
                            resolve_value_id(cat_id, type_id, target.id, p) for p in parts
                        ]
                        if any(r is not None for r in resolved):
                            value_ids = resolved
                    except Exception as exc:
                        logger.debug("[YandexMarket] resolve_value_id (list) failed: %s", exc)
            else:
                value_out = char_val
                if cat_id and type_id:
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                    except Exception as exc:
                        logger.debug("[YandexMarket] resolve_value_id failed: %s", exc)

            results.append(AttributeValue(
                attribute_id=target.id,
                value=value_out,
                confidence=conf,
                source=Source.OZON_CARD,
                evidence=evidence_short,
                semantic_type=target.semantic_type,
                is_collection=target.is_collection,
                value_id=value_id,
                value_ids=value_ids,
            ))

        logger.info(
            "[YandexMarket] %s mode → %d характеристик скопировано (из %d chars, %d targets)",
            mode, len(results), len(chars), len(targets),
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

    def _cache_put(self, key: tuple[str, str], value: list[AttributeValue]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)
