"""WbCardSource — копирует характеристики из живой Wildberries-карточки похожего товара.

Использует ПУБЛИЧНЫЕ JSON-эндпоинты WB через Scrappey.com proxy.
WB rate-limits бот-трафик (HTTP 429 на batch >5 запросов с одного IP),
поэтому прямые httpx-вызовы непригодны для production-eval. Scrappey
ротация IP + browser fingerprint обходит rate limit ценой ~1 credit/запрос.

Алгоритм:
  1. Skip-guard: если already_filled покрывает ≥80% targets → return [].
  2. Search step: Scrappey GET https://search.wb.ru/exactmatch/ru/common/v9/search?query=<q>
     &resultset=catalog&limit=10&dest=-1257786&curr=rub
     → JSON `data.products[]` → каждый {id (nm), name, brand}.
  3. Match step: rapidfuzz (partial_ratio + token_sort_ratio) vs product_name:
     - ≥78 → "exact": full card, copy ALL.
     - 60-77 → "brand_line": full card, только safe attrs.
     - <60 → skip.
  4. Detail step: Scrappey GET basket-NN.wbbasket.ru/vol{vol}/part{part}/{nm_id}/info/ru/card.json
     с retry по basket NN от 1 до 30 (vol→basket sharding известен,
     но WB периодически перешивает диапазоны → fallback brute force).
  5. JSON содержит:
       imt_name, description, options[{name,value}], grouped_options[],
       media.photos[] → image_urls для downstream VisionSource.
  6. Mapping по русским именам через get_ozon_characteristics_for_type
     (lowercase + substring + fuzzy WRatio≥88) — финальная цель Ozon API.
  7. resolve_value_id для (attr_id, value) → словарный value_id Ozon.
  8. Confidence: 0.93 (exact), 0.85 (brand_line). Source: WB_CARD.
  9. Evidence: f"wb:{title[:50]} | match={score}".
  10. In-process LRU cache по (brand, normalized_model), max 256.

Scrappey cost: 1 credit/запрос. Бюджет на товар:
  - search: 1 credit
  - card.json: 1-N credits (basket retry — обычно 1, иногда 2-3 при brute force)
Latency: 3-10s end-to-end (Scrappey JSON быстрее browser bypass ~3-5s/call).

Anti-block: WB не имеет DataDome, но имеет per-IP rate-limit. Scrappey ротация
IP решает rate-limit без full browser cookie challenge.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import OrderedDict
from typing import Any, Optional

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
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    resolve_value_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

# WB search API v9. dest=-1257786 — Москва (стабильный fallback).
# curr=rub. resultset=catalog → отдаёт products[] с id/name/brand.
_WB_SEARCH_URL = (
    "https://search.wb.ru/exactmatch/ru/common/v9/search"
    "?query={query}&resultset=catalog&limit=10&dest=-1257786&curr=rub"
)

# Scrappey endpoint — единая точка для всех WB запросов.
_SCRAPPEY_ENDPOINT = "https://publisher.scrappey.com/api/v1"

# Известный sharding table «vol → basket-NN». См. scripts/build_wb_dictionary_lib/
# card_parser.py — используется в bulk-загрузчике словаря.
# ПРЕДПОЛОЖЕНИЕ: WB иногда расширяет диапазоны (новые товары попадают на
# basket-18+). Для unknown vol fallback пытается basket-18..30. Если найдёте
# обновлённую таблицу — добавьте сюда.
_BASKET_THRESHOLDS: list[tuple[int, str]] = [
    (143,  "basket-01"),
    (287,  "basket-02"),
    (431,  "basket-03"),
    (719,  "basket-04"),
    (1007, "basket-05"),
    (1061, "basket-06"),
    (1115, "basket-07"),
    (1169, "basket-08"),
    (1313, "basket-09"),
    (1601, "basket-10"),
    (1655, "basket-11"),
    (1919, "basket-12"),
    (2045, "basket-13"),
    (2189, "basket-14"),
    (2405, "basket-15"),
    (2621, "basket-16"),
    (2837, "basket-17"),
]
_BASKET_FALLBACK_RANGE = range(18, 31)  # basket-18 .. basket-30

_CONTENT_FALLBACK_URL = "https://wbx-content-v2.wbstatic.net/ru/{nm_id}.json"

# Scrappey browser bypass обычно 3-10s; редкие spike до 30s.
_HTTP_TIMEOUT = 180.0
_MAX_SEARCH_TILES = 5
_MAX_RETRIES = 0  # retry отключён: search.wb.ru через Scrappey стабильно возвращает
                  # envelope без statusCode (Scrappey не справляется с этим endpoint)
                  # — 4 credits per product экономия. Если retry понадобится — поднять до 1.

# Confidence — параллельно с OzonCardSource (одна логика match → conf).
_CONF_EXACT = 0.93
_CONF_BRAND_LINE = 0.85

_EXACT_THRESHOLD = 78.0
_BRAND_LINE_THRESHOLD = 60.0

# Skip-guard
_SKIP_FILL_RATIO = 0.80

# LRU
_CACHE_MAX = 256

# Brand-line BLACKLIST — model-specific атрибуты, которые точно отличаются
# между моделями одной линейки.
_BRAND_LINE_BLACKLIST: frozenset[str] = frozenset(name.lower() for name in {
    "Артикул",
    "Артикул WB",
    "Артикул товара",
    "Код производителя",
    "MPN",
    "Партномер",
    "Серийный номер",
    "EAN",
    "GTIN",
    "ASIN",
    "Штрихкод",
    "Дата производства",
    "Модель",
    "Название модели",
    "ID товара",
    "ID карточки",
})

# Generic-префиксы которые срезаются при нормализации для cache key.
_GENERIC_PREFIXES = (
    "блок питания", "блок", "питания",
    "power supply", "power", "supply", "unit",
)


def _compress_search_query(product_name: str, brand: Optional[str], max_tokens: int = 5) -> str:
    """Сжимает product_name для WB search (brand + model + 1-2 спеки)."""
    result = product_name.strip()
    low = result.lower()
    for prefix in _GENERIC_PREFIXES:
        if low.startswith(prefix):
            result = result[len(prefix):].strip()
            low = result.lower()
            break
    tokens = result.split()[:max_tokens]
    compact = " ".join(tokens).strip()
    if brand and brand.strip():
        b = brand.strip()
        if b.lower() not in compact.lower():
            compact = f"{b} {compact}".strip()
    return compact or product_name.strip()


def _normalize_model(product_name: str, brand: Optional[str]) -> str:
    """Убирает generic-префиксы и бренд, lowercase, для cache key."""
    result = product_name.strip()
    low = result.lower()
    for prefix in _GENERIC_PREFIXES:
        if low.startswith(prefix):
            result = result[len(prefix):].strip()
            low = result.lower()
            break
    if brand:
        b = brand.strip().lower()
        if low.startswith(b):
            result = result[len(brand):].strip()
    return re.sub(r"\s+", " ", result).strip().lower()


def _basket_host_from_table(nm_id: int) -> Optional[str]:
    """Возвращает basket-host из известной таблицы (vol < 2837) или None."""
    vol = nm_id // 100_000
    for threshold, name in _BASKET_THRESHOLDS:
        if vol <= threshold:
            return f"{name}.wbbasket.ru"
    return None  # требует brute-force fallback


def _card_url(host: str, nm_id: int) -> str:
    vol = nm_id // 100_000
    part = nm_id // 1000
    return f"https://{host}/vol{vol}/part{part}/{nm_id}/info/ru/card.json"


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class WbCardSource(AttributeSource):
    """Копия характеристик с live WB-карточки похожего товара через Scrappey.

    Cost: 2-5 credits/product (search + card.json, иногда + basket retry).
    Latency: 3-10s end-to-end.
    """

    def __init__(
        self,
        scrappey_key: Optional[str] = None,
        **kwargs: Any,
    ):
        _ = kwargs  # backward-compat

        self._scrappey_key = scrappey_key or os.environ.get("SCRAPPEY_KEY")
        if not self._scrappey_key:
            logger.warning(
                "[WbCard] SCRAPPEY_KEY не задан (ни параметром, ни в env) — "
                "extract() всегда вернёт []."
            )

        self._judge = WbCardJudge()
        # LRU cache: (brand_lower, model_lower) → list[AttributeValue]
        self._cache: "OrderedDict[tuple[str, str], list[AttributeValue]]" = OrderedDict()

    @property
    def source_type(self) -> Source:
        return Source.WB_CARD

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим если product_name есть и не пустой, и SCRAPPEY_KEY доступен."""
        return bool(
            self._scrappey_key
            and context.product_name
            and len(context.product_name.strip()) >= 5
        )

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        if not targets or not context.product_name or not self._scrappey_key:
            return []

        already_filled = already_filled or []

        # Skip-guard
        filled_ids = {av.attribute_id for av in already_filled if av.confidence >= 0.85}
        if targets and len(filled_ids & {t.id for t in targets}) / len(targets) >= _SKIP_FILL_RATIO:
            logger.debug(
                "[WbCard] skip (≥%.0f%% targets уже filled)",
                _SKIP_FILL_RATIO * 100,
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
            all_values = await self._do_extract(context, targets)
        except Exception as exc:
            logger.warning(
                "[WbCard] unexpected error для '%s': %s",
                context.product_name[:60], exc,
            )
            self._cache_put(cache_key, [])
            return []

        self._cache_put(cache_key, all_values)
        return self._filter_for_targets(all_values, effective)

    def get_judge(self) -> LlmJudge:
        return self._judge

    # ------------------------------------------------------------------
    # Network transport via Scrappey
    # ------------------------------------------------------------------

    async def _scrappey_fetch_json(
        self,
        client: httpx.AsyncClient,
        target_url: str,
    ) -> Optional[dict]:
        """POST к Scrappey с retry, возвращает parsed JSON ответа upstream.

        Retry если HTTP 429 / 500 от WB (через Scrappey envelope.solution.statusCode).
        Network errors / Scrappey HTTP 4xx также реtraется.
        """
        logger.info("[WbCard] via Scrappey: %s", target_url[:120])
        for attempt in range(_MAX_RETRIES + 1):
            result = await self._scrappey_fetch_once(client, target_url)
            if result is None:
                # Hard fail (network, Scrappey 4xx) — имеет смысл retry
                if attempt < _MAX_RETRIES:
                    logger.info(
                        "[WbCard] retry #%d (hard fail) %s",
                        attempt + 1, target_url[:80],
                    )
                    continue
                return None
            status_code, data = result
            if status_code in (429, 500, 502, 503, 504):
                # Upstream rate-limit / server err — retry
                if attempt < _MAX_RETRIES:
                    logger.info(
                        "[WbCard] retry #%d (upstream HTTP %s) %s",
                        attempt + 1, status_code, target_url[:80],
                    )
                    continue
                logger.info(
                    "[WbCard] final upstream HTTP %s for %s — give up",
                    status_code, target_url[:80],
                )
                return None
            if status_code != 200:
                logger.info(
                    "[WbCard] upstream HTTP %s for %s — skip",
                    status_code, target_url[:80],
                )
                return None
            return data
        return None

    async def _scrappey_fetch_once(
        self,
        client: httpx.AsyncClient,
        target_url: str,
    ) -> Optional[tuple[int, Optional[dict]]]:
        """Один POST к Scrappey. Возвращает (upstream_status, parsed_json) или None.

        None — Scrappey-уровневая ошибка (network, HTTP 4xx, пустой envelope).
        (status, None) — upstream вернул не-JSON (но это не должно случаться для WB API).
        (status, dict) — upstream JSON распарсен.
        """
        payload = {"cmd": "request.get", "url": target_url}
        try:
            r = await client.post(
                _SCRAPPEY_ENDPOINT,
                params={"key": self._scrappey_key},
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            logger.info("[WbCard] Scrappey network err: %s", exc)
            return None

        if r.status_code >= 400:
            logger.warning(
                "[WbCard] Scrappey HTTP %s for %s — skip (body[:200]=%s)",
                r.status_code, target_url[:80], r.text[:200],
            )
            return None

        try:
            envelope = r.json()
        except (ValueError, json.JSONDecodeError):
            logger.info("[WbCard] Scrappey returned non-json envelope")
            return None

        # Credit usage logging (Scrappey возвращает в envelope creditsUsed/credits/cost)
        for credits_key in ("creditsUsed", "credits", "cost"):
            if credits_key in envelope:
                logger.info(
                    "[WbCard] credits used: %s (%s)",
                    envelope.get(credits_key), credits_key,
                )
                break

        solution = envelope.get("solution") or {}
        upstream_status = solution.get("statusCode")
        content = solution.get("response") or ""

        if not isinstance(upstream_status, int):
            logger.info(
                "[WbCard] Scrappey envelope без statusCode для %s",
                target_url[:80],
            )
            return None

        if not content:
            logger.info(
                "[WbCard] Scrappey empty content (upstream=%s) for %s",
                upstream_status, target_url[:80],
            )
            return (upstream_status, None)

        try:
            data = json.loads(content)
        except (ValueError, json.JSONDecodeError):
            logger.info(
                "[WbCard] Scrappey response не JSON (upstream=%s) for %s",
                upstream_status, target_url[:80],
            )
            return (upstream_status, None)

        return (upstream_status, data if isinstance(data, dict) else None)

    async def _do_extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Полный flow: search → match → card.json → map → AVs."""
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
        ) as client:
            # ---- SEARCH ----
            full_name = context.product_name.strip()
            primary_query = _compress_search_query(full_name, context.brand, max_tokens=5)
            fallback_query = _compress_search_query(full_name, context.brand, max_tokens=3)

            queries_to_try: list[str] = [primary_query]
            if fallback_query and fallback_query != primary_query:
                queries_to_try.append(fallback_query)

            tiles: list[dict] = []
            query: Optional[str] = None
            for q in queries_to_try:
                logger.info(
                    "[WbCard] search query: '%s' (was: '%s')",
                    q, full_name[:80],
                )
                parsed = await self._search(client, q)
                if parsed:
                    tiles, query = parsed, q
                    break
                logger.info("[WbCard] no tiles на query='%s' — пробую fallback", q[:60])

            if not tiles or query is None:
                logger.info("[WbCard] no search tiles ни для primary ни для fallback")
                return []

            top_tile, top_score = self._pick_best_match(query, tiles[:_MAX_SEARCH_TILES])
            if top_tile is None:
                return []
            mode = self._classify_match(top_score)
            if mode == "skip":
                logger.info(
                    "[WbCard] best score=%.1f < %.0f — skip",
                    top_score, _BRAND_LINE_THRESHOLD,
                )
                return []

            title = (top_tile.get("title") or "").strip()
            nm_id = top_tile.get("nm_id")
            if not nm_id:
                logger.info("[WbCard] no nm_id in top tile")
                return []

            logger.info(
                "[WbCard] match=%s score=%.1f title='%s' nm=%s",
                mode, top_score, title[:80], nm_id,
            )

            # ---- CARD JSON ----
            card_data = await self._fetch_card(client, int(nm_id))
            if not card_data:
                logger.info("[WbCard] card.json failed for nm=%s", nm_id)
                return []

            chars = self._extract_options(card_data)
            if not chars:
                logger.info("[WbCard] no options/characteristics в card.json nm=%s", nm_id)
                return []

            # ---- IMAGES (для downstream VisionSource) ----
            new_image_urls = self._extract_image_urls(card_data, int(nm_id))
            if new_image_urls:
                existing = set(context.image_urls or [])
                added = [u for u in new_image_urls if u not in existing]
                if added:
                    context.image_urls = list(context.image_urls or []) + added
                    logger.info(
                        "[WbCard] +%d image URLs для VisionSource (nm=%s)",
                        len(added), nm_id,
                    )

            # ---- MAP & EMIT ----
            return self._map_characteristics(
                chars, targets, context, mode, title, top_score,
            )

    async def _search(self, client: httpx.AsyncClient, query: str) -> list[dict]:
        """Scrappey → WB search API → нормализованные tiles [{title, nm_id, brand}]."""
        url = _WB_SEARCH_URL.format(query=query)
        data = await self._scrappey_fetch_json(client, url)
        if not data:
            return []

        products = (((data or {}).get("data") or {}).get("products")) or []
        tiles: list[dict] = []
        for p in products:
            if not isinstance(p, dict):
                continue
            nm_id = p.get("id")
            name = (p.get("name") or "").strip()
            brand = (p.get("brand") or "").strip()
            if not nm_id or not name:
                continue
            title = f"{brand} {name}".strip() if brand and brand.lower() not in name.lower() else name
            tiles.append({"title": title[:200], "nm_id": int(nm_id), "brand": brand})
        return tiles

    async def _fetch_card(
        self,
        client: httpx.AsyncClient,
        nm_id: int,
    ) -> Optional[dict]:
        """Скачать card.json через Scrappey с retry по basket-NN.

        Стратегия:
          1. Сначала пробуем basket из known таблицы (если nm_id в покрытом диапазоне).
          2. Если 404 / нет таблицы → brute force basket-NN от 18 до 30.
          3. Если всё мимо → fallback на wbx-content-v2.wbstatic.net.
        """
        # 1. Known basket
        primary_host = _basket_host_from_table(nm_id)
        if primary_host:
            data = await self._try_basket(client, primary_host, nm_id)
            if data:
                return data

        # 2. Brute force fallback range (basket-18..30) — для свежих nm_id
        for n in _BASKET_FALLBACK_RANGE:
            host = f"basket-{n:02d}.wbbasket.ru"
            if host == primary_host:
                continue
            data = await self._try_basket(client, host, nm_id)
            if data:
                logger.info("[WbCard] nm=%s found in fallback %s", nm_id, host)
                return data

        # 3. Static content fallback
        url = _CONTENT_FALLBACK_URL.format(nm_id=nm_id)
        return await self._scrappey_fetch_json(client, url)

    async def _try_basket(
        self,
        client: httpx.AsyncClient,
        host: str,
        nm_id: int,
    ) -> Optional[dict]:
        """Один basket-attempt через Scrappey. None если 404/non-200."""
        url = _card_url(host, nm_id)
        # NB: используем _scrappey_fetch_once напрямую (без retry) — basket-brute-force
        # сам по себе цикл retry'ов по разным хостам, плюс 404 ожидаем и
        # ретраить его смысла нет (на этом хосте товара просто нет).
        result = await self._scrappey_fetch_once(client, url)
        if result is None:
            return None
        status_code, data = result
        if status_code != 200 or not isinstance(data, dict):
            return None
        return data

    # ------------------------------------------------------------------
    # Card JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_options(card: dict) -> list[dict]:
        """Извлечь characteristics из WB card.json.

        Структуры:
          - options: [{"name": "Цвет", "value": "Белый"}, ...]
          - grouped_options: [{"group_name": "...", "options": [...]}]
            (иерархия — собираем options[] из каждой группы).
          - compositions: [{"name", "value"}] — fallback из wbx-content-v2.

        Возвращает [{name, value, value_ids=[]}], deduped по lowercase name.
        """
        out: list[dict] = []
        seen: set[str] = set()

        def _push(name: Any, value: Any) -> None:
            if not isinstance(name, str):
                return
            name = name.strip()
            if not name:
                return
            name_low = name.lower()
            if name_low in seen:
                return
            # value может быть string | list | dict
            if isinstance(value, list):
                texts = [str(v).strip() for v in value if str(v).strip()]
                if not texts:
                    return
                value_str = ", ".join(texts)
            elif isinstance(value, (str, int, float)):
                value_str = str(value).strip()
                if not value_str:
                    return
            else:
                return
            seen.add(name_low)
            out.append({"name": name, "value": value_str, "value_ids": []})

        # 1. Плоские options
        for o in card.get("options") or []:
            if isinstance(o, dict):
                _push(o.get("name"), o.get("value"))

        # 2. grouped_options
        for grp in card.get("grouped_options") or []:
            if isinstance(grp, dict):
                for o in grp.get("options") or []:
                    if isinstance(o, dict):
                        _push(o.get("name"), o.get("value"))

        # 3. compositions (wbx-content-v2 fallback)
        for c in card.get("compositions") or []:
            if isinstance(c, dict):
                _push(c.get("name"), c.get("value"))

        return out

    @staticmethod
    def _extract_image_urls(card: dict, nm_id: int, limit: int = 5) -> list[str]:
        """Собрать high-res photo URLs.

        WB card.json иногда содержит media.photos[] (полные URL'ы) или просто
        media.photo_count (число — тогда конструируем URL'ы из basket-host).
        Возвращает первые `limit` URL'ов в порядке появления.
        """
        out: list[str] = []
        media = card.get("media") or {}

        # 1. Готовые photo URLs
        for p in media.get("photos") or []:
            if isinstance(p, dict):
                # WB иногда отдаёт {url} / {big}
                for k in ("url", "big", "src"):
                    url = p.get(k)
                    if isinstance(url, str) and url.startswith("http"):
                        if url not in out:
                            out.append(url)
                        break
            elif isinstance(p, str) and p.startswith("http"):
                if p not in out:
                    out.append(p)
            if len(out) >= limit:
                return out

        # 2. Конструкция из photo_count + basket-host
        photo_count = media.get("photo_count")
        if isinstance(photo_count, int) and photo_count > 0:
            host = _basket_host_from_table(nm_id)
            if host:
                vol = nm_id // 100_000
                part = nm_id // 1000
                base = f"https://{host}/vol{vol}/part{part}/{nm_id}/images/big"
                for i in range(1, min(photo_count, limit - len(out)) + 1):
                    url = f"{base}/{i}.webp"
                    if url not in out:
                        out.append(url)
                    if len(out) >= limit:
                        break
        return out[:limit]

    # ------------------------------------------------------------------
    # Match scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_best_match(
        query: str,
        tiles: list[dict],
    ) -> tuple[Optional[dict], float]:
        """Top-1 по rapidfuzz (partial_ratio + token_sort_ratio averaged)."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return (tiles[0], 100.0) if tiles else (None, 0.0)

        best_tile: Optional[dict] = None
        best_score = 0.0
        q = query.lower()
        for tile in tiles:
            title = (tile.get("title") or "").strip()
            if not title:
                continue
            t = title.lower()
            score = (fuzz.partial_ratio(q, t) + fuzz.token_sort_ratio(q, t)) / 2.0
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
        """Сопоставить WB-char names с target.name через Ozon dictionary.

        Да, маппим WB-характеристики на наш Ozon словарь — финальное API
        publish'a у нас Ozon, а WB-карточка лишь источник данных.
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

        # attr_id → canonical_name из dict
        attr_id_to_dict_name: dict[int, str] = {}
        for oc in ozon_chars:
            if isinstance(oc, dict) and "id" in oc and "name" in oc:
                attr_id_to_dict_name[int(oc["id"])] = str(oc["name"])

        # target_id → lowercase set имён
        target_names_low: dict[int, set[str]] = {}
        for t in targets:
            names = {t.name.lower()}
            dn = attr_id_to_dict_name.get(t.id)
            if dn:
                names.add(dn.lower())
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

        evidence_short = f"wb:{title[:50]} | match={score:.1f}"
        conf = _CONF_EXACT if mode == "exact" else _CONF_BRAND_LINE

        results: list[AttributeValue] = []
        used_ids: set[int] = set()

        for c in chars:
            char_name = c["name"].strip()
            char_val = c["value"].strip()
            char_name_low = char_name.lower()

            if mode == "brand_line" and char_name_low in _BRAND_LINE_BLACKLIST:
                continue

            # 1) Exact lowercase
            target_id = name_to_target_id.get(char_name_low)
            # 2) Substring
            if target_id is None:
                for tn, tid in name_to_target_id.items():
                    if char_name_low in tn or tn in char_name_low:
                        target_id = tid
                        break
            # 3) Fuzzy
            if target_id is None and process is not None and all_target_names:
                best = process.extractOne(
                    char_name_low, all_target_names, scorer=fuzz.WRatio,
                )
                if best is not None and best[1] >= 88:
                    target_id = name_to_target_id[best[0]]

            if target_id is None or target_id in used_ids:
                continue

            target = target_by_id.get(target_id)
            if target is None:
                continue
            used_ids.add(target_id)

            value_id: Optional[int] = None
            if cat_id and type_id:
                try:
                    value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                except Exception as exc:
                    logger.debug("[WbCard] resolve_value_id failed: %s", exc)

            results.append(AttributeValue(
                attribute_id=target.id,
                value=char_val,
                confidence=conf,
                source=Source.WB_CARD,
                evidence=evidence_short,
                semantic_type=target.semantic_type,
                is_collection=target.is_collection,
                value_id=value_id,
            ))

        logger.info(
            "[WbCard] %s mode → %d характеристик скопировано (из %d candidate chars, %d targets)",
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
