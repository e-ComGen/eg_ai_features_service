"""WbCardSource — копирует характеристики из живой Wildberries-карточки похожего товара.

Старый путь ходил на ``search.wb.ru`` — WB банит серверный IP (0 заполнений).
Новый путь полностью минует anti-bot WB:

  1. **Сжать запрос**: ``_compress_search_query`` (category-aware, из ozon_card_source)
     чистит «грязный» product_name (срез категорийного префикса, стоп-слов, спеков).
  2. **Найти nm_id через Serper** (НЕ search.wb.ru). Запрос вида
     ``inurl:catalog detail.aspx <query>`` → ~100% прямых ссылок на карточки WB.
     Из organic-результатов nm_id вытаскивается регэкспом, предпочитая домен
     ``wildberries.ru`` (зеркала .ge/.am/.by дают тот же nm_id — fallback).
     Берём топ-10 уникальных nm_id (многие архивные → 404).
  3. **Скачать card.json с CDN** простым httpx GET (Chrome User-Agent, БЕЗ Scrappey —
     CDN не банится). URL:
     ``https://basket-{NN}.wbbasket.ru/vol{vol}/part{part}/{nm_id}/info/ru/card.json``,
     где ``vol = nm_id // 100000``, ``part = nm_id // 1000``. NN по диапазону vol
     из таблицы ниже; на 404 перебираются остальные NN (01..21).
  4. **Распарсить характеристики**: ``options[]``, ``grouped_options[].options``,
     ``compositions[]`` (Состав) → пары name→value.
  5. **Выбрать лучший кандидат** среди скачанных card.json: ``_pick_best_card``
     сначала отсекает нерелевантные по матч-скору (``_pick_best_match``: model-token
     бонус, type-mismatch penalty по ``imt_name``/``subj_name``), затем среди
     релевантных при сопоставимом скоре предпочитает карточку с бОльшим числом
     полезных options (богатую, не пустую). card.json качаем по кандидатам по
     порядку, останавливаясь на _MAX_FETCHED_CARDS успешно скачанных (404 не
     прекращает перебор).
  6. **Маппинг WB→Ozon**: ``_map_characteristics`` мапит русские имена характеристик на
     Ozon-словарь (lowercase + substring + fuzzy WRatio≥88) — финальное API публикации
     у нас Ozon, WB лишь источник данных.

Стоимость: 1 Serper-запрос (~$0.001) + N бесплатных CDN GET'ов. Latency: ~1-3s.
Все сетевые вызовы — с таймаутами и graceful-fallback (возврат [] вместо падения).
"""
from __future__ import annotations

import asyncio
import logging
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
from app.services.enrichment.sources.ozon_card_source import (
    _compress_search_query,
    _extract_model_tokens,
    _normalize_model,
)
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    resolve_value_id,
)
from app.services.providers.factory import get_web_search_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

# Известный sharding table «vol → basket-NN». На 404 перебираем все NN (01..21).
# Формат: (верхняя_граница_vol_включительно, NN).
_BASKET_THRESHOLDS: list[tuple[int, str]] = [
    (143,  "01"),
    (287,  "02"),
    (431,  "03"),
    (719,  "04"),
    (1007, "05"),
    (1061, "06"),
    (1115, "07"),
    (1169, "08"),
    (1313, "09"),
    (1601, "10"),
    (1655, "11"),
    (1919, "12"),
    (2045, "13"),
    (2189, "14"),
    (2405, "15"),
    (2621, "16"),
    (2837, "17"),
    (3053, "18"),
    (3473, "19"),
    (3793, "20"),
]
# vol >= 3794 → basket-21. Полный список NN для brute-force перебора на 404.
_BASKET_DEFAULT = "21"
_ALL_BASKET_NN: list[str] = [f"{n:02d}" for n in range(1, 22)]  # 01..21

# Regex для извлечения nm_id из ссылки на карточку WB.
# Покрывает .ru/.ge/.am/.by зеркала и относительные ссылки.
_NM_ID_RE = re.compile(
    r"(?:wildberries\.\w+/catalog/|/catalog/)(\d{6,12})/detail\.aspx",
    re.IGNORECASE,
)

# Chrome User-Agent для CDN GET (CDN не банит, но без UA иногда 403).
_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_HTTP_TIMEOUT = 15.0
_SERPER_NUM_RESULTS = 10
_MAX_CANDIDATES = 10  # топ-N уникальных nm_id из Serper (многие дадут 404)

# Ретрай Serper-поиска. Serper флакает при concurrency (троттлинг, пустые
# ответы): один и тот же запрос то даёт 10 nm_id, то 0. Если поиск вернул 0
# organic ИЛИ из них извлеклось 0 nm_id — повторяем с экспоненциальным бэкоффом.
# Покрывает временные пустышки, не залипая на флаке. Бэкофф короткий, попыток
# мало — время растёт максимум в _SERPER_MAX_ATTEMPTS раз только в худшем случае
# (стабильно пустой товар), на успехе ретраев нет.
_SERPER_MAX_ATTEMPTS = 3        # всего попыток (1 основная + 2 ретрая)
_SERPER_BACKOFF_BASE = 1.0      # сек: задержки 1с, 2с (экспонента 2^n)

# Сколько card.json реально скачать прежде чем выбирать лучший. Многие nm_id
# архивные/несуществующие → 404 по всем basket. Перебираем кандидатов по порядку,
# но останавливаемся, набрав _MAX_FETCHED_CARDS успешно скачанных карточек.
_MAX_FETCHED_CARDS = 4

# Карточка считается «богатой» если у неё ≥ _RICH_OPTIONS_THRESHOLD полезных
# options. При сопоставимом матч-скоре богатую предпочитаем бедной.
_RICH_OPTIONS_THRESHOLD = 6
# Разрешённый разрыв в матч-скоре, при котором богатство решает исход. Если
# кандидат с большим числом options отстаёт по скору не более чем на эту
# величину — берём его (богатую карточку), а не пустого лидера по fuzzy.
_SCORE_TIE_BAND = 12.0

# Confidence — параллельно с OzonCardSource.
_CONF_EXACT = 0.93
_CONF_BRAND_LINE = 0.85

_EXACT_THRESHOLD = 78.0
_BRAND_LINE_THRESHOLD = 60.0

# Skip-guard
_SKIP_FILL_RATIO = 0.80

# LRU
_CACHE_MAX = 256

# Brand-line BLACKLIST — model-specific атрибуты.
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


def _basket_nn_from_table(nm_id: int) -> str:
    """Возвращает basket-NN из known таблицы по vol (fallback basket-21)."""
    vol = nm_id // 100_000
    for threshold, nn in _BASKET_THRESHOLDS:
        if vol <= threshold:
            return nn
    return _BASKET_DEFAULT


def _card_url(nn: str, nm_id: int) -> str:
    vol = nm_id // 100_000
    part = nm_id // 1000
    return (
        f"https://basket-{nn}.wbbasket.ru"
        f"/vol{vol}/part{part}/{nm_id}/info/ru/card.json"
    )


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class WbCardSource(AttributeSource):
    """Копия характеристик с live WB-карточки похожего товара.

    Путь: Serper (поиск nm_id) → CDN card.json (httpx, бесплатно).
    Cost: 1 Serper-запрос/товар. Latency: ~1-3s.
    """

    def __init__(
        self,
        web_search_client: Any = None,
        **kwargs: Any,
    ):
        _ = kwargs  # backward-compat (старые коды передавали scrappey_key и т.п.)

        # Serper-клиент для поиска nm_id. Лениво создаём через factory, если не
        # передан явно. Если SERPER_API_KEY не задан — factory кинет/вернёт None,
        # extract() тогда всегда вернёт [].
        self._search_client = web_search_client
        if self._search_client is None:
            try:
                self._search_client = get_web_search_client()
            except Exception as exc:  # SERPER_API_KEY не задан и т.п.
                logger.warning(
                    "[WbCard] web search client недоступен (%s) — extract() вернёт [].",
                    exc,
                )
                self._search_client = None
        if self._search_client is None:
            logger.warning(
                "[WbCard] SERPER не сконфигурирован (PROVIDER_WEB_SEARCH != 'serper' "
                "или нет ключа) — extract() всегда вернёт []."
            )

        self._judge = WbCardJudge()
        # LRU cache: (brand_lower, model_lower) → list[AttributeValue]
        self._cache: "OrderedDict[tuple[str, str], list[AttributeValue]]" = OrderedDict()

    @property
    def source_type(self) -> Source:
        return Source.WB_CARD

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим если product_name достаточный и search-клиент доступен."""
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
            # НЕ кэшируем пустышку: ошибка часто транзиентная (Serper-флак,
            # сетевой сбой). Кэширование [] «залипает» и блокирует ретрай в
            # следующем прогоне. Возвращаем пусто, кэш не трогаем.
            return []

        # Кэшируем только непустой результат. Пустой список — обычно следствие
        # флака Serper (троттлинг/пустой ответ), уже отретраенного в _search;
        # если всё равно пусто, кэшировать [] нельзя — иначе флак-пустышка
        # залипнет в LRU и следующий прогон не сделает повторный поиск. Непустой
        # результат кэшируем как раньше, чтобы не бить Serper повторно.
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
        """Полный flow: compress → Serper(nm_id) → CDN card.json → pick → map → AVs."""
        full_name = context.product_name.strip()
        cat_leaf = context.category_path[-1] if context.category_path else None
        primary_query = _compress_search_query(
            full_name, context.brand, max_tokens=5, category_name=cat_leaf
        )
        fallback_query = _compress_search_query(
            full_name, context.brand, max_tokens=3, category_name=cat_leaf
        )

        queries_to_try: list[str] = [primary_query]
        if fallback_query and fallback_query != primary_query:
            queries_to_try.append(fallback_query)

        # ---- SEARCH (Serper → nm_id) ----
        nm_ids: list[int] = []
        used_query: Optional[str] = None
        for q in queries_to_try:
            logger.info("[WbCard] search query: '%s' (was: '%s')", q, full_name[:80])
            candidates = await self._search(q)
            if candidates:
                nm_ids, used_query = candidates, q
                break
            logger.info("[WbCard] no nm_id на query='%s' — пробую fallback", q[:60])

        if not nm_ids or used_query is None:
            logger.info("[WbCard] no nm_id ни для primary ни для fallback")
            return []

        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _CHROME_UA},
        ) as client:
            # ---- CARD JSON (CDN) для кандидатов ----
            # Перебираем nm_id по порядку. 404 на одном не прекращает перебор —
            # идём к следующему. Останавливаемся, набрав _MAX_FETCHED_CARDS
            # успешно скачанных карточек (чтобы было из чего выбирать богатую,
            # но не качать все 10 кандидатов).
            cards: list[tuple[int, dict]] = []
            attempted = 0
            for nm_id in nm_ids:
                attempted += 1
                card = await self._fetch_card(client, nm_id)
                if card:
                    cards.append((nm_id, card))
                    if len(cards) >= _MAX_FETCHED_CARDS:
                        break

            if not cards:
                logger.info(
                    "[WbCard] ни один card.json не скачался (%d из %d кандидатов перебрано)",
                    attempted, len(nm_ids),
                )
                return []

            logger.info(
                "[WbCard] скачано %d card.json из %d перебранных кандидатов (всего %d)",
                len(cards), attempted, len(nm_ids),
            )

            # ---- PICK BEST (match-фильтр → среди релевантных богатую) ----
            best = self._pick_best_card(used_query, cat_leaf, cards)
            if best is None:
                return []
            nm_id, card, title, score = best
            mode = self._classify_match(score)
            if mode == "skip":
                logger.info("[WbCard] best score=%.1f < %.0f — skip", score, _BRAND_LINE_THRESHOLD)
                return []

            logger.info(
                "[WbCard] match=%s score=%.1f title='%s' nm=%s",
                mode, score, title[:80], nm_id,
            )

            chars = self._extract_options(card)
            if not chars:
                logger.info("[WbCard] no options/characteristics в card.json nm=%s", nm_id)
                return []

            # ---- IMAGES (для downstream VisionSource) ----
            new_image_urls = self._extract_image_urls(card, nm_id)
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
                chars, targets, context, mode, title, score,
            )

    async def _search(self, query: str) -> list[int]:
        """Serper → nm_id с ретраем при пустом результате (троттлинг/флак).

        Serper при concurrency нестабилен: тот же запрос то возвращает 10 nm_id,
        то 0 organic / 0 извлечённых nm_id. Если попытка дала 0 — повторяем с
        экспоненциальным бэкоффом (1с, 2с) до _SERPER_MAX_ATTEMPTS. На успехе
        (≥1 nm_id) выходим сразу. После всех ретраев пусто → graceful [] (выше
        по стеку это fallback, не падение).
        """
        for attempt in range(1, _SERPER_MAX_ATTEMPTS + 1):
            nm_ids = await self._search_once(query)
            if nm_ids:
                if attempt > 1:
                    logger.info(
                        "[WbCard] Serper непустой результат с попытки %d/%d",
                        attempt, _SERPER_MAX_ATTEMPTS,
                    )
                return nm_ids
            if attempt < _SERPER_MAX_ATTEMPTS:
                delay = _SERPER_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.info(
                    "[WbCard] Serper → 0 nm_id (попытка %d/%d), ретрай через %.1fс",
                    attempt, _SERPER_MAX_ATTEMPTS, delay,
                )
                await asyncio.sleep(delay)
        logger.info(
            "[WbCard] Serper → 0 nm_id после %d попыток (флак/нет результатов)",
            _SERPER_MAX_ATTEMPTS,
        )
        return []

    async def _search_once(self, query: str) -> list[int]:
        """Одна Serper-попытка → список уникальных nm_id (top-N).

        Запрос ``inurl:catalog detail.aspx <query>`` даёт ~100% прямых ссылок
        на карточки WB. Предпочитаем домен wildberries.ru, зеркала — fallback.
        """
        serper_query = f"inurl:catalog detail.aspx {query}".strip()
        try:
            results = await self._search_client.search(
                serper_query, num_results=_SERPER_NUM_RESULTS
            )
        except Exception as exc:
            logger.info("[WbCard] Serper search err: %s", exc)
            return []

        organic = getattr(results, "organic_results", None) or []

        # Собираем (nm_id, prefer_ru) в порядке появления, дедуп.
        ordered_main: list[int] = []   # с wildberries.ru
        ordered_mirror: list[int] = []  # зеркала / относительные
        seen: set[int] = set()
        for item in organic:
            link = getattr(item, "link", "") or ""
            m = _NM_ID_RE.search(link)
            if not m:
                continue
            try:
                nm_id = int(m.group(1))
            except (ValueError, TypeError):
                continue
            if nm_id in seen:
                continue
            seen.add(nm_id)
            if "wildberries.ru" in link.lower():
                ordered_main.append(nm_id)
            else:
                ordered_mirror.append(nm_id)

        nm_ids = (ordered_main + ordered_mirror)[:_MAX_CANDIDATES]
        logger.info("[WbCard] Serper → %d уникальных nm_id: %s", len(nm_ids), nm_ids)
        return nm_ids

    async def _fetch_card(
        self,
        client: httpx.AsyncClient,
        nm_id: int,
    ) -> Optional[dict]:
        """Скачать card.json с CDN. Простой GET; на 404 перебор NN (01..21)."""
        primary_nn = _basket_nn_from_table(nm_id)
        # Сначала known NN, затем остальные (без повтора primary).
        order = [primary_nn] + [nn for nn in _ALL_BASKET_NN if nn != primary_nn]
        for nn in order:
            data = await self._try_basket(client, nn, nm_id)
            if data is not None:
                if nn != primary_nn:
                    logger.info("[WbCard] nm=%s найден на basket-%s (fallback)", nm_id, nn)
                return data
        logger.info("[WbCard] card.json не найден ни на одном basket для nm=%s", nm_id)
        return None

    async def _try_basket(
        self,
        client: httpx.AsyncClient,
        nn: str,
        nm_id: int,
    ) -> Optional[dict]:
        """Один CDN GET. None если non-200 / не JSON / network err."""
        url = _card_url(nn, nm_id)
        try:
            r = await client.get(url)
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            logger.debug("[WbCard] CDN network err %s: %s", url, exc)
            return None
        if r.status_code != 200:
            return None
        try:
            data = r.json()
        except ValueError:  # включает json.JSONDecodeError
            return None
        except Exception as exc:  # noqa: BLE001 — на всякий случай не падаем
            logger.debug("[WbCard] card.json parse err %s: %s", url, exc)
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------------
    # Card JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_options(card: dict) -> list[dict]:
        """Извлечь characteristics из WB card.json.

        Структуры:
          - options: [{"name": "Цвет", "value": "Белый"}, ...]
          - grouped_options: [{"group_name": "...", "options": [...]}]
          - compositions: [{"name": "хлопок", "value": "100"}] ИЛИ ["хлопок 100%"]
            → Состав.

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

        # 3. compositions → Состав. WB отдаёт либо список {name,value} (доля
        #    материала), либо список строк. Собираем в одну пару «Состав».
        comp_parts: list[str] = []
        for c in card.get("compositions") or []:
            if isinstance(c, dict):
                cname = str(c.get("name") or "").strip()
                cval = c.get("value")
                cval_str = str(cval).strip() if isinstance(cval, (str, int, float)) else ""
                if cname and cval_str:
                    comp_parts.append(f"{cname} {cval_str}")
                elif cname:
                    comp_parts.append(cname)
            elif isinstance(c, str) and c.strip():
                comp_parts.append(c.strip())
        if comp_parts and "состав" not in seen:
            seen.add("состав")
            out.append({"name": "Состав", "value": ", ".join(comp_parts), "value_ids": []})

        return out

    @staticmethod
    def _extract_image_urls(card: dict, nm_id: int, limit: int = 5) -> list[str]:
        """Собрать high-res photo URLs из media.photos[] или media.photo_count."""
        out: list[str] = []
        media = card.get("media") or {}

        for p in media.get("photos") or []:
            if isinstance(p, dict):
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

        photo_count = media.get("photo_count")
        if isinstance(photo_count, int) and photo_count > 0:
            nn = _basket_nn_from_table(nm_id)
            vol = nm_id // 100_000
            part = nm_id // 1000
            base = f"https://basket-{nn}.wbbasket.ru/vol{vol}/part{part}/{nm_id}/images/big"
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

    def _pick_best_card(
        self,
        query: str,
        cat_leaf: Optional[str],
        cards: list[tuple[int, dict]],
    ) -> Optional[tuple[int, dict, str, float]]:
        """Выбрать карточку среди скачанных: сначала матч, потом богатство.

        Прежняя стратегия брала ОДИН лучший по fuzzy-скору вслепую — если у
        лидера была бедная карточка (3 options), мы теряли соседнюю богатую с
        чуть меньшим скором. Новая логика:

          1. Скорим каждый кандидат по заголовку (imt_name/subj_name) через
             _pick_best_match (model-token бонус, type-mismatch штраф) и считаем
             число полезных options (_extract_options).
          2. Отсекаем нерелевантные (score < _BRAND_LINE_THRESHOLD) — это
             type/brand-mismatch, такие карточки не нужны даже если богатые.
          3. Среди релевантных выбираем «лучшую»: при сопоставимом скоре
             (разрыв ≤ _SCORE_TIE_BAND) предпочитаем карточку с бОльшим числом
             options. Иначе — карточку с максимальным скором.

        Возвращает (nm_id, card, title, score) или None.
        """
        scored: list[dict] = []
        for nm_id, card in cards:
            title = self._card_title(card)
            _, score = self._pick_best_match(
                query, [{"title": title}], category_leaf=cat_leaf
            )
            n_opts = len(self._extract_options(card))
            scored.append({
                "nm_id": nm_id,
                "card": card,
                "title": title,
                "score": score,
                "n_opts": n_opts,
            })

        # Отсечь нерелевантные по матчу (type/brand-mismatch).
        relevant = [c for c in scored if c["score"] >= _BRAND_LINE_THRESHOLD]
        if not relevant:
            # Все кандидаты ниже порога матча — деградируем к старому поведению:
            # вернуть абсолютного лидера по скору, дальше _classify_match → skip.
            top = max(scored, key=lambda c: c["score"], default=None)
            if top is None:
                return None
            return (top["nm_id"], top["card"], top["title"], top["score"])

        max_score = max(c["score"] for c in relevant)

        # Кандидаты в пределах tie-band от лидера → решает богатство (n_opts),
        # при равенстве — скор. Так пустой лидер уступает богатому соседу.
        contenders = [c for c in relevant if c["score"] >= max_score - _SCORE_TIE_BAND]
        best = max(contenders, key=lambda c: (c["n_opts"], c["score"]))

        if logger.isEnabledFor(logging.INFO):
            ranking = ", ".join(
                f"nm={c['nm_id']}(score={c['score']:.1f},opts={c['n_opts']})"
                for c in sorted(relevant, key=lambda c: -c["score"])
            )
            logger.info(
                "[WbCard] pick: %d релевантных → выбран nm=%s (score=%.1f, opts=%d) | %s",
                len(relevant), best["nm_id"], best["score"], best["n_opts"], ranking,
            )

        return (best["nm_id"], best["card"], best["title"], best["score"])

    @staticmethod
    def _card_title(card: dict) -> str:
        """Заголовок карточки для fuzzy-сравнения: brand + imt_name + subj_name."""
        parts: list[str] = []
        for key in ("selling", "imt_name", "subj_name", "subj_root_name"):
            val = card.get(key)
            if key == "selling" and isinstance(val, dict):
                val = val.get("brand_name")
            if isinstance(val, str) and val.strip():
                if val.strip().lower() not in " ".join(parts).lower():
                    parts.append(val.strip())
        return " ".join(parts).strip()

    @staticmethod
    def _pick_best_match(
        query: str,
        tiles: list[dict],
        category_leaf: Optional[str] = None,
    ) -> tuple[Optional[dict], float]:
        """rapidfuzz match с model-token бонусом и type-mismatch штрафом.

        Логика идентична OzonCardSource._pick_best_match: partial_ratio +
        token_sort_ratio, +5 за общий артикул, -30 за несовпадение типа товара
        (только для одежды без артикула).
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return (tiles[0], 100.0) if tiles else (None, 0.0)

        _MODEL_BONUS = 5.0
        _TYPE_MISMATCH_PENALTY = 30.0
        q_models = _extract_model_tokens(query)
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
            if cat_leaf_low and not q_models and cat_leaf_low not in t:
                score -= _TYPE_MISMATCH_PENALTY
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

        Маппим WB-характеристики на наш Ozon словарь — финальное API
        публикации у нас Ozon, а WB-карточка лишь источник данных.
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

            target_id = name_to_target_id.get(char_name_low)
            if target_id is None:
                for tn, tid in name_to_target_id.items():
                    if char_name_low in tn or tn in char_name_low:
                        target_id = tid
                        break
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
