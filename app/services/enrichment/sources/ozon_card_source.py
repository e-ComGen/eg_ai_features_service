"""OzonCardSource — копирует характеристики из живой Ozon-карточки похожего товара.

Использует ПУБЛИЧНЫЕ HTML-страницы Ozon через Scrappey.com proxy (TRUE PAYG,
~$0.0002-0.001/page). Composer-api.bx заблокирован DataDome даже через
премиум-прокси; HTML-страницы Ozon содержат полный widget state в SSR.

Алгоритм:
  1. Skip-guard: если already_filled покрывает ≥80% targets → return [].
  2. Search step: GET https://www.ozon.ru/search/?text=<name> через Scrappey
     → regex по `<a href="/product/<slug>-<pid>/">` → list of tiles.
  3. Match step: rapidfuzz (partial_ratio + token_sort_ratio) vs product_name:
     - ≥85 → "exact": full card, copy ALL.
     - 70-84 → "brand_line": full card, только safe attrs.
     - <70 → skip.
  4. Detail step: GET https://www.ozon.ru/product/<slug>/features/ через Scrappey
     → regex `<div id="state-webCharacteristics-..." data-state='<JSON>'>`
     → распарсить characteristics[].short / .long / .full → [{name, value}].
  5. Mapping по русским именам через get_ozon_characteristics_for_type
     (lowercase + substring + fuzzy WRatio≥88).
  6. resolve_value_id для (attr_id, value) → словарный value_id Ozon.
  7. Confidence: 0.93 (exact), 0.85 (brand_line). Source: OZON_CARD.
  8. Evidence: f"ozon:{title[:50]} | match={score}".
  9. In-process LRU cache по (brand, normalized_model), max 256.

Scrappey cost: 1 credit / запрос → 2 credits / товар (search + features).
Free trial: 150 credits = 75 товаров. PAYG top-ups доступны без подписки.

Anti-block:
  - SCRAPPEY_KEY читаем из env, можно передать в __init__.
  - DataDome detection: incidentId в первых 1500 символах → []
  - HTTP 4xx из Scrappey → warn + [].
  - Timeout 180s (Scrappey full browser bypass занимает 8-20s).
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
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
from app.services.enrichment.judges.ozon_card_judge import OzonCardJudge
from app.services.enrichment.prompt_router import filter_already_filled_targets
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    resolve_value_id,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

_OZON_SEARCH_URL = "https://www.ozon.ru/search/"
_OZON_PRODUCT_BASE = "https://www.ozon.ru/product/"
_SCRAPPEY_ENDPOINT = "https://publisher.scrappey.com/api/v1"
_MAX_SEARCH_TILES = 5
_HTTP_TIMEOUT = 180.0  # Scrappey browser bypass обычно 8-20s, иногда до 60s

# Regex для парсинга
_PRODUCT_LINK_RE = re.compile(
    r'<a[^>]+href="(/product/([a-z0-9\-]+)-(\d+)/)[^"]*"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_FEATURES_STATE_RE = re.compile(
    r'<div\s+id="state-webCharacteristics-[^"]+"\s+data-state=\'([^\']+)\'',
    re.DOTALL,
)
# Product photos на /features/ странице — `<img src="https://ir.ozone.ru/s3/multimedia-X/wcY/...jpg">`.
# wc1200 = высокое разрешение (1200px) — нужно для Vision LLM. Берём первые ~5 уникальных
# (товар обычно имеет 5-15 фото — front/back/side/box/detail). Vision дополнит attrs
# которые сложно достать из текста: цвет, RGB-подсветка, форм-фактор SFX vs ATX.
_OZON_IMAGE_RE = re.compile(
    r'<img[^>]+src="(https://ir(?:-\d+)?\.ozone\.ru/[^"]+\.(?:jpg|jpeg|png|webp))"',
    re.IGNORECASE,
)

# Confidence.
# brand_line conf=0.85 — ровно на pipeline `filter_already_filled_targets`
# threshold 0.85, чтобы OzonCard fills попадали в filled_so_far и AttributeMerger
# выбирал их как backup, если позже более уверенный источник не нашёл.
# IceCat skip-guard теперь использует is_confident() (>=0.90 для IceCat),
# а не naive set membership, поэтому OzonCard brand_line 0.85 НЕ блокирует
# IceCat от заполнения этих же attrs с conf 0.92 — merger выберет IceCat.
# exact conf=0.93 — точный match того же товара, не переписываем.
_CONF_EXACT = 0.93
_CONF_BRAND_LINE = 0.85

# Similarity thresholds (понижены для (V2/V3/Plus/Bronze) вариаций — title часто
# содержит "Блок питания + brand + model + V3 80 Plus Gold (MPE-XXX-...)", т.е.
# много шума вокруг query "brand + model").
_EXACT_THRESHOLD = 78.0
_BRAND_LINE_THRESHOLD = 65.0  # v17 had 75.0 — слишком жёстко, убил 73→0 OzonCard fills.
                              # 65 — компромисс: brand_line whitelist всё равно ограничивает
                              # копирование model-specific attrs.

# Retry: HTML < этого размера или 0 tiles → retry (Ozon **рандомно** отдаёт
# обрезанную SPA-страницу 10KB без SSR data; та же query 2-3 попытки спустя
# возвращает полные 480+KB SSR). Эмпирически 3 попыток достаточно.
_MIN_VALID_HTML_LEN = 50_000
_MAX_RETRIES = 3

# Skip-guard
_SKIP_FILL_RATIO = 0.80

# LRU
_CACHE_MAX = 256

# Brand-line BLACKLIST (универсальный для всех категорий).
#
# В brand_line режиме (match score 60-78 — товар похож, но не точно тот же)
# копируем ВСЁ что Ozon /features/ карточки соседнего товара отдаёт, **кроме**
# явно model-specific атрибутов которые гарантированно отличаются между
# моделями даже одной линейки бренда (Артикул конкретного товара, MPN,
# серийник, ID-шник).
#
# Фильтрация по targets+Ozon dict работает естественно: если char_name
# с tile не сматчился ни с одним target.name из загруженного Ozon dict для
# текущей категории — этот char просто пропускается в _map_characteristics().
# Поэтому whitelist в source избыточен — здесь только защита от перетирания
# реальных полей шумом из чужой карточки.
_BRAND_LINE_BLACKLIST: frozenset[str] = frozenset(name.lower() for name in {
    "Артикул",
    "Код производителя",
    "MPN",
    "Партномер",
    "Серийный номер",
    "EAN",
    "GTIN",
    "ASIN",
    "Дата производства",
    "Модель",
    "Название модели",
    "ID товара",
    "ID карточки",
})

# Brand-line STRICT WHITELIST — главный фильтр для brand_line режима.
#
# Threshold 75-77 (brand_line) означает «близкая модель того же бренда, но
# НЕ тот же товар». Spec attrs (Мощность, Длина/Ширина/Высота, Кол-во SATA/
# Molex, Гарантия, Сертификат 80 PLUS, Подсветка, Разъёмы) — модель-
# специфичны и копирование их с соседней модели = галлюцинация.
#
# Безопасны для копирования с brand-line карточки только brand-/линейка-
# уровневые атрибуты: бренд, производитель, страна-изготовитель, цвет,
# ТН ВЭД (классификация на уровне типа товара), назначение.
#
# Применяется case-insensitive substring match: char_name из Ozon-карточки
# проходит если ЛЮБОЙ entry из whitelist встречается как substring в нём
# (например «Цвет товара» → match «цвет товара», «Цвет товара (основной)»).
#
# Exact режим (≥78) не использует whitelist — копируется всё.
_BRAND_LINE_STRICT_WHITELIST: frozenset[str] = frozenset(name.lower() for name in {
    # Brand/manufacturer (как было)
    "Бренд",
    "Производитель",
    "Страна-изготовитель",
    # Color/appearance (как было)
    "Цвет товара",
    # Classification (как было)
    "ТН ВЭД",
    "Назначение",
    # NEW Phase 2 #1: brand-line-safe spec attrs.
    # Эти атрибуты одинаковы в продуктовой линейке бренда независимо от
    # конкретной модели (Phase 1 #5 recovery: вернули 30+ valid OzonCard
    # fills для Zalman/Deepcool/FSP — гарантия/PFC/охлаждение/80PLUS
    # фактически brand-line-уровневые, а не модель-специфичные).
    "Гарантия",                              # одинакова в брендовой линейке
    "Гарантийный срок",                      # синоним «Гарантии»
    "Подсветка",                             # enum-based на дизайне линейки
    "Оплётка проводов",                      # одинакова в продуктовом классе
    "Сертификат 80 PLUS",                    # часто одинаков (Bronze/Gold/Platinum)
    "Корректор коэффициента мощности (PFC)", # 99% Активный для современных БП
    "Система охлаждения",                    # 95% Активная (с вентилятором)
    "Тип",                                   # «Блок питания компьютера» по умолчанию
})

# Generic-префиксы — универсальные слова, которые срезаются как fallback
# (когда category_name не задан или не покрывает случай)
_GENERIC_PREFIXES = (
    "блок питания", "блок", "питания",
    "power supply", "power", "supply", "unit",
)


def _strip_category_prefix(text: str, category_name: Optional[str]) -> str:
    """Срезает ведущие слова категории из начала text (case-insensitive).

    Алгоритм:
      1. Попробовать совпадение с полной фразой категории (напр. «Микроволновая печь»).
      2. Если не совпало — итеративно срезать каждое слово категории, пока
         оно стоит в начале text. Это покрывает случай «Блок питания Cooler Master»
         при leaf «Блок питания компьютера»: срезается «Блок», затем «питания».

    Возвращает текст после среза (или исходный, если ничего не срезалось).
    """
    if not category_name:
        return text
    result = text.strip()
    low = result.lower()
    cat_low = category_name.strip().lower()

    # Попытка 1: полная фраза
    if low.startswith(cat_low):
        return result[len(cat_low):].strip()

    # Попытка 2: итеративный побуквенный срез слов категории
    cat_words = [w for w in cat_low.split() if len(w) > 2]
    for word in cat_words:
        low = result.lower()
        if low.startswith(word):
            result = result[len(word):].strip()
        # не break — пробуем следующее слово категории тоже (если оно стоит следом)

    return result


def _compress_search_query(
    product_name: str,
    brand: Optional[str],
    max_tokens: int = 5,
    category_name: Optional[str] = None,
) -> str:
    """Сжимает product_name для Ozon search до «бренд + модель».

    Ozon отдаёт пустую SPA-страницу для слишком специфичных запросов
    ("Блок питания Cooler Master MWE Gold 750 V2 Full Modular 750W ATX"
    → 10KB пустота). Нормальный SSR приходит для запросов
    «brand + model line» (3-5 терминов).

    Логика:
      1. Strip leading category prefix (из category_name, напр. «Монитор», «Наушники»).
         Если category_name не задан — fallback на _GENERIC_PREFIXES (БП-кейс).
      2. Взять первые max_tokens токенов.
      3. Если brand задан и его нет в результате — prepend.

    category_name — leaf из ExtractionContext.category_path (последний элемент).
    """
    result = product_name.strip()

    # ---- Шаг 1: срезать категорийный префикс ----
    # Основной сигнал: имя категории из контекста (generic, без хардкода)
    after_cat = _strip_category_prefix(result, category_name)
    stripped = after_cat != result
    if stripped:
        result = after_cat
    else:
        # Fallback: старые _GENERIC_PREFIXES (работают как раньше для БП и похожих)
        low = result.lower()
        for prefix in _GENERIC_PREFIXES:
            if low.startswith(prefix):
                result = result[len(prefix):].strip()
                break

    # ---- Шаг 2: взять первые max_tokens токенов (бренд + модель) ----
    tokens = result.split()[:max_tokens]
    compact = " ".join(tokens).strip()

    # ---- Шаг 3: если бренд не попал в начало — prepend ----
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


def _is_datadome_block(content: str) -> bool:
    """Detect Ozon's DataDome challenge in response body.

    DataDome возвращает JSON `{"incidentId":"...","blockURL":"..."}` или
    `{"incidentId":"...","supportURL":"..."}` вместо composer-api layout.
    Валидный composer-api начинается с `{"layout":[...]}` и `incidentId` в нём
    не встречается. Достаточно проверить `incidentId` в первых 500 символах.
    """
    if not content:
        return False
    head = content[:500]
    return "incidentId" in head


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class OzonCardSource(AttributeSource):
    """Копия характеристик с live Ozon-карточки похожего товара через Scrappey.

    Cost: 2 credits/product (search + features), 0 на failed.
    Latency: 10-40s end-to-end (8-20s per Scrappey call).
    """

    def __init__(
        self,
        scrappey_key: Optional[str] = None,
        # backward-compat (старые коды передавали эти kwargs, игнорируем):
        scrapfly_key: Optional[str] = None,
        apify_token: Optional[str] = None,
        ozon_api_base: Optional[str] = None,
        **kwargs: Any,
    ):
        _ = scrapfly_key
        _ = apify_token
        _ = ozon_api_base
        _ = kwargs

        self._scrappey_key = scrappey_key or os.environ.get("SCRAPPEY_KEY")
        if not self._scrappey_key:
            logger.warning(
                "[OzonCard] SCRAPPEY_KEY не задан (ни параметром, ни в env) — "
                "extract() всегда вернёт []."
            )

        self._judge = OzonCardJudge()

        # LRU cache: (brand_lower, model_lower) → list[AttributeValue]
        self._cache: "OrderedDict[tuple[str, str], list[AttributeValue]]" = OrderedDict()

    @property
    def source_type(self) -> Source:
        return Source.OZON_CARD

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

        # Skip-guard: ≥80% targets уже filled с high confidence
        filled_ids = {av.attribute_id for av in already_filled if av.confidence >= 0.85}
        if targets and len(filled_ids & {t.id for t in targets}) / len(targets) >= _SKIP_FILL_RATIO:
            logger.debug(
                "[OzonCard] skip (≥%.0f%% targets уже filled)",
                _SKIP_FILL_RATIO * 100,
            )
            return []

        effective = filter_already_filled_targets(targets, already_filled)
        if not effective:
            return []

        brand = (context.brand or "").strip()
        model_norm = _normalize_model(context.product_name, brand)
        cache_key = (brand.lower(), model_norm)

        # Cache hit?
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            # Фильтруем по полному targets (не effective): already_filled атрибуты
            # тоже должны дойти до merger-а — он выберет лучшее через consensus.
            return self._filter_for_targets(self._cache[cache_key], targets)

        # Network calls
        try:
            all_values = await self._do_extract(context, targets)
        except Exception as exc:
            logger.warning(
                "[OzonCard] unexpected error для '%s': %s",
                context.product_name[:60], exc,
            )
            self._cache_put(cache_key, [])
            return []

        self._cache_put(cache_key, all_values)
        # Аналогично: фильтруем по полному targets — merger решает через consensus,
        # а не отбрасываем уже заполненные до merger-а.
        return self._filter_for_targets(all_values, targets)

    def get_judge(self) -> LlmJudge:
        return self._judge

    # ------------------------------------------------------------------
    # Network orchestration via Scrappey
    # ------------------------------------------------------------------

    async def _scrappey_fetch(
        self,
        client: httpx.AsyncClient,
        target_url: str,
        min_len: int = _MIN_VALID_HTML_LEN,
    ) -> Optional[str]:
        """POST к Scrappey с retry, возвращает HTML response от target_url.

        Ozon иногда отдаёт обрезанную SPA-страницу (10KB без SSR data),
        особенно при долгих запросах. Retry 1 раз если content слишком короткий.

        Возвращает None если все попытки fail.
        """
        for attempt in range(_MAX_RETRIES + 1):
            content = await self._scrappey_fetch_once(client, target_url)
            if content is None:
                # Hard fail (network, HTTP4xx, DataDome) — retry имеет смысл
                if attempt < _MAX_RETRIES:
                    logger.info(
                        "[OzonCard] retry #%d (hard fail) %s",
                        attempt + 1, target_url[:80],
                    )
                    continue
                return None
            if len(content) < min_len:
                # Soft fail — короткий HTML, бывает = пустая SPA. Retry.
                if attempt < _MAX_RETRIES:
                    logger.info(
                        "[OzonCard] retry #%d (short %d chars) %s",
                        attempt + 1, len(content), target_url[:80],
                    )
                    continue
                logger.info(
                    "[OzonCard] final HTML still too short (%d chars) for %s",
                    len(content), target_url[:80],
                )
                return None
            return content
        return None

    async def _scrappey_fetch_once(
        self,
        client: httpx.AsyncClient,
        target_url: str,
    ) -> Optional[str]:
        """Один POST к Scrappey, без retry."""
        payload = {"cmd": "request.get", "url": target_url}
        try:
            r = await client.post(
                _SCRAPPEY_ENDPOINT,
                params={"key": self._scrappey_key},
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            logger.info("[OzonCard] Scrappey network err: %s", exc)
            return None

        if r.status_code >= 400:
            logger.warning(
                "[OzonCard] Scrappey HTTP %s for %s — skip (body[:200]=%s)",
                r.status_code, target_url[:80], r.text[:200],
            )
            return None

        try:
            envelope = r.json()
        except (ValueError, json.JSONDecodeError):
            logger.info("[OzonCard] Scrappey returned non-json envelope")
            return None

        solution = envelope.get("solution") or {}
        upstream_status = solution.get("statusCode")
        content = solution.get("response") or ""

        if not content:
            logger.info(
                "[OzonCard] Scrappey empty content (upstream=%s) for %s",
                upstream_status, target_url[:80],
            )
            return None

        if upstream_status != 200:
            logger.info(
                "[OzonCard] upstream HTTP %s for %s — likely block",
                upstream_status, target_url[:80],
            )
            return None

        if _is_datadome_block(content):
            logger.info(
                "[OzonCard] DataDome challenge in response for %s",
                target_url[:80],
            )
            return None

        return content

    async def _do_extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Полный flow: search HTML → match → /features/ HTML → map → AVs."""
        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
        ) as client:
            # ---- SEARCH (HTML) ----
            # Ozon SSR пустой для длинных/слишком специфичных запросов —
            # сжимаем product_name до brand+model+ключевая_спека.
            # Если 5-токенный compress + 3 retries дали SPA — пробуем
            # ультра-короткий 3-токенный fallback (часто помогает на
            # редких/старых моделях вроде «AeroCool Cylon 600W»).
            full_name = context.product_name.strip()
            # Берём leaf-имя категории (последний элемент category_path) для
            # category-aware компрессии запроса (срезаем «Монитор», «Наушники» и т.д.)
            cat_leaf = context.category_path[-1] if context.category_path else None
            primary_query = _compress_search_query(full_name, context.brand, max_tokens=5, category_name=cat_leaf)
            fallback_query = _compress_search_query(full_name, context.brand, max_tokens=3, category_name=cat_leaf)

            queries_to_try: list[str] = [primary_query]
            if fallback_query and fallback_query != primary_query:
                queries_to_try.append(fallback_query)

            search_html: Optional[str] = None
            query: Optional[str] = None
            tiles: list[dict] = []
            for q in queries_to_try:
                logger.info(
                    "[OzonCard] search query: '%s' (was: '%s')",
                    q, full_name[:80],
                )
                html = await self._scrappey_fetch(client, f"{_OZON_SEARCH_URL}?text={q}")
                if html is None:
                    continue
                parsed = self._parse_search_tiles_html(html)
                if parsed:
                    search_html, query, tiles = html, q, parsed
                    break
                logger.info("[OzonCard] no tiles на query='%s' — пробую fallback", q[:60])

            if not tiles or query is None:
                logger.info("[OzonCard] no search tiles ни для primary ни для fallback")
                return []

            top_tile, top_score = self._pick_best_match(query, tiles[:_MAX_SEARCH_TILES])
            if top_tile is None:
                return []
            mode = self._classify_match(top_score)
            if mode == "skip":
                logger.info(
                    "[OzonCard] best score=%.1f < %.0f — skip",
                    top_score, _BRAND_LINE_THRESHOLD,
                )
                return []

            title = (top_tile.get("title") or "").strip()
            slug = (top_tile.get("slug") or "").strip()
            pid = (top_tile.get("pid") or "").strip()
            if not slug or not pid:
                logger.info("[OzonCard] no slug/pid in top tile")
                return []

            logger.info(
                "[OzonCard] match=%s score=%.1f title='%s' pid=%s",
                mode, top_score, title[:80], pid,
            )

            # ---- FEATURES (HTML SSR) ----
            features_url = f"{_OZON_PRODUCT_BASE}{slug}-{pid}/features/"
            features_html = await self._scrappey_fetch(client, features_url)
            if features_html is None:
                return []

            chars = self._parse_characteristics_html(features_html)
            if not chars:
                logger.info("[OzonCard] no characteristics в /features/ for pid=%s", pid)
                return []

            # ---- IMAGES (для downstream VisionSource) ----
            # Mutating context.image_urls — pipeline передаёт context по ссылке
            # между stages, поэтому Stage 3 (VisionSource) увидит эти фотки
            # на товарах где OzonCard нашёл tile. Vision дополнит «визуальные»
            # attrs (цвет, RGB-подсветка, форм-фактор) которые сложно достать
            # из текста характеристик.
            new_image_urls = self._extract_image_urls(features_html)
            if new_image_urls:
                existing = set(context.image_urls or [])
                added = [u for u in new_image_urls if u not in existing]
                if added:
                    context.image_urls = list(context.image_urls or []) + added
                    logger.info(
                        "[OzonCard] +%d image URLs для VisionSource (pid=%s)",
                        len(added), pid,
                    )

            # ---- MAP & EMIT ----
            return self._map_characteristics(
                chars, targets, context, mode, title, top_score,
            )

    # ------------------------------------------------------------------
    # HTML parsing (Scrappey HTML pages)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_search_tiles_html(html: str) -> list[dict]:
        """Извлекает product tiles из Ozon search HTML страницы.

        Каждый товар на search page имеет 2+ anchor'а с одинаковым href:
        первый — image link (короткий badge text), второй — title link
        (`<span class="tsBody...">{real title}</span>`). Для каждого pid
        берём anchor с самым длинным распарсенным title.

        Возвращает [{title, slug, pid}] в порядке появления на странице.
        """
        # pid → (slug, best_title, first_offset)
        by_pid: dict[str, dict] = {}
        for m in _PRODUCT_LINK_RE.finditer(html):
            slug, pid, body = m.group(2), m.group(3), m.group(4)
            # Strip inner HTML tags для title
            title = re.sub(r"<[^>]+>", " ", body)
            title = re.sub(r"\s+", " ", title).strip()
            existing = by_pid.get(pid)
            if existing is None:
                by_pid[pid] = {"slug": slug, "title": title, "offset": m.start()}
            elif len(title) > len(existing["title"]):
                existing["title"] = title
                existing["slug"] = slug
        tiles = sorted(by_pid.items(), key=lambda kv: kv[1]["offset"])
        out: list[dict] = []
        for pid, info in tiles:
            title = info["title"]
            # Skip tiles without meaningful title (badge-only anchors)
            if not title or len(title) < 15:
                continue
            out.append({"title": title[:200], "slug": info["slug"], "pid": pid})
        return out

    @staticmethod
    def _extract_image_urls(html: str, limit: int = 5) -> list[str]:
        """Извлекает URL'ы фоток товара из Ozon /features/ HTML.

        Возвращает первые `limit` уникальных high-res URL'ов. Vision LLM
        обычно достаточно 3-5 фоток (front/back/box) — больше = дороже без
        прироста. Дедупим по basename файла (один товар имеет ту же фотку
        в нескольких разрешениях wc50/wc300/wc1200).
        """
        seen_basenames: set[str] = set()
        out: list[str] = []
        for m in _OZON_IMAGE_RE.finditer(html):
            url = m.group(1)
            # Dedup по imagename (https://ir.ozone.ru/s3/multimedia-X/wc1200/{filename})
            basename = url.rsplit("/", 1)[-1].split("?", 1)[0]
            if basename in seen_basenames:
                continue
            seen_basenames.add(basename)
            out.append(url)
            if len(out) >= limit:
                break
        return out

    @classmethod
    def _parse_characteristics_html(cls, html: str) -> list[dict]:
        """Извлекает характеристики из /features/ HTML.

        Ozon рендерит каждый widget с `<div id="state-webCharacteristics-..."
        data-state='<JSON>'>`. JSON структура:
          {"link":"...","characteristics":[
              {"short":[{key,name,values:[{text,id}]}],
               "long":[...], "full":[...]}
          ]}

        Объединяем short+long+full, dedupe по name.
        Возвращает [{name, value, value_ids}] где value — comma-joined text.
        """
        out: list[dict] = []
        seen: set[str] = set()
        for raw in _FEATURES_STATE_RE.findall(html):
            try:
                data = json.loads(raw)
            except (ValueError, json.JSONDecodeError):
                decoded = (
                    raw.replace("&quot;", '"')
                    .replace("&#39;", "'")
                    .replace("&amp;", "&")
                )
                try:
                    data = json.loads(decoded)
                except (ValueError, json.JSONDecodeError):
                    continue
            for c in data.get("characteristics", []) or []:
                if not isinstance(c, dict):
                    continue
                for kind in ("short", "long", "full"):
                    items = c.get(kind) or []
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        name = item.get("name")
                        if not isinstance(name, str):
                            continue
                        name = name.strip()
                        if not name:
                            continue
                        name_low = name.lower()
                        if name_low in seen:
                            continue
                        values = item.get("values") or []
                        if not isinstance(values, list) or not values:
                            continue
                        texts = []
                        value_ids = []
                        for v in values:
                            if not isinstance(v, dict):
                                continue
                            t = v.get("text")
                            if isinstance(t, str) and t.strip():
                                texts.append(t.strip())
                            vid = v.get("id")
                            if isinstance(vid, (int, str)) and str(vid).strip():
                                value_ids.append(str(vid))
                        if not texts:
                            continue
                        seen.add(name_low)
                        out.append({
                            "name": name,
                            "value": ", ".join(texts),
                            "value_ids": value_ids,
                        })
        return out

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
    # Mapping: char name → target attribute_id → value_id
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
        """Сопоставить Ozon-char names с target.name через Ozon dictionary."""
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

        # Inverted: lowercase name → target.id
        name_to_target_id: dict[str, int] = {}
        for tid, names in target_names_low.items():
            for n in names:
                name_to_target_id.setdefault(n, tid)

        # Fuzzy fallback
        try:
            from rapidfuzz import process, fuzz
            all_target_names = list(name_to_target_id.keys())
        except ImportError:
            process = None
            fuzz = None
            all_target_names = []

        target_by_id: dict[int, TargetAttribute] = {t.id: t for t in targets}

        evidence_short = f"ozon:{title[:50]} | match={score:.1f}"
        conf = _CONF_EXACT if mode == "exact" else _CONF_BRAND_LINE

        results: list[AttributeValue] = []
        used_ids: set[int] = set()

        for c in chars:
            char_name = c["name"].strip()
            char_val = c["value"].strip()
            char_name_low = char_name.lower()

            # brand_line: STRICT WHITELIST — копируем ТОЛЬКО brand-/линейка-
            # уровневые атрибуты (бренд, цвет, страна, ТН ВЭД, назначение).
            # Spec attrs (мощность, размеры, кол-во разъёмов, гарантия,
            # сертификат 80 PLUS, подсветка) — модель-специфичны и брать
            # их с соседней модели = галлюцинация. Whitelist match —
            # case-insensitive substring (любой entry из whitelist должен
            # встречаться как substring в char_name_low). BLACKLIST остаётся
            # дополнительным фильтром (страховка от Артикул/MPN/EAN если они
            # случайно проходят whitelist substring match).
            if mode == "brand_line":
                if char_name_low in _BRAND_LINE_BLACKLIST:
                    continue
                if not any(
                    allowed in char_name_low for allowed in _BRAND_LINE_STRICT_WHITELIST
                ):
                    continue

            # 1) Exact lowercase match
            target_id = name_to_target_id.get(char_name_low)
            # 2) Substring match
            if target_id is None:
                for tn, tid in name_to_target_id.items():
                    if char_name_low in tn or tn in char_name_low:
                        target_id = tid
                        break
            # 3) Fuzzy fallback
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

            # value_id через ozon_loader
            value_id: Optional[int] = None
            if cat_id and type_id:
                try:
                    value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                except Exception as exc:
                    logger.debug("[OzonCard] resolve_value_id failed: %s", exc)

            results.append(AttributeValue(
                attribute_id=target.id,
                value=char_val,
                confidence=conf,
                source=Source.OZON_CARD,
                evidence=evidence_short,
                semantic_type=target.semantic_type,
                is_collection=target.is_collection,
                value_id=value_id,
            ))

        logger.info(
            "[OzonCard] %s mode → %d характеристик скопировано (из %d candidate chars, %d targets)",
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
