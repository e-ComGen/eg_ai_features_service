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

import asyncio
import json
import logging
import os
import re
import ssl
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

# Штраф за несовпадение типа товара в tile-заголовке.
# Применяется к тайлам ТОЛЬКО когда нет model-токенов (артикулов) в query —
# т.е. типично для одежды («Куртка The North Face Resolve») где тип = ключевой
# дискриминатор. Для электроники с артикулом (WH-1000XM5, RTX 4060) штраф
# НЕ срабатывает: артикул уже уникален, тип «Наушники» vs «Колонка» не нужен.
_TYPE_MISMATCH_PENALTY = 30.0

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

# Стоп-слова (русская грамматика): предлоги, союзы, гендерные прилагательные.
# Срезаются как ведущие токены ПОСЛЕ стрипа категории, если они не бренд
# и не модельный идентификатор. Это НЕ категорийный хардкод — чисто грамматика.
_LEADING_STOPWORDS: frozenset[str] = frozenset({
    # предлоги / союзы
    "для", "и", "с", "со", "в", "на", "к", "по", "из", "от", "под", "при",
    # гендерные/возрастные прилагательные (одиночные токены)
    "мужской", "мужская", "мужское", "мужские",
    "женский", "женская", "женское", "женские",
    "детский", "детская", "детское", "детские",
    "унисекс",
    # относительные прилагательные-описания (тип товара как adjective перед брендом)
    "городской", "городская", "городское",
    "спортивный", "спортивная", "спортивное",
    "модель",
    # разговорные/сленговые наименования типа товара (не бренды!)
    "худи",   # «Толстовка худи Champion» — «худи» = тип, не бренд
})

# Единицы измерения — суффиксы после числа в одном токене-спеке.
# Порядок важен: более длинные варианты первыми, чтобы regex жадно захватил max.
_UNIT_SUFFIXES_RE = re.compile(
    r"(?:"
    r"гб|тб|мб|gb|tb|mb"         # объём хранения (перед g/m/w чтобы не смешались)
    r"|квт|kw|вт"                 # мощность
    r"|мгц|ггц|mhz|ghz"           # частота
    r"|мм|mm|см|cm"               # длина
    r"|кг|kg"                     # масса
    r"|дб|db"                     # уровень шума
    r"|rpm"                       # обороты
    r"|дюйм(?:а|ов)?"             # дюймы
    r"|inch(?:es)?"               # дюймы en
    r"|мл|ml"                     # жидкость
    r"|нм|nm"                     # нанометры
    r"|ампер|amp"                 # ток
    r"|w"                         # ватт (одна буква — в конце чтобы не смешать с kw)
    r")$",
    re.IGNORECASE,
)

# Спек-токен: число + единица слитно, или «NxM GB», или чистое число, или NxG (5G/4G)
_SPEC_TOKEN_RE = re.compile(
    r"^\d+(?:[.,]\d+)?(?:"
    r"гб|тб|мб|gb|tb|mb"
    r"|квт|kw"
    r"|мгц|ггц|mhz|ghz"
    r"|мм|mm|см|cm"
    r"|кг|kg"
    r"|дб|db"
    r"|rpm"
    r"|дюйм(?:а|ов)?"
    r"|inch(?:es)?"
    r"|мл|ml"
    r"|нм|nm"
    r"|ампер|amp"
    r"|вт|w"           # ватт — в конце
    r")$"
    r"|^\d+[gG]$"              # 5G, 4G
    r"|^\d+/\d+[gGbB]+$"      # 8/256GB, 6/128GB
    r"|^\d+$",                 # чисто числовой (22, 27, 750)
    re.IGNORECASE,
)

# Модельный идентификатор: содержит хотя бы одну пару букв (не единица) + цифры.
# Примеры: TAT3A011, 27GP850-B, WGG2540MOE, HR2470, RMC-M90, EC685.M, DST7050/20, V2, A55.
# НЕ модельные: 5G, 850W, 100ml, 8/256GB (они ловятся _SPEC_TOKEN_RE первыми).
_MODEL_ID_RE = re.compile(r"(?:[A-Za-zА-Яа-яёЁ]\d|\d[A-Za-zА-Яа-яёЁ])")


def _is_spec_token(tok: str) -> bool:
    """True если токен — чистый спек (число+единица или чисто цифровой).

    Спеки: 27, 22, 5G, 8/256GB, 850W, 750W, 100ml, 45mm
    НЕ спеки (модельные ID): 27GP850-B, WGG2540MOE, TAT3A011, V2, A55, HR2470
    """
    # Spec-token regex проверяем ПЕРВЫМ — он точнее (число+единица не является моделью)
    if _SPEC_TOKEN_RE.match(tok):
        return True
    # Если есть смесь букв+цифр НЕ покрытая спек-паттерном — это модельный ID
    if _MODEL_ID_RE.search(tok):
        return False
    return False


# Standalone unit words (единицы измерения без числа) — тоже спек-хвосты.
# Используется для «LG UltraGear 27GP850-B 27 дюймов»: «дюймов» — standalone unit.
_UNIT_STANDALONE_RE = re.compile(
    r"^(?:"
    r"гб|тб|мб|gb|tb|mb"
    r"|квт|kw|вт|w"
    r"|мгц|ггц|mhz|ghz"
    r"|мм|mm|см|cm|м"
    r"|кг|kg|г"
    r"|дб|db"
    r"|rpm"
    r"|дюйм(?:а|ов|е)?"
    r"|дюймов|дюйма"
    r"|inch(?:es)?"
    r"|мл|ml|л"
    r"|нм|nm"
    r"|ампер|amp"
    r")$",
    re.IGNORECASE,
)


def _is_spec_or_unit_token(tok: str) -> bool:
    """True если токен — спек-число+единица, чистое число, или standalone единица."""
    if _is_spec_token(tok):
        return True
    # Standalone unit word (дюймов, мм, W, kg и т.п.)
    if _UNIT_STANDALONE_RE.match(tok):
        return True
    return False


def _count_meaningful_tokens(tokens: list[str]) -> int:
    """Считает «значимые» токены — не спеки и не стоп-слова.

    Значимые = бренд, модель, описательное слово (то, что несёт смысл запроса).
    Чистые числа/единицы и ведущие стоп-слова не считаются.
    """
    return sum(
        1 for t in tokens
        if not _is_spec_or_unit_token(t) and t.lower() not in _LEADING_STOPWORDS
    )


def _strip_trailing_specs(tokens: list[str]) -> list[str]:
    """Убирает хвостовые спек-токены из конца списка, сохраняя модельные ID.

    Примеры:
      ['LG', 'UltraGear', '27GP850-B', '27', 'дюймов']
        → срезаем 'дюймов' (unit), '27' (число) → ['LG', 'UltraGear', '27GP850-B']
      ['Samsung', 'Galaxy', 'A55', '5G', '8/256GB']
        → срезаем '8/256GB' (спек), '5G' (спек) → ['Samsung', 'Galaxy', 'A55']
      ['Bosch', 'WGG2540MOE'] → без изменений (WGG2540MOE — модельный ID)
      ['Penny', 'Board', '22'] → НЕ срезаем '22', т.к. после среза осталось бы
        только 2 значимых токена ('Penny', 'Board') — минимальный порог. Вернётся
        оригинал если бы было < 2 значимых; здесь ровно 2 → всё равно не срезаем.

    Защита от over-strip: если после потенциального среза остаётся ≤ 2 значимых
    токенов (не-спек, не-стоп) — стрип прекращается. Это сохраняет «Penny Board 22»
    как есть: «22» — числовой спек, но без него осталось бы ровно 2 значимых слова.
    Правило «≤ 2» означает: стрипуем только если значимых токенов останется ≥ 3.
    Это гарантирует бренд + линейка + хотя бы одно отличительное слово в запросе.

    Samsung Galaxy A55: meaningful=['Samsung','Galaxy','A55'] → 3 ≥ 3, стрипуем '5G'
      → meaningful=['Samsung','Galaxy','A55'] 3 ≥ 3, стрипуем '8/256GB' → OK.
    LG UltraGear 27GP850-B 27 дюймов: meaningful=['LG','UltraGear','27GP850-B'] → 3,
      стрипуем 'дюймов' → 3 ≥ 3, стрипуем '27' → 3 ≥ 3 → OK (27GP850-B — модельный ID).
    Penny Board 22: meaningful=['Penny','Board'] → 2 ≤ 2, НЕ стрипуем '22' → остаётся.
    """
    result = list(tokens)
    while result and _is_spec_or_unit_token(result[-1]):
        # Проверяем: сколько значимых токенов останется ПОСЛЕ среза этого хвоста?
        candidate = result[:-1]
        if _count_meaningful_tokens(candidate) <= 2:
            # Срезать нельзя — слишком мало значимых токенов (нужно ≥ 3: бренд + модель + что-то)
            break
        result.pop()
    return result or tokens  # не возвращаем пустой список


def _strip_category_prefix(text: str, category_name: Optional[str]) -> str:
    """Срезает ведущие слова категории из начала text (case-insensitive).

    Алгоритм:
      1. Попробовать совпадение с полной фразой категории (напр. «Микроволновая печь»).
      2. Если не совпало — итеративно срезать любые токены text, которые встречаются
         в категории (как набор значимых слов), пока они стоят в начале text.
         Это покрывает «Стиральная машина Bosch» при leaf «Стиральная машина»:
         оба слова «стиральная» и «машина» присутствуют в категории → срезаются оба.

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

    # Попытка 2: срезать ведущие токены text, которые входят в НАБОР слов категории.
    # «Стиральная машина Bosch» + leaf «Стиральная машина» →
    #   cat_word_set = {'стиральная', 'машина'}, tokens = ['стиральная', 'машина', 'bosch']
    #   'стиральная' ∈ set → срезать, 'машина' ∈ set → срезать, 'bosch' ∉ set → стоп.
    cat_word_set = {w for w in cat_low.split() if len(w) > 2}
    if cat_word_set:
        words = result.split()
        i = 0
        while i < len(words) and words[i].lower() in cat_word_set:
            i += 1
        if i > 0:
            return " ".join(words[i:]).strip()

    return result


def _strip_leading_stopwords(tokens: list[str]) -> list[str]:
    """Срезает ведущие стоп-слова (предлоги, гендерные adj) из списка токенов.

    Останавливается на первом токене, которого нет в _LEADING_STOPWORDS.
    Никогда не возвращает пустой список — если все токены стоп-слова, возвращает исходный.
    """
    i = 0
    while i < len(tokens) and tokens[i].lower() in _LEADING_STOPWORDS:
        i += 1
    return tokens[i:] if i < len(tokens) else tokens


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
      2. Strip ведущих стоп-слов (предлоги/гендерные adj) после стрипа категории.
      3. Strip хвостовых спек-токенов (число+единица: «27 дюймов», «8/256GB», «850W»).
         Модельные идентификаторы (27GP850-B, WGG2540MOE) сохраняются.
      4. Взять первые max_tokens токенов.
      5. Если brand задан, не пустой, не является стоп-словом и его нет в результате
         — prepend.

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

    # ---- Шаг 2: срезать ведущие стоп-слова ----
    tokens_after_cat = result.split()
    tokens_after_cat = _strip_leading_stopwords(tokens_after_cat)
    result = " ".join(tokens_after_cat)

    # ---- Шаг 3: срезать хвостовые спек-токены ----
    all_tokens = result.split()
    all_tokens = _strip_trailing_specs(all_tokens)

    # ---- Шаг 4: взять первые max_tokens токенов (бренд + модель) ----
    tokens = all_tokens[:max_tokens]
    compact = " ".join(tokens).strip()

    # ---- Шаг 5: если бренд не попал в начало — prepend ----
    # Не добавляем бренд если он:
    #  а) сам является стоп-словом (bad brand_guess от words[1]: «для», «мужская»)
    #  б) является словом из категории (brand_guess «машина» для «Стиральная машина»,
    #     «книга» для «Электронная книга», «гитара» для «Акустическая гитара» и т.п.)
    if brand and brand.strip():
        b = brand.strip()
        b_low = b.lower()
        is_stopword_brand = b_low in _LEADING_STOPWORDS
        # Проверяем: не является ли brand словом из категории
        cat_word_set: set[str] = set()
        if category_name:
            cat_word_set = {w.lower() for w in category_name.split() if len(w) > 2}
        is_cat_word_brand = b_low in cat_word_set
        if not is_stopword_brand and not is_cat_word_brand and b_low not in compact.lower():
            compact = f"{b} {compact}".strip()

    return compact or product_name.strip()


def _normalize_for_fuzzy(s: str) -> str:
    """Нормализация строки перед fuzzy-сравнением.

    Цель — убрать пунктуационный шум, чтобы точные совпадения модели не
    тонули из-за апострофов/дефисов/пробелов.  Не меняет смысловые цифры
    и буквы, только пунктуацию.

    Примеры:
        "De’Longhi EC685.M"  → "delonghi ec685.m"
        "Delonghi Dedica EC685.M" → "delonghi dedica ec685.m"
        "REDMOND RMC-M902S"  → "redmond rmc-m902s"  (дефис в артикуле сохраняется)
    """
    # Убираем апострофы всех видов (U+0027, U+2019, U+02BC)
    s = s.replace("’", "").replace("ʼ", "").replace("’", "")
    # lower
    return s.lower()


def _extract_model_tokens(s: str) -> set:
    """Извлечь токены-артикулы (латиница+цифры, длина ≥ 3) из строки.

    Используется для бонуса: если query и tile имеют общий артикул —
    это надёжный сигнал совпадения (EC685.M, WH-1000XM5, RMC-M90).
    Бонус применяется только к точным токен-пересечениям, поэтому
    соседние модели (M90 vs M902S, 4624 vs 4621) бонуса не получают.
    """
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*[0-9][A-Za-z0-9.\-]*", s)
    return {t.lower() for t in tokens if len(t) >= 3}


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
        _RETRY_DELAYS = (1.0, 2.0, 4.0)  # backoff seconds для попыток 1, 2, 3

        for attempt in range(_MAX_RETRIES + 1):
            content = await self._scrappey_fetch_once(client, target_url)
            if content is None:
                # Hard fail (network/SSL, HTTP4xx, DataDome) — retry с backoff
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                    logger.info(
                        "[OzonCard] retry #%d (hard fail, backoff %.0fs) %s",
                        attempt + 1, delay, target_url[:80],
                    )
                    await asyncio.sleep(delay)
                    continue
                return None
            if len(content) < min_len:
                # Soft fail — короткий HTML, бывает = пустая SPA. Retry.
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)]
                    logger.info(
                        "[OzonCard] retry #%d (short %d chars, backoff %.0fs) %s",
                        attempt + 1, len(content), delay, target_url[:80],
                    )
                    await asyncio.sleep(delay)
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
        except (
            ssl.SSLError,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.TransportError,
            httpx.TimeoutException,
            httpx.HTTPError,
        ) as exc:
            logger.info("[OzonCard] Scrappey network/SSL err (transient): %s", exc)
            # Возвращаем специальный sentinel чтобы _scrappey_fetch сделал retry
            # с backoff. None означает hard-fail (см. _scrappey_fetch).
            # Используем тот же None — caller уже retry-ует на None.
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

    async def _fetch_card_raw(
        self,
        context: ExtractionContext,
        client: httpx.AsyncClient,
    ) -> dict:
        """Внутренний helper: search → match → /features/ → сырые характеристики.

        Возвращает dict с полями:
          query, tiles_count, match_score, match_class, card_url,
          raw_chars (list[dict]), image_urls (list[str]),
          stage ("no_tiles"|"low_match"|"fetch_fail"|"parse_empty"|"ok")

        Не вызывает LLM. Используется и в _do_extract, и в probe.
        """
        full_name = context.product_name.strip()
        cat_leaf = context.category_path[-1] if context.category_path else None
        primary_query = _compress_search_query(full_name, context.brand, max_tokens=5, category_name=cat_leaf)
        fallback_query = _compress_search_query(full_name, context.brand, max_tokens=3, category_name=cat_leaf)

        queries_to_try: list[str] = [primary_query]
        if fallback_query and fallback_query != primary_query:
            queries_to_try.append(fallback_query)

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
                query, tiles = q, parsed
                break
            logger.info("[OzonCard] no tiles на query='%s' — пробую fallback", q[:60])

        used_query = query or primary_query

        if not tiles or query is None:
            logger.info("[OzonCard] no search tiles ни для primary ни для fallback")
            return {
                "query": used_query,
                "tiles_count": 0,
                "match_score": None,
                "match_class": "none",
                "card_url": None,
                "card_title": "",
                "raw_chars": [],
                "image_urls": [],
                "stage": "no_tiles",
            }

        top_tile, top_score = self._pick_best_match(query, tiles[:_MAX_SEARCH_TILES], category_leaf=cat_leaf)
        mode = self._classify_match(top_score) if top_tile is not None else "skip"

        if top_tile is None or mode == "skip":
            logger.info(
                "[OzonCard] best score=%.1f < %.0f — skip",
                top_score, _BRAND_LINE_THRESHOLD,
            )
            return {
                "query": used_query,
                "tiles_count": len(tiles),
                "match_score": top_score,
                "match_class": "none",
                "card_url": None,
                "card_title": (top_tile.get("title") or "") if top_tile else "",
                "raw_chars": [],
                "image_urls": [],
                "stage": "low_match",
            }

        title = (top_tile.get("title") or "").strip()
        slug = (top_tile.get("slug") or "").strip()
        pid = (top_tile.get("pid") or "").strip()

        card_url: Optional[str] = None
        if slug and pid:
            card_url = f"{_OZON_PRODUCT_BASE}{slug}-{pid}/"

        if not slug or not pid:
            logger.info("[OzonCard] no slug/pid in top tile")
            return {
                "query": used_query,
                "tiles_count": len(tiles),
                "match_score": top_score,
                "match_class": mode,
                "card_url": None,
                "card_title": title,
                "raw_chars": [],
                "image_urls": [],
                "stage": "fetch_fail",
            }

        logger.info(
            "[OzonCard] match=%s score=%.1f title='%s' pid=%s",
            mode, top_score, title[:80], pid,
        )

        # ---- FEATURES (HTML SSR) ----
        features_url = f"{_OZON_PRODUCT_BASE}{slug}-{pid}/features/"
        features_html = await self._scrappey_fetch(client, features_url)
        if features_html is None:
            return {
                "query": used_query,
                "tiles_count": len(tiles),
                "match_score": top_score,
                "match_class": mode,
                "card_url": card_url,
                "card_title": title,
                "raw_chars": [],
                "image_urls": [],
                "stage": "fetch_fail",
            }

        chars = self._parse_characteristics_html(features_html)
        image_urls = self._extract_image_urls(features_html)

        if not chars:
            logger.info("[OzonCard] no characteristics в /features/ for pid=%s", pid)
            return {
                "query": used_query,
                "tiles_count": len(tiles),
                "match_score": top_score,
                "match_class": mode,
                "card_url": card_url,
                "card_title": title,
                "raw_chars": [],
                "image_urls": image_urls,
                "stage": "parse_empty",
            }

        return {
            "query": used_query,
            "tiles_count": len(tiles),
            "match_score": top_score,
            "match_class": mode,
            "card_url": card_url,
            "card_title": title,
            "raw_chars": chars,
            "image_urls": image_urls,
            "stage": "ok",
        }

    async def probe(self, context: ExtractionContext) -> dict:
        """Диагностика матчинга карточки БЕЗ LLM-экстракции.

        Выполняет: search → match → /features/ → parse chars.
        Не вызывает LLM. Возвращает диагностический dict:
          {
            "query": str,
            "tiles_count": int,
            "match_score": float|None,
            "match_class": "exact"|"brand_line"|"none",
            "card_url": str|None,
            "raw_chars": int,
            "image_urls": int,
            "stage": "no_tiles"|"low_match"|"fetch_fail"|"parse_empty"|"ok"|"error",
            "found": bool,
          }
        """
        try:
            async with httpx.AsyncClient(
                timeout=_HTTP_TIMEOUT,
                follow_redirects=True,
            ) as client:
                raw = await self._fetch_card_raw(context, client)
        except Exception as exc:
            return {
                "query": "",
                "tiles_count": 0,
                "match_score": None,
                "match_class": "none",
                "card_url": None,
                "raw_chars": 0,
                "image_urls": 0,
                "stage": "error",
                "found": False,
                "error": str(exc),
            }

        return {
            "query": raw["query"],
            "tiles_count": raw["tiles_count"],
            "match_score": raw["match_score"],
            "match_class": raw["match_class"],
            "card_url": raw["card_url"],
            "best_tile_title": raw.get("card_title") or "",
            "raw_chars": len(raw["raw_chars"]),
            "image_urls": len(raw["image_urls"]),
            "stage": raw["stage"],
            "found": raw["stage"] == "ok" and len(raw["raw_chars"]) > 0,
        }

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
            raw = await self._fetch_card_raw(context, client)

            if raw["stage"] != "ok":
                return []

            chars = raw["raw_chars"]
            top_score = raw["match_score"] or 0.0
            mode = raw["match_class"]
            title = raw.get("card_title") or ""

            # ---- IMAGES (для downstream VisionSource) ----
            # Mutating context.image_urls — pipeline передаёт context по ссылке
            # между stages, поэтому Stage 3 (VisionSource) увидит эти фотки
            # на товарах где OzonCard нашёл tile. Vision дополнит «визуальные»
            # attrs (цвет, RGB-подсветка, форм-фактор) которые сложно достать
            # из текста характеристик.
            new_image_urls = raw["image_urls"]
            if new_image_urls:
                existing = set(context.image_urls or [])
                added = [u for u in new_image_urls if u not in existing]
                if added:
                    context.image_urls = list(context.image_urls or []) + added
                    logger.info(
                        "[OzonCard] +%d image URLs для VisionSource",
                        len(added),
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
        category_leaf: Optional[str] = None,
    ) -> tuple[Optional[dict], float]:
        """Top-1 по rapidfuzz (partial_ratio + token_sort_ratio averaged).

        Бонус +5 за общий артикул (model token) между query и tile —
        чтобы точные совпадения модели (EC685.M, WH-1000XM5 и т.п.)
        не тонули из-за описательных слов в tile-title.
        Соседние модели (M90 vs M902S, 4624 vs 4621) бонуса не получают.

        Штраф _TYPE_MISMATCH_PENALTY за несовпадение типа товара:
        если category_leaf задан И в query нет model-токенов (одежда без
        артикула) И category_leaf отсутствует в tile-title → score -= 30.
        Для электроники с артикулом штраф не применяется.
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return (tiles[0], 100.0) if tiles else (None, 0.0)

        _MODEL_BONUS = 5.0
        q_models = _extract_model_tokens(query)
        best_tile: Optional[dict] = None
        best_score = 0.0
        q = _normalize_for_fuzzy(query)
        cat_leaf_low = category_leaf.strip().lower() if category_leaf else None
        for tile in tiles:
            title = (tile.get("title") or "").strip()
            if not title:
                continue
            t = _normalize_for_fuzzy(title)
            score = (fuzz.partial_ratio(q, t) + fuzz.token_sort_ratio(q, t)) / 2.0
            # Бонус: есть хотя бы один общий артикул-токен → точное совпадение модели
            if q_models and q_models & _extract_model_tokens(title):
                score += _MODEL_BONUS
            # Штраф за тип товара: только для товаров без артикула (одежда)
            # и когда тип (category_leaf) отсутствует в заголовке тайла.
            # Пример: query «Куртка The North Face» + tile «Шорты The North Face» →
            #   cat_leaf_low «куртка» не входит в «шорты the north face» → штраф.
            # Для электроники с артикулом (q_models непустое) штраф не срабатывает.
            if cat_leaf_low and not q_models and cat_leaf_low not in title.lower():
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
