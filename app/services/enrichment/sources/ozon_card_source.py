"""OzonCardSource — копирует характеристики из живой Ozon-карточки похожего товара.

Использует ПУБЛИЧНЫЕ HTML-страницы Ozon через scrape.do proxy (TRUE PAYG,
~$0.0002-0.001/page). Composer-api.bx заблокирован DataDome даже через
премиум-прокси; HTML-страницы Ozon содержат полный widget state в SSR.

Алгоритм:
  1. Skip-guard: если already_filled покрывает ≥80% targets → return [].
  2. Search step: GET https://www.ozon.ru/search/?text=<name> через scrape.do
     → regex по `<a href="/product/<slug>-<pid>/">` → list of tiles.
  3. Match step: rapidfuzz (partial_ratio + token_sort_ratio) vs product_name:
     - ≥85 → "exact": full card, copy ALL.
     - 70-84 → "brand_line": full card, только safe attrs.
     - <70 → skip.
  4. Detail step: GET https://www.ozon.ru/product/<slug>/features/ через scrape.do
     → regex `<div id="state-webCharacteristics-..." data-state='<JSON>'>` (или data-state="<HTML-escaped JSON>")`
     → распарсить characteristics[].short / .long / .full → [{name, value}].
  5. Mapping по русским именам через get_ozon_characteristics_for_type
     (lowercase + substring + fuzzy WRatio≥88).
  6. resolve_value_id для (attr_id, value) → словарный value_id Ozon.
  7. Confidence: 0.93 (exact), 0.85 (brand_line). Source: OZON_CARD.
  8. Evidence: f"ozon:{title[:50]} | match={score}".
  9. In-process LRU cache по (brand, normalized_model), max 256.

Cost: 1 scrape.do-запрос / шаг (search + features) — 2 запроса/товар.

Anti-block:
  - SCRAPEDO_TOKEN читаем из env (см. app/services/providers/scrapedo_client.py),
    можно передать backward-compat kwarg scrappey_key в __init__ (игнорируется —
    он больше не источник токена).
  - DataDome detection: incidentId в первых 1500 символах → []
  - HTTP 4xx / короткое тело / DataDome → scrape.do сам вернёт success=False → []
  - Таймаут: до 120s на попытку (scrape.do render+residential-proxy),
    до 3 внутренних ретраев на транзиентных 429/5xx.
"""
from __future__ import annotations

import asyncio
import html as _html
import json
import logging
import os
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
from app.services.providers.scrapedo_client import scrapedo_fetch
from app.services.providers.deepseek_provider import DeepSeekProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

_OZON_SEARCH_URL = "https://www.ozon.ru/search/"
_OZON_PRODUCT_BASE = "https://www.ozon.ru/product/"
_MAX_SEARCH_TILES = 8
# Score the top-N parsed SSR tiles and pick the HIGHEST-scoring one (best-of-N),
# instead of relying on the single top tile. Ozon's SSR tile ORDERING jitters
# run-to-run: the exact card is present every run but not always ranked #1.
# Scoring N tiles and taking the best kills that ordering dependence — the exact
# card (which scores ~91 whenever present) wins regardless of its rank. N=8 is a
# small buffer above the typical jitter window (the exact card has been observed
# ranking up to ~6th) while staying cheap (scoring is pure-CPU rapidfuzz, no I/O).
_MATCH_TOP_N = 8

# Hard cap on the entire _do_extract (scrape.do path). Scrape.do render+residential-proxy is
# slower per call than the old (retired) proxy (~15-60s typical, up to 120s worst-case per
# scrapedo_fetch call incl. its own internal retries) and _do_extract can issue up to ~2-3
# sequential scrapedo_fetch calls (search + retry-search + features, or 1 call when Serper-first
# finds the card directly). 200s gives a realistic budget for that; on timeout, falls through to
# the Serper-snippet fallback (unchanged).
_OZON_CARD_TOTAL_TIMEOUT = 200.0


# Regex для парсинга
_PRODUCT_LINK_RE = re.compile(
    r'<a[^>]+href="(/product/([a-z0-9\-]+)-(\d+)/)[^"]*"[^>]*>(.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)
_FEATURES_STATE_RE = re.compile(
    r'<div\s+id="state-webCharacteristics-[^"]+"\s+data-state='
    r'(?:\'([^\']+)\'|"([^"]+)")',
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
# Пониженный confidence для поля «Пол», навеянного гендером brand_line-карточки
# при нейтральном имени товара (гендер-страховка). Ниже порога, чтобы
# merger/judge не предпочли его более надёжным источникам (vision/llm).
_CONF_GENDER_DOWNWEIGHT = 0.40

_MULTIVALUE_SPLIT_RE = re.compile(r"[;,]")


def _split_multivalue(raw: str) -> list[str]:
    """Сплит карточной multi-value строки в дедуплицированный список.

    Разделители ";" и ",". Чистит пробелы, регистронезависимый дедуп. WB/Ozon
    отдают коллекционные характеристики одной строкой (", ".join(...)) — сплит
    нужен чтобы значение участвовало в union merge поэлементно.
    """
    parts = _MULTIVALUE_SPLIT_RE.split(raw)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        low = p.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(p)
    return out or [raw.strip()]

# Serper-snippet fallback confidence.
# Срабатывает ТОЛЬКО когда scrape.do-путь вернул 0 характеристик (карточка не найдена
# или отдал пустоту). Сниппет частичный (3-6 полей из карточки), поэтому conf
# ниже brand_line, НО ≥ 0.80 — достаточно, чтобы fill попал в filled_so_far и
# merger его засчитал, если более уверенный источник не нашёл значение.
_CONF_SNIPPET = 0.80

# Сколько organic-результатов запрашивать у Serper для fallback.
_SERPER_NUM_RESULTS = 5
_SERPER_TIMEOUT = 15

# Regex для извлечения pid из ссылки вида /product/<slug>-<pid>/ или
# product/...-123456789 в любом тексте (link или snippet).
_SERPER_PID_RE = re.compile(r"/product/[a-z0-9\-]*?-(\d{5,})/?", re.IGNORECASE)
# Slug + pid из URL карточки: /product/<slug>-<pid>/ → (slug, pid).
_SERPER_SLUG_PID_RE = re.compile(r"/product/([a-z0-9\-]+?)-(\d{5,})/?", re.IGNORECASE)
# fallback: «Артикул: 1240096510» в сниппете.
_SERPER_ARTICUL_RE = re.compile(r"(?:артикул|sku)\D{0,3}(\d{5,})", re.IGNORECASE)

# Serper-assisted card-finding: когда внутренний поиск Ozon флачит (no_tiles/low_match),
# найти точный URL товара через Google (он индексирует ozon.ru надёжнее, чем держится
# их антибот-search), затем scrape.do тащит ТОЛЬКО /features/ найденного URL. Цель —
# поднять 66%-потолок за счёт search/match-промахов (не антибот). Гард: Serper-карточка
# проходит ТОТ ЖЕ match-скоринг, что и Ozon-tile → чужой бренд/тип отсекается.
# Дефолт ON (только ДОБАВЛЯет fallback при провале основного пути; гард не пускает мусор).
_OZON_SERPER_CARD_FINDING = os.getenv(
    "OZON_SERPER_CARD_FINDING", "1"
).strip().lower() not in ("0", "false", "no", "off", "")

# Serper-FIRST: дёргать Serper-card-finding ПЕРВЫМ, до внутреннего поиска Ozon.
# Внутренний поиск Ozon — самая флаковая/троттлимая часть (2 scrape.do-вызова с ретраями
# на блокируемой search-странице). Если Google и так надёжно находит URL — идём сразу
# на /features/ (1 фетч вместо search+features) → ВДВОЕ меньше запросов к Ozon → меньше
# троттла, быстрее, дешевле. Ozon-поиск остаётся фоллбэком (когда Google не индексирует
# товар / гард отверг). Гард по бренду тот же. Дефолт ON. Требует _OZON_SERPER_CARD_FINDING.
_OZON_SERPER_FIRST = os.getenv(
    "OZON_SERPER_FIRST", "1"
).strip().lower() not in ("0", "false", "no", "off", "")

# FIX-16: LLM-верификатор идентичности товара — финальный гейт после _pick_best_match.
# DIFFERENT -> abstain (карта возвращает пусто), UNKNOWN -> fail-safe (см. _do_extract).
# Дефолт ON. Рубильник на случай недоступности/стоимости DeepSeek.
_OZON_CARD_LLM_IDENTITY_ENABLED: bool = os.getenv(
    "OZON_CARD_LLM_IDENTITY_ENABLED", "1"
).strip().lower() not in ("0", "false", "no", "off", "")

# "deepseek-chat" — рабочий production-алиас DeepSeek V4-flash в этом движке
# (см. deepseek_provider.py: прямые имена v4-flash/v4-pro сейчас 200+пустой content).
_OZON_CARD_LLM_IDENTITY_MODEL: str = (
    os.getenv("OZON_CARD_LLM_IDENTITY_MODEL", "deepseek-chat").strip()
    or "deepseek-chat"
)

_OZON_CARD_LLM_IDENTITY_TIMEOUT: float = float(
    os.getenv("OZON_CARD_LLM_IDENTITY_TIMEOUT", "20.0").strip() or "20.0"
)

# Similarity thresholds (понижены для (V2/V3/Plus/Bronze) вариаций — title часто
# содержит "Блок питания + brand + model + V3 80 Plus Gold (MPE-XXX-...)", т.е.
# много шума вокруг query "brand + model").
_EXACT_THRESHOLD = 78.0
_BRAND_LINE_THRESHOLD = 65.0  # v17 had 75.0 — слишком жёстко, убил 73→0 OzonCard fills.
                              # 65 — компромисс: brand_line whitelist всё равно ограничивает
                              # копирование model-specific attrs.

# (Retry-on-short-HTML logic removed: scrape.do/scrapedo_fetch already enforces
# a minimum body length and retries transient failures internally.)

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

    # ---- Шаг 2.5: восстановить тип категории если осталось < 3 значимых токенов ----
    # «Платье женское befree летнее» → после среза категории «Платье» и стоп-слова
    # «женское» → «befree летнее» (2 токена, нет типа товара). Ozon-заголовки начинаются
    # с типа («Платье befree…»), поэтому без него fuzzy-score падает ниже порога.
    # Препендируем ПЕРВОЕ слово category_name обратно — generic, без хардкода категорий.
    if category_name and _count_meaningful_tokens(tokens_after_cat) < 3:
        first_cat_word = category_name.strip().split()[0]
        if first_cat_word.lower() not in result.lower():
            result = f"{first_cat_word} {result}".strip()

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


# Стоп-слова-классификаторы: общая русская грамматика, НЕ категорийный хардкод.
# Убираем первое слово-классификатор, если оно одно из этих — оно несёт
# нулевую семантическую нагрузку относительно самого атрибута.
_CHAR_NAME_STOPWORDS = frozenset({
    "тип", "вид", "количество", "число", "наличие",
    "способ", "метод", "класс", "степень",
})

import re as _re
_SPLIT_RE = _re.compile(r"[\s\-–—/]+")


def _norm_char_name(s: str) -> str:
    """Нормализация имени характеристики для сопоставления.

    Алгоритм:
    1. lower
    2. split по пробелам и разделителям [-–—/]
    3. удалить стоп-слова-классификаторы из НАЧАЛА токен-цепочки
       (убираем только пока идут стоп-слова подряд, не трогаем середину)
    4. склеить токены пробелом

    Примеры:
        «Тип интерфейса USB»  → «интерфейс usb»
        «Количество SIM-карт» → «sim карт»
        «USB-интерфейс»        → «usb интерфейс»
        «Комплектация»         → «комплектация»
        «Что в комплекте»      → «что в комплекте»  (нет стоп-слова в начале)
        «В комплекте»          → «в комплекте»
        «Наличие Bluetooth»    → «bluetooth»
    """
    tokens = _SPLIT_RE.split(s.lower().strip())
    # удаляем ведущие стоп-слова
    while tokens and tokens[0] in _CHAR_NAME_STOPWORDS:
        tokens = tokens[1:]
    return " ".join(tokens) if tokens else s.lower()


def _extract_model_tokens(s: str) -> set:
    """Извлечь токены-артикулы (латиница+цифры, длина ≥ 3) из строки.

    Используется для бонуса: если query и tile имеют общий артикул —
    это надёжный сигнал совпадения (EC685.M, WH-1000XM5, RMC-M90).
    Бонус применяется только к точным токен-пересечениям, поэтому
    соседние модели (M90 vs M902S, 4624 vs 4621) бонуса не получают.
    """
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*[0-9][A-Za-z0-9.\-]*", s)
    return {t.lower() for t in tokens if len(t) >= 3}


def _common_prefix_len(a: str, b: str) -> int:
    """Возвращает длину общего префикса двух строк."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# RU-словоизменительные суффиксы: короткий корень категории («сок», «нож») отличается
# от словоформы в заголовке («соки», «ножи») именно такой флексией. Матчим 3-символьные
# корни, НЕ ловя ложные пары («сок↔сокол» — «ол» не флексия; «сок↔соска» — не префикс).
_RU_INFLECTIONS = frozenset({"и", "ы", "а", "я", "е", "у", "ю", "ов", "ев", "ей",
                             "ам", "ям", "ах", "ях", "ка", "ки"})


def _category_present_in_title(cat_leaf_low: str, title_low: str) -> bool:
    r"""FIX-12: stem/prefix-aware проверка присутствия категории в заголовке карточки.

    Таксономия CS-Cart (category_leaf) часто в другой словоформе/числе, чем заголовок
    карточки Ozon («Смартфоны» vs «Смартфон POCO...», «Дрели ударные» vs «Дрель ударная...»).
    Буквальная подстрока это не ловит; вместо этого сравниваем общий префикс токенов.

    Токенизация по [\s\-–—/]+, токены длиной >= 3 символа. Матч двумя правилами:
    (A) общий префикс >= 4 симв. (смартфоны↔смартфон, дрели↔дрель, куртки↔куртка);
    (B) короткий токен — полный префикс длинного, остаток <= 2 симв. и он ∈ RU-флексий
    (сок↔соки, нож↔ножи) — чинит 3-символьные корни, которые старое правило теряло.
    Чистая функция: без I/O, без внешних зависимостей. Пустые строки → False.
    """
    if not cat_leaf_low or not title_low:
        return False
    cat_tokens = [t for t in re.split(r"[\s\-–—/]+", cat_leaf_low.lower()) if len(t) >= 3]
    title_tokens = [t for t in re.split(r"[\s\-–—/]+", title_low.lower()) if len(t) >= 3]
    for ct in cat_tokens:
        for tt in title_tokens:
            if _common_prefix_len(ct, tt) >= 4:
                return True
            short, long = (ct, tt) if len(ct) <= len(tt) else (tt, ct)
            if (long.startswith(short) and 0 < len(long) - len(short) <= 2
                    and long[len(short):] in _RU_INFLECTIONS):
                return True
    return False


# Чисто-алфавитный токен (латиница ИЛИ кириллица), без цифр, длиной ≥ 4.
# Словесные модели не имеют цифр: Resolve, Ultraboost, Sauvage, Triclimate.
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яёЁ]{4,}")


def _extract_alpha_model_tokens(s: str) -> set:
    """Извлечь АЛФАВИТНЫЕ модель-токены (словесные модели без цифр) из строки.

    Дополняет _extract_model_tokens (тот ловит только токены с цифрой). Здесь —
    содержательные словесные модели: «Resolve», «Ultraboost», «Sauvage»,
    «Triclimate». Цель — дать бонус при совпадении такого слова между запросом
    и заголовком карточки, когда у модели нет цифрового артикула.

    Консервативная фильтрация (переиспользует существующие стоп-листы), чтобы
    НЕ ловить бренд/тип товара/служебные слова:
      - длина ≥ 4 символа (короткие шумные слова отсекаются);
      - не ведущий стоп-слово/гендер/предлог (_LEADING_STOPWORDS);
      - не классификатор-имя характеристики (_CHAR_NAME_STOPWORDS);
      - не спек/единица (_is_spec_or_unit_token — на всякий случай).

    Бренд НЕ отсеиваем явным списком (его тут нет), но это безопасно: бонус
    применяется только к ПЕРЕСЕЧЕНИЮ токенов запроса и заголовка, и он меньше
    цифрового (см. _MODEL_BONUS_ALPHA < _MODEL_BONUS), поэтому надёжный
    цифровой артикул всегда перевешивает.
    """
    out: set = set()
    for tok in _ALPHA_TOKEN_RE.findall(s):
        low = tok.lower()
        if low in _LEADING_STOPWORDS:
            continue
        if low in _CHAR_NAME_STOPWORDS:
            continue
        if _is_spec_or_unit_token(low):
            continue
        out.add(low)
    return out


# ---------------------------------------------------------------------------
# Гендер-сигнал (генеральный, без хардкода брендов/категорий)
# ---------------------------------------------------------------------------
# Стем-паттерны гендера: одна основа покрывает все словоформы без лемматизатора
# (мужск-ой/ая/ие/ой пол → "male"; женск-ий/ая/ое + женщин → "female";
# мальчик/парн → male; девочк/девуш → female). Унисекс — нейтральный сигнал,
# совместим с любым, поэтому в конфликт НЕ вступает.
_GENDER_PATTERNS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("unisex", re.compile(r"унисекс|unisex", re.IGNORECASE)),
    ("male", re.compile(
        r"мужск|мужчин|мальчик|парн(?:ой|ям|и|ь)|\bmale\b|\bmen('?s)?\b|\bman\b|\bboy",
        re.IGNORECASE,
    )),
    ("female", re.compile(
        r"женск|женщин|девочк|девуш|\bfemale\b|\bwomen('?s)?\b|\bwoman\b|\bgirl",
        re.IGNORECASE,
    )),
)


def _extract_gender_signal(text: str) -> Optional[str]:
    """Извлечь гендер-сигнал из произвольного текста (name / subj_name карточки).

    Возвращает 'male' / 'female' / 'unisex' / None. Генеральный, без хардкода
    конкретных брендов/категорий — только грамматические стемы пола, покрывающие
    все словоформы (мужск-* / женск-* / мальчик / девочк / men / women / ...).

    Логика:
      - если найдено явное «унисекс/unisex» → 'unisex' (нейтрально, ни с чем не
        конфликтует);
      - если найден ровно один из male/female → он;
      - если найдены ОБА (mixed-листинг «мужские и женские») → None (неоднозначно,
        в конфликт не вступаем — нельзя утверждать гендер);
      - ничего не найдено → None (нейтральное имя, напр. «Ultraboost 22»).
    """
    if not text:
        return None
    found: set[str] = set()
    for label, pat in _GENDER_PATTERNS:
        if pat.search(text):
            found.add(label)
    if "unisex" in found:
        return "unisex"
    if found == {"male"}:
        return "male"
    if found == {"female"}:
        return "female"
    # ноль сигналов ИЛИ оба пола сразу (mixed) → не утверждаем гендер
    return None


def _gender_conflict(name_text: str, card_text: str) -> bool:
    """True если ОБА текста несут гендер-сигнал и они ПРОТИВОРЕЧАТ.

    Конфликт = {male vs female}. Если у одного из текстов сигнала нет (None) или
    он 'unisex' — конфликта НЕТ (нельзя утверждать, что карточка неверного пола).
    Используется как страховочный сигнал в матче карточки и при даунвейте поля
    «Пол» (gender-target), пришедшего из brand_line-карточки.
    """
    g_name = _extract_gender_signal(name_text)
    g_card = _extract_gender_signal(card_text)
    if g_name is None or g_card is None:
        return False
    if g_name == "unisex" or g_card == "unisex":
        return False
    return g_name != g_card


def _is_gender_target_name(name: str) -> bool:
    """True если имя таргета — поле «Пол» (gender). Без хардкода attribute_id.

    Определяем по имени: содержит «пол» как отдельное слово / «gender» / «род».
    Гард: «пол» матчим как целое слово (через границы), чтобы не задеть
    «полнота», «наполнитель», «потолок» и пр., где «пол» — лишь подстрока.
    """
    low = name.lower()
    if "gender" in low:
        return True
    # «пол» / «род» как самостоятельное слово (а не часть «наПОЛнитель», «ПОЛнота»)
    return bool(re.search(r"(?<![а-яёa-z])(пол|род)(?![а-яёa-z])", low))


# ---------------------------------------------------------------------------
# Бренд-консистентность (генеральный wrong-SKU гард, без хардкода брендов)
# ---------------------------------------------------------------------------
# Карточка соседнего товара иногда оказывается ДРУГИМ брендом (Ozon search вернул
# «Футболка Shilla» на запрос «Футболка Nike» — fuzzy-score высокий по типу+полу,
# но это не тот товар). Копировать её attrs = перетереть верные значения чужими.
# Гард зеркалит стиль гендер-гарда: извлекаем бренд запроса и сравниваем с брендом
# карточки; явный конфликт → штраф (карточка проигрывает / уходит ниже порога).

# Токенайзер бренда (зеркало pipeline._brand_norm_tokens / matcher._TOKEN_RE —
# не импортируем, чтобы не тянуть лишние модули). ё→е, lower, latin/cyrillic/digits.
_BRAND_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
# Минимальная длина склеенного бренд-кандидата (отсекает 1-2-буквенные ложняки).
_BRAND_MIN_LEN = 3
# Штраф за конфликт бренда: как type/gender-mismatch, но крупнее — карточка
# другого бренда должна гарантированно уйти ниже _BRAND_LINE_THRESHOLD (65),
# чтобы _classify_match вернул "skip" и значения вообще не копировались.
_BRAND_MISMATCH_PENALTY = 60.0
# Латинский бренд-кандидат: непрерывная цепочка латиница/цифры/амперсанд длиной ≥3.
# Бренды одежды/электроники почти всегда латиница (Nike, Shilla, Levi's, ASUS).
# Кириллические токены НЕ берём в кандидаты бренда: тип товара и описательные
# слова в RU-заголовках кириллические («Футболка», «мужская») — чтобы не принять
# тип за бренд. Это делает гард консервативным (срабатывает на латин-vs-латин).
_LATIN_BRAND_RE = re.compile(r"[a-z][a-z0-9&]{2,}", re.IGNORECASE)


def _brand_tokens(text: str) -> list[str]:
    """Токены строки для brand-матчинга (зеркало pipeline._brand_norm_tokens)."""
    return _BRAND_TOKEN_RE.findall((text or "").lower().replace("ё", "е"))


def _brand_present(brand: str, text_tokens: list[str]) -> bool:
    """True если бренд присутствует в тексте как непрерывная цепочка токенов.

    Многословные бренды («The North Face», «Calvin Klein») матчатся как
    contiguous-подпоследовательность. Бренд короче _BRAND_MIN_LEN символов
    (после склейки токенов) игнорируется. Зеркало pipeline._brand_in_name.
    """
    b_tokens = _brand_tokens(brand)
    if not b_tokens:
        return False
    if sum(len(t) for t in b_tokens) < _BRAND_MIN_LEN:
        return False
    n = len(b_tokens)
    for i in range(len(text_tokens) - n + 1):
        if text_tokens[i:i + n] == b_tokens:
            return True
    return False


def _latin_brand_candidates(text: str) -> set[str]:
    """Множество латинских бренд-кандидатов из текста (lower, ≥3 символов).

    Только латиница: в RU-заголовках Ozon бренд почти всегда латиницей
    («Футболка мужская Nike Sportswear»), а кириллица — тип/описание. Это
    делает гард консервативным — он сравнивает латин-бренды, не принимая
    кириллический тип товара за бренд.
    """
    return {m.group(0).lower() for m in _LATIN_BRAND_RE.finditer(text or "")}


def _brand_conflict(query_brand: Optional[str], query_name: str, card_title: str) -> bool:
    """True если бренд карточки ЯВНО конфликтует с брендом запроса (wrong-SKU).

    Консервативный гард (минимум ложняков):
      1. Определяем бренд запроса: explicit `query_brand` (из контекста) если он
         латинский ≥3 символов; иначе НЕ определяем → конфликта нет (не угадываем
         бренд из имени, чтобы не словить ложняк на типе/описании).
      2. Если бренд запроса присутствует в title карточки (как цепочка токенов) →
         НЕТ конфликта (это тот самый бренд, возможно другая модель — ОК).
      3. Если бренд запроса в title ОТСУТСТВУЕТ, НО в title есть другой латинский
         бренд-кандидат (≥3 символов), которого нет в самом имени запроса →
         КОНФЛИКТ (карточка другого бренда).
      4. Если у карточки вообще нет латинских бренд-кандидатов → НЕ конфликт
         (бренд карточки неопределим — не отвергаем, избегаем ложного reject).

    Симметрично безопасно: если бренд запроса неопределим (нет explicit brand или
    он кириллический/короткий) → всегда False (никогда не reject вслепую).
    """
    if not card_title:
        return False
    qb = (query_brand or "").strip()
    # Бренд запроса должен быть латинским и достаточно длинным, иначе не судим.
    if not qb or not _LATIN_BRAND_RE.fullmatch(qb.replace(" ", "")):
        return False
    if sum(len(t) for t in _brand_tokens(qb)) < _BRAND_MIN_LEN:
        return False

    title_tokens = _brand_tokens(card_title)
    if _brand_present(qb, title_tokens):
        return False  # тот же бренд (другая модель) — ОК

    # Бренд запроса в карточке отсутствует. Есть ли в карточке ДРУГОЙ латин-бренд?
    card_brands = _latin_brand_candidates(card_title)
    if not card_brands:
        return False  # бренд карточки неопределим — не отвергаем

    qb_tokens = set(_brand_tokens(qb))
    name_brands = _latin_brand_candidates(query_name)
    # Конфликтующие кандидаты = латин-токены карточки, которых нет ни в бренде
    # запроса, ни (как латин-токен) в самом имени запроса.
    conflicting = {
        cb for cb in card_brands
        if cb not in qb_tokens and cb not in name_brands
    }
    return bool(conflicting)


# ---------------------------------------------------------------------------
# Model-conflict guard (FIX-15, sibling of _brand_conflict) -- tier-0 authored
# ---------------------------------------------------------------------------

_VARIANT_MODIFIERS = {"pro", "max", "plus", "ultra", "lite", "mini", "neo", "note", "air", "se", "fe", "prime"}
_MODEL_CONFLICT_PENALTY = 60.0

def _conflict_codes(s: str) -> set[str]:
    """Извлекает токены, содержащие одновременно буквы и цифры — потенциальные коды моделей."""
    tokens = re.findall(r'[A-Za-z0-9]+', (s or "").lower())
    return {t for t in tokens if any(c.isalpha() for c in t) and any(c.isdigit() for c in t)}

def _variant_mods(s: str) -> set[str]:
    """Возвращает пересечение токенов строки с известными модификаторами моделей (pro, max и т.д.)."""
    tokens = set(re.findall(r'[A-Za-z0-9]+', (s or "").lower()))
    return tokens & _VARIANT_MODIFIERS

def _model_conflict(query: str, title: str) -> bool:
    """Проверяет конфликт моделей: разные коды или одинаковый код с разными модификаторами."""
    qc = _conflict_codes(query)
    tc = _conflict_codes(title)
    if qc and tc and (qc - tc) and (tc - qc):
        return True
    if (qc & tc) and _variant_mods(query) != _variant_mods(title):
        return True
    return False


# ---------------------------------------------------------------------------
# LLM identity verifier (FIX-16, sibling of the deterministic guards above)
# ---------------------------------------------------------------------------

_IDENTITY_PROMPT_TEMPLATE: str = (
    "Сверь идентичность товара. Запрос пользователя: «{query}». Заголовок карточки маркетплейса: «{title}».\n"
    "Это ОДИН И ТОТ ЖЕ товар (та же модель/поколение/линейка) или РАЗНЫЕ товары?\n"
    "ПО УМОЛЧАНИЮ — РАЗНЫЕ; отвечай \"same\" только если уверен, что это одна модель.\n"
    "ИГНОРИРУЙ косметику: объём памяти (256/512 ГБ), цвет, регион, год в названии, слово «Смартфон/Дрель» —\n"
    "это ТОТ ЖЕ товар. РАЗНЫЕ = другая модель (X6 vs M8), другое ПОКОЛЕНИЕ (iPhone 15 vs 14),\n"
    "вариант линейки (X6 vs X6 Pro/Max/Ultra), или АКСЕССУАР vs само устройство (чехол/плёнка/зарядка).\n"
    "Ответь СТРОГО JSON: {{\"verdict\":\"same|different\",\"distinguishing\":\"<конкретный различающий признак, или пусто>\"}}"
)

_IDENTITY_JSON_RE: "re.Pattern[str]" = re.compile(r"\{.*\}", re.DOTALL)


def _parse_identity_verdict(raw: Optional[str]) -> tuple[str, str]:
    """Робастный парс ответа LLM-верификатора в (verdict, distinguishing).

    Всегда возвращает verdict строго "same"|"different"|"unknown" — никогда не
    бросает исключение наружу. Любой сбой парсинга (пустой ответ, битый JSON,
    неожиданное значение verdict) трактуется как "unknown" (fail-safe).
    """
    if not raw or not raw.strip():
        return ("unknown", "")

    text = raw.strip()
    match = _IDENTITY_JSON_RE.search(text)
    json_str = match.group(0) if match else text

    try:
        data = json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        return ("unknown", "")

    if not isinstance(data, dict):
        return ("unknown", "")

    verdict = data.get("verdict")
    if verdict not in ("same", "different"):
        return ("unknown", "")

    distinguishing = data.get("distinguishing")
    if not isinstance(distinguishing, str):
        distinguishing = ""
    else:
        distinguishing = distinguishing.strip()

    return (verdict, distinguishing)


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
    """Копия характеристик с live Ozon-карточки похожего товара через scrape.do.

    Cost: 2 credits/product (search + features), 0 на failed.
    Latency: 15-60s end-to-end typical (scrape.do render+super_proxy per call).
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
        _ = scrappey_key
        _ = scrapfly_key
        _ = apify_token
        _ = ozon_api_base
        _ = kwargs

        self._scrapedo_token = os.environ.get("SCRAPEDO_TOKEN")
        if not self._scrapedo_token:
            logger.warning(
                "[OzonCard] SCRAPEDO_TOKEN не задан — extract() всегда вернёт []."
            )

        self._judge = OzonCardJudge()

        # LRU cache: (brand_lower, model_lower) → list[AttributeValue]
        self._cache: "OrderedDict[tuple[str, str], list[AttributeValue]]" = OrderedDict()

        # FIX-16: in-process memoization of identity-verdicts within this
        # instance's lifetime — (query, title) -> "same"|"different"|"unknown".
        # Winner-only calls land here so an identical pair never re-hits the LLM.
        self._identity_cache: dict[tuple[str, str], str] = {}

    @property
    def source_type(self) -> Source:
        return Source.OZON_CARD

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим если product_name есть и не пустой, и SCRAPEDO_TOKEN доступен."""
        return bool(
            self._scrapedo_token
            and context.product_name
            and len(context.product_name.strip()) >= 5
        )

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        if not targets or not context.product_name or not self._scrapedo_token:
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

        # Network calls — bounded by hard total timeout to prevent stalls.
        # scrape.do (with its own internal retry) is bounded by _OZON_CARD_TOTAL_TIMEOUT
        # (200s). If the total cap still fires (extreme degradation), fall through to
        # the Serper-snippet fallback instead of surrendering the product.
        try:
            all_values = await asyncio.wait_for(
                self._do_extract(context, targets),
                timeout=_OZON_CARD_TOTAL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[OzonCard] total timeout (%.0fs) for '%s' — scrape.do path failed; "
                "trying serper-fallback",
                _OZON_CARD_TOTAL_TIMEOUT,
                context.product_name[:60],
            )
            # Serper fallback does not touch scrape.do and has its own 15s timeout
            # — safe to call even after the scrape.do path timed out.
            snippet_values = await self._serper_snippet_fallback(context, targets)
            logger.info(
                "[OzonCard] serper-fallback-used after total-timeout: %d values for '%s'",
                len(snippet_values), context.product_name[:60],
            )
            self._cache_put(cache_key, snippet_values)
            return self._filter_for_targets(snippet_values, targets)
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
    # Network orchestration via scrape.do
    # ------------------------------------------------------------------

    async def _fetch_page(
        self, client: Any, target_url: str, session: Optional[str] = None
    ) -> Optional[str]:
        """Один scrape.do вызов (render+super_proxy, geo=ru) с собственным retry внутри scrapedo_fetch.

        scrapedo_fetch сам ретраит транзиентные 429/5xx (backoff 2/4/8s) и сам
        проверяет минимальную длину тела — здесь никакого дополнительного
        retry-обёртывания не нужно. HTML SSR-парсинг ниже по файлу не меняется.

        Возвращает HTML-контент страницы или None при неудаче.
        Параметры client и session принимаются для backward-compat с существующими
        call-sites/тестами, но игнорируются — scrape.do использует свой внутренний
        httpx-клиент и не поддерживает session-reuse в этой интеграции.
        """
        _ = client
        _ = session
        res = await scrapedo_fetch(target_url, render=True, super_proxy=True, geo="ru")
        if res.success and res.content and not _is_datadome_block(res.content):
            logger.info(
                "[OzonCard] scrape.do success (%d chars, credits=%s) for %s",
                len(res.content), res.credits_used, target_url[:80],
            )
            return res.content
        logger.info(
            "[OzonCard] scrape.do failed (success=%s, err=%s) for %s",
            res.success, res.error, target_url[:80],
        )
        return None

    async def _fetch_card_raw(
        self,
        context: ExtractionContext,
        client: Any = None,
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

        # scrape.do не использует session-reuse в этой интеграции — параметр оставлен
        # для сигнатур-совместимости нижестоящих вызовов _fetch_page/_try_serper_card,
        # всегда None.
        session = None

        # Serper-FIRST: пробуем Google-card ДО флакового внутреннего поиска Ozon.
        # Успех → 1 scrape.do-фетч (/features/) вместо search+features → меньше троттла.
        # serper_tried гасит повторный Serper-вызов в фоллбэках ниже.
        serper_tried = False
        if _OZON_SERPER_FIRST and _OZON_SERPER_CARD_FINDING:
            serper = await self._try_serper_card(context, client, session)
            serper_tried = True
            if serper is not None and serper.get("stage") == "ok":
                return serper
            logger.info("[OzonCard] Serper-first не дал карточку → внутренний поиск Ozon")

        query: Optional[str] = None
        tiles: list[dict] = []
        for q in queries_to_try:
            logger.info(
                "[OzonCard] search query: '%s' (was: '%s')",
                q, full_name[:80],
            )
            html = await self._fetch_page(client, f"{_OZON_SEARCH_URL}?text={q}", session=session)
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
            # Поиск Ozon флакнул — Google (Serper), если ещё не пробовали (serper-first).
            if not serper_tried:
                serper = await self._try_serper_card(context, client, session)
                if serper is not None and serper.get("stage") == "ok":
                    return serper
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

        top_tile, top_score = self._pick_best_match(
            query, tiles[:_MATCH_TOP_N], category_leaf=cat_leaf,
            query_brand=context.brand, query_name=full_name,
        )
        mode = self._classify_match(top_score) if top_tile is not None else "skip"

        # Secondary net (bounded, 1 extra fetch): if even the best of top-N is
        # below threshold, the SSR ordering/content jitter may have starved this
        # particular fetch of the exact card. Re-run the SAME search ONCE and
        # re-score; keep the better of the two attempts. The hard total timeout
        # (_OZON_CARD_TOTAL_TIMEOUT) still bounds the whole _do_extract.
        if (top_tile is None or mode == "skip") and query is not None:
            retry_html = await self._fetch_page(client, f"{_OZON_SEARCH_URL}?text={query}", session=session)
            if retry_html is not None:
                retry_tiles = self._parse_search_tiles_html(retry_html)
                if retry_tiles:
                    r_tile, r_score = self._pick_best_match(
                        query, retry_tiles[:_MATCH_TOP_N], category_leaf=cat_leaf,
                        query_brand=context.brand, query_name=full_name,
                    )
                    logger.info(
                        "[OzonCard] retry search best score=%.1f (prev=%.1f)",
                        r_score, top_score,
                    )
                    if r_tile is not None and r_score > top_score:
                        top_tile, top_score, tiles = r_tile, r_score, retry_tiles
                        mode = self._classify_match(top_score)

        if top_tile is None or mode == "skip":
            logger.info(
                "[OzonCard] best score=%.1f < %.0f — skip",
                top_score, _BRAND_LINE_THRESHOLD,
            )
            # Ozon-tile ниже порога — Google может найти точную карточку (если ещё не пробовали).
            if not serper_tried:
                serper = await self._try_serper_card(context, client, session)
                if serper is not None and serper.get("stage") == "ok":
                    return serper
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
        features_html = await self._fetch_page(client, features_url, session=session)
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
            raw = await self._fetch_card_raw(context, None)
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

    async def _verify_product_identity(
        self, query: str, candidate_title: str, extra: str = "",
    ) -> str:
        """FIX-16: LLM-гейт идентичности товара (финальный гейт после _pick_best_match).

        Спрашивает DeepSeek "это тот же товар?" на выжившем кандидате (winner-only,
        редкий путь — копейки). Транспорт: DIRECT DeepSeekProvider (переиспользует
        существующий движковый клиент/ключ из .env, НЕ gateway/ocl_call — gateway
        EMPTY-флак дал бы ложный "same").

        Returns:
            "same" | "different" | "unknown" — строго одно из трёх, НИКОГДА не
            бросает исключение наружу. Любой сбой транспорта/парсинга/таймаута
            fail-safe'ится в "unknown" (никогда не "same" по умолчанию).
        """
        query_stripped = query.strip()
        title_stripped = candidate_title.strip()
        if not query_stripped or not title_stripped:
            return "unknown"

        prompt = _IDENTITY_PROMPT_TEMPLATE.format(
            query=query_stripped,
            title=title_stripped,
        )
        extra_stripped = extra.strip()
        if extra_stripped:
            prompt += f"\n{extra_stripped}"

        try:
            provider = DeepSeekProvider()
            resp = await asyncio.wait_for(
                provider.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=_OZON_CARD_LLM_IDENTITY_MODEL,
                    temperature=0.0,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                    timeout=int(_OZON_CARD_LLM_IDENTITY_TIMEOUT),
                ),
                timeout=_OZON_CARD_LLM_IDENTITY_TIMEOUT,
            )
            verdict, distinguishing = _parse_identity_verdict(resp.content)
            logger.info(
                "[OzonCard][FIX-16] verdict=%s query=%.80s title=%.80s dist=%s cost=%.6f",
                verdict,
                query_stripped,
                title_stripped,
                distinguishing,
                resp.cost_usd,
            )
            return verdict
        except Exception as exc:
            logger.warning(
                "[OzonCard][FIX-16] identity check failed query=%.80s title=%.80s exc=%s",
                query_stripped,
                title_stripped,
                exc,
            )
            return "unknown"

    async def _get_identity_verdict(self, query: str, title: str) -> str:
        """FIX-16: мемоизированная обёртка над _verify_product_identity.

        Кеш по (query, title) в рамках жизни инстанса source — идентичная пара
        не бьёт LLM дважды (см. self._identity_cache в __init__).
        """
        key = (query, title)
        if key in self._identity_cache:
            return self._identity_cache[key]

        verdict = await self._verify_product_identity(query, title)
        self._identity_cache[key] = verdict
        return verdict

    async def _resolve_identity_gate(
        self,
        context: ExtractionContext,
        mode: str,
        title: str,
    ) -> Optional[list[AttributeValue]]:
        """FIX-16: identity-гейт, вынесенный из _do_extract (extract-method, CC).

        Если LLM-гейт включён и mode в ("exact", "brand_line"), запрашивает
        вердикт идентичности query vs title. Возвращает [] (пустой список),
        если winner должен быть abstain'нут (identity=different, ИЛИ
        identity=unknown без fail-safe accept); иначе None — extraction
        должен продолжиться как обычно (гейт выключен / mode=skip недостижим
        здесь / verdict=="same" / fail-safe accept для "unknown").

        Args:
            context: Контекст извлечения (для product_name).
            mode: Класс совпадения winner-кандидата ("exact"|"brand_line").
            title: Заголовок winner-карточки.

        Returns:
            [] если нужно abstain, None если продолжать extraction.
        """
        query_for_identity = (context.product_name or "").strip()

        if not _OZON_CARD_LLM_IDENTITY_ENABLED or mode not in ("exact", "brand_line"):
            return None

        verdict = await self._get_identity_verdict(query_for_identity, title)

        if verdict == "different":
            logger.info(
                "[OzonCard][FIX-16] query=%.80s title=%.80s abstain (identity=different)",
                query_for_identity,
                title,
            )
            return []

        if verdict == "unknown":
            # FAIL-SAFE: без судьи проходит ТОЛЬКО высоко-уверенное совпадение
            # (exact + FIX-15 model_conflict==False); brand_line (двусмысленная
            # полоса — ровно где нужен был LLM) → abstain.
            fail_safe_ok = mode == "exact" and not _model_conflict(query_for_identity, title)
            if not fail_safe_ok:
                logger.info(
                    "[OzonCard][FIX-16] query=%.80s title=%.80s abstain "
                    "(unknown + mode=%s model_conflict=%s)",
                    query_for_identity,
                    title,
                    mode,
                    _model_conflict(query_for_identity, title),
                )
                return []
            logger.info(
                "[OzonCard][FIX-16] query=%.80s title=%.80s fail-safe accept "
                "(unknown but exact without model-conflict)",
                query_for_identity,
                title,
            )

        return None

    async def _do_extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Полный flow: search HTML → match → /features/ HTML → identity-gate → map → AVs."""
        raw = await self._fetch_card_raw(context, None)

        if raw["stage"] != "ok":
            # scrape.do недоступен/пуст (no_tiles / fetch_fail / parse_empty / low_match).
            # БЕСПЛАТНАЯ подстраховка: Serper-сниппет Ozon-страницы. Не трогает
            # scrape.do-путь — срабатывает ТОЛЬКО на его нулевом результате.
            logger.info(
                "[OzonCard] scrape.do-путь дал stage=%s (0 chars) → snippet-fallback",
                raw["stage"],
            )
            return await self._serper_snippet_fallback(context, targets)

        chars = raw["raw_chars"]
        top_score = raw["match_score"] or 0.0
        mode = raw["match_class"]
        title = raw.get("card_title") or ""

        # ---- FIX-16: LLM identity gate (ПОСЛЕ _pick_best_match, ДО отдачи карты) ----
        # Идёт ДО блока IMAGES, чтобы different/fail-safe abstain не успел
        # замусорить context.image_urls чужими фотками.
        gate_result = await self._resolve_identity_gate(context, mode, title)
        if gate_result is not None:
            return gate_result

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
    # HTML parsing (scrape.do HTML pages)
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
        data-state='<JSON>'>` (старая разметка, одинарная кавычка) ИЛИ
        `data-state="<HTML-escaped JSON>"` (новая разметка, двойная кавычка ---
        `html.unescape()` применяется перед `json.loads`). JSON структура:
          {"link":"...","characteristics":[
              {"short":[{key,name,values:[{text,id}]}],
               "long":[...], "full":[...]}
          ]}

        Объединяем short+long+full, dedupe по name.
        Возвращает [{name, value, value_ids}] где value — comma-joined text.
        """
        out: list[dict] = []
        seen: set[str] = set()
        for match in _FEATURES_STATE_RE.finditer(html):
            raw = (
                match.group(1)
                if match.group(1) is not None
                else _html.unescape(match.group(2))
            )
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

    @staticmethod
    def _parse_snippet_pairs(snippet: str) -> list[dict]:
        """Генерично парсит пары «Имя: значение» из Serper-сниппета Ozon-страницы.

        Сниппет /features/ страницы выглядит как:
          «Артикул: 1240096510; Сезон: На любой сезон; Материал: Трикотаж;
           Состав материала: полиэстер 67%, акрил 20%; Коллекция: ...»

        Парсинг полностью generic (без хардкода имён/значений):
          1. split по разделителям пар: `;`, `•`, переводы строк.
          2. для каждого куска split по ПЕРВОМУ `:` → (name, value).
          3. чистка: trim, схлопывание пробелов, отбрасывание пустых и слишком
             длинных «значений» (вероятно мусор/маркетинг-текст без ':').

        Возвращает [{name, value, value_ids:[]}] совместимо с _map_characteristics.
        """
        if not snippet:
            return []
        # Разделители между парами. Запятая НЕ используется как разделитель пар —
        # она встречается внутри значений («полиэстер 67%, акрил 20%»).
        chunks = re.split(r"[;•·|\n\r]+", snippet)
        out: list[dict] = []
        seen: set[str] = set()
        for chunk in chunks:
            if ":" not in chunk:
                continue
            name_part, _, value_part = chunk.partition(":")
            name = re.sub(r"\s+", " ", name_part).strip(" \t-–—.")
            value = re.sub(r"\s+", " ", value_part).strip(" \t-–—.")
            if not name or not value:
                continue
            # name характеристики — короткое словосочетание, не предложение.
            if len(name) > 40 or len(name.split()) > 5:
                continue
            # value слишком длинное → вероятно обрезанный маркетинг-текст, не спек.
            if len(value) > 120:
                continue
            name_low = name.lower()
            if name_low in seen:
                continue
            seen.add(name_low)
            out.append({"name": name, "value": value, "value_ids": []})
        return out

    async def _serper_find_card(self, context: ExtractionContext) -> Optional[dict]:
        """Найти точный URL Ozon-товара через Serper (Google индексирует ozon.ru
        надёжнее их собственного антибот-поиска).

        Возвращает {card_url, slug, pid, title} первого organic-результата
        ozon.ru/product/...-<pid>/ или None. Никогда не падает.
        """
        try:
            from app.services.providers.factory import get_web_search_client
            client = get_web_search_client()
        except Exception as exc:
            logger.info("[OzonCard] serper-find: web_search client unavailable: %s", exc)
            return None
        if client is None:
            logger.info("[OzonCard] serper-find: PROVIDER_WEB_SEARCH != serper → skip")
            return None

        product_name = context.product_name.strip()
        try:
            res = await client.search(
                f"{product_name} ozon",
                num_results=_SERPER_NUM_RESULTS,
                timeout=_SERPER_TIMEOUT,
            )
        except Exception as exc:
            logger.info("[OzonCard] serper-find: Serper search failed: %s", exc)
            return None

        for org in res.organic_results:
            link = org.link or ""
            m = _SERPER_SLUG_PID_RE.search(link)
            if not m:
                continue
            slug, pid = m.group(1), m.group(2)
            return {
                "card_url": f"{_OZON_PRODUCT_BASE}{slug}-{pid}/",
                "slug": slug,
                "pid": pid,
                "title": (org.title or "").strip(),
            }
        logger.info("[OzonCard] serper-find: no ozon.ru/product/ organic для '%s'", product_name[:60])
        return None

    async def _try_serper_card(
        self,
        context: ExtractionContext,
        client: Any = None,
        session: Optional[str] = None,
    ) -> Optional[dict]:
        """Serper-assisted card-finding: URL товара через Google (в обход флакового
        поиска Ozon) → scrape.do-фетч /features/ → parse полной карточки.

        Гард: Serper-карточка проходит ТОТ ЖЕ _pick_best_match/_classify_match скоринг,
        что и Ozon-tile — чужой бренд/тип/модель уходит ниже порога → отвергаем (пусто
        честнее мусорной донор-карточки). Возвращает raw-dict в формате _fetch_card_raw
        (stage=ok|fetch_fail|parse_empty) или None (не нашли / гард отверг / выключено).
        """
        if not _OZON_SERPER_CARD_FINDING:
            return None
        found = await self._serper_find_card(context)
        if not found:
            return None

        full_name = context.product_name.strip()
        cat_leaf = context.category_path[-1] if context.category_path else None
        title = found["title"]
        slug, pid, card_url = found["slug"], found["pid"], found["card_url"]

        # Гард: тот же скоринг, что для Ozon-tile (бренд/тип/модель-mismatch штрафы).
        pseudo_tile = {"title": title, "slug": slug, "pid": pid}
        _t, score = self._pick_best_match(
            full_name, [pseudo_tile], category_leaf=cat_leaf,
            query_brand=context.brand, query_name=full_name,
        )
        mode = self._classify_match(score)
        if mode == "skip":
            logger.info(
                "[OzonCard] Serper-card '%s' score=%.1f < %.0f — отвергнут гардом",
                title[:60], score, _BRAND_LINE_THRESHOLD,
            )
            return None

        logger.info(
            "[OzonCard] Serper-card найден: '%s' score=%.1f mode=%s pid=%s → фетч /features/",
            title[:60], score, mode, pid,
        )
        base = {
            "query": f"serper:{full_name[:40]}",
            "tiles_count": 0,
            "match_score": score,
            "match_class": mode,
            "card_url": card_url,
            "card_title": title,
        }
        features_url = f"{_OZON_PRODUCT_BASE}{slug}-{pid}/features/"
        features_html = await self._fetch_page(client, features_url, session=session)
        if features_html is None:
            return {**base, "raw_chars": [], "image_urls": [], "stage": "fetch_fail"}

        chars = self._parse_characteristics_html(features_html)
        image_urls = self._extract_image_urls(features_html)
        if not chars:
            return {**base, "raw_chars": [], "image_urls": image_urls, "stage": "parse_empty"}

        logger.info("[OzonCard] Serper-card HIT: '%s' → %d chars", title[:60], len(chars))
        return {**base, "raw_chars": chars, "image_urls": image_urls, "stage": "ok"}

    async def _serper_snippet_fallback(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """БЕСПЛАТНАЯ подстраховка: достать ключевые характеристики из
        Serper-сниппета Ozon-страницы, когда scrape.do мёртв/пуст.

        Поток:
          1. Serper-запрос «<product_name> ozon» → из organic вытащить ссылку
             ozon.ru/product/...-<pid>/ и pid.
          2. Серия generic name→value пар уже может прийти прямо в snippet того
             же organic-результата; дополнительно — второй запрос
             «ozon product <pid> состав материал» за сниппетом /features/.
          3. Распарсить пары → прогнать через ТУ ЖЕ мапилку _map_characteristics
             с пониженным confidence (_CONF_SNIPPET).

        Никогда не падает: любые сетевые/quota ошибки → []. Не дёргается, если
        SERPER_API_KEY не задан или web_search не serper.
        """
        try:
            from app.services.providers.factory import get_web_search_client
            client = get_web_search_client()
        except Exception as exc:
            logger.info("[OzonCard] snippet-fallback: web_search client unavailable: %s", exc)
            return []
        if client is None:
            logger.info("[OzonCard] snippet-fallback: PROVIDER_WEB_SEARCH != serper → skip")
            return []

        product_name = context.product_name.strip()

        # ---- Запрос 1: найти Ozon-товар + pid ----
        snippets: list[str] = []
        pid: Optional[str] = None
        try:
            res = await client.search(
                f"{product_name} ozon",
                num_results=_SERPER_NUM_RESULTS,
                timeout=_SERPER_TIMEOUT,
            )
        except Exception as exc:
            logger.info("[OzonCard] snippet-fallback: Serper search #1 failed: %s", exc)
            return []

        for org in res.organic_results:
            link = org.link or ""
            if "ozon.ru/product/" not in link.lower():
                continue
            if pid is None:
                m = _SERPER_PID_RE.search(link)
                if m:
                    pid = m.group(1)
            if org.snippet:
                snippets.append(org.snippet)
            # Артикул в сниппете — тоже источник pid.
            if pid is None and org.snippet:
                ma = _SERPER_ARTICUL_RE.search(org.snippet)
                if ma:
                    pid = ma.group(1)

        # ---- Запрос 2 (опц.): сниппет /features/ страницы по pid ----
        # Делаем только если pid найден — точечный запрос к features-странице,
        # где сниппет содержит «Имя: значение;» пары.
        if pid:
            try:
                res2 = await client.search(
                    f"ozon product {pid} состав материал характеристики",
                    num_results=_SERPER_NUM_RESULTS,
                    timeout=_SERPER_TIMEOUT,
                )
                for org in res2.organic_results:
                    link = (org.link or "").lower()
                    if "ozon" in link and org.snippet:
                        snippets.append(org.snippet)
            except Exception as exc:
                logger.info("[OzonCard] snippet-fallback: Serper search #2 failed: %s", exc)

        if not snippets:
            logger.info("[OzonCard] snippet-fallback: no Ozon snippets found")
            return []

        # ---- Парсинг пар из всех собранных сниппетов ----
        chars: list[dict] = []
        seen_names: set[str] = set()
        for snip in snippets:
            for pair in self._parse_snippet_pairs(snip):
                nm = pair["name"].lower()
                if nm in seen_names:
                    continue
                seen_names.add(nm)
                chars.append(pair)

        if not chars:
            logger.info("[OzonCard] snippet-fallback: 0 name:value pairs parsed")
            return []

        logger.info(
            "[OzonCard] snippet-fallback: pid=%s, %d пар распарсилось из %d сниппетов",
            pid, len(chars), len(snippets),
        )

        # ---- Маппинг через ту же логику (exact mode → без brand_line фильтров) ----
        evidence = f"ozon-snippet:pid={pid or '?'} | serper-fallback"
        return self._map_characteristics(
            chars, targets, context,
            mode="exact",          # без brand_line BLACKLIST/numeric-skip: сниппет = тот же товар
            title=product_name[:50],
            score=0.0,
            conf_override=_CONF_SNIPPET,
            evidence_override=evidence,
        )

    # ------------------------------------------------------------------
    # Match scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_best_match(
        query: str,
        tiles: list[dict],
        category_leaf: Optional[str] = None,
        query_brand: Optional[str] = None,
        query_name: Optional[str] = None,
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

        Штраф _BRAND_MISMATCH_PENALTY за конфликт бренда (wrong-SKU гард):
        если бренд запроса (query_brand) латинский и явно ОТСУТСТВУЕТ в title
        карточки, а в title есть ДРУГОЙ латинский бренд-кандидат → score -= 60
        (карточка чужого бренда уходит ниже порога → "skip", attrs не копируются).
        query_name нужен чтобы не считать конфликтом латин-токены, уже стоящие в
        самом имени запроса (модель-линейка).
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return (tiles[0], 100.0) if tiles else (None, 0.0)

        _MODEL_BONUS = 5.0
        _MODEL_BONUS_ALPHA = 3.0  # словесная модель — слабее цифрового артикула
        _GENDER_MISMATCH_PENALTY = 30.0  # как type-mismatch: карточка чужого пола проигрывает
        q_models = _extract_model_tokens(query)
        q_alpha = _extract_alpha_model_tokens(query)
        # Гендер-сигнал ЗАПРОСА (имени товара). Если у имени пол нейтрален
        # («Ultraboost 22») → q_gender=None → штраф не применяется (нельзя
        # утверждать, что карточка неверного пола).
        q_gender = _extract_gender_signal(query)
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
            # Алфавитный (словесный) модель-бонус: меньше цифрового, отдельно от
            # q_models — чтобы НЕ отключить type-mismatch штраф ниже (он гейтится
            # по отсутствию ЦИФРОВЫХ артикулов, q_models, а не словесных моделей).
            elif q_alpha and q_alpha & _extract_alpha_model_tokens(title):
                score += _MODEL_BONUS_ALPHA
            # Штраф за тип товара: только для товаров без артикула (одежда)
            # и когда тип (category_leaf) отсутствует в заголовке тайла.
            # Пример: query «Куртка The North Face» + tile «Шорты The North Face» →
            #   cat_leaf_low «куртка» не входит в «шорты the north face» → штраф.
            # Для электроники с артикулом (q_models непустое) штраф не срабатывает.
            if (cat_leaf_low and not q_models
                    and not _category_present_in_title(cat_leaf_low, title.lower())):
                score -= _TYPE_MISMATCH_PENALTY
            # Гендер-штраф: имя товара несёт явный пол И заголовок карточки несёт
            # ПРОТИВОРЕЧАЩИЙ пол (мужской vs женский) → карточка не того гендера
            # проигрывает. Унисекс/нейтральное имя/неоднозначность — без штрафа.
            if q_gender is not None and _gender_conflict(query, title):
                score -= _GENDER_MISMATCH_PENALTY
            # Бренд-штраф (wrong-SKU гард): бренд запроса латинский, отсутствует
            # в title карточки, а в title есть другой латин-бренд → карточка
            # чужого бренда. Крупный штраф (60) гарантирует уход ниже порога.
            if _brand_conflict(query_brand, query_name or query, title):
                score -= _BRAND_MISMATCH_PENALTY
            # FIX-15: model-conflict guard (sibling-of-same-brand)
            if _model_conflict(query, title):
                score -= _MODEL_CONFLICT_PENALTY
            # Per-tile score logging under existing instrumentation — makes the
            # best-of-top-N selection (chosen tile + its score) visible run-to-run,
            # so SSR ordering jitter can be confirmed/diagnosed.
            logger.info(
                "[OzonCard] tile score=%.1f title='%s'",
                score, title[:80],
            )
            if score > best_score:
                best_score = score
                best_tile = tile
        if best_tile is not None:
            logger.info(
                "[OzonCard] best-of-top-N score=%.1f title='%s'",
                best_score, (best_tile.get("title") or "")[:80],
            )
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
        conf_override: Optional[float] = None,
        evidence_override: Optional[str] = None,
    ) -> list[AttributeValue]:
        """Сопоставить Ozon-char names с target.name через Ozon dictionary.

        conf_override / evidence_override — для snippet-fallback пути: там данные
        приходят не из полной карточки, а из Serper-сниппета (частичные), поэтому
        confidence ниже и evidence другой. Логика name→attribute_id→value_id
        полностью переиспользуется.
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

        # target_id → lowercase set имён (raw + нормализованные)
        target_names_low: dict[int, set[str]] = {}
        for t in targets:
            raw_low = t.name.lower()
            names = {raw_low, _norm_char_name(t.name)}
            dn = attr_id_to_dict_name.get(t.id)
            if dn:
                names.add(dn.lower())
                names.add(_norm_char_name(dn))
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

        evidence_short = evidence_override or f"ozon:{title[:50]} | match={score:.1f}"
        conf = conf_override if conf_override is not None else (
            _CONF_EXACT if mode == "exact" else _CONF_BRAND_LINE
        )

        # Гендер-страховка для поля «Пол»: имя товара vs гендер карточки (title).
        name_gender = _extract_gender_signal(context.product_name or "")
        card_gender = _extract_gender_signal(title)

        results: list[AttributeValue] = []
        used_ids: set[int] = set()

        for c in chars:
            char_name = c["name"].strip()
            char_val = c["value"].strip()
            char_name_low = char_name.lower()
            char_name_norm = _norm_char_name(char_name)

            # brand_line: пропускаем ТОЛЬКО numeric-атрибуты (мощность,
            # размеры, объём и т.п. — модель-специфичны, брать с соседней
            # модели = галлюцинация). enum/text/bool (бренд, цвет, материал,
            # состав, сезон, гарантия, ...) — безопасны: они одинаковы в
            # рамках бренд-линейки и merger/judge отфильтруют чужое.
            # BLACKLIST (артикул/MPN/EAN) остаётся — страховка во всех режимах.
            if mode == "brand_line":
                if char_name_low in _BRAND_LINE_BLACKLIST:
                    continue

            # 1) Exact lowercase match (raw, затем нормализованное)
            target_id = name_to_target_id.get(char_name_low)
            if target_id is None and char_name_norm != char_name_low:
                target_id = name_to_target_id.get(char_name_norm)
            # 2) Substring match (raw и нормализованное против всего словаря)
            if target_id is None:
                for tn, tid in name_to_target_id.items():
                    if char_name_low in tn or tn in char_name_low:
                        target_id = tid
                        break
            if target_id is None and char_name_norm != char_name_low:
                for tn, tid in name_to_target_id.items():
                    if char_name_norm in tn or tn in char_name_norm:
                        target_id = tid
                        break
            # 3) Fuzzy fallback (нормализованная форма как кандидат)
            if target_id is None and process is not None and all_target_names:
                # пробуем нормализованную форму первой — она ближе семантически
                query = char_name_norm if char_name_norm else char_name_low
                best = process.extractOne(
                    query, all_target_names, scorer=fuzz.WRatio,
                )
                if best is not None and best[1] >= 88:
                    target_id = name_to_target_id[best[0]]
                # если не нашли по норме — пробуем сырое имя
                if target_id is None and query != char_name_low:
                    best2 = process.extractOne(
                        char_name_low, all_target_names, scorer=fuzz.WRatio,
                    )
                    if best2 is not None and best2[1] >= 88:
                        target_id = name_to_target_id[best2[0]]

            if target_id is None:
                logger.debug(
                    "[OzonCard] MISS char=%r norm=%r", char_name, char_name_norm
                )
            if target_id is None or target_id in used_ids:
                continue

            target = target_by_id.get(target_id)
            if target is None:
                continue

            # brand_line: skip numeric targets — они модель-специфичны
            # (мощность/объём/размеры) и взятые с соседней модели = галлюцинация.
            # enum/text/bool пропускаем без ограничений: merger/judge отфильтруют.
            if mode == "brand_line" and target.type == "numeric":
                continue

            used_ids.add(target_id)

            # Гендер-страховка для поля «Пол» из brand_line-карточки (не exact).
            # (a) имя несёт явный пол, конфликтующий со значением → СКИП;
            # (b) имя нейтрально, но карточка-донор сама гендерная и значение
            #     повторяет её пол → понижаем confidence (значение навеяно донором).
            target_conf = conf
            if mode == "brand_line" and _is_gender_target_name(target.name):
                value_gender = _extract_gender_signal(char_val)
                if value_gender is not None and value_gender != "unisex":
                    if name_gender is not None and value_gender != name_gender \
                            and name_gender != "unisex":
                        logger.info(
                            "[OzonCard] гендер-страховка: СКИП '%s'='%s' "
                            "(имя='%s' пол=%s vs значение=%s)",
                            target.name, char_val, context.product_name,
                            name_gender, value_gender,
                        )
                        continue
                    if name_gender is None and card_gender is not None \
                            and card_gender == value_gender:
                        target_conf = min(conf, _CONF_GENDER_DOWNWEIGHT)
                        logger.info(
                            "[OzonCard] гендер-страховка: ПОНИЖЕН conf '%s'='%s'→%.2f "
                            "(нейтральное имя, brand_line-карточка пол=%s)",
                            target.name, char_val, target_conf, card_gender,
                        )

            # Коллекционные характеристики Ozon отдаёт одной строкой (", ".join).
            # Сплитим в список, чтобы значение участвовало в union merge поэлементно
            # и не проигрывало vision/llm целиком. value_id(s) дорезолвит
            # resolve_value_ids в _finalize (он умеет per-element для списков).
            value_id: Optional[int] = None
            value_out: Union[str, list[str]]
            if target.is_collection:
                value_out = _split_multivalue(char_val)
            else:
                value_out = char_val
                if cat_id and type_id:
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                    except Exception as exc:
                        logger.debug("[OzonCard] resolve_value_id failed: %s", exc)

            results.append(AttributeValue(
                attribute_id=target.id,
                value=value_out,
                confidence=target_conf,
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
