"""IceCatSource — извлекает brand-verified характеристики из IceCat Open API.

IceCat Open Tier покрывает крупные бренды: ASUS, be quiet!, MSI, Gigabyte, HP, Lenovo и др.
Для брендов только в Full IceCat возвращает 403 → graceful skip (бренд логируется).

Алгоритм:
  1. Извлечь кандидатов product code из product_name (многоуровневая стратегия):
       a) Полное название модели как-есть (напр. "MPG A850G PCIE5")
       b) Дефисованный вариант (напр. "ROG-STRIX-850G")
       c) Отдельные значимые токены (напр. "UD850GM", "PM650D")
  2. Для каждого кандидата: GET IceCat API. На первый 200 — парсим и выходим.
     - 403 → бренд в Full tier, логируем, возвращаем [].
     - 404 → продукт не найден, пробуем следующий кандидат.
  3. Если все кандидаты 404 → MPN Lookup через Serper + LLM (DeepSeek cheap):
       Serper ищет "{brand} {product_name} MPN Part Number",
       LLM извлекает реальный MPN из сниппетов (напр. "90YE00A4-B0NA00"),
       затем повторяем запрос IceCat с найденным MPN.
  4. Парсим FeaturesGroups → flat list (icecat_name, value).
  5. Маппируем icecat_name → TargetAttribute через fuzzy/semantic match (rapidfuzz).
  6. Строим AttributeValue(confidence=0.92, source=Source.ICECAT).

IceCat API не поддерживает текстовый поиск по названию в Open tier.
Поиск работает только по Brand+ProductCode или icecat_id.
Правильный ProductCode: "ROG-STRIX-850G", "MPG A850G PCIE5", "GP-UD850GM" и т.д.

Credentials: ICECAT_EMAIL, ICECAT_TOKEN из .env (никогда не хардкодить).
Spec: docs/architecture/pipeline.md (IceCatSource — cheap pre-LLM stage).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import Counter
from typing import Optional

from pydantic import BaseModel

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.icecat_judge import IceCatJudge
from app.services.enrichment.sources.icecat_numeric_normalizer import normalize_icecat_numeric

logger = logging.getLogger(__name__)

# Confidence для всех IceCat-значений (brand-verified данные высокого качества)
_ICECAT_CONFIDENCE = 0.92

# ---------------------------------------------------------------------------
# LEVER 1: General bilingual EN→RU attribute-name normalization
#
# Purpose: IceCat Open returns attribute names in English for many brands while
# our target attributes are in Russian. A fuzzy match on dissimilar-language
# strings produces low scores and drops real data. This compact, GENERAL
# EN→RU concept map (not per-category) converts common tech-spec attribute
# names to Russian so the existing rapidfuzz matcher can operate on same-
# language strings. No per-category or per-product hardcode.
#
# Maintenance rule: entries must be GENERAL concepts that appear across product
# categories (power, weight, color, dimensions, …). Per-category terms belong
# in the marketplace dictionary, not here.
# ---------------------------------------------------------------------------

_EN_RU_ATTR_NAME_MAP: dict[str, str] = {
    # Power / energy
    "power": "мощность",
    "power output": "мощность",
    "rated power": "мощность",
    "wattage": "мощность",
    "power consumption": "потребляемая мощность",
    "standby power": "мощность в режиме ожидания",
    "energy efficiency": "энергоэффективность",
    "energy class": "класс энергопотребления",
    "energy rating": "класс энергопотребления",
    # Dimensions / weight
    "weight": "вес",
    "product weight": "вес",
    "net weight": "вес нетто",
    "width": "ширина",
    "height": "высота",
    "depth": "глубина",
    "length": "длина",
    "diameter": "диаметр",
    "thickness": "толщина",
    # Color / appearance
    "color": "цвет",
    "colour": "цвет",
    "product colour": "цвет",
    "product color": "цвет",
    # Connectivity / interfaces
    "interface": "интерфейс",
    "connector": "разъём",
    "connector type": "тип разъёма",
    "port": "порт",
    "cable length": "длина кабеля",
    "cable type": "тип кабеля",
    # Display
    "display diagonal": "диагональ экрана",
    "screen size": "диагональ экрана",
    "resolution": "разрешение",
    "refresh rate": "частота обновления",
    "response time": "время отклика",
    "brightness": "яркость",
    "contrast ratio": "контрастность",
    # Memory / storage
    "memory": "память",
    "storage": "объём памяти",
    "capacity": "ёмкость",
    "storage capacity": "объём памяти",
    "ram": "оперативная память",
    "internal memory": "внутренняя память",
    # Processor
    "processor": "процессор",
    "processor speed": "частота процессора",
    "number of cores": "количество ядер",
    "clock speed": "тактовая частота",
    # Battery
    "battery capacity": "ёмкость аккумулятора",
    "battery life": "время работы аккумулятора",
    "battery voltage": "напряжение аккумулятора",
    "battery type": "тип аккумулятора",
    # General specs
    "form factor": "форм-фактор",
    "material": "материал",
    "operating temperature": "рабочая температура",
    "operating humidity": "рабочая влажность",
    "noise level": "уровень шума",
    "frequency": "частота",
    "voltage": "напряжение",
    "current": "ток",
    "efficiency": "кпд",
    "efficiency rating": "сертификат эффективности",
    "certification": "сертификат",
    "compatibility": "совместимость",
    "warranty": "гарантия",
    "country of origin": "страна производства",
    "brand": "бренд",
    "manufacturer": "производитель",
    "model": "модель",
    "product line": "линейка продуктов",
    "type": "тип",
    "fan speed": "скорость вентилятора",
    "fan size": "размер вентилятора",
    "number of fans": "количество вентиляторов",
    "modular": "модульность",
    "protection": "защита",
    "input voltage": "входное напряжение",
    "output voltage": "выходное напряжение",
    "input frequency": "входная частота",
    "output power": "выходная мощность",
    "peak power": "пиковая мощность",
    "dimensions": "габариты",
    "packaging dimensions": "размер упаковки",
}

# Pre-build lowercase version for O(1) exact lookup
_EN_RU_LOWER: dict[str, str] = {k.lower(): v for k, v in _EN_RU_ATTR_NAME_MAP.items()}


def _bilingual_normalize(name: str) -> str:
    """Return a Russian equivalent of an English attribute name, or the original.

    Algorithm (general, no per-category hardcode):
      1. Exact lowercase lookup in the EN→RU map.
      2. Partial-phrase lookup: check if any map key is a substring of the name
         (for names like "Total power output" → "power output" → "мощность").
      3. If no match: return the original name unchanged (Russian names pass through).

    This is used as a query-time normalization before rapidfuzz matching so that
    EN IceCat names can fuzzy-match against RU target names.
    """
    stripped = name.strip()
    lower = stripped.lower()

    # Step 1: exact match
    if lower in _EN_RU_LOWER:
        return _EN_RU_LOWER[lower]

    # Step 2: longest matching phrase contained in the name
    best_key: Optional[str] = None
    best_len = 0
    for key, ru_val in _EN_RU_LOWER.items():
        if key in lower and len(key) > best_len:
            best_key = key
            best_len = len(key)
    if best_key is not None:
        return _EN_RU_LOWER[best_key]

    return stripped


# ---------------------------------------------------------------------------
# LEVER 2: Stated-negative ("Нет") detection for boolean attributes
#
# FEATURE FLAG: ICECAT_STATED_NEGATIVE_ENABLED (default "0" = OFF)
# When ON: if an IceCat feature has an EXPLICIT verbatim negative value for a
# boolean/yes-no target, we fill "Нет" (false). We NEVER infer absence from
# missing data — only verbatim stated-negative triggers a fill.
#
# Conservative detection: only a small set of unambiguous explicit-negative
# literals counts. Any value not in this set → treated as positive or unknown.
# ---------------------------------------------------------------------------

_ICECAT_STATED_NEGATIVE_ENABLED: bool = (
    os.environ.get("ICECAT_STATED_NEGATIVE_ENABLED", "0").strip().lower()
    in ("1", "true", "yes", "on")
)

# Verbatim negative literals returned by IceCat for absent features.
# ONLY these exact strings (case-insensitive, stripped) trigger a "Нет" fill.
# "n/a" and "-" are already filtered in _parse_response (ambiguous: not explicit negatives).
_ICECAT_VERBATIM_NEGATIVES: frozenset[str] = frozenset({
    "no",
    "none",
    "нет",
    "false",
    "not available",
    "not supported",
    "не поддерживается",
    "не предусмотрен",
    "не предусмотрено",
    "без",
})

# Максимальное число candidates кодов которые пробуем (увеличено для multi-strategy)
_MAX_CANDIDATES = 8

# Счётчик брендов которые не в Open IceCat (глобальный для статистики)
closed_brands: Counter = Counter()

# Счётчик брендов у которых были успешные запросы
open_brands: Counter = Counter()

# Русские и английские обобщённые слова-префиксы — не являются частью product code
_GENERIC_PREFIX_WORDS = {
    "блок", "питания", "блок питания",
    "power", "supply", "unit",
}

# Слова-исключения: не являются product code
# ВАЖНО: "pcie5", "pcie4" НЕ включаем — они часть модельных кодов MSI (MPG A850G PCIE5)
_STOPWORDS = {
    # Сертификаты
    "gold", "silver", "bronze", "platinum", "titanium",
    # Технические термины (только generic, не модельные)
    "modular", "fully", "semi", "non", "atx", "sfx", "sfx-l", "eatx",
    "psu", "plus", "ultra", "pro", "max",
    "series", "edition", "full", "v2", "v3",
    "rgb", "argb", "aura",
    # Единицы мощности и числа
    "80+",
}

# Шаблон для поиска значимых токенов (alphanum + дефис, длина >= 3)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{2,}")

# Watt-суффикс: "850W", "750 W" и т.д. — убираем из модельного названия
_WATT_SUFFIX = re.compile(r"\s*\d+\s*W(?:att)?\b", re.IGNORECASE)

# Хвостовые слова которые обрезаем из конца модельного названия
# (сертификаты, рейтинги, ATX-тип)
# ВАЖНО: PCIE5/PCIE4 НЕ убираем — они часть модельных кодов (MPG A850G PCIE5)
_TRAILING_NOISE = re.compile(
    r"\s*(80\+?\s*(?:Gold|Silver|Bronze|Platinum|Titanium)?|Gold|Silver|Bronze|Platinum|Titanium|ATX|SFX|Modular|Fully|Semi|Gen\.?\s*\d+|RGB|ARGB)\s*$",
    re.IGNORECASE,
)


def _strip_brand_and_prefix(product_name: str, brand: str) -> str:
    """Убрать бренд и обобщённые префиксы из названия продукта.

    Пример: "Блок питания ASUS ROG STRIX 850G" → "ROG STRIX 850G"
    """
    result = product_name.strip()
    brand_lower = brand.strip().lower()

    # Убираем русские/английские обобщённые префиксы
    for prefix in sorted(_GENERIC_PREFIX_WORDS, key=len, reverse=True):
        if result.lower().startswith(prefix):
            result = result[len(prefix):].strip()
            break

    # Убираем бренд если стоит в начале
    if result.lower().startswith(brand_lower):
        result = result[len(brand):].strip()

    return result.strip()


def _extract_model_tokens(model_part: str, brand: str) -> list[str]:
    """Извлечь значимые токены из модельной части названия.

    Исключаем: стоп-слова, сам бренд, чисто цифровые токены, слишком короткие.
    """
    brand_lower = brand.lower()
    tokens: list[str] = []
    seen: set[str] = set()

    for match in _TOKEN_PATTERN.finditer(model_part):
        token = match.group()
        token_lower = token.lower()

        if token_lower == brand_lower:
            continue
        if token_lower in _STOPWORDS:
            continue
        # Пропускаем чисто цифровые (типа "850", "750")
        if token.isdigit():
            continue
        # Пропускаем слишком короткие без букв
        if len(token) < 3:
            continue

        if token not in seen:
            seen.add(token)
            tokens.append(token)

    return tokens


def _clean_model_name(model_part: str) -> str:
    """Очистить модельную часть от ватт-суффикса и хвостовых шумовых слов.

    Пример: "MPG A850G PCIE5 850W 80+ Gold" → "MPG A850G PCIE5"
    Пример: "ROG STRIX 850G 850W 80+ Gold ATX" → "ROG STRIX 850G"
    """
    result = model_part.strip()
    # Итерационно убираем хвостовые шумовые паттерны
    changed = True
    while changed:
        prev = result
        result = _WATT_SUFFIX.sub("", result).strip()
        result = _TRAILING_NOISE.sub("", result).strip()
        changed = result != prev
    return result.strip()


def _build_code_candidates(product_name: str, brand: str) -> list[str]:
    """Построить список кандидатов product code из product_name.

    Стратегии (в порядке убывания вероятности совпадения):
      1. Чистое имя модели без ватт и рейтингов (напр. "MPG A850G PCIE5", "ROG STRIX 850G")
      2. Дефисованный вариант из токенов: "ROG-STRIX-850G"
      3. Полное имя модели со всеми словами (запасной вариант)
      4. Каждый значимый токен отдельно (напр. "UD850GM", "PM650D")

    Возвращаем до _MAX_CANDIDATES уникальных кандидатов.
    """
    # Убираем бренд и обобщённые слова
    model_part = _strip_brand_and_prefix(product_name, brand)

    # Чистое имя модели (без ватт и хвостового шума)
    clean_model = _clean_model_name(model_part)

    candidates: list[str] = []
    seen: set[str] = set()

    def add(code: str) -> None:
        code = code.strip()
        if code and code not in seen and len(code) >= 3:
            seen.add(code)
            candidates.append(code)

    # 1. Чистое имя модели (без ватт и рейтингов) — наиболее вероятное совпадение
    add(clean_model)

    # 2. Дефисованный вариант из значимых токенов (ROG-STRIX-850G)
    tokens = _extract_model_tokens(clean_model, brand)
    if tokens:
        add("-".join(tokens))

    # 3. Полное имя модели как-есть (с ваттами, может совпасть для некоторых брендов)
    if model_part != clean_model:
        add(model_part)

    # 4. Токены с ваттами (для кандидатов типа ROG-STRIX-850G-850W)
    tokens_with_watts = _extract_model_tokens(model_part, brand)
    if tokens_with_watts and tokens_with_watts != tokens:
        add("-".join(tokens_with_watts))

    # 5. Каждый значимый токен отдельно (только длинные >= 5 символов)
    for token in tokens_with_watts:
        if len(token) >= 5:
            add(token)

    return candidates[:_MAX_CANDIDATES]


# Оставляем для обратной совместимости (используется в тестах)
def _extract_code_candidates(product_name: str, brand: str) -> list[str]:
    """Извлечь кандидатов product code (обёртка над _build_code_candidates).

    Оставлен для обратной совместимости с тестами.
    """
    return _build_code_candidates(product_name, brand)


# ---------------------------------------------------------------------------
# Pydantic модель для MPN извлечения (структурированный ответ LLM)
# ---------------------------------------------------------------------------

class _MpnResponse(BaseModel):
    """Структурированный ответ LLM при поиске MPN производителя."""
    mpn: Optional[str] = None


class IceCatSource(AttributeSource):
    """Извлекает характеристики продукта из IceCat Open API.

    Brand-verified данные: ASUS ROG, be quiet!, MSI, Gigabyte, HP, Lenovo и т.д.
    Работает без LLM — прямой HTTP запрос к IceCat API.

    Кэш трёх уровней:
      - per-(brand, product_name): результат всего поиска (избегаем повторных попыток)
      - per-(brand, code): результат конкретного HTTP запроса к API
      - per-(brand, product_name): результат MPN lookup (избегаем повторных Serper+LLM вызовов)

    IceCat API не имеет текстового поиска в Open tier.
    Используем многоуровневую стратегию генерации кандидатов ProductCode.
    Если все кандидаты 404 → MPN lookup через Serper + LLM.

    Порядок в pipeline: после DescriptionSource (бренд уже извлечён),
    до LlmKnowledgeSource (brand-verified лучше LLM-памяти).
    """

    def __init__(
        self,
        email: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 20,
        serper_client=None,
        mpn_llm_manager=None,
    ):
        # Читаем из env если не переданы напрямую
        self._email = email or os.environ.get("ICECAT_EMAIL", "")
        self._token = token or os.environ.get("ICECAT_TOKEN", "") or os.environ.get("ICECAT_CONTENT_ACCESS_TOKEN", "")
        self._timeout = timeout
        self._judge = IceCatJudge()
        # Кэш HTTP запросов: (brand, code) → list[tuple[str, str]] или "403" или "404"
        self._cache: dict[tuple[str, str], list[tuple[str, str]] | str] = {}
        # Кэш результатов поиска: (brand_lower, name_lower) → list[tuple[str,str]] | None | "403"
        # None = не найдено, "403" = бренд заблокирован
        self._search_cache: dict[tuple[str, str], list[tuple[str, str]] | None | str] = {}
        # Кэш MPN lookup: (brand_lower, name_lower) → str (MPN) или None (не найден)
        # Sentinel: ключ присутствует, значение None → поиск уже делался, MPN не найден
        self._mpn_cache: dict[tuple[str, str], Optional[str]] = {}
        # Клиент Serper и LLM для MPN lookup — lazy init при первом вызове если не заданы
        self._serper_client = serper_client  # можно передать явно (для тестов)
        self._mpn_llm_manager = mpn_llm_manager  # можно передать явно (для тестов)

    @property
    def source_type(self) -> Source:
        return Source.ICECAT

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим если есть бренд и product_name достаточной длины."""
        return bool(
            context.brand and context.brand.strip()
            and context.product_name and len(context.product_name.strip()) >= 5
        )

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Извлечь характеристики из IceCat API и смапить на targets.

        Алгоритм:
        1. Пропустить если нет бренда.
        2. Генерировать кандидатов product code: полное имя модели, дефисованный, токены.
        3. Для каждого кандидата: запросить IceCat API.
           - 200 → парсим, останавливаемся.
           - 403 → бренд в Full tier, логируем, возвращаем [].
           - 404 → продукт не найден, пробуем следующий.
        4. Парсим FeaturesGroups → flat list (name, value).
        5. Маппируем через fuzzy/cosine на targets.
        6. Пропускаем already_filled.
        """
        if not (context.brand and context.brand.strip()):
            logger.debug("[IceCat] нет бренда для product_id=%s", context.product_id)
            return []

        if not targets:
            return []

        brand = context.brand.strip()

        # Определяем уже заполненные — только high-confidence (is_confident()).
        # Source threshold для IceCat = 0.90. Если ранее запущенный source
        # положил attr с conf < 0.90 (например OzonCard brand_line 0.85),
        # IceCat ВСЁ РАВНО пытается заполнить — потом AttributeMerger выберет
        # winner по confidence. Раньше naive set membership блокировал IceCat
        # даже на низкоуверенных prior fills, теряя его 0.92 точность.
        already_filled_ids: set[int] = set()
        if already_filled:
            already_filled_ids = {av.attribute_id for av in already_filled if av.is_confident()}

        effective_targets = [t for t in targets if t.id not in already_filled_ids]
        if not effective_targets:
            return []

        # Кэш на уровне (brand, product_name) — повторные вызовы для одного продукта
        search_key = (brand.lower(), context.product_name.lower())
        if search_key in self._search_cache:
            cached = self._search_cache[search_key]
            if cached is None:
                return []
            if cached == "403":
                return []
            features = cached
        else:
            features = await self._search_and_fetch(brand, context.product_name, context=context)
            self._search_cache[search_key] = features if features != "403" else "403"

        if not features or features == "403":
            return []

        # Маппируем IceCat features на targets
        return self._map_features_to_targets(features, effective_targets)

    async def _search_and_fetch(
        self,
        brand: str,
        product_name: str,
        context: Optional[ExtractionContext] = None,
    ) -> list[tuple[str, str]] | None | str:
        """Перебрать кандидатов product code и вернуть первый успешный результат.

        Стратегия:
          0. Если context.mpn задан (например, обогащён Vision из фото) — пробуем его
             ПЕРВЫМ. Это бесплатный точный MPN, экономит Serper+LLM lookup.
          1. Генерируем кандидатов из product_name (многоуровневая эвристика).
          2. Для каждого кандидата: GET IceCat API.
             - 200 → возвращаем features.
             - 403 → бренд в Full tier, прекращаем.
             - 404 → пробуем следующий.
          3. Если все 404 → MPN Lookup: Serper + LLM извлекают реальный ProductCode.
             Если MPN найден → повторяем _fetch_features(brand, mpn).

        Возвращает:
          - list[tuple[str, str]] при первом 200 (список (feature_name, value))
          - "403" при 403 (бренд в Full tier) — прекращаем все попытки
          - None если ни один кандидат не дал 200
        """
        # Step 0: если context принёс точный MPN (например, Vision прочитал с коробки) —
        # пробуем его первым. Это free shortcut, минует и эвристики, и платный Serper+LLM lookup.
        context_mpn = (context.mpn.strip() if context and context.mpn else "")
        if context_mpn:
            logger.info(
                "[IceCat] using context.mpn='%s' for brand='%s' (skip heuristics + LLM lookup)",
                context_mpn, brand,
            )
            result = await self._fetch_features(brand, context_mpn)
            if result == "403":
                closed_brands[brand] += 1
                logger.info(
                    "[IceCat] 403 brand='%s' — Full IceCat only (не в Open tier)",
                    brand,
                )
                return "403"
            if isinstance(result, list):
                open_brands[brand] += 1
                logger.info(
                    "[IceCat] 200 brand='%s' mpn='%s' → %d features (via context.mpn)",
                    brand, context_mpn, len(result),
                )
                return result
            # 404 на context.mpn — продолжаем эвристический поиск как fallback
            logger.debug(
                "[IceCat] 404 на context.mpn='%s', продолжаем эвристический поиск",
                context_mpn,
            )

        # Генерируем кандидатов по многоуровневой стратегии
        candidates = _build_code_candidates(product_name, brand)
        if not candidates:
            logger.debug("[IceCat] нет кандидатов для '%s'", product_name[:60])
            return None

        logger.debug(
            "[IceCat] кандидаты для '%s': %s",
            product_name[:60],
            candidates[:5],
        )

        for code in candidates:
            result = await self._fetch_features(brand, code)
            if result == "403":
                # Бренд не в Open IceCat — логируем и прекращаем
                closed_brands[brand] += 1
                logger.info(
                    "[IceCat] 403 brand='%s' — Full IceCat only (не в Open tier)",
                    brand,
                )
                return "403"
            elif result == "404":
                logger.debug("[IceCat] 404 brand='%s' code='%s' — не найден", brand, code)
                continue  # пробуем следующий кандидат
            elif isinstance(result, list):
                open_brands[brand] += 1
                logger.info(
                    "[IceCat] 200 brand='%s' code='%s' → %d features",
                    brand, code, len(result),
                )
                return result

        logger.debug(
            "[IceCat] не найден ни один кандидат для '%s' (пробовали %d вариантов)",
            product_name[:60],
            len(candidates),
        )

        # Все кандидаты дали 404 → пробуем MPN Lookup через Serper + LLM
        mpn = await self.lookup_mpn(brand, product_name)
        if mpn:
            logger.info(
                "[IceCat] MPN lookup нашёл '%s' для '%s', повторяем запрос",
                mpn,
                product_name[:60],
            )
            result = await self._fetch_features(brand, mpn)
            if isinstance(result, list):
                open_brands[brand] += 1
                logger.info(
                    "[IceCat] 200 brand='%s' mpn='%s' → %d features (via MPN lookup)",
                    brand, mpn, len(result),
                )
                return result
            else:
                logger.debug(
                    "[IceCat] MPN '%s' также дал %s для brand='%s'",
                    mpn, result, brand,
                )

        return None

    async def lookup_mpn(self, brand: str, product_name: str) -> Optional[str]:
        """Найти реальный MPN (Manufacturer Part Number) через Serper + LLM.

        Когда IceCat не находит продукт по эвристическим кандидатам, ищем
        настоящий ProductCode производителя (напр. "90YE00A4-B0NA00" для ASUS ROG STRIX 850G).

        Алгоритм:
          1. Serper search: "{brand} {product_name} MPN Part Number"
          2. Берём сниппеты топ-5 результатов
          3. LLM (DeepSeek cheap) извлекает MPN из сниппетов → _MpnResponse
          4. Возвращаем mpn или None

        Кэш per-(brand_lower, name_lower): повторный вызов для одного продукта
        не делает новых сетевых запросов даже если предыдущий вернул None.

        Если SERPER_API_KEY не задан → graceful None (не кидаем исключение).
        """
        mpn_key = (brand.lower(), product_name.lower())
        # Проверяем кэш (sentinel: ключ присутствует → поиск уже был)
        if mpn_key in self._mpn_cache:
            return self._mpn_cache[mpn_key]

        from app import config as _cfg

        # Проверяем наличие Serper API ключа — graceful skip если не задан
        serper_key = _cfg.SERPER_API_KEY
        if not serper_key:
            logger.debug("[IceCat/MPN] SERPER_API_KEY не задан — MPN lookup пропущен для '%s'", product_name[:60])
            self._mpn_cache[mpn_key] = None
            return None

        # Инициализируем Serper клиент если не был передан явно
        if self._serper_client is None:
            try:
                from app.services.providers.serper_client import SerperClient
                self._serper_client = SerperClient(api_key=serper_key)
            except Exception as exc:
                logger.warning("[IceCat/MPN] не удалось создать SerperClient: %s", exc)
                self._mpn_cache[mpn_key] = None
                return None

        # Инициализируем LLM менеджер если не был передан явно
        if self._mpn_llm_manager is None:
            try:
                from app.services.providers.factory import get_main_manager
                self._mpn_llm_manager = get_main_manager()
            except Exception as exc:
                logger.warning("[IceCat/MPN] не удалось создать LLM manager: %s", exc)
                self._mpn_cache[mpn_key] = None
                return None

        # Запрос Serper: ищем MPN производителя
        query = f"{brand} {product_name} MPN Part Number"
        try:
            search_results = await asyncio.wait_for(
                self._serper_client.search(query, num_results=5),
                timeout=15,
            )
        except asyncio.TimeoutError:
            logger.warning("[IceCat/MPN] Serper timeout для '%s'", product_name[:60])
            self._mpn_cache[mpn_key] = None
            return None
        except Exception as exc:
            logger.warning("[IceCat/MPN] Serper ошибка для '%s': %s", product_name[:60], exc)
            self._mpn_cache[mpn_key] = None
            return None

        if not search_results.organic_results:
            logger.debug("[IceCat/MPN] Serper не вернул результатов для '%s'", product_name[:60])
            self._mpn_cache[mpn_key] = None
            return None

        # Собираем сниппеты топ-5 результатов
        snippets = [
            f"[{r.position}] {r.title}\n{r.snippet}"
            for r in search_results.organic_results[:5]
        ]
        search_context = "\n\n".join(snippets)

        # LLM prompt: извлечь MPN из сниппетов
        system_prompt = (
            "Ты — эксперт по спецификациям электронных товаров. "
            "Твоя задача: найти MPN (Manufacturer Part Number) или артикул производителя "
            "в предоставленных фрагментах веб-страниц. "
            "MPN — это буквенно-цифровой код вида 90YE00A4-B0NA00, MPY-750G-AFAAG-EU, "
            "CP-9020264-EU и т.п. (обычно содержит дефисы, буквы и цифры). "
            "Верни ТОЛЬКО JSON с полем 'mpn': сам код или null если не нашёл."
        )
        user_text = (
            f"Товар: {product_name}\n"
            f"Бренд: {brand}\n\n"
            f"Фрагменты веб-поиска:\n{search_context}\n\n"
            "Из этих сниппетов найди Manufacturer Part Number (MPN) или артикул производителя. "
            "Это код типа 90YE00A4-B0NA00 или MPY-750G-AFAAG-EU. "
            "Верни только сам код. Если не нашёл — null."
        )

        try:
            parsed, _tokens = await self._mpn_llm_manager.structured_request(
                system_prompt=system_prompt,
                user_text=user_text,
                response_model=_MpnResponse,
            )
        except Exception as exc:
            logger.warning("[IceCat/MPN] LLM ошибка при извлечении MPN для '%s': %s", product_name[:60], exc)
            self._mpn_cache[mpn_key] = None
            return None

        mpn: Optional[str] = None
        if parsed is not None and parsed.mpn:
            mpn = parsed.mpn.strip() or None

        logger.info(
            "[IceCat/MPN] brand='%s' product='%s' → mpn=%r",
            brand, product_name[:60], mpn,
        )
        self._mpn_cache[mpn_key] = mpn
        return mpn

    async def _fetch_features(
        self,
        brand: str,
        code: str,
    ) -> list[tuple[str, str]] | str:
        """Запросить IceCat API для конкретного (brand, code).

        Endpoint: GET https://live.icecat.biz/api
        Параметры: UserName, content_token, lang, Brand, ProductCode
        Альтернатива: icecat_id=<int> вместо Brand+ProductCode

        Возвращает:
          - list[tuple[str, str]] при 200 (список (feature_name, value))
          - "403" при 403 (бренд в Full tier)
          - "404" при 404 / пустом ответе / любой ошибке (graceful)

        Результат кэшируется на уровне экземпляра.
        """
        cache_key = (brand.lower(), code.lower())
        if cache_key in self._cache:
            return self._cache[cache_key]

        import aiohttp  # lazy import

        url = "https://live.icecat.biz/api"
        params = {
            "UserName": self._email,
            "lang": "ru",
            "Brand": brand,
            "ProductCode": code,
            "content_token": self._token,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=self._timeout)) as resp:
                    if resp.status == 403:
                        self._cache[cache_key] = "403"
                        return "403"
                    if resp.status == 404:
                        self._cache[cache_key] = "404"
                        return "404"
                    if resp.status != 200:
                        logger.warning(
                            "[IceCat] неожиданный HTTP %d для brand='%s' code='%s'",
                            resp.status, brand, code,
                        )
                        self._cache[cache_key] = "404"
                        return "404"

                    data = await resp.json(content_type=None)
        except Exception as e:
            logger.warning("[IceCat] HTTP ошибка для brand='%s' code='%s': %s", brand, code, e)
            return "404"

        features = self._parse_response(data)
        self._cache[cache_key] = features
        return features

    def _parse_response(self, data: dict) -> list[tuple[str, str]]:
        """Распарсить JSON-ответ IceCat API в flat list (feature_name, value).

        Структура ответа:
          data.FeaturesGroups[] → FeatureGroup.Name.Value + Features[].{Feature.Name.Value, PresentationValue}
        """
        features: list[tuple[str, str]] = []

        try:
            payload = data.get("data", {})
            groups = payload.get("FeaturesGroups", []) or []

            for group in groups:
                group_features = group.get("Features", []) or []
                for feat in group_features:
                    feat_name = (feat.get("Feature", {}) or {}).get("Name", {}) or {}
                    name = feat_name.get("Value", "").strip()
                    value = str(feat.get("PresentationValue", "") or "").strip()

                    if name and value and value.lower() not in ("n/a", "-", ""):
                        features.append((name, value))

        except Exception as e:
            logger.warning("[IceCat] ошибка парсинга ответа: %s", e)

        return features

    def _map_features_to_targets(
        self,
        features: list[tuple[str, str]],
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Смапить IceCat features на TargetAttribute через fuzzy/semantic matching.

        Для каждой IceCat (name, value) ищем ближайший target по имени.

        EN↔RU bilingual normalization (LEVER 1):
          Перед fuzzy-матчингом IceCat-имя прогоняется через _bilingual_normalize().
          Если IceCat вернул EN-имя (напр. "Power"), функция транслирует его в RU
          ("мощность") через компактный general EN→RU словарь (не per-category).
          Затем фаззи-матч работает на одноязычных строках, находя совпадения
          которые ранее терялись из-за языковой разницы.

        Stated-negative ("Нет") для bool-атрибутов (LEVER 2):
          Включается только при ICECAT_STATED_NEGATIVE_ENABLED=1 (по умолчанию OFF).
          Если IceCat явно указывает отсутствие признака (verbatim "No"/"Нет"/...),
          заполняем "Нет" для boolean-цели. НИКОГДА не выводим "Нет" из отсутствия
          данных — только из явного verbatim-отрицания.

        Один target может получить только одно значение (первое совпадение с наивысшим score).
        """
        if not targets or not features:
            return []

        target_names = [t.name for t in targets]
        target_by_name = {t.name: t for t in targets}

        # Пробуем использовать rapidfuzz для fuzzy matching (без тяжёлых зависимостей)
        try:
            from rapidfuzz import fuzz, process as fuzz_process
            _has_rapidfuzz = True
        except ImportError:
            _has_rapidfuzz = False

        results: list[AttributeValue] = []
        used_target_ids: set[int] = set()  # каждый target заполняем только один раз

        for icecat_name, value in features:
            if not icecat_name or not value:
                continue

            # LEVER 2: stated-negative guard for bool targets (flag-gated OFF by default)
            value_lower_stripped = value.strip().lower()
            is_stated_negative = value_lower_stripped in _ICECAT_VERBATIM_NEGATIVES
            if is_stated_negative and not _ICECAT_STATED_NEGATIVE_ENABLED:
                # Explicit negatives are dropped when the feature flag is OFF.
                # We don't want to incorrectly skip non-negative values that
                # happen to partially match, so only skip verbatim negatives here.
                logger.debug(
                    "[IceCat] stated-negative skipped (flag OFF): attr=%r value=%r",
                    icecat_name, value,
                )
                continue

            # LEVER 1: bilingual normalization — translate EN attr name to RU
            # before fuzzy matching so EN IceCat attrs match RU targets.
            normalized_name = _bilingual_normalize(icecat_name)
            if normalized_name != icecat_name:
                logger.debug(
                    "[IceCat] bilingual normalize: %r → %r",
                    icecat_name, normalized_name,
                )

            matched_target: Optional[TargetAttribute] = None

            if _has_rapidfuzz:
                # First attempt: match with the (possibly translated) normalized name.
                match = fuzz_process.extractOne(
                    normalized_name,
                    target_names,
                    scorer=fuzz.WRatio,
                    score_cutoff=65,
                )
                # For short translated queries (≤8 chars) WRatio under-scores when
                # the translated word is a substring of a longer target name.
                # Use partial_ratio as a secondary scorer to catch these cases
                # (e.g. "вес" → "Вес, кг" scores 67 with partial_ratio vs ~60 WRatio).
                if match is None and len(normalized_name) <= 8:
                    match = fuzz_process.extractOne(
                        normalized_name,
                        target_names,
                        scorer=fuzz.partial_ratio,
                        score_cutoff=65,
                    )
                # Third attempt: if normalized name did not improve score, also try
                # the original IceCat name (guards against over-aggressive translation).
                if match is None and normalized_name != icecat_name:
                    match = fuzz_process.extractOne(
                        icecat_name,
                        target_names,
                        scorer=fuzz.WRatio,
                        score_cutoff=65,
                    )
                if match:
                    matched_name = match[0]
                    matched_target = target_by_name.get(matched_name)
            else:
                # Fallback: простой lowercase substring match (try translated first)
                for query in (normalized_name, icecat_name):
                    q_lower = query.lower()
                    for t in targets:
                        if (
                            t.name.lower() == q_lower
                            or t.name.lower() in q_lower
                            or q_lower in t.name.lower()
                        ):
                            matched_target = t
                            break
                    if matched_target is not None:
                        break

            if matched_target is None:
                continue
            if matched_target.id in used_target_ids:
                continue  # уже заполнили этот target

            # LEVER 2: stated-negative fill for boolean targets
            # Only when flag is ON AND value is a verbatim explicit negative.
            if is_stated_negative:
                # Only fill "Нет" for bool-type targets.
                if matched_target.type not in ("bool", "boolean"):
                    logger.debug(
                        "[IceCat] stated-negative skipped (target not bool): "
                        "attr=%r type=%r value=%r",
                        matched_target.name, matched_target.type, value,
                    )
                    continue
                logger.info(
                    "[IceCat] stated-negative fill: attr=%r value=%r → Нет",
                    matched_target.name, value,
                )
                used_target_ids.add(matched_target.id)
                results.append(AttributeValue(
                    attribute_id=matched_target.id,
                    value="Нет",
                    confidence=_ICECAT_CONFIDENCE,
                    source=Source.ICECAT,
                    evidence=f"icecat:stated_negative:{icecat_name}={value}",
                    semantic_type=matched_target.semantic_type,
                    is_collection=matched_target.is_collection,
                ))
                continue

            used_target_ids.add(matched_target.id)
            # Normalize numeric values: strip units, convert if needed
            # (e.g. "6,3 kg" → "6.3", "68,6 cm (27\")" → "27", "165 Hz" → "165")
            normalized_value = normalize_icecat_numeric(matched_target.name, value)
            if normalized_value != value:
                logger.debug(
                    "[IceCat] numeric normalize: attr=%r raw=%r → %r",
                    matched_target.name, value, normalized_value,
                )
            results.append(AttributeValue(
                attribute_id=matched_target.id,
                value=normalized_value,
                confidence=_ICECAT_CONFIDENCE,
                source=Source.ICECAT,
                evidence=f"icecat:{icecat_name}={value}",
                semantic_type=matched_target.semantic_type,
                is_collection=matched_target.is_collection,
            ))

        return results

    def get_judge(self) -> LlmJudge:
        return self._judge
