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

logger = logging.getLogger(__name__)

# Confidence для всех IceCat-значений (brand-verified данные высокого качества)
_ICECAT_CONFIDENCE = 0.92

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

        # Определяем уже заполненные
        already_filled_ids: set[int] = set()
        if already_filled:
            already_filled_ids = {av.attribute_id for av in already_filled}

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
            features = await self._search_and_fetch(brand, context.product_name)
            self._search_cache[search_key] = features if features != "403" else "403"

        if not features or features == "403":
            return []

        # Маппируем IceCat features на targets
        return self._map_features_to_targets(features, effective_targets)

    async def _search_and_fetch(
        self,
        brand: str,
        product_name: str,
    ) -> list[tuple[str, str]] | None | str:
        """Перебрать кандидатов product code и вернуть первый успешный результат.

        Стратегия:
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
        Используем rapidfuzz + cosine similarity (через MatcherService если доступен).
        Если совпадение найдено — эмитируем AttributeValue.

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

            matched_target: Optional[TargetAttribute] = None

            if _has_rapidfuzz:
                # rapidfuzz: WRatio хорошо работает с русскими строками разной длины
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
                # Fallback: простой lowercase substring match
                icecat_lower = icecat_name.lower()
                for t in targets:
                    if t.name.lower() == icecat_lower or t.name.lower() in icecat_lower or icecat_lower in t.name.lower():
                        matched_target = t
                        break

            if matched_target is None:
                continue
            if matched_target.id in used_target_ids:
                continue  # уже заполнили этот target

            used_target_ids.add(matched_target.id)
            results.append(AttributeValue(
                attribute_id=matched_target.id,
                value=value,
                confidence=_ICECAT_CONFIDENCE,
                source=Source.ICECAT,
                evidence=f"icecat:{icecat_name}={value}",
                semantic_type=matched_target.semantic_type,
                is_collection=matched_target.is_collection,
            ))

        return results

    def get_judge(self) -> LlmJudge:
        return self._judge
