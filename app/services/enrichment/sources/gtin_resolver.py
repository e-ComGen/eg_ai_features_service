"""GTINResolver — резолвер имя→EAN через Serper.

Находит штрихкод (EAN-13 / UPC-A / EAN-8) для товара по brand + product_name,
используя Serper Google Search API.  Результат кладётся в context.ean, после
чего IceCatSource использует его в Step 4 (_fetch_features_by_gtin).

Логика:
  1. Строит Serper-запрос: '<brand> <product_name> штрихкод EAN'
  2. Сканирует title+snippet всех органических результатов на 8/12/13-значные числа.
  3. Валидирует checksum (EAN-13 / UPC-A / EAN-8) — переиспользует validate_barcode().
  4. Выбирает наиболее частый валидный кандидат (tie → первый).
  5. Кэш по (brand, model): одинаковый продукт не ищется дважды в рамках процесса.
  6. При ошибке / отсутствии ключа / пустой выдаче → graceful None.

Активация:
  GTIN_RESOLVE_ENABLED=1  (env-флаг, по умолчанию off).

Интеграция в pipeline:
  Stage 0.54 — после BarcodeSource (0.46), перед IceCat (0.55).
  Не заполняет никаких AttributeValue; только обогащает context.ean.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from typing import Optional

from app.services.enrichment.base import ExtractionContext
from app.services.enrichment.sources.barcode_source import validate_barcode
from app.services.providers.factory import get_web_search_client

logger = logging.getLogger(__name__)

# Regex: 8, 12 или 13 цифр, не обрамлённых другими цифрами.
# Захватываем только сам числовой блок (группа 1) для длинно-блочной фильтрации.
_DIGIT_RE = re.compile(r"(?<!\d)(\d{8}|\d{12}|\d{13})(?!\d)")

# Максимальное количество organic-результатов для сканирования.
_SERPER_NUM_RESULTS = 5

# In-process LRU-подобный кэш: (brand_lower, model_lower) → EAN | None
_CACHE: dict[tuple[str, str], Optional[str]] = {}


def _extract_gtin_candidates(text: str) -> list[str]:
    """Собирает все 8/12/13-значные числовые кандидаты из строки.

    Не фильтрует по checksum — caller должен вызвать validate_barcode().
    """
    return [m.group(1) for m in _DIGIT_RE.finditer(text)]


def _pick_best_candidate(candidates: list[str]) -> Optional[str]:
    """Выбирает наиболее частый checksum-валидный кандидат.

    Среди всех кандидатов находим те, что прошли validate_barcode().
    Из них берём наиболее часто встречающийся (tie-break: первый по порядку).
    """
    valid: list[str] = [c for c in candidates if validate_barcode(c)]
    if not valid:
        return None
    # Counter сохраняет порядок first-seen при равном count (Python 3.7+).
    counter = Counter(valid)
    return counter.most_common(1)[0][0]


async def resolve_gtin(context: ExtractionContext) -> Optional[str]:
    """Возвращает EAN-13/UPC-A/EAN-8 для товара или None при неудаче.

    Делает ровно 1 Serper-вызов.  Результат кэшируется по (brand, product_name).

    Args:
        context: ExtractionContext с заполненными brand и product_name.

    Returns:
        Checksum-валидный EAN-строкой или None.
    """
    brand = (context.brand or "").strip()
    product_name = (context.product_name or "").strip()
    if not product_name:
        return None

    cache_key = (brand.lower(), product_name.lower())
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    try:
        client = get_web_search_client()
    except Exception as exc:
        logger.warning("[GTINResolver] Не удалось получить Serper client: %s", exc)
        _CACHE[cache_key] = None
        return None

    if client is None:
        logger.debug("[GTINResolver] Serper client недоступен (нет SERPER_API_KEY).")
        _CACHE[cache_key] = None
        return None

    query_parts = [p for p in (brand, product_name) if p]
    query = " ".join(query_parts) + " штрихкод EAN"

    try:
        results = await client.search(query, num_results=_SERPER_NUM_RESULTS)
    except Exception as exc:
        logger.warning("[GTINResolver] Serper search failed для %r: %s", query, exc)
        _CACHE[cache_key] = None
        return None

    if not results.organic_results:
        logger.debug("[GTINResolver] Пустая выдача Serper для %r.", query)
        _CACHE[cache_key] = None
        return None

    # Сканируем title + snippet каждого результата.
    all_candidates: list[str] = []
    for r in results.organic_results:
        text = f"{r.title} {r.snippet}"
        all_candidates.extend(_extract_gtin_candidates(text))

    ean = _pick_best_candidate(all_candidates)

    if ean:
        logger.info(
            "[GTINResolver] Найден EAN %s для brand=%r product=%r (query=%r)",
            ean, brand, product_name, query,
        )
    else:
        logger.debug(
            "[GTINResolver] EAN не найден для %r (кандидатов: %d, валидных: 0).",
            query, len(all_candidates),
        )

    _CACHE[cache_key] = ean
    return ean
