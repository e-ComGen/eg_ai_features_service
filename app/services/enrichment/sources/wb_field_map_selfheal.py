"""
LIVE self-heal WB-имя-поля -> Ozon attribute_id через DeepSeek LLM.

Портировано из eg-importer/mapping/field_map_builder.py (self-heal ядро),
адаптировано под родной web-fetch провайдер DeepSeekProvider.

Схема кэша:
- Директория: <repo_root>/data/field_maps/
- Файл: <cache_key>.json
- Ключ: <wb_subject_slug>__<ozon_cat_id>_<ozon_type_id>
- Содержимое: JSON-объект {"<wb_field_name>": <ozon_attr_id_or_null>, ...}
- Запись атомарная (write-then-rename), чтение с fallback на None при ошибках.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

# Путь к корню репозитория: от __file__ (wb_field_map_selfheal.py) поднимаемся:
# sources/ -> enrichment/ -> services/ -> app/ -> корень проекта
# _DATA_DIR = <корень>/data/field_maps/
_DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "field_maps"

# Административные WB-поля, которые не нужно маппить (lowercase)
_WB_SKIP_FIELDS: frozenset[str] = frozenset([
    "высота упаковки",
    "длина упаковки",
    "ширина упаковки",
    "вес упаковки",
    "вес с упаковкой",
    "вес с упаковкой (кг)",
    "группа",
    "киз",
    "18+",
    "только для ип и юрлиц",
    "подтверждаю что товар промаркирован",
    "предмет",
    "предмет 1",
    "предмет 2",
    "количество штук в упаковке",
    "баркоды",
    "код тн вэд",
    "цена",
    "скидка",
])


def _cache_key(
    wb_subject: Optional[str],
    ozon_cat_id: int,
    ozon_type_id: int,
) -> str:
    """Формирует безопасный ключ кэша из subject и идентификаторов Ozon."""
    subject_slug = re.sub(r"[^\w\-]", "_", (wb_subject or "_none_").lower())
    return f"{subject_slug}__{ozon_cat_id}_{ozon_type_id}"


def _cache_path(
    wb_subject: Optional[str],
    ozon_cat_id: int,
    ozon_type_id: int,
) -> Path:
    """Возвращает путь к файлу кэша для заданных параметров."""
    return _DATA_DIR / f"{_cache_key(wb_subject, ozon_cat_id, ozon_type_id)}.json"


def _load_cache(
    wb_subject: Optional[str],
    ozon_cat_id: int,
    ozon_type_id: int,
) -> Optional[dict[str, Optional[int]]]:
    """Загружает кэш из JSON-файла. При ошибке возвращает None."""
    path = _cache_path(wb_subject, ozon_cat_id, ozon_type_id)
    if not path.exists():
        return None
    try:
        data = path.read_text(encoding="utf-8")
        return json.loads(data)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "[WbFieldMapSelfHeal] Failed to load cache from %s: %s",
            path,
            exc,
        )
        return None


def _save_cache(
    wb_subject: Optional[str],
    ozon_cat_id: int,
    ozon_type_id: int,
    field_map: dict,
) -> None:
    """Атомарно сохраняет field_map в кэш. Ошибки логируются, исключения не пробрасываются."""
    path = _cache_path(wb_subject, ozon_cat_id, ozon_type_id)
    tmp_path = path.with_suffix(".tmp")
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            json.dumps(field_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except OSError as exc:
        logger.warning(
            "[WbFieldMapSelfHeal] Failed to save cache to %s: %s",
            path,
            exc,
        )
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _build_prompt(
    missing_wb_fields: dict[str, str],
    ozon_attrs: list[dict],
    wb_subject: str,
    ozon_cat_id: int,
    ozon_type_id: int,
) -> str:
    """Формирует промпт для DeepSeek на основе отсутствующих полей WB и атрибутов Ozon."""
    wb_lines = []
    for name, sample in missing_wb_fields.items():
        wb_lines.append(f'- "{name}": "{sample}"')

    ozon_lines = []
    for attr in ozon_attrs:
        attr_id = attr.get("id")
        attr_name = attr.get("name", "")
        attr_type = attr.get("type", "String")
        is_collection = attr.get("is_collection", False)
        has_dictionary = bool(attr.get("values"))
        ozon_lines.append(
            f'  [{attr_id}] "{attr_name}" (type={attr_type}, '
            f'collection={str(is_collection).lower()}, '
            f'has_dictionary={str(has_dictionary).lower()})'
        )

    prompt = (
        f"Map the following WB source fields to Ozon target attributes.\n"
        f"WB subject: {wb_subject}\n"
        f"Ozon category ID: {ozon_cat_id}, type ID: {ozon_type_id}\n\n"
        f"WB source fields:\n" + "\n".join(wb_lines) + "\n\n"
        "Ozon target attributes:\n" + "\n".join(ozon_lines) + "\n\n"
        "RULES:\n"
        "- Map by SEMANTIC MEANING and VALUE TYPE, not by name similarity.\n"
        "- If no suitable Ozon attribute exists -> null (do NOT invent, do NOT force).\n"
        "- Never map two different WB fields to the same Ozon attr in the response.\n"
        "- The value shown for each WB field is a SAMPLE, not the only valid value -- map by the field meaning, not by that one example value.\n"
        "- Return ONLY a JSON object: {\"<wb_field_name>\": <ozon_attr_id_or_null>, ...}\n"
        "- Keys in the response MUST be exactly the names given in WB source fields (echo exact).\n"
        "- No explanations, no markdown, no extra text."
    )
    return prompt


def _parse_json_response(raw: str) -> Optional[dict]:
    """Пытается распарсить JSON из ответа LLM. Возвращает dict или None."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _validate_mapping(
    result: dict,
    missing: dict[str, str],
    valid_ozon_ids: set[int],
    claimed_ozon_ids: set[int],
) -> dict[str, Optional[int]]:
    """Валидирует и корректирует маппинг от LLM, обеспечивая 1-к-1 и grounding."""
    new_pairs: dict[str, Optional[int]] = {}
    for wb_field, ozon_id in result.items():
        if wb_field not in missing:
            continue
        if ozon_id is None:
            new_pairs[wb_field] = None
        elif isinstance(ozon_id, (int, float)):
            attr_id = int(ozon_id)
            if attr_id in valid_ozon_ids:
                if attr_id in claimed_ozon_ids:
                    new_pairs[wb_field] = None  # коллизия
                else:
                    new_pairs[wb_field] = attr_id
                    claimed_ozon_ids.add(attr_id)
            else:
                new_pairs[wb_field] = None  # невалидный ID
        else:
            new_pairs[wb_field] = None  # нечисловой ID
    # Поля, которые LLM пропустила -> negative cache
    for k in missing:
        if k not in new_pairs:
            new_pairs[k] = None
    return new_pairs


def _prepare_missing(
    persisted: dict[str, Optional[int]],
    static_map: Mapping[str, Optional[int]],
    wb_fields: dict[str, str],
) -> tuple[dict[str, Optional[int]], dict[str, str]]:
    """Строит combined-маппинг и определяет отсутствующие WB-поля.

    Объединяет persisted (накопленный self-heal-кэш) со static_map
    (static_map имеет приоритет при коллизии ключа), затем фильтрует
    wb_fields до имён, которых ещё нет в combined (исключая административные
    _WB_SKIP_FIELDS).

    Args:
        persisted: Ранее закэшированные маппинги (из _load_cache).
        static_map: Read-only статический маппинг (высший приоритет).
        wb_fields: Все WB-поля-кандидаты на маппинг.

    Returns:
        Кортеж (combined-маппинг, missing-поля).
    """
    combined = {**persisted, **static_map}
    missing = {
        k: v
        for k, v in wb_fields.items()
        if k.strip().lower() not in _WB_SKIP_FIELDS and k not in combined
    }
    return combined, missing


async def _query_deepseek(prompt: str, timeout: int) -> Optional[str]:
    """Запрашивает DeepSeek LLM и возвращает сырой JSON-текст ответа.

    Инкапсулирует вызов + все ошибки (сеть/API/пустой ответ) внутри —
    на любой сбой логирует warning и возвращает None (единый сигнал сбоя
    для caller-а, который тогда идёт по static-only fallback пути).

    Args:
        prompt: Промпт для LLM.
        timeout: Таймаут запроса в секундах.

    Returns:
        Сырой текст ответа при успехе, None при любом сбое.
    """
    from app.services.providers.deepseek_provider import DeepSeekProvider
    from app import config

    try:
        provider = DeepSeekProvider(api_key=config.DEEPSEEK_API_KEY)
        messages = [
            {
                "role": "system",
                "content": "You are a precise data mapping assistant. You output only valid JSON objects with no additional text.",
            },
            {"role": "user", "content": prompt},
        ]
        response = await provider.complete(
            messages=messages,
            model=config.DEEPSEEK_DEFAULT_MODEL,
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        raw = (response.content or "").strip()
        if not raw:
            logger.warning("[WbFieldMapSelfHeal] static-only fallback: empty response")
            return None
        return raw
    except Exception as exc:
        logger.warning(
            "[WbFieldMapSelfHeal] static-only fallback: LLM call failed: %s", exc
        )
        return None


def _finalize_healed(
    parsed: dict,
    missing: dict[str, str],
    ozon_attrs: list[dict],
    claimed_ozon_ids: set[int],
    persisted: dict[str, Optional[int]],
    wb_subject: Optional[str],
    ozon_cat_id: int,
    ozon_type_id: int,
) -> dict[str, Optional[int]]:
    """Валидирует LLM-результат, накопительно сохраняет self-heal-кэш, возвращает new_pairs.

    Args:
        parsed: Распарсенный ответ LLM.
        missing: Словарь отсутствующих WB-полей {имя: пример_значения}.
        ozon_attrs: Список атрибутов Ozon.
        claimed_ozon_ids: Множество уже занятых Ozon attr id (1-к-1 инвариант).
        persisted: Существующий персистентный self-heal-кэш.
        wb_subject: Предмет WB.
        ozon_cat_id: ID категории Ozon.
        ozon_type_id: ID типа Ozon.

    Returns:
        new_pairs: Словарь новых пар {wb_name: ozon_attr_id_or_null}.
    """
    valid_ozon_ids: set[int] = {
        a["id"] for a in ozon_attrs if isinstance(a, dict) and isinstance(a.get("id"), int)
    }
    new_pairs = _validate_mapping(parsed, missing, valid_ozon_ids, claimed_ozon_ids)

    updated_persisted = dict(persisted)
    for k, v in new_pairs.items():
        if k not in updated_persisted:
            updated_persisted[k] = v
    _save_cache(wb_subject, ozon_cat_id, ozon_type_id, updated_persisted)

    return new_pairs


async def self_heal(
    static_map: Mapping[str, Optional[int]],
    wb_fields: dict[str, str],
    ozon_attrs: list[dict],
    wb_subject: Optional[str],
    ozon_cat_id: int,
    ozon_type_id: int,
    timeout: int = 20,
) -> dict[str, Optional[int]]:
    """
    Главная публичная функция self-heal.

    Управляет lifecycle персистентного self-heal-кэша самостоятельно.
    Загружает накопленный кэш с прошлых вызовов, дополняет его новыми
    маппингами для отсутствующих WB-полей через DeepSeek LLM,
    и сохраняет накопленный результат обратно в кэш.

    Args:
        static_map: Read-only верифицированный статический маппинг
                    (из eg_get_field_map). Самим self_heal не персистится.
        wb_fields: Все WB-поля с примерами значений {имя: пример}.
        ozon_attrs: Список атрибутов Ozon (dict с ключами id, name, type, is_collection, values).
        wb_subject: Предмет WB (может быть None).
        ozon_cat_id: ID категории Ozon.
        ozon_type_id: ID типа Ozon.
        timeout: Таймаут для LLM вызова в секундах.

    Returns:
        Комбинированный маппинг: static_map ∪ persisted ∪ новые пары.
        Исходный static_map не модифицируется.
    """
    # 1. Загружаем накопленный self-heal-кэш
    persisted = _load_cache(wb_subject, ozon_cat_id, ozon_type_id) or {}

    # 2. Комбинируем и определяем отсутствующие поля
    combined, missing = _prepare_missing(persisted, static_map, wb_fields)
    if not missing:
        return combined

    # 3. Формируем промпт
    claimed_ozon_ids: set[int] = {v for v in combined.values() if v is not None}
    prompt = _build_prompt(
        missing, ozon_attrs, wb_subject or "", ozon_cat_id, ozon_type_id
    )

    # 4. Вызов LLM
    raw = await _query_deepseek(prompt, timeout)
    if raw is None:
        return combined

    # 5. Парсинг ответа
    parsed = _parse_json_response(raw)
    if parsed is None:
        logger.warning(
            "[WbFieldMapSelfHeal] static-only fallback: failed to parse JSON from LLM response"
        )
        return combined

    # 6-7. Валидация + накопительное сохранение self-heal-кэша
    new_pairs = _finalize_healed(
        parsed, missing, ozon_attrs, claimed_ozon_ids, persisted,
        wb_subject, ozon_cat_id, ozon_type_id,
    )

    # 8. Возвращаем комбинированный маппинг
    return {**combined, **new_pairs}


def get_persisted_cache(
    wb_subject: Optional[str],
    ozon_cat_id: int,
    ozon_type_id: int,
) -> dict[str, Optional[int]]:
    """Возвращает текущий персистентный self-heal-кэш для ключа, или {} если отсутствует/битый."""
    return _load_cache(wb_subject, ozon_cat_id, ozon_type_id) or {}


def is_enabled() -> bool:
    """
    Проверяет, включён ли self-heal.

    Читает переменную окружения WB_CARD_SELFHEAL_ENABLED.
    По умолчанию включено (True).
    """
    value = os.getenv("WB_CARD_SELFHEAL_ENABLED", "true").strip().lower()
    return value not in ("0", "false", "no", "off")


def self_heal_sync(
    static_map: Mapping[str, Optional[int]],
    wb_fields: dict[str, str],
    ozon_attrs: list[dict],
    wb_subject: Optional[str],
    ozon_cat_id: int,
    ozon_type_id: int,
    timeout: int = 20,
) -> dict[str, Optional[int]]:
    """Synchronous bridge around self_heal() — for callers running outside asyncio
    (e.g. WbCardSource._map_characteristics, which stays a plain sync def).
    Mirrors eg-importer's build_field_map() sync wrapper."""
    import asyncio
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        self_heal(static_map, wb_fields, ozon_attrs, wb_subject, ozon_cat_id, ozon_type_id, timeout),
                    )
                    return future.result()
            return loop.run_until_complete(
                self_heal(static_map, wb_fields, ozon_attrs, wb_subject, ozon_cat_id, ozon_type_id, timeout)
            )
        except RuntimeError:
            return asyncio.run(
                self_heal(static_map, wb_fields, ozon_attrs, wb_subject, ozon_cat_id, ozon_type_id, timeout)
            )
    except Exception as exc:
        logger.warning("[WbFieldMapSelfHeal] sync bridge failed: %s — static-only fallback", exc)
        return dict(static_map)
