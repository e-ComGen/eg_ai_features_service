"""Loader for the Ozon category characteristics dictionary.

The dictionary is built by scripts/build_ozon_dictionary.py and stored at
app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json.

Schema v2 (current):
{
    "schema_version": 2,
    "source": "ozon_seller_api",
    "categories": {
        "<description_category_id>:<type_id>": {
            "description_category_id": int,
            "type_id": int,
            "name": str,
            "path": [str, ...],
            "characteristics": [
                {"id": int, "name": str, "type": str,
                 "is_required": bool, "is_collection": bool, "description": str},
                ...
            ]
        }
    }
}
"""
import gzip
import json
from pathlib import Path
from typing import Optional
from functools import lru_cache


DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def load_ozon_dictionary() -> dict:
    """Load the Ozon dictionary from .json.gz (preferred) or plain .json. Cached."""
    gz_path = DATA_DIR / "ozon_dictionary.json.gz"
    plain_path = DATA_DIR / "ozon_dictionary.json"
    if gz_path.exists():
        with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    elif plain_path.exists():
        data = json.loads(plain_path.read_text(encoding="utf-8"))
    else:
        return {}
    # Оба формата (legacy flat и schema_version с оберткой) возвращают одинаково
    if "categories" in data:
        return data["categories"]
    return data


def get_ozon_characteristics_for_type(
    description_category_id: int,
    type_id: int,
) -> list[dict]:
    """Вернуть список характеристик для конкретной (category_id, type_id) пары.

    Используется когда ExtractionContext содержит оба поля (основной путь для v2).
    Возвращает [] если пара не найдена в словаре.
    """
    d = load_ozon_dictionary()
    key = f"{description_category_id}:{type_id}"
    entry = d.get(key)
    return entry["characteristics"] if entry else []


def get_ozon_entries_for_category(description_category_id: int) -> list[dict]:
    """Вернуть все type-записи для данного description_category_id.

    Используется как fallback когда type_id неизвестен — перебираем все type-варианты
    категории и возвращаем характеристики первого найденного типа (обычно type_id
    единственный или характеристики совпадают между типами одной категории).
    """
    d = load_ozon_dictionary()
    prefix = f"{description_category_id}:"
    return [v for k, v in d.items() if k.startswith(prefix)]


def get_ozon_characteristics_for_category(description_category_id: int) -> list[dict]:
    """Вернуть характеристики для category_id (без type_id).

    Совместимый API для случаев когда type_id недоступен в контексте.
    При нескольких type_id для одной категории возвращает характеристики первого
    найденного type — достаточно для normalize_target (обогащение метаданными).

    Также поддерживает legacy-формат (plain '<cat_id>' ключи).
    """
    d = load_ozon_dictionary()
    # Сначала пробуем legacy-формат (ключ = строка category_id)
    entry = d.get(str(description_category_id))
    if entry:
        return entry["characteristics"]
    # Затем v2 compound-ключи — берём первый подходящий type
    entries = get_ozon_entries_for_category(description_category_id)
    return entries[0]["characteristics"] if entries else []


def get_ozon_category_name(description_category_id: int) -> Optional[str]:
    """Вернуть human-readable имя категории, или None."""
    d = load_ozon_dictionary()
    # Legacy-формат
    entry = d.get(str(description_category_id))
    if entry:
        return entry["name"]
    # v2 compound-ключи
    entries = get_ozon_entries_for_category(description_category_id)
    return entries[0]["name"] if entries else None


def resolve_value_id(
    cat_id: int,
    type_id: int,
    attribute_id: int,
    value: str,
) -> Optional[int]:
    """Найти словарный id для строкового значения характеристики.

    Ищет по (cat_id, type_id, attribute_id), затем сравнивает value
    case-insensitive. Возвращает None если список values отсутствует или совпадения нет.
    """
    chars = get_ozon_characteristics_for_type(cat_id, type_id)
    char = next((c for c in chars if c.get("id") == attribute_id), None)
    if not char:
        return None
    values_list = char.get("values")
    if not values_list:
        return None
    value_lower = value.lower()
    for entry in values_list:
        if str(entry.get("value", "")).lower() == value_lower:
            return entry.get("id")
    return None


def is_truncated(cat_id: int, type_id: int, attribute_id: int) -> bool:
    """Return True if the cached values for this attribute were truncated at 5000.

    A characteristic is considered truncated when its dict entry contains
    ``"values_truncated": true``, set by scripts/build_ozon_values.py when
    the Ozon API returned ``has_next=true`` after fetching 5000 values.

    Returns False when:
    - The (cat_id, type_id) pair is not in the dictionary.
    - The attribute_id is not found in characteristics.
    - The characteristic has no ``values_truncated`` flag (i.e. values were not
      yet enriched, or the full set fit within 5000).
    """
    d = load_ozon_dictionary()
    key = f"{cat_id}:{type_id}"
    entry = d.get(key)
    if not entry:
        return False
    for char in entry.get("characteristics", []):
        if char.get("id") == attribute_id:
            return bool(char.get("values_truncated", False))
    return False
