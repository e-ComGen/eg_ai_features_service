"""Loader for the Ozon category characteristics dictionary.

The dictionary is built by scripts/build_ozon_dictionary.py and stored at
app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json.

Structure of ozon_dictionary.json:
{
    "<category_id>": {
        "name": "Смартфоны",
        "path": ["Электроника", "Смартфоны"],
        "characteristics": [
            {"key": "brand", "name": "Бренд"},
            {"key": "color", "name": "Цвет"},
            ...
        ]
    }
}
"""
import json
from pathlib import Path
from typing import Optional
from functools import lru_cache


DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def load_ozon_dictionary() -> dict:
    """Load the Ozon dictionary from JSON.  Cached per process lifetime."""
    path = DATA_DIR / "ozon_dictionary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_ozon_characteristics_for_category(category_id: int) -> list[dict]:
    """Return characteristics list for a given Ozon category id.

    Returns an empty list when the category is not in the dictionary or the
    dictionary file does not yet exist (run build_ozon_dictionary.py first).
    """
    d = load_ozon_dictionary()
    entry = d.get(str(category_id))
    return entry["characteristics"] if entry else []


def get_ozon_category_name(category_id: int) -> Optional[str]:
    """Return the human-readable name for an Ozon category id, or None."""
    d = load_ozon_dictionary()
    entry = d.get(str(category_id))
    return entry["name"] if entry else None
