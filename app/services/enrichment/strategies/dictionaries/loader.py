"""Loader for marketplace dictionaries.

Loads JSON dictionaries into memory and provides a lookup API.
The WB dictionary is built by scripts/build_wb_dictionary.py.
"""
import json
from pathlib import Path
from typing import Optional
from functools import lru_cache


DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def load_wb_dictionary() -> dict:
    """Load WB dictionary from JSON.  Cached per process lifetime."""
    path = DATA_DIR / "wb_dictionary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_wb_characteristics_for_category(subj_id: int) -> list[dict]:
    """Return characteristics list for a given WB subject id.

    Returns an empty list when the subject is not in the dictionary or the
    dictionary file does not yet exist.
    """
    d = load_wb_dictionary()
    entry = d.get(str(subj_id))
    return entry["characteristics"] if entry else []


def get_wb_subject_name(subj_id: int) -> Optional[str]:
    """Return the human-readable name for a WB subject id, or None."""
    d = load_wb_dictionary()
    entry = d.get(str(subj_id))
    return entry["name"] if entry else None
