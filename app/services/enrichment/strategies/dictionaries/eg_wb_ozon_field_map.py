"""Loader for the vendored WB→Ozon field map (verified name→attr-id mapping).

The map is consolidated from eg-importer by
scripts/eg_build_wb_ozon_field_map.py and stored at
app/services/enrichment/strategies/dictionaries/data/eg_wb_ozon_field_map.json.

Schema:
{
    "<wb_subject>__<ozon_catid>_<ozon_typeid>": {
        "<WB field name>": <Ozon attribute id (int)>,
        ...
    },
    ...
}

Only verified (non-null) mappings are present in the file. Resolution by
(wb_subject, cat_id, type_id):
  1. exact key ``{wb_subject}__{catid}_{typeid}`` when wb_subject is known;
  2. fallback — union of ALL keys ending with ``__{catid}_{typeid}`` (several
     WB subjects often route to the same Ozon type).

Field names in the returned dict are normalized (lower + strip) so callers can
look up directly by a normalized WB characteristic name.
"""
import json
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional


DATA_DIR = Path(__file__).parent / "data"

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_eg_field_map() -> dict:
    """Load the consolidated WB→Ozon field map. Cached per process lifetime.

    Returns {} when the file is absent (graceful degradation — callers fall
    back to fuzzy matching).
    """
    path = DATA_DIR / "eg_wb_ozon_field_map.json"
    if not path.exists():
        logger.warning("eg_wb_ozon_field_map.json not found at %s", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_field_name(name: str) -> str:
    """Normalize a WB field name for lookup: lower + strip."""
    return name.lower().strip()


# Cache of resolved (wb_subject, cat_id, type_id) → normalized name→id maps.
_resolved_cache: dict[tuple, dict[str, int]] = {}


def eg_get_field_map(
    wb_subject: Optional[str],
    ozon_cat_id: Optional[int],
    ozon_type_id: Optional[int],
) -> dict[str, int]:
    """Return {normalized WB field name: Ozon attr id} for the given context.

    Resolution:
      1. exact key ``{wb_subject}__{catid}_{typeid}`` when wb_subject is given;
      2. fallback — merge ALL keys ending with ``__{catid}_{typeid}`` (union of
         all WB subjects routing to that Ozon type).

    Returns {} when cat/type are missing or no key matches. Cached per
    (wb_subject, cat_id, type_id) triple.
    """
    if not ozon_cat_id or not ozon_type_id:
        return {}

    cache_key = (
        _normalize_field_name(wb_subject) if wb_subject else None,
        ozon_cat_id,
        ozon_type_id,
    )
    cached = _resolved_cache.get(cache_key)
    if cached is not None:
        return cached

    data = load_eg_field_map()
    suffix = f"__{ozon_cat_id}_{ozon_type_id}"
    result: dict[str, int] = {}

    # 1. Exact key when wb_subject is known.
    if wb_subject:
        exact_key = f"{_normalize_field_name(wb_subject)}{suffix}"
        inner = data.get(exact_key)
        if inner:
            for name, attr_id in inner.items():
                if attr_id is not None:
                    result[_normalize_field_name(name)] = int(attr_id)

    # 2. Fallback — union of all subjects routing to this Ozon type.
    if not result:
        for key, inner in data.items():
            if not key.endswith(suffix) or not isinstance(inner, dict):
                continue
            for name, attr_id in inner.items():
                if attr_id is not None:
                    # setdefault: first subject wins on conflict (rare).
                    result.setdefault(_normalize_field_name(name), int(attr_id))

    _resolved_cache[cache_key] = result
    return result
