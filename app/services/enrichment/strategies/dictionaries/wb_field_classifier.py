"""Classifier for WildBerries characteristic fields: platform/admin vs extractable.

Mirrors the structure of ozon_field_classifier.py for consistent reporting.

USAGE — REPORTING ONLY. This classifier must NEVER be used to drop targets
from the extraction pipeline. It is used solely to compute an honest reporting
denominator (optional_honest) for WB eval metrics — the same purpose as
is_platform_field() on Ozon.

The filter is deliberately HIGH-PRECISION / conservative: it only fires on
characteristic names that are unambiguously administrative (barcode, certificate
numbers, fiscal codes, etc.) and cannot plausibly appear in a real product spec.

Guard: charc.get("popular") is True → return False.
WB marks "popular" characteristics as commonly used by sellers — any popular
field is almost certainly a real extractable spec field, never an admin field.
This mirrors the Ozon guard (is_required → False).

Markers are loaded from data/wb_platform_field_markers.json (module-level cache).
Each marker is a lowercase substring matched against the lowercased charc name.

Known platform fields caught by markers (validated against live WB API,
15 subjectIDs, June 2026):
  Баркод                                  — product barcode (logistics/seller)
  Номер декларации соответствия           — conformity declaration number
  Дата окончания действия сертификата/... — certificate expiry date
  Дата регистрации сертификата/...        — certificate registration date
  Номер сертификата соответствия          — certificate number
  ИКПУ                                    — fiscal commodity code (Uzbekistan)
  NTIN                                    — National Trade Item Number
  Ставка НДС                              — VAT rate (fiscal)
  Код упаковки                            — packaging code (logistics)
  Артикул OZON                            — cross-platform SKU link field
  Код ТРУ 1 / Код ТРУ 2                  — RU fiscal goods/work/service code
  Количество штук в товаре по ЭС          — unit count per electronic invoice

Fields intentionally NOT excluded (conservative):
  ТН ВЭД / ТНВЭД / Код ТН ВЭД           — customs code, extractable from specs
  Код производителя                       — manufacturer article (extractable)
  Вес с упаковкой (кг) / Высота упаковки — logistics dimensions (real spec data)
  Упаковка / Вид упаковки                 — packaging type (extractable)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"
_MARKERS_FILE = _DATA_DIR / "wb_platform_field_markers.json"


@lru_cache(maxsize=1)
def _load_markers() -> tuple[str, ...]:
    """Load platform-field markers from JSON. Cached at module level."""
    data = json.loads(_MARKERS_FILE.read_text(encoding="utf-8"))
    return tuple(m.lower() for m in data["markers"])


def is_wb_platform_field(charc: dict) -> bool:
    """Return True if *charc* is a non-extractable platform/admin WB field.

    REPORTING ONLY — never use to filter pipeline targets (see module docstring).

    Conservative criteria:
    - charc.get("popular") is True  → NEVER platform (popular = real seller work)
    - name (lowercased) contains any marker substring → platform admin field
    - otherwise → False (keep in denominator)

    Args:
        charc: WB charc dict as returned by get_subject_charcs(). Expected keys:
               charcID (int), name (str), required (bool), popular (bool), ...

    Returns:
        True if the characteristic is an administrative/platform-only field
        that a content-enrichment pipeline cannot and should not fill.
    """
    # Guard: popular fields are real product characteristics — never platform-only
    if charc.get("popular"):
        return False

    name: str = charc.get("name", "").lower()
    if not name:
        return False

    markers = _load_markers()
    return any(marker in name for marker in markers)
