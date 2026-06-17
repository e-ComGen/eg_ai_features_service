"""Deterministic unit normalizer for card-sourced attribute values.

WB/Ozon card extraction maps a card spec line to an attribute by fuzzy name
match without checking unit compatibility, so a value carrying its own unit
("5.4 см") can land in a field that declares a different unit ("Ширина, мм").
This module fixes ONLY that one class — a value whose explicit unit differs
from the field's explicit unit, same physical dimension — by lossless
conversion (5.4 см → 54). It does NOT guess: a bare number with no unit token
is left untouched, because nothing in the data tells us which unit it is.

The conversion TABLE is data, not code — it lives in
``data/unit_conversions.json``. This module is the generic engine. Adding a
unit/dimension is a JSON edit, no code change.

Firing conditions (ALL must hold, else value returned unchanged):
  1. the field name carries an explicit unit (token after the last comma);
  2. the value carries exactly ONE explicit unit token (whole-token, with
     non-letter boundaries so "м" inside "мм"/"см" never false-matches);
  3. both units belong to the SAME dimension;
  4. the units actually differ;
  5. the value is a single scalar — no range ("5–7 см"), no list, no extra
     numbers.

Safe by construction: arithmetic only (no LLM), fires on an explicit-explicit
unit mismatch only, and logs every conversion (before → after) so a run can be
eyeballed rather than trusted blindly.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def load_unit_conversions() -> dict:
    """Load the declarative unit table. Cached per process; {} if absent."""
    path = DATA_DIR / "unit_conversions.json"
    if not path.exists():
        logger.warning("unit_conversions.json not found at %s", path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("dimensions", {})


@lru_cache(maxsize=1)
def _token_index() -> dict[str, tuple[str, float]]:
    """{unit token (lower) -> (dimension, factor-to-base)}.

    Built once from the JSON. Used to look up the dimension/factor of any
    detected token.
    """
    idx: dict[str, tuple[str, float]] = {}
    for dim, spec in load_unit_conversions().items():
        for tok, factor in spec.get("units", {}).items():
            idx[tok.lower()] = (dim, float(factor))
    return idx


@lru_cache(maxsize=1)
def _tokens_longest_first() -> list[str]:
    """All known unit tokens, longest first (greedy boundary matching)."""
    return sorted(_token_index().keys(), key=len, reverse=True)


def _field_unit(field_name: str) -> Optional[str]:
    """Return the explicit unit token declared by a field name, else None.

    The unit conventionally trails the last comma: "Ширина профиля, мм".
    Matched as the WHOLE tail token (exact), never a substring.
    """
    if not field_name or "," not in field_name:
        return None
    tail = field_name.rsplit(",", 1)[-1].strip().lower()
    return tail if tail in _token_index() else None


# One number, optional decimal (dot or comma), e.g. "5.4", "94,5", "1000".
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _value_unit_tokens(value_str: str) -> list[str]:
    """All distinct unit tokens present in the value, whole-token matched.

    Boundaries: a token must not be flanked by Cyrillic/Latin letters, so "м"
    cannot match inside "мм"/"см" and "г" cannot match inside "кг".
    """
    s = value_str.lower()
    found: list[str] = []
    for tok in _tokens_longest_first():
        if re.search(r"(?<![а-яёa-z])" + re.escape(tok) + r"(?![а-яёa-z])", s):
            found.append(tok)
    return found


def _fmt(num: float) -> str:
    """Format a converted number: drop a trailing .0, trim float noise."""
    r = round(num, 6)
    if r == int(r):
        return str(int(r))
    # strip trailing zeros from the decimal part
    return f"{r:.6f}".rstrip("0").rstrip(".")


@dataclass
class UnitNormResult:
    changed: bool
    value: str            # converted value (number only) or the original
    note: Optional[str] = None  # human-readable "5.4 см → 54 мм" for logging


def normalize_value(field_name: str, value) -> UnitNormResult:
    """Convert ``value`` into the field's declared unit when (and only when)
    an explicit-explicit, same-dimension unit mismatch is present.

    Returns ``UnitNormResult(changed=False, value=str(value))`` for everything
    else — bare numbers, lists, ranges, matching units, unknown units.
    """
    # Lists/non-scalars: never touch.
    if isinstance(value, (list, dict, tuple)):
        return UnitNormResult(False, str(value))

    sval = str(value).strip()
    if not sval:
        return UnitNormResult(False, sval)

    field_tok = _field_unit(field_name)
    if field_tok is None:
        return UnitNormResult(False, sval)

    field_dim, field_factor = _token_index()[field_tok]

    # Range guard: "5–7 см" — two numbers separated by a dash.
    if re.search(r"\d\s*[-–—]\s*\d", sval):
        return UnitNormResult(False, sval)

    nums = _NUM_RE.findall(sval)
    if len(nums) != 1:  # must be exactly one scalar
        return UnitNormResult(False, sval)

    vtoks = _value_unit_tokens(sval)
    if len(vtoks) != 1:  # exactly one explicit unit in the value
        return UnitNormResult(False, sval)

    vtok = vtoks[0]
    vdim, vfactor = _token_index()[vtok]
    if vdim != field_dim:  # different physical dimension — never convert
        return UnitNormResult(False, sval)
    if vtok == field_tok:  # already in the field's unit
        return UnitNormResult(False, sval)

    try:
        amount = float(nums[0].replace(",", "."))
    except ValueError:
        return UnitNormResult(False, sval)

    converted = amount * vfactor / field_factor
    out = _fmt(converted)
    note = f"{sval} → {out} {field_tok}"
    return UnitNormResult(True, out, note)
