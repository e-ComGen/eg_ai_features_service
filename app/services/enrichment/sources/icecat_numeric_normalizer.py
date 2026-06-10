"""icecat_numeric_normalizer — conservative unit-aware normalizer for IceCat free-text specs.

IceCat returns numeric specs with units baked in, e.g.:
  "6,3 kg"  "165 Hz"  "320 cd/m²"  "68,6 cm (27\")"  "1 ms"

Ozon numeric attrs want the bare number in the attr's own unit (which the attr NAME
encodes: "..., кг", "..., Гц", "..., дюймы", etc.).

Conservatism rules:
  - Only activate when the attr name clearly encodes a known target unit.
  - Only activate when the raw value contains a clearly parseable single number
    (or a known pattern like the diagonal "68,6 cm (27\")").
  - Ambiguous multi-value strings ("0,2331 x 0,2331 mm") → pass through unchanged.
  - Unknown units → pass through unchanged.
  - Never fabricate. If regex fails to find a number → pass through.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Unit aliases: each entry maps display variants → canonical key
# ---------------------------------------------------------------------------

# (pattern_in_value, canonical_key)
# Ordered: longer/more-specific patterns first to avoid partial matches.
_VALUE_UNIT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"cd/m²|кд/м²|кд/м2|cd/m2", re.IGNORECASE), "cd/m2"),
    (re.compile(r"mah|мач", re.IGNORECASE), "mah"),
    (re.compile(r"\bkwh\b|\bквтч\b", re.IGNORECASE), "kwh"),
    (re.compile(r"\bw\b|\bвт\b|\bватт\b", re.IGNORECASE), "w"),
    (re.compile(r"\bv\b|\bвольт\b|\bв\b", re.IGNORECASE), "v"),
    (re.compile(r"\bkhz\b|\bкгц\b", re.IGNORECASE), "khz"),
    (re.compile(r"\bghz\b|\bггц\b", re.IGNORECASE), "ghz"),
    (re.compile(r'\bhz\b|\bгц\b|\bhertz\b', re.IGNORECASE), "hz"),
    (re.compile(r'\bms\b|\bмс\b|\bмилли\s*сек', re.IGNORECASE), "ms"),
    (re.compile(r'\bcm\b|\bсм\b|\bсантиметр', re.IGNORECASE), "cm"),
    (re.compile(r'\bmm\b|\bмм\b|\bмиллиметр', re.IGNORECASE), "mm"),
    (re.compile(r'\bm\b|\bметр\b(?!\w)', re.IGNORECASE), "m"),
    (re.compile(r'\bkg\b|\bкг\b|\bкилограмм', re.IGNORECASE), "kg"),
    (re.compile(r'\bg\b|\bгр\b|\bграмм\b', re.IGNORECASE), "g"),
    (re.compile(r'"|inch|inches|дюйм', re.IGNORECASE), "inch"),
    (re.compile(r"'|foot|feet|фут", re.IGNORECASE), "ft"),
]

# ---------------------------------------------------------------------------
# Attr name → expected unit key
# ---------------------------------------------------------------------------
# Maps a keyword (substring of attr name, lowercased) to the canonical unit.
# First match wins (ordered by specificity).
_ATTR_NAME_TO_UNIT: list[tuple[str, str]] = [
    # Electronics specs most commonly seen in IceCat
    ("дюйм", "inch"),
    ("inch", "inch"),
    ("кд/м", "cd/m2"),
    ("кд/м2", "cd/m2"),
    ("cd/m", "cd/m2"),
    ("гц", "hz"),
    ("hz", "hz"),
    ("мс", "ms"),
    (" мс", "ms"),
    ("мillis", "ms"),
    ("кг", "kg"),
    (" kg", "kg"),
    (",кг", "kg"),
    (", кг", "kg"),
    ("г,", "g"),          # e.g. "Масса, г"
    (", г", "g"),
    (" мм", "mm"),
    (", мм", "mm"),
    (" mm", "mm"),
    (", mm", "mm"),
    (" см", "cm"),
    (", см", "cm"),
    (" cm", "cm"),
    (", cm", "cm"),
    ("вт", "w"),
    (", w", "w"),
    ("в,", "v"),
    (", в", "v"),
    ("мах", "mah"),
    ("mah", "mah"),
]

# ---------------------------------------------------------------------------
# Conversion table: (source_unit, target_unit) → callable
# ---------------------------------------------------------------------------

def _cm_to_inch(x: float) -> float:
    return x / 2.54

def _mm_to_inch(x: float) -> float:
    return x / 25.4

def _g_to_kg(x: float) -> float:
    return x / 1000.0

def _kg_to_g(x: float) -> float:
    return x * 1000.0

_CONVERSIONS: dict[tuple[str, str], object] = {
    ("cm", "inch"): _cm_to_inch,
    ("mm", "inch"): _mm_to_inch,
    ("g", "kg"):    _g_to_kg,
    ("kg", "g"):    _kg_to_g,
}

# ---------------------------------------------------------------------------
# Rounding: sensible precision per unit
# ---------------------------------------------------------------------------
_ROUND_DIGITS: dict[str, int] = {
    "inch": 1,   # 27.0 → "27", 13.3 → "13.3"
    "cm": 1,
    "mm": 2,
    "kg": 3,
    "g": 0,
    "hz": 0,
    "ms": 1,
    "cd/m2": 0,
    "w": 0,
    "v": 1,
    "mah": 0,
}

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Generic number: optional sign, digits, optional decimal part (dot or comma)
_NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

# Inch explicit: "(27\")" or "(27 inch)" or "27\""
_INCH_EXPLICIT_RE = re.compile(
    r'[\(\[]?\s*([\d]+(?:[.,]\d+)?)\s*(?:"|inch|inches|дюйм)["\s\)]*',
    re.IGNORECASE,
)

# Multi-value guard: "x" between numbers (e.g. "0.23 x 0.23 mm")
_MULTI_VALUE_RE = re.compile(r"\d\s*[x×]\s*\d", re.IGNORECASE)


def _parse_float(s: str) -> Optional[float]:
    """Parse a number string that may use comma as decimal separator."""
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def _format_number(val: float, unit: str) -> str:
    """Format a float to string, removing trailing zeros after decimal."""
    digits = _ROUND_DIGITS.get(unit, 2)
    rounded = round(val, digits)
    if digits == 0:
        return str(int(rounded))
    formatted = f"{rounded:.{digits}f}"
    # Strip trailing zeros: "6.300" → "6.3", "27.0" → "27"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _detect_attr_unit(attr_name: str) -> Optional[str]:
    """Return the canonical expected unit from attr name, or None if unknown."""
    lower = attr_name.lower()
    for keyword, unit in _ATTR_NAME_TO_UNIT:
        if keyword in lower:
            return unit
    return None


def _detect_value_unit(raw: str) -> Optional[str]:
    """Return the canonical unit found in the raw value string, or None."""
    for pattern, unit_key in _VALUE_UNIT_PATTERNS:
        if pattern.search(raw):
            return unit_key
    return None


def normalize_icecat_numeric(attr_name: str, raw_value: str) -> str:
    """Normalize a raw IceCat numeric value to match the attr's expected unit.

    Conservative: returns raw_value unchanged when:
      - attr name doesn't encode a known unit
      - raw value doesn't contain a parseable number
      - raw value has multiple numbers (ambiguous dimension pair)
      - conversion would be needed but isn't defined
    """
    raw = raw_value.strip()

    # Guard: ambiguous multi-value strings like "0,2331 x 0,2331 mm"
    if _MULTI_VALUE_RE.search(raw):
        return raw

    target_unit = _detect_attr_unit(attr_name)
    if target_unit is None:
        # Attr name doesn't encode a known unit — pass through
        return raw

    # --- Special case: diagonal with explicit inch notation ---
    # e.g. "68,6 cm (27\")"  → want "27"
    if target_unit == "inch":
        m = _INCH_EXPLICIT_RE.search(raw)
        if m:
            val = _parse_float(m.group(1))
            if val is not None:
                return _format_number(val, "inch")

    # --- Extract the (single) number from raw ---
    numbers = _NUM_RE.findall(raw)
    if len(numbers) != 1:
        # Zero or multiple numbers — ambiguous, pass through
        return raw

    val = _parse_float(numbers[0])
    if val is None:
        return raw

    # Detect source unit in raw value
    value_unit = _detect_value_unit(raw)

    if value_unit is None:
        # No unit in value — already bare number, just fix decimal separator
        normalized_bare = numbers[0].replace(",", ".")
        # Ensure valid float representation
        try:
            float(normalized_bare)
        except ValueError:
            return raw
        return normalized_bare

    if value_unit == target_unit:
        # Same unit — strip the unit suffix, fix decimal
        return _format_number(val, target_unit)

    # Different unit — attempt conversion
    conv = _CONVERSIONS.get((value_unit, target_unit))
    if conv is None:
        # No conversion defined — pass through unchanged
        return raw

    converted = conv(val)  # type: ignore[operator]
    return _format_number(converted, target_unit)
