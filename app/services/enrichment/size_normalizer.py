# -*- coding: utf-8 -*-
"""Size normalization: GOST intl→RU clothing sizes and explicit-size extraction.

GOST R 52774-2007 / GOST R 55972-2014 clothing size table:
  International letter → RU numeric range (even numbers, women's standard sizing).

AMBIGUITY RULE (critical for fail-closed behaviour):
  Letter sizes map to RANGES, not single values.  Because the target attr
  4295 is is_collection=True we CAN emit ALL candidates in the range — we let
  resolve_value_id on the full Ozon enum decide which ones are real dict values.
  What we NEVER do is silently pick one number from a range and present it as
  the definitive single value.  So:
    - expand_intl_to_ru("M")  → ["46", "48"]  (both candidates, collection)
    - expand_intl_to_ru("44") → ["44"]         (direct numeric, unambiguous)

  Callers that require a single value for a non-collection field should not use
  this table — they should leave the field empty instead.

WB sizes_table field path (discovered by live probe, 2026-06-09):
  card.json  →  sizes_table.values[]  →  .tech_size (intl: "XS"/"S"/"M"/"L"/"XL"/"2XL"…)
                                      →  .details[]  (list aligned with details_props)
  details_props[0] == "RU"  →  details[0] is the Russian size range ("42-44", "44-46")

  Strategy:
    1. If details_props[0] == "RU" and details[0] is present, parse the RU column.
    2. Fall back to expanding tech_size via the GOST table.
    3. Prefer the RU column value when both are present.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# GOST R 52774-2007 / GOST R 55972-2014 intl→RU numeric size table.
#
# Sources: GOST R 52774-2007 (women), GOST R 55972-2014 (men).
# Each entry: letter → tuple of RU numeric sizes (even numbers).
# Ambiguity: each letter covers a waist/chest range that straddles two
# or three even-number RU sizes.  The table lists ALL valid candidates.
# Footwear uses separate sizing (attr 4298) and is NOT covered here.
# ---------------------------------------------------------------------------

_INTL_TO_RU: dict[str, tuple[str, ...]] = {
    # Women (GOST R 52774-2007) — chest size anchors:
    "XS":    ("40", "42"),
    "S":     ("42", "44"),
    "M":     ("46", "48"),
    "L":     ("48", "50"),
    "XL":    ("50", "52"),
    "XXL":   ("52", "54"),
    "XXXL":  ("54", "56"),
    "2XL":   ("52", "54"),
    "3XL":   ("54", "56"),
    "4XL":   ("56", "58"),
    "5XL":   ("58", "60"),
    "6XL":   ("60", "62"),
    # Men (GOST R 55972-2014) — chest sizes overlap with women numerics.
    # Same table is used; context from the product/card drives which is correct.
    # No separate men/women split needed: the Ozon dict for 4295 has the same
    # values for all genders (44/46/48/50 …) so the resolved value_ids are shared.
}

# Normalize aliases: lowercase key → canonical key
_INTL_ALIASES: dict[str, str] = {}
for _k in list(_INTL_TO_RU):
    _INTL_ALIASES[_k.lower()] = _k
# Extra common aliases
_EXTRA_ALIASES = {
    "xxs":  "XS",
    "one size": None,        # → "универсальный"
    "onesize": None,
    "os": None,
    "универсальный": None,
    "универс": None,
    "free size": None,
    "freesize": None,
    "no size": None,
}
for _k, _v in _EXTRA_ALIASES.items():
    _INTL_ALIASES[_k] = _v  # type: ignore[assignment]


# RU numeric size range: "42-44", "44-46", "46-48", "50-52" etc.
# These come directly from WB sizes_table.details[0] when details_props[0]=="RU".
_RU_SIZE_RANGE_RE = re.compile(r"^(\d+)[-–—](\d+)$")

# Valid RU clothing numeric size range for the 4295 attribute.
# We accept even numbers 38–68 (standard adult clothing) plus common children's
# (68, 74, 80, 86, 92, 98, 104, 110, 116, 122, 128, 134, 140, 146, 152, 158,
#  164, 170, 176) and extend a few for tall/plus sizes (62–68 adults, up to 72).
# Outside this range the value may still exist in the Ozon dict — resolve_value_id
# will confirm. We just don't hard-reject here; resolve will return None for trash.
_RU_CLOTHING_EVEN_RE = re.compile(r"^\d{2,3}$")

# Minimum / maximum sensible adult numeric RU clothing sizes to accept from name-parsing.
# These bounds prevent catching years (2024), article numbers (501), etc.
_ADULT_RU_MIN = 38
_ADULT_RU_MAX = 72

# Standalone letter sizes that appear in product names as size tokens.
# Must be preceded or be surrounded by clear size-context signals.
_STANDALONE_LETTER_RE = re.compile(r"(?<![A-Za-z])(2XL|3XL|4XL|5XL|6XL|XXL|XXXL|XL|XS|[SML])(?![A-Za-z])", re.IGNORECASE)

# Explicit size context patterns in Russian / English product names.
# "размер M", "р. 44", "size XL", "р-р 48" etc.
_EXPLICIT_SIZE_CONTEXT_RE = re.compile(
    r"(?:р(?:азмер)?[:\.\s]*|р-р[:\s]*|size[:\s]+)([A-Za-z0-9/\-,]+)",
    re.IGNORECASE,
)


def expand_intl_to_ru(token: str) -> list[str]:
    """Convert an international size token to a list of Russian numeric sizes.

    Returns [] when the token is not a recognised size.
    Returns ["универсальный"] for "one size" / "os" / "universal" variants.
    For unambiguous numeric RU tokens (e.g. "44", "46") returns the token as-is.
    For letter sizes returns ALL candidates (collection — caller resolves via dict).
    """
    t = token.strip()
    if not t:
        return []

    # Direct RU numeric ("44", "46", "50" …)?
    if re.fullmatch(r"\d+", t):
        return [t]

    # Range "42-44" → expand to both endpoints
    m = _RU_SIZE_RANGE_RE.match(t)
    if m:
        lo, hi = m.group(1), m.group(2)
        # Deduplicate in case lo==hi (shouldn't happen but be safe)
        return list(dict.fromkeys([lo, hi]))

    # Canonical alias lookup (case-insensitive)
    key = t.lower()
    canonical = _INTL_ALIASES.get(key)
    if canonical is None and key in _EXTRA_ALIASES:
        return ["универсальный"]
    if canonical is not None:
        ru_vals = _INTL_TO_RU.get(canonical)
        if ru_vals is not None:
            return list(ru_vals)
        # canonical mapped to None → universal
        return ["универсальный"]

    # Try as-is in table (already uppercase canonical form)
    ru_vals = _INTL_TO_RU.get(t.upper())
    if ru_vals is not None:
        return list(ru_vals)

    return []


def extract_wb_sizes(card: dict) -> list[str]:
    """Extract RU size tokens from a WB card.json payload.

    Field path (confirmed by live probe 2026-06-09):
      card["sizes_table"]["values"][n]["details"][0]   when
      card["sizes_table"]["details_props"][0] == "RU"  (first column is RU size)

    Fallback:
      card["sizes_table"]["values"][n]["tech_size"]  → expand_intl_to_ru()

    Returns a deduplicated list of RU size strings (e.g. ["44", "46", "48"]).
    Returns [] when the card has no sizes_table or sizes cannot be parsed.
    """
    sizes_table = card.get("sizes_table")
    if not isinstance(sizes_table, dict):
        return []

    values = sizes_table.get("values")
    if not isinstance(values, list) or not values:
        return []

    details_props = sizes_table.get("details_props") or []
    ru_column_idx: Optional[int] = None
    for i, prop in enumerate(details_props):
        if isinstance(prop, str) and prop.strip().upper() == "RU":
            ru_column_idx = i
            break

    seen: set[str] = set()
    result: list[str] = []

    def _push(val: str) -> None:
        v = val.strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)

    for entry in values:
        if not isinstance(entry, dict):
            continue

        # Primary: RU column in details
        collected: list[str] = []
        if ru_column_idx is not None:
            details = entry.get("details")
            if isinstance(details, list) and len(details) > ru_column_idx:
                raw = str(details[ru_column_idx]).strip()
                if raw:
                    expanded = expand_intl_to_ru(raw)
                    if expanded:
                        collected = expanded

        # Fallback: tech_size → GOST table
        if not collected:
            tech = str(entry.get("tech_size", "")).strip()
            if tech:
                collected = expand_intl_to_ru(tech)

        for v in collected:
            _push(v)

    return result


def parse_explicit_size(name: str) -> list[str]:
    """Extract size token(s) from a product name ONLY when unambiguously a size.

    Conservative parser — empty is FAR better than wrong.

    Patterns accepted (in order of priority):
      1. Explicit context: «размер M», «р. 44», «size XL», «р-р 48» — the word
         "размер"/"р."/"size" immediately precedes the token. Most reliable.
      2. Standalone unambiguous numeric: a 2-digit even number in [38, 72]
         that is CLEARLY a size — i.e. it does NOT look like an article/year/
         model number.  Strictness rules:
           - The number must be preceded by a non-digit separator (space, comma,
             dot, slash) OR be at the end of the string.
           - Must NOT be preceded by a brand/article context (4-digit number,
             or directly after a letter-digit combo like "501").
           - Must be even (GOST clothing sizes are even).
      3. Standalone letter sizes (S/M/L/XL/XXL/2XL…) — ONLY when there is
         no preceding alphanumeric context that would make it an article suffix.

    Returns a list of raw size tokens (may be ranges like "44-46", letter sizes
    like "M", or numeric like "44").  Caller must expand via expand_intl_to_ru().
    Returns [] when unsure.
    """
    if not name:
        return []

    results: list[str] = []

    # Pattern 1: explicit "размер X" / "р. X" / "size X" — highest confidence
    for m in _EXPLICIT_SIZE_CONTEXT_RE.finditer(name):
        token = m.group(1).strip()
        if token:
            results.append(token)
    if results:
        return list(dict.fromkeys(results))

    # Pattern 2: standalone letter sizes (S/M/L/XL/XXL/2XL…)
    # Only accept when the token is clearly isolated (not part of an article).
    for m in _STANDALONE_LETTER_RE.finditer(name):
        tok = m.group(1).upper()
        start = m.start(1)
        end = m.end(1)
        # Reject if the letter is directly adjacent to digits on either side
        # (e.g. "T5XL" or "XL3" suggests an article suffix, not a size).
        before = name[max(0, start - 1):start]
        after = name[end:end + 1]
        if (before and before.isdigit()) or (after and after.isdigit()):
            continue
        results.append(tok)
    if results:
        return list(dict.fromkeys(results))

    # Pattern 3: standalone even 2-digit number in valid clothing range
    # Very strict to avoid model numbers, years, quantities.
    for m in re.finditer(r"(?<![a-zA-Z\d])(\d{2})(?![,/\d])", name):
        num_str = m.group(1)
        try:
            num = int(num_str)
        except ValueError:
            continue
        # Must be even and within adult clothing range
        if num % 2 != 0:
            continue
        if num < _ADULT_RU_MIN or num > _ADULT_RU_MAX:
            continue
        # Reject if preceded by another digit (e.g. "501" → "50" would match
        # at position 1, but the full token is 3 digits)
        start = m.start(1)
        pre = name[:start]
        if pre and pre[-1].isdigit():
            continue
        # Reject if it looks like a well-known model number context:
        # directly preceded by a letter (e.g. "A48" or "N52")
        if pre and pre[-1].isalpha():
            continue
        results.append(num_str)

    return list(dict.fromkeys(results))
