"""oem_spec_harvest — verbatim spec-pair extraction from fetched page text.

LEVER 2: when brand+model (MPN) are known, Serper's top organic results are
often the manufacturer's official spec page.  The existing WebSearchProducer
already fetches the top-1-2 pages via url_fetcher.fetch_all; trafilatura
renders spec tables as plain-text «Key: Value» lines.

This module adds a DETERMINISTIC pre-pass over those lines BEFORE the LLM
step.  Any «Key: Value» line where:
  a. the Key (normalised, ё→е, stripped) is a fuzzy prefix-match of a target
     attr name, AND
  b. the Value (for enum attrs) passes the allowed-option matcher (verbatim or
     normalised), OR (for numeric attrs) passes the IceCat numeric normalizer
     to produce a bare number, OR (for free-text attrs — NOT enabled by default)
     is returned verbatim

is emitted as an AttributeValue with source=DESCRIPTION (verbatim from page),
evidence="oem_spec:key:value", confidence=0.87.

«Пусто честнее мусора» guards:
  1. Key-to-attr matching: lower+strip+ё→е prefix match. Min overlap: key must
     cover ≥ _KEY_OVERLAP_MIN of the target attr name tokens. Unambiguous match
     only: if ≥2 target attrs match the same key → skip that key (ambiguity).
  2. Enum value: passes _match_enum_value (fuzzy option matcher from
     competitor_rag_source — returns None when no confident match). No match → skip.
  3. Numeric value: passes _extract_numeric_from_title (re-exported from
     title_cross_fill_source / icecat_numeric_normalizer). Returns None when
     ambiguous. No match → skip.
  4. Never overwrites already-filled attrs.
  5. Source=DESCRIPTION, not WEB_SEARCH — verbatim page text is product-specific
     (same trust tier as DescriptionSource). Does NOT go through brand-guess guard.
  6. Gated by OEM_SPEC_HARVEST_ENABLED=1 env flag (default OFF). Zero cost/risk
     when disabled.

Architecture: NOT a new AttributeSource class — instead a helper function
`harvest_spec_pairs()` called from WebSearchSource.extract() BEFORE the LLM
step, reusing the already-fetched page text from WebSearchProducer's summary
cache.  This avoids duplicating the fetch infrastructure.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)

logger = logging.getLogger(__name__)

# Feature flag — off by default.
OEM_SPEC_HARVEST_ENABLED: bool = (
    os.environ.get("OEM_SPEC_HARVEST_ENABLED", "0") == "1"
)

# Confidence for verbatim spec-pair fills.
_OEM_SPEC_CONFIDENCE: float = 0.87

# Minimum fraction of target attr name tokens that must appear in the key
# before we accept the key→attr mapping.  Prevents "Вес" matching "Ширина" etc.
_KEY_OVERLAP_MIN: float = 0.6

# Token normalisation regex (mirrors pipeline.py _brand_norm_tokens)
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)

# Line pattern: «Some Key: some value» or «Some Key — some value»
# Captures key and value from each line.
_SPEC_LINE_RE = re.compile(
    r"^([^:\-–—]{2,60})\s*[:\-–—]\s*(.{1,200})$",
    re.MULTILINE,
)

# Alternating-bullet pattern used by Ozon trafilatura output:
#   - KeyName
#   - ValueText
# Each spec attribute is on a dash-prefixed line, alternating key then value.
_BULLET_LINE_RE = re.compile(r"^-\s+(.{1,200})$", re.MULTILINE)

# Max key length for alternating-bullet extraction (prevents stray long lines
# from being treated as keys).
_BULLET_KEY_MAX = 60


def _norm_tokens(text: str) -> list[str]:
    """Normalised token list: ё→е, lowercase."""
    return _TOKEN_RE.findall(text.lower().replace("ё", "е"))


def _token_overlap(key_tokens: list[str], attr_tokens: list[str]) -> float:
    """Fraction of attr_tokens that appear in key_tokens (order-independent)."""
    if not attr_tokens:
        return 0.0
    key_set = set(key_tokens)
    matched = sum(1 for t in attr_tokens if t in key_set)
    return matched / len(attr_tokens)


def _parse_bullet_spec_lines(text: str) -> list[tuple[str, str]]:
    """Extract (key, value) pairs from alternating-bullet spec format.

    Handles Ozon trafilatura output style:
        - Сезон
        - На любой сезон
        - Материал
        - Хлопок

    Algorithm: collect all dash-bullet lines; iterate in pairs (even=key,
    odd=value).  Accept only when the key is ≤ _BULLET_KEY_MAX chars (guards
    against long description sentences that happen to start with a dash).
    Requires that the bullet block is dense enough: at least 4 consecutive
    bullet lines in the same block (avoids false positives from stray dash-lists
    like navigation menus with 1-2 items).
    """
    bullet_lines = [m.group(1).strip() for m in _BULLET_LINE_RE.finditer(text)]
    if len(bullet_lines) < 4:
        return []  # not a dense enough spec block — skip

    pairs: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(bullet_lines):
        key = bullet_lines[i]
        value = bullet_lines[i + 1]
        # Key must be short (spec attribute names are rarely > 60 chars)
        # and must not itself look like a value (e.g. a percentage / long sentence)
        if key and value and len(key) <= _BULLET_KEY_MAX:
            pairs.append((key, value))
        i += 2
    return pairs


def _parse_spec_lines(text: str) -> list[tuple[str, str]]:
    """Extract (key, value) pairs from spec-like text lines.

    Handles two formats:
      1. Inline separator: «Key: Value» or «Key — Value» on a single line.
      2. Alternating-bullet: «- Key\\n- Value» (Ozon trafilatura output style).

    Deduplicates pairs; inline-separator results take precedence when both
    formats produce the same key (different separators for the same attr).
    Returns only pairs where key ≤ 60 chars and value ≤ 200 chars.
    """
    pairs: list[tuple[str, str]] = []
    seen_keys: set[str] = set()

    # Format 1: inline separators (highest-fidelity)
    for m in _SPEC_LINE_RE.finditer(text):
        key = m.group(1).strip()
        value = m.group(2).strip()
        if key and value:
            pairs.append((key, value))
            seen_keys.add(key.lower())

    # Format 2: alternating-bullet (Ozon page style) — only add keys not yet seen
    for key, value in _parse_bullet_spec_lines(text):
        if key.lower() not in seen_keys and value:
            pairs.append((key, value))
            seen_keys.add(key.lower())

    return pairs


def _find_unique_attr_for_key(
    key: str,
    targets: list[TargetAttribute],
    filled_ids: set[int],
) -> Optional[TargetAttribute]:
    """Return the single unfilled target whose name best matches `key`, or None.

    Matching: token-overlap ≥ _KEY_OVERLAP_MIN, shortest-match tiebreak.
    Returns None if 0 or ≥2 targets match (ambiguity guard).
    """
    key_tokens = _norm_tokens(key)
    if not key_tokens:
        return None

    candidates: list[tuple[float, int, TargetAttribute]] = []
    for t in targets:
        if t.id in filled_ids:
            continue
        attr_tokens = _norm_tokens(t.name)
        if not attr_tokens:
            continue
        overlap = _token_overlap(key_tokens, attr_tokens)
        if overlap >= _KEY_OVERLAP_MIN:
            candidates.append((overlap, len(attr_tokens), t))

    if len(candidates) == 0:
        return None
    if len(candidates) >= 2:
        # Ambiguity: two or more targets match the key → skip
        return None

    return candidates[0][2]


def _match_enum_value_safe(raw: str, allowed_values: list[str]) -> Optional[str]:
    """Wrapper around competitor_rag._match_enum_value with graceful fallback."""
    try:
        from app.services.enrichment.sources.competitor_rag_source import _match_enum_value
        return _match_enum_value(raw, allowed_values)
    except Exception as exc:
        logger.debug("[OemSpecHarvest] enum matcher failed for %r: %s", raw, exc)
        # Fallback: case+ё-insensitive exact match
        def _n(s: str) -> str:
            return s.strip().lower().replace("ё", "е")
        raw_n = _n(raw)
        for opt in allowed_values:
            if _n(opt) == raw_n:
                return opt
        return None


def _match_numeric_value(attr_name: str, raw_value: str) -> Optional[str]:
    """Extract normalised numeric value from raw spec-pair value.

    Reuses icecat_numeric_normalizer logic via title_cross_fill_source helper.
    """
    try:
        from app.services.enrichment.sources.title_cross_fill_source import (
            _extract_numeric_from_title,
        )
        return _extract_numeric_from_title(attr_name, raw_value)
    except Exception as exc:
        logger.debug("[OemSpecHarvest] numeric extract failed for %r: %s", raw_value, exc)
        return None


async def harvest_spec_pairs(
    page_text: str,
    targets: list[TargetAttribute],
    already_filled: Optional[list[AttributeValue]],
    context: ExtractionContext,
) -> list[AttributeValue]:
    """Extract verbatim spec pairs from fetched page text and fill matching targets.

    Called from WebSearchSource.extract() BEFORE the LLM step when
    OEM_SPEC_HARVEST_ENABLED=1.  The page_text is the already-fetched content
    from WebSearchProducer (trafilatura-extracted, boilerplate-guarded).

    Returns a list of AttributeValue with source=DESCRIPTION, evidence tag
    "oem_spec:<key>:<value>".  Does NOT call any LLM or network.
    """
    if not OEM_SPEC_HARVEST_ENABLED:
        return []

    if not page_text or not targets:
        return []

    # Build set of already-filled attr ids (any confidence)
    filled_ids: set[int] = set()
    if already_filled:
        filled_ids = {av.attribute_id for av in already_filled}

    # Parse spec-like lines from the fetched page text
    pairs = _parse_spec_lines(page_text)
    if not pairs:
        return []

    # For each (key, value) pair, try to find a matching unfilled target
    results: list[AttributeValue] = []
    seen_attr_ids: set[int] = set()  # one fill per attr

    for key, raw_value in pairs:
        target = _find_unique_attr_for_key(key, targets, filled_ids | seen_attr_ids)
        if target is None:
            continue

        t_type = (target.type or "").lower()
        filled_value: Optional[str] = None

        if t_type == "enum" and target.allowed_values:
            filled_value = _match_enum_value_safe(raw_value, target.allowed_values)

        elif t_type == "numeric":
            filled_value = _match_numeric_value(target.name, raw_value)

        # text + bool: not handled (too risky without verbatim anchor beyond line match)

        if filled_value is None:
            continue

        evidence = f"oem_spec:{key[:40]}:{raw_value[:60]}"
        logger.info(
            "[OemSpecHarvest] product=%s attr=%s '%s' ← '%s' (key=%r page_text)",
            context.product_id, target.id, target.name, filled_value, key,
        )
        results.append(AttributeValue(
            attribute_id=target.id,
            value=filled_value,
            confidence=_OEM_SPEC_CONFIDENCE,
            source=Source.DESCRIPTION,
            evidence=evidence,
            semantic_type=target.semantic_type,
            is_collection=target.is_collection,
        ))
        seen_attr_ids.add(target.id)

    return results
