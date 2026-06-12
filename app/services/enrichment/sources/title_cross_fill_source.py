"""TitleCrossFillSource — verbatim attribute extraction from the product title.

Fills still-empty target attributes whose values appear LITERALLY in the
product title (product_name).  No LLM, no network — pure deterministic
string matching.  Zero cost.

Design principles («пусто честнее мусора»):
  - For ENUM attrs: an allowed value is accepted ONLY when it appears as a
    whole normalised token sequence in the title AND no other allowed value
    of the same attr also matches (ambiguity → skip).  Matching uses the
    same _brand_norm_tokens normalisation (ё→е, case-insensitive,
    latin+cyrillic+digits).  Min token-char total: 3 (avoids accidental hits
    on short tokens).
  - For NUMERIC attrs: the IceCat numeric normalizer is applied to extract a
    bare number+unit pair from the title token window.  Only fires when the
    attr name encodes a known unit (detect via _detect_attr_unit) AND the
    title contains exactly one matching number in that unit.
  - For FREE-TEXT attrs: fill ONLY when a single clean verbatim token (≥4
    chars, not a stop-word) appears that is not present in any other filled
    or known-noisy value.  The bar here is intentionally high — free-text
    fills from titles are too risky in general, so this path is OFF by
    default and gated by TITLE_FREETEXT_FILL_ENABLED=1.

Guards (all):
  1. Never overwrites an already-filled attr (only fills empties).
  2. Brand targets are excluded — those are handled by _apply_brand_from_name
     with richer dict-scan + containment logic.
  3. Min value token length ≥ _TITLE_MIN_VALUE_LEN (3 chars total).
  4. Ambiguity guard: ≥2 different allowed values match → skip attr entirely.
  5. source=DESCRIPTION, evidence="title:verbatim" — highest-priority source
     tier, no judge needed (deterministic).
  6. Pipeline placement: Stage 0.47 — after BarcodeSource (Stage 0.46),
     before WbCardSource (Stage 0.45).

attr types handled:
  - "enum"    → whole-token allowed-value match (same as _spec_value_in_title
                 but run as an EARLY STAGE, not post-merge)
  - "numeric" → regex+unit extraction via icecat_numeric_normalizer helpers
  - "bool"    → NOT handled here (bool values must come from authoritative
                 specs, not title inference)
  - "text"    → OFF by default (TITLE_FREETEXT_FILL_ENABLED=1 to enable)
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.sources.icecat_numeric_normalizer import (
    _NUM_RE,
    _MULTI_VALUE_RE,
    _detect_attr_unit,
    _detect_value_unit,
    _parse_float,
    _format_number,
)

logger = logging.getLogger(__name__)

# Feature flag: free-text attr filling from title (risky — off by default).
_TITLE_FREETEXT_FILL_ENABLED: bool = (
    os.environ.get("TITLE_FREETEXT_FILL_ENABLED", "0") == "1"
)

# Confidence: verbatim from title, deterministic, no inference.
_TITLE_FILL_CONFIDENCE: float = 0.88

# Minimum total character count across all normalised tokens of a value
# before we consider it safe to match (avoids accidental hits on 1-2 char tokens).
_TITLE_MIN_VALUE_LEN: int = 3

# Attr-name brand detection — exclude brand targets (handled by dedicated logic).
_BRAND_NAME_RE = re.compile(r"бренд|brand|торгов\w*\s+марк", re.IGNORECASE)

# Token normalisation regex — mirrors _brand_norm_tokens in pipeline.py.
_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _norm_tokens(text: str) -> list[str]:
    """Normalised token list: ё→е, lowercase, latin+cyrillic+digits only."""
    return _TOKEN_RE.findall(text.lower().replace("ё", "е"))


def _value_in_title(value: str, title_tokens: list[str]) -> bool:
    """True if `value` appears as a contiguous normalised token sequence in title.

    Mirrors _spec_value_in_title from pipeline.py — kept local to avoid
    cross-module coupling.  Min total chars: _TITLE_MIN_VALUE_LEN.
    """
    v_tokens = _norm_tokens(value)
    if not v_tokens:
        return False
    if sum(len(t) for t in v_tokens) < _TITLE_MIN_VALUE_LEN:
        return False
    n = len(v_tokens)
    for i in range(len(title_tokens) - n + 1):
        if title_tokens[i: i + n] == v_tokens:
            return True
    return False


def _is_brand_target(target: TargetAttribute) -> bool:
    """True if this target is a brand/trademark field."""
    if target.id == 31:  # Ozon brand attr id
        return True
    return bool(_BRAND_NAME_RE.search(target.name))


def _extract_numeric_from_title(
    attr_name: str,
    title: str,
) -> Optional[str]:
    """Try to extract a bare number for a numeric attr from the title.

    Uses icecat_numeric_normalizer helpers:
      - _detect_attr_unit: what unit does this attr name expect?
      - _detect_value_unit: what unit is in the title fragment?
      - normalize via _parse_float + _format_number.

    Verbatim-only contract:
      - If the title contains multiple numbers → too ambiguous, return None.
      - If attr unit and value unit mismatch AND no conversion defined → None.
      - Guard: _MULTI_VALUE_RE (e.g. «0.2 x 0.3 mm» dimension) → None.

    Returns the normalised bare number string, or None when uncertain.
    """
    # Guard: multi-value (dimension) expressions are ambiguous
    if _MULTI_VALUE_RE.search(title):
        return None

    target_unit = _detect_attr_unit(attr_name)
    if target_unit is None:
        return None  # attr doesn't declare a unit — skip

    # Find all number tokens in title
    numbers = _NUM_RE.findall(title)
    if len(numbers) != 1:
        return None  # zero or multiple numbers → ambiguous

    val = _parse_float(numbers[0])
    if val is None:
        return None

    value_unit = _detect_value_unit(title)

    if value_unit is None:
        # Bare number in title without explicit unit — accept only if attr
        # unit context makes it unambiguous. Too risky without unit label.
        return None

    if value_unit == target_unit:
        # Same unit — strip unit suffix, return bare number
        return _format_number(val, target_unit)

    # Different unit — attempt conversion via normalizer's conversion table
    from app.services.enrichment.sources.icecat_numeric_normalizer import _CONVERSIONS
    conv = _CONVERSIONS.get((value_unit, target_unit))
    if conv is None:
        return None  # no conversion defined — don't guess

    converted = conv(val)  # type: ignore[operator]
    return _format_number(converted, target_unit)


# ---------------------------------------------------------------------------
# Judge — deterministic passthrough (verbatim extraction needs no LLM judge)
# ---------------------------------------------------------------------------

class _TitleFillJudge(LlmJudge):
    """Deterministic passthrough judge for title-verbatim fills."""

    source = Source.DESCRIPTION

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        # Always accept — deterministic extraction with built-in ambiguity guards.
        return True


# ---------------------------------------------------------------------------
# TitleCrossFillSource
# ---------------------------------------------------------------------------

class TitleCrossFillSource(AttributeSource):
    """Fills still-empty attributes from verbatim tokens in the product title.

    Zero cost (no LLM, no network).  Called as Stage 0.47, after BarcodeSource,
    before WbCardSource.

    Handles three attr types:
      - enum:    whole-token allowed-value match (ambiguity → skip)
      - numeric: unit-aware number extraction from title via icecat normalizer
      - text:    OFF by default (TITLE_FREETEXT_FILL_ENABLED=1 to enable)
    """

    @property
    def source_type(self) -> Source:
        return Source.DESCRIPTION

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Applicable whenever the product has a non-trivial title."""
        return bool(context.product_name) and len(context.product_name.strip()) >= 3

    def get_judge(self) -> LlmJudge:
        return _TitleFillJudge()

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        title = (context.product_name or "").strip()
        if not title:
            return []

        # Skip targets already filled
        already_filled_ids: set[int] = set()
        if already_filled:
            already_filled_ids = {
                av.attribute_id
                for av in already_filled
                if av.is_confident()
            }

        title_tokens = _norm_tokens(title)
        if not title_tokens:
            return []

        results: list[AttributeValue] = []

        for target in targets:
            # Only process empty targets
            if target.id in already_filled_ids:
                continue

            # Never touch brand targets — dedicated _apply_brand_from_name handles those
            if _is_brand_target(target):
                continue

            av = self._try_fill(target, title, title_tokens)
            if av is not None:
                results.append(av)

        if results:
            logger.info(
                "[TitleCrossFill] product=%s: filled %d attrs from title=%r",
                context.product_id,
                len(results),
                title[:60],
            )
        return results

    def _try_fill(
        self,
        target: TargetAttribute,
        title: str,
        title_tokens: list[str],
    ) -> Optional[AttributeValue]:
        """Attempt to fill one target from the title. Returns AV or None."""
        t_type = (target.type or "").lower()

        if t_type == "enum" and target.allowed_values:
            return self._fill_enum(target, title_tokens)

        if t_type == "numeric":
            return self._fill_numeric(target, title)

        if t_type == "text" and _TITLE_FREETEXT_FILL_ENABLED:
            return self._fill_freetext(target, title_tokens)

        return None

    def _fill_enum(
        self,
        target: TargetAttribute,
        title_tokens: list[str],
    ) -> Optional[AttributeValue]:
        """Fill enum target: exactly one allowed value matches title token sequence."""
        hitting = [
            v for v in target.allowed_values
            if _value_in_title(v, title_tokens)
        ]

        if not hitting:
            return None

        if len(hitting) >= 2:
            # Ambiguity guard: multiple matches → skip (empty > wrong)
            logger.debug(
                "[TitleCrossFill] AMBIG enum attr=%s '%s' — %d values hit title %s → skip",
                target.id, target.name, len(hitting), hitting[:4],
            )
            return None

        chosen = hitting[0]
        logger.info(
            "[TitleCrossFill] enum attr=%s '%s' ← '%s' (title:verbatim)",
            target.id, target.name, chosen,
        )
        return AttributeValue(
            attribute_id=target.id,
            value=chosen,
            confidence=_TITLE_FILL_CONFIDENCE,
            source=Source.DESCRIPTION,
            evidence="title:verbatim",
            semantic_type=target.semantic_type,
            is_collection=target.is_collection,
        )

    def _fill_numeric(
        self,
        target: TargetAttribute,
        title: str,
    ) -> Optional[AttributeValue]:
        """Fill numeric target: unit-aware bare number extracted from title."""
        result = _extract_numeric_from_title(target.name, title)
        if result is None:
            return None

        logger.info(
            "[TitleCrossFill] numeric attr=%s '%s' ← '%s' (title:verbatim+unit)",
            target.id, target.name, result,
        )
        return AttributeValue(
            attribute_id=target.id,
            value=result,
            confidence=_TITLE_FILL_CONFIDENCE,
            source=Source.DESCRIPTION,
            evidence="title:verbatim+unit",
            semantic_type=target.semantic_type,
        )

    def _fill_freetext(
        self,
        target: TargetAttribute,
        title_tokens: list[str],
    ) -> Optional[AttributeValue]:
        """Fill free-text target: single meaningful token from title.

        Conservative: only fires when TITLE_FREETEXT_FILL_ENABLED=1.
        Token must be ≥4 chars and not a common stop-word.
        This is intentionally minimal — free-text fills from titles are
        high-risk and this path is off by default.
        """
        # This is deliberately not implemented beyond the flag check in _try_fill.
        # Free-text from title is too ambiguous for a general verbatim fill
        # (any noun in the title could look like a "value").  The flag exists
        # as a future extension point but the logic stays empty until there is
        # a concrete use-case with a proper safety design.
        return None
