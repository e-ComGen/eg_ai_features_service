"""BarcodeSource — verbatim EAN/barcode extractor from already-fetched text.

Scans product_description, context.ean (if pre-populated), and any
`source_text` string passed by the pipeline for EAN-13, EAN-8, and UPC-A
barcodes that appear VERBATIM in the text.  Applies standard checksum
validation to avoid matching random 13-digit numbers (phone numbers, IDs,
timestamps, etc.).

Design constraints (hard):
  - NO guessing, NO LLM, NO external calls — verbatim extraction only.
  - Only fills target attributes whose name contains a barcode keyword
    (штрихкод / штрих-код / barcode / ean / gtin — case-insensitive).
    This prevents injecting a barcode value into an unrelated attribute.
  - Validates EAN-13 / UPC-A / EAN-8 checksum (standard GS1 algorithm).
  - Returns high confidence (0.98) because the value came verbatim from
    product data and passed checksum — no inference involved.

Pipeline placement: called from _run_barcode_stage (Stage 0.46), right after
DescriptionSource, before any LLM source.  Runs even when description is empty
because context.ean may be pre-populated by the caller (e.g. from a marketplace
card).  Zero cost (no LLM, no network).
"""
from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

# Confidence for verbatim+checksum-validated barcode fills.
_BARCODE_CONFIDENCE: float = 0.98

# Regex: sequences of 8, 12, or 13 digits that are preceded and followed by
# a non-digit boundary (word boundary on digit end).  We pull all such
# sequences and then validate by checksum.
_DIGIT_SEQ_RE = re.compile(r"(?<!\d)(\d{8}|\d{12}|\d{13})(?!\d)")

# Keyword pattern for attribute-name matching (barcode-type targets).
_BARCODE_NAME_RE = re.compile(
    r"штрихкод|штрих.код|barcode|гтин|gtin|\bean\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Checksum helpers (GS1 standard)
# ---------------------------------------------------------------------------


def _ean13_checksum_valid(digits: str) -> bool:
    """Return True if the 13-digit string has a valid GS1 EAN-13 check digit.

    Algorithm: alternating weights 1 and 3 on the first 12 digits, check that
    (10 - (total % 10)) % 10 == 13th digit.
    """
    if len(digits) != 13 or not digits.isdigit():
        return False
    total = sum(
        int(d) * (3 if i % 2 else 1)
        for i, d in enumerate(digits[:12])
    )
    expected = (10 - (total % 10)) % 10
    return int(digits[12]) == expected


def _upc_a_checksum_valid(digits: str) -> bool:
    """Return True if the 12-digit string has a valid GS1 UPC-A check digit.

    UPC-A uses the same algorithm as EAN-13 but with the *opposite* weight
    pattern: odd positions (1-based) get weight 3, even positions get weight 1.
    That is: first digit weight 3, second digit weight 1, etc.
    """
    if len(digits) != 12 or not digits.isdigit():
        return False
    total = sum(
        int(d) * (3 if i % 2 else 1)
        for i, d in enumerate(digits[:11])
    )
    expected = (10 - (total % 10)) % 10
    return int(digits[11]) == expected


def _ean8_checksum_valid(digits: str) -> bool:
    """Return True if the 8-digit string has a valid GS1 EAN-8 check digit.

    GS1 EAN-8 weight pattern: positions 1,3,5,7 (1-based, i.e. indices 0,2,4,6)
    get weight 3; positions 2,4,6 (indices 1,3,5) get weight 1.
    This is the opposite of EAN-13/UPC-A where position 1 (index 0) has weight 1.
    """
    if len(digits) != 8 or not digits.isdigit():
        return False
    total = sum(
        int(d) * (3 if i % 2 == 0 else 1)
        for i, d in enumerate(digits[:7])
    )
    expected = (10 - (total % 10)) % 10
    return int(digits[7]) == expected


def validate_barcode(digits: str) -> bool:
    """Return True if `digits` is a checksummed EAN-13, UPC-A, or EAN-8."""
    length = len(digits)
    if length == 13:
        return _ean13_checksum_valid(digits)
    if length == 12:
        return _upc_a_checksum_valid(digits)
    if length == 8:
        return _ean8_checksum_valid(digits)
    return False


# ---------------------------------------------------------------------------
# Target-name detection
# ---------------------------------------------------------------------------


def is_barcode_target(target: TargetAttribute) -> bool:
    """True if the target attribute is a barcode/EAN/GTIN field."""
    return bool(_BARCODE_NAME_RE.search(target.name))


# ---------------------------------------------------------------------------
# Text scanning
# ---------------------------------------------------------------------------


def extract_barcodes_from_text(text: str) -> list[str]:
    """Return all distinct valid barcode strings found verbatim in `text`.

    Scans for 8-, 12-, and 13-digit sequences, validates checksum for each,
    and returns deduplicated results preserving order of first occurrence.
    """
    if not text:
        return []
    seen: set[str] = set()
    results: list[str] = []
    for m in _DIGIT_SEQ_RE.finditer(text):
        candidate = m.group(1)
        if candidate not in seen and validate_barcode(candidate):
            seen.add(candidate)
            results.append(candidate)
    return results


# ---------------------------------------------------------------------------
# Judge — deterministic: only accepts strings that pass checksum
# ---------------------------------------------------------------------------


class BarcodeJudge(LlmJudge):
    """Deterministic judge: accepts only checksum-valid barcode strings."""

    source = Source.DESCRIPTION  # barcode fills carry Source.DESCRIPTION

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        raw = str(value.value).strip()
        digits = re.sub(r"\D", "", raw)
        return validate_barcode(digits)


# ---------------------------------------------------------------------------
# BarcodeSource
# ---------------------------------------------------------------------------


class BarcodeSource(AttributeSource):
    """Verbatim EAN/barcode extractor — zero LLM, zero network.

    Scans three text pools in priority order:
      1. context.ean — already extracted by caller (highest authority).
      2. product_description — seller-provided description text.
      3. `source_text` kwarg — any additional fetched text passed by pipeline.

    Only fills barcode-type target attributes (detected by name keyword).
    Only outputs checksum-valid EAN-13 / UPC-A / EAN-8 values.
    """

    @property
    def source_type(self) -> Source:
        return Source.DESCRIPTION

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        return is_barcode_target(target)

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
        source_text: Optional[str] = None,
    ) -> list[AttributeValue]:
        # Only barcode-type targets
        barcode_targets = [t for t in targets if is_barcode_target(t)]
        if not barcode_targets:
            return []

        # Skip targets already filled with high confidence
        already_filled_ids: set[int] = set()
        if already_filled:
            already_filled_ids = {
                av.attribute_id
                for av in already_filled
                if av.is_confident()
            }
        unfilled = [t for t in barcode_targets if t.id not in already_filled_ids]
        if not unfilled:
            return []

        # Gather all valid barcodes from available text pools
        candidates: list[str] = []
        seen: set[str] = set()

        def _add_from(text: Optional[str]) -> None:
            for bc in extract_barcodes_from_text(text or ""):
                if bc not in seen:
                    seen.add(bc)
                    candidates.append(bc)

        # Pool 1: context.ean (pre-populated by caller or vision stage)
        ean_val = (context.ean or "").strip()
        if ean_val:
            digits = re.sub(r"\D", "", ean_val)
            if validate_barcode(digits) and digits not in seen:
                seen.add(digits)
                candidates.append(digits)

        # Pool 2: product description
        _add_from(context.product_description)

        # Pool 3: extra source_text (web_search summary, card text, etc.)
        _add_from(source_text)

        if not candidates:
            logger.debug(
                "[BarcodeSource] product=%s: no valid barcodes found in any text pool",
                context.product_id,
            )
            return []

        # Use the first candidate (highest-priority pool came first).
        # If context.ean was valid it will be first; otherwise description-first.
        best = candidates[0]

        results: list[AttributeValue] = []
        for target in unfilled:
            results.append(AttributeValue(
                attribute_id=target.id,
                value=best,
                confidence=_BARCODE_CONFIDENCE,
                source=Source.DESCRIPTION,
                evidence=f"barcode:verbatim+checksum: {best}",
                semantic_type=target.semantic_type or "ean",
                is_collection=target.is_collection,
            ))
            logger.info(
                "[BarcodeSource] product=%s: filled attr=%s name=%r value=%r",
                context.product_id, target.id, target.name, best,
            )
        return results

    def get_judge(self) -> LlmJudge:
        return BarcodeJudge()
