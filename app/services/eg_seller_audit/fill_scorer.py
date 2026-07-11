"""Per-product fill-rate scoring against the WB characteristics schema
(required vs optional_honest denominator)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from app.services.enrichment.sources.wb_card_cdn import extract_options
from app.services.enrichment.strategies.dictionaries.loader import get_wb_characteristics_for_category
from app.services.enrichment.strategies.dictionaries.wb_field_classifier import is_wb_platform_field


@dataclass
class ProductFillResult:
    nm_id: int
    subject_id: int
    required_total: int
    required_filled: int
    optional_total: int
    optional_filled: int
    combined: float
    missing_required: list[str]
    missing_optional: list[str]

    @property
    def required_applicable(self) -> bool:
        """True when the WB required-schema is applicable to this product at all
        (required_total > 0). WB systematically emits zero required=true
        characteristics for many real categories (fashion/accessories) — this is a
        real property of the data, not a bug. Downstream aggregation must NOT treat
        required_total==0 as req_fill=0.0 (N/A != 0%); use this property instead of
        re-deriving `required_total > 0` ad hoc in multiple places."""
        return self.required_total > 0


def score_product(card: Optional[dict], subject_id: int) -> Optional[ProductFillResult]:
    """Score one product's field-fill rate against its WB category schema.

    NOTE: this function does not know the product's nm_id (it only receives the raw
    card dict). It sets ProductFillResult.nm_id=0 as a placeholder; the CALLER (which
    already has the nm_id, e.g. the audit_seller pipeline) is responsible for setting
    `result.nm_id = nm_id` on the returned object before using it (the dataclass is
    mutable, this is a cheap post-construction assignment, not a design flaw).
    """
    if card is None:
        return None

    chars = get_wb_characteristics_for_category(subject_id)
    if not chars:
        return None

    required_chars = [c for c in chars if c.get("required")]
    optional_chars = [c for c in chars if c.get("popular") and not is_wb_platform_field(c)]

    filled_names = {o["name"].strip().lower() for o in extract_options(card) if isinstance(o, dict) and o.get("name")}

    required_total = len(required_chars)
    required_filled = 0
    missing_required = []
    for c in required_chars:
        name = c.get("name", "")
        if name.strip().lower() in filled_names:
            required_filled += 1
        else:
            missing_required.append(name)

    optional_total = len(optional_chars)
    optional_filled = 0
    missing_optional = []
    for c in optional_chars:
        name = c.get("name", "")
        if name.strip().lower() in filled_names:
            optional_filled += 1
        else:
            missing_optional.append(name)

    denominator = required_total + optional_total
    combined = (required_filled + optional_filled) / denominator if denominator > 0 else 0.0

    return ProductFillResult(
        nm_id=0,
        subject_id=subject_id,
        required_total=required_total,
        required_filled=required_filled,
        optional_total=optional_total,
        optional_filled=optional_filled,
        combined=combined,
        missing_required=missing_required,
        missing_optional=missing_optional,
    )
