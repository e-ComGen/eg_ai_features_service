"""Public JSON contract v1 for eg_seller_audit. Pydantic models — do not change field names/types without bumping schema_version."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel
import re


class SellerInfo(BaseModel):
    id: str
    display_name: str
    display_name_masked: str
    rating: Optional[float] = None


class SamplingInfo(BaseModel):
    method: str
    seed: str


class ScoreDistBucket(BaseModel):
    bucket: str
    n: int


class Scores(BaseModel):
    combined_fill_median: float
    combined_fill_mean: float
    required_fill_median: Optional[float] = None
    optional_honest_fill_median: float
    distribution: list[ScoreDistBucket]


class MissingField(BaseModel):
    name: str
    missing_share: float


class CategoryBreakdown(BaseModel):
    subject_id: int
    subject_name: Optional[str]
    n_products: int
    required_fill: Optional[float] = None
    optional_honest_fill: float
    top_missing_fields: list[MissingField]


class ProductScore(BaseModel):
    nm_id: int
    subject_id: int
    req: str
    opt: str
    combined: float


class GateInfo(BaseModel):
    thin_content_ok: bool
    sample_size_threshold: int = 30
    category_threshold: int = 2
    reason: Optional[str] = None


class UnresolvedInfo(BaseModel):
    schema_unresolved: int = 0
    card_fetch_failed: int = 0


class LegalInfo(BaseModel):
    data_source: str = "public_marketplace_pages"
    opt_out_url: Optional[str] = None


class AuditReport(BaseModel):
    schema_version: int = 1
    marketplace: str
    seller: SellerInfo
    generated_at: str
    catalog_total: int
    sample_size: int
    sampling: SamplingInfo
    scores: Scores
    by_category: list[CategoryBreakdown]
    top_missing_fields_overall: list[MissingField]
    products: list[ProductScore]
    gate: GateInfo
    unresolved: UnresolvedInfo
    legal: LegalInfo


def mask_seller_display_name(name: str) -> str:
    """Mask a seller display name if it looks like a Russian individual entrepreneur's
    full name (FIO): exactly 3 whitespace-separated tokens, each token starts with an
    uppercase Cyrillic letter followed by lowercase Cyrillic letters only (a simple
    regex per token: ^[А-ЯЁ][а-яё]+$). If it matches -> return
    f"{tokens[0]} {tokens[1][0]}.{tokens[2][0]}." (surname + first-name-initial +
    patronymic-initial). Otherwise return name unchanged (this covers legal entities,
    brand names, already-masked names, non-3-token names, non-Cyrillic names, etc.)."""
    if not name:
        return name

    tokens = name.split()
    if len(tokens) != 3:
        return name

    token_pattern = re.compile(r'^[А-ЯЁ][а-яё]+$')
    for token in tokens:
        if not token_pattern.match(token):
            return name

    return f"{tokens[0]} {tokens[1][0]}.{tokens[2][0]}."
