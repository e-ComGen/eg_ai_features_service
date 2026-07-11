"""Seller-level aggregation: medians, distribution, category breakdown,
thin-content gate."""

from __future__ import annotations

import statistics
from collections import defaultdict, Counter

from .fill_scorer import ProductFillResult
from .report import (
    AuditReport,
    SellerInfo,
    SamplingInfo,
    ScoreDistBucket,
    Scores,
    MissingField,
    CategoryBreakdown,
    ProductScore,
    GateInfo,
    UnresolvedInfo,
    LegalInfo,
)
from app.services.enrichment.strategies.dictionaries.loader import get_wb_subject_name


def _bucket_distribution(combined_list: list[float]) -> list[ScoreDistBucket]:
    buckets = [
        ("0-20", 0.0, 20.0),
        ("20-40", 20.0, 40.0),
        ("40-60", 40.0, 60.0),
        ("60-80", 60.0, 80.0),
        ("80-100", 80.0, 100.0),
    ]
    result: list[ScoreDistBucket] = []
    for label, lo, hi in buckets:
        count = 0
        for val in combined_list:
            pct = val * 100.0
            if label == "80-100":
                if lo <= pct <= hi:
                    count += 1
            else:
                if lo <= pct < hi:
                    count += 1
        result.append(ScoreDistBucket(bucket=label, n=count))
    return result


def _top_missing(
    results_group: list[ProductFillResult],
    n_products: int,
    top_n: int,
) -> list[MissingField]:
    counter: Counter[str] = Counter()
    for r in results_group:
        for name in r.missing_required:
            counter[name] += 1
        for name in r.missing_optional:
            counter[name] += 1
    most_common = counter.most_common(top_n)
    missing_fields: list[MissingField] = []
    for name, count in most_common:
        missing_share = count / n_products if n_products > 0 else 0.0
        missing_fields.append(MissingField(name=name, missing_share=missing_share))
    return missing_fields


def aggregate(
    results: list[ProductFillResult],
    catalog_total: int,
    sample_size: int,
    seller: SellerInfo,
    marketplace: str,
    seed: str,
    generated_at: str,
    unresolved: UnresolvedInfo,
) -> AuditReport:
    # a) Per-product fill fractions
    combined_list: list[float] = [r.combined for r in results]
    req_list: list[float] = []
    opt_list: list[float] = []
    for r in results:
        # required_total==0 -> WB required-schema not applicable to this product;
        # exclude entirely from req_list (N/A != 0%), don't zero-pad the stats.
        if r.required_applicable:
            req_fill = r.required_filled / r.required_total
            req_list.append(req_fill)
        opt_fill = r.optional_filled / r.optional_total if r.optional_total > 0 else 0.0
        opt_list.append(opt_fill)

    # b) Medians and means
    if results:
        combined_fill_median = statistics.median(combined_list)
        combined_fill_mean = statistics.mean(combined_list)
        required_fill_median = statistics.median(req_list) if req_list else None
        optional_honest_fill_median = statistics.median(opt_list)
    else:
        combined_fill_median = 0.0
        combined_fill_mean = 0.0
        required_fill_median = None
        optional_honest_fill_median = 0.0

    # c) Distribution
    distribution_list = _bucket_distribution(combined_list)

    # d) By category
    grouped: dict[int, list[ProductFillResult]] = defaultdict(list)
    for r in results:
        grouped[r.subject_id].append(r)

    by_category_list: list[CategoryBreakdown] = []
    for subject_id in sorted(grouped.keys()):
        group = grouped[subject_id]
        subject_name = get_wb_subject_name(subject_id)
        n_products = len(group)
        group_req_fills = []
        group_opt_fills = []
        for r in group:
            if r.required_applicable:
                req_fill = r.required_filled / r.required_total
                group_req_fills.append(req_fill)
            opt_fill = r.optional_filled / r.optional_total if r.optional_total > 0 else 0.0
            group_opt_fills.append(opt_fill)
        required_fill = statistics.mean(group_req_fills) if group_req_fills else None
        optional_honest_fill = statistics.mean(group_opt_fills) if group_opt_fills else 0.0
        top_missing_fields = _top_missing(group, n_products, 5)
        by_category_list.append(
            CategoryBreakdown(
                subject_id=subject_id,
                subject_name=subject_name,
                n_products=n_products,
                required_fill=required_fill,
                optional_honest_fill=optional_honest_fill,
                top_missing_fields=top_missing_fields,
            )
        )

    # e) Top missing fields overall
    n_total = len(results)
    top_missing_overall_list = _top_missing(results, n_total, 10)

    # f) Products
    products_list: list[ProductScore] = []
    for r in results:
        products_list.append(
            ProductScore(
                nm_id=r.nm_id,
                subject_id=r.subject_id,
                req=f"{r.required_filled}/{r.required_total}",
                opt=f"{r.optional_filled}/{r.optional_total}",
                combined=r.combined,
            )
        )

    # g) Gate
    unique_categories = {r.subject_id for r in results}
    category_count = len(unique_categories)
    if sample_size >= 30 and category_count >= 2:
        gate_info = GateInfo(
            thin_content_ok=True,
            sample_size_threshold=30,
            category_threshold=2,
            reason=None,
        )
    else:
        if sample_size < 30 and category_count < 2:
            reason = "sample_size<30 and categories<2"
        elif sample_size < 30:
            reason = "sample_size<30"
        else:
            reason = "categories<2"
        gate_info = GateInfo(
            thin_content_ok=False,
            sample_size_threshold=30,
            category_threshold=2,
            reason=reason,
        )

    # h) Return
    return AuditReport(
        schema_version=1,
        marketplace=marketplace,
        seller=seller,
        generated_at=generated_at,
        catalog_total=catalog_total,
        sample_size=sample_size,
        sampling=SamplingInfo(method="stratified_by_subject", seed=seed),
        scores=Scores(
            combined_fill_median=combined_fill_median,
            combined_fill_mean=combined_fill_mean,
            required_fill_median=required_fill_median,
            optional_honest_fill_median=optional_honest_fill_median,
            distribution=distribution_list,
        ),
        by_category=by_category_list,
        top_missing_fields_overall=top_missing_overall_list,
        products=products_list,
        gate=gate_info,
        unresolved=unresolved,
        legal=LegalInfo(data_source="public_marketplace_pages", opt_out_url=None),
    )
