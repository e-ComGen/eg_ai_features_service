"""eg_seller_audit — stateless seller card-fill audit engine (WB, Phase 1).
Public entry point: audit_seller()."""

from __future__ import annotations

from .report import AuditReport, SellerInfo, UnresolvedInfo
from .report import mask_seller_display_name
from .catalog_provider import WbSellerCatalog
from .sampler import stratified_sample, DEFAULT_SAMPLE_TARGET
from .card_fetcher import fetch_cards
from .fill_scorer import score_product, ProductFillResult
from .aggregator import aggregate

__all__ = ["AuditReport", "audit_seller"]


async def audit_seller(
    seller_id: str,
    marketplace: str,
    generated_at: str,
    sample_size: int = DEFAULT_SAMPLE_TARGET,
) -> AuditReport:
    """Run the full WB seller card-fill audit pipeline end-to-end.

    Phase 1 supports marketplace=="wb" only; any other value raises ValueError.
    """
    if marketplace != "wb":
        raise ValueError("Phase 1: only wb supported")

    provider = WbSellerCatalog()
    products, catalog_total = await provider.fetch_products(seller_id)

    seed = f"{marketplace}:{seller_id}"
    sample = stratified_sample(products, sample_size, seed)
    nm_ids = [p["nm_id"] for p in sample]
    cards = await fetch_cards(nm_ids)

    results: list[ProductFillResult] = []
    card_fetch_failed = 0
    schema_unresolved = 0

    for p in sample:
        nm_id = p["nm_id"]
        subject_id = p.get("subject_id")
        card = cards.get(nm_id)

        if card is None:
            card_fetch_failed += 1
            continue

        if subject_id is None:
            schema_unresolved += 1
            continue

        result = score_product(card, subject_id)
        if result is None:
            schema_unresolved += 1
            continue

        result.nm_id = nm_id
        results.append(result)

    unresolved = UnresolvedInfo(
        schema_unresolved=schema_unresolved,
        card_fetch_failed=card_fetch_failed,
    )

    display_name = f"Seller {seller_id}"
    masked = mask_seller_display_name(display_name)
    seller = SellerInfo(
        id=str(seller_id),
        display_name=display_name,
        display_name_masked=masked,
        rating=None,
    )

    return aggregate(
        results=results,
        catalog_total=catalog_total,
        sample_size=len(sample),
        seller=seller,
        marketplace=marketplace,
        seed=seed,
        generated_at=generated_at,
        unresolved=unresolved,
    )
