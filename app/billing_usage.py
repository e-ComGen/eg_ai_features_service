"""Billing usage aggregation for /process-batch responses.

Pure functions — no I/O, no side effects. Importable and unit-testable in isolation.
"""

from __future__ import annotations


def _count_product_features(debug_info: dict) -> tuple[int, int]:
    """Count generated and cached features for a single product.

    Args:
        debug_info: Mapping of feature_name -> {source: str, ...}.

    Returns:
        (generated_count, cached_count) for this product.
    """
    generated = 0
    cached = 0
    for feature_data in debug_info.values():
        source = feature_data.get("source", "") if isinstance(feature_data, dict) else ""
        if source == "cache":
            cached += 1
        else:
            generated += 1
    return generated, cached


def compute_usage(results: list) -> dict:
    """Aggregate billing usage from /process-batch gather results.

    Processes each result entry and computes totals across all products:
    - generated_count: total features with source != "cache"
    - cached_count: total features with source == "cache"
    - generated_products: products with >= 1 generated feature
    - total_tokens: sum of tokens_used across all non-Exception results

    Exception entries and error dicts without debug_info contribute 0 features
    but still add tokens_used if that key is present.

    Args:
        results: Raw list from asyncio.gather — may contain dicts or Exception objects.

    Returns:
        Dict with keys: generated_count, generated_products, cached_count, total_tokens.
    """
    total_generated = 0
    total_cached = 0
    products_with_generated = 0
    total_tokens = 0

    for result in results:
        if isinstance(result, Exception):
            continue

        if not isinstance(result, dict):
            continue

        total_tokens += result.get("tokens_used", 0) or 0

        debug_info = result.get("debug_info")
        if not debug_info or not isinstance(debug_info, dict):
            continue

        gen, cached = _count_product_features(debug_info)
        total_generated += gen
        total_cached += cached
        if gen >= 1:
            products_with_generated += 1

    return {
        "generated_count": total_generated,
        "generated_products": products_with_generated,
        "cached_count": total_cached,
        "total_tokens": total_tokens,
    }
