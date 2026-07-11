"""Deterministic stratified sampling of seller products by WB subject_id."""

from __future__ import annotations
import random

DEFAULT_SAMPLE_TARGET = 300


def stratified_sample(products: list[dict], target_size: int, seed: str) -> list[dict]:
    """Deterministically sample `target_size` products from `products`, stratified
    proportionally by the "subject_id" key of each product dict (a product with
    subject_id=None goes into a group keyed "unknown" -- do not drop it).

    Behavior:
    - rng = random.Random(seed) for full determinism given the same seed.
    - If len(products) <= target_size: return ALL products, sorted by the "nm_id" key
      ascending (for deterministic output order).
    - Otherwise: group products by subject_id (None -> "unknown" group, an explicit
      dict key alongside real subject_id values -- Python allows None as a dict key too,
      just be consistent, use the string "unknown" as the group key when subject_id is
      None). For each group, compute a proportional quota:
      quota = max(1, round(target_size * len(group) / len(products))) but never more
      than len(group) (cannot sample more than the group has). Then within each group
      use rng.sample(group, k=min(quota, len(group))) to pick a random subset
      (Random.sample requires a sequence; list(group) is fine).
    - After computing per-group picks, the total may slightly exceed or fall short of
      target_size due to rounding + the "minimum 1 per non-empty group" rule -- that is
      ACCEPTABLE, do not do complex rebalancing, just concatenate all per-group samples.
    - Sort the final combined list by "nm_id" key ascending before returning (deterministic
      output order regardless of dict iteration order).
    - Grouping must be built by iterating products in their given order and appending to
      a dict[str, list[dict]] keyed by str(subject_id) if subject_id is not None else
      "unknown" (str() call keeps the group key type consistent and hashable regardless
      of whether subject_id is int or already str in the input dicts).
    - Do not mutate the input `products` list or its dicts.
    """
    if len(products) <= target_size:
        return sorted(products, key=lambda p: p["nm_id"])

    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for product in products:
        key = str(product["subject_id"]) if product["subject_id"] is not None else "unknown"
        if key not in groups:
            groups[key] = []
        groups[key].append(product)

    total = len(products)
    sampled: list[dict] = []
    for group_key, group in groups.items():
        quota = max(1, round(target_size * len(group) / total))
        quota = min(quota, len(group))
        sampled.extend(rng.sample(group, k=quota))

    sampled.sort(key=lambda p: p["nm_id"])
    return sampled
