"""Aggregator — combines per-category characteristics into the final Ozon dictionary."""


def aggregate_by_category(
    chars_by_category: dict,
    categories: list[dict],
) -> dict:
    """Build the final Ozon dictionary structure.

    Input:
        chars_by_category: {category_id: set of (key, display_name) tuples}
        categories: list of category metadata dicts from seed_loader

    Output:
    {
        "category_id": {
            "name": "Смартфоны",
            "path": ["Электроника", "Смартфоны"],
            "characteristics": [
                {"key": "brand", "name": "Бренд"},
                {"key": "color", "name": "Цвет"},
                ...
            ]
        }
    }

    Categories with no collected characteristics are omitted.
    Characteristics are sorted by key for deterministic, diff-friendly output.
    """
    cats_by_id = {c["id"]: c for c in categories if c.get("id") is not None}
    final: dict[str, dict] = {}

    for cat_id, chars in chars_by_category.items():
        if not chars:
            continue

        cat_meta = cats_by_id.get(cat_id, {})
        # Sort by (key, display_name) for stable output
        sorted_chars = sorted(chars, key=lambda x: (str(x[0]).lower(), str(x[1]).lower()))

        final[str(cat_id)] = {
            "name": cat_meta.get("name", ""),
            "path": cat_meta.get("path", []),
            "characteristics": [
                {"key": key, "name": display_name}
                for key, display_name in sorted_chars
            ],
        }

    return final
