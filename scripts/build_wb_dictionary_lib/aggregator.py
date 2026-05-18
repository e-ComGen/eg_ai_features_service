"""Aggregator — combines per-subject characteristics into the final dictionary."""


def aggregate_characteristics_by_subject(
    characteristics_by_subj: dict,
    subjects: list[dict],
) -> dict:
    """Build final dictionary structure.

    Input:
        characteristics_by_subj: {subj_id: set of (char_id, char_name) tuples}
        subjects: list of subject metadata dicts from menu_fetcher

    Output:
    {
        "subject_id": {
            "name": "Кроссовки мужские",
            "path": ["Обувь", "Мужская обувь", "Кроссовки"],
            "characteristics": [
                {"id": 14177419, "name": "Цвет"},
                {"id": 14177421, "name": "Материал верха"},
                ...
            ]
        }
    }
    """
    subjects_by_id = {s["id"]: s for s in subjects}
    final: dict[str, dict] = {}

    for subj_id, chars in characteristics_by_subj.items():
        subj_meta = subjects_by_id.get(subj_id, {})
        # Sort by (char_id, char_name) for stable, deterministic output
        sorted_chars = sorted(chars, key=lambda x: (x[0] or 0, x[1]))
        final[str(subj_id)] = {
            "name": subj_meta.get("name", ""),
            "path": subj_meta.get("path", []),
            "characteristics": [
                {"id": cid, "name": cname} for cid, cname in sorted_chars
            ],
        }

    return final
