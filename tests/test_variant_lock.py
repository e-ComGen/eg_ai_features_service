"""
Variant-lock regression tests.

matcher.py constrains dropdown (type='select') features to the provided
options list using fuzzy + cosine match. If someone breaks the lock,
free-text from the description (e.g. brand 'Apple') will leak through.

These tests guard against both directions:
- negative: must NOT return a brand outside the variant list
- positive: must correctly pick the variant when the brand IS in the list
"""

import pytest

from tests.conftest import extract_value


pytestmark = [pytest.mark.integration, pytest.mark.real_ai]


def _product(pid: int, name: str, description: str, category_id: int = 999) -> dict:
    return {
        "id": pid,
        "category_id": category_id,
        "id_path": "",
        "name": name,
        "description": description,
        "price": 0,
        "context": {"existing_features": {}, "company_id": 0},
        "languages": ["en"],
    }


def _dropdown(options: list[str]) -> dict:
    """Build a select-type feature schema with allowed variants."""
    return {"type": "select", "options": options, "suffix": "", "prefix": ""}


def test_dropdown_does_not_invent_brand(call_worker):
    """
    Product clearly mentions brand 'Apple'. Variant list is {Dyson, Samsung,
    LG}. The pipeline MUST NOT return 'Apple' as the Brand — the dropdown
    lock should either pick one of the variants (by fuzzy mismatch fallback)
    or return empty / None. It must never leak free-text.
    """
    product = _product(
        pid=9101,
        name="Apple MacBook Pro 16",
        description=(
            "Apple MacBook Pro 16-inch laptop with M3 Max chip, 36GB unified "
            "memory and 1TB SSD storage. Manufactured by Apple."
        ),
    )
    allowed = ["Dyson", "Samsung", "LG"]
    schema = {"Brand": _dropdown(allowed)}

    body = call_worker(product, schema)
    value = extract_value(body, 9101, "Brand")

    if value in (None, ""):
        return  # acceptable — lock kicked in by refusing to assign

    assert value in allowed, (
        f"\nVARIANT LOCK BREACH:\n"
        f"  Brand returned {value!r}, which is OUTSIDE allowed variants "
        f"{allowed}.\n"
        f"  This means the dropdown constraint in matcher.py is broken: "
        f"the pipeline leaked free-text from the description "
        f"(probably 'Apple') instead of constraining to options."
    )
    # Extra explicit guard against the specific leak pattern.
    assert value != "Apple", "Pipeline returned literal 'Apple' as Brand"


def test_dropdown_picks_correct_variant_when_present(call_worker):
    """
    Positive sanity check: with 'Dyson' both in the description AND in the
    variant list, the pipeline MUST pick 'Dyson'. Without this test, the
    negative one above could pass via an over-strict pipeline that just
    returns None for every dropdown.
    """
    product = _product(
        pid=9102,
        name="Dyson V15 Detect cordless vacuum",
        description=(
            "Dyson V15 Detect cordless vacuum cleaner with 240AW suction "
            "power. Brand: Dyson."
        ),
    )
    allowed = ["Dyson", "Samsung", "LG"]
    schema = {"Brand": _dropdown(allowed)}

    body = call_worker(product, schema)
    value = extract_value(body, 9102, "Brand")

    assert value == "Dyson", (
        f"\nExpected pipeline to pick 'Dyson' from variants {allowed} when "
        f"the description literally says 'Brand: Dyson'.\n"
        f"  got value={value!r}\n"
        f"  Either matcher.py is over-rejecting, or the dropdown selection "
        f"path isn't reached."
    )
