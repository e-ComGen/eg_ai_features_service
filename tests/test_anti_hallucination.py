"""
Anti-hallucination regression tests.

Operator bug report (product 403): features with vague names like "NewFeature",
"new_feature", "Feature 1" were getting AI-populated values even when nothing
in the product description could justify them. AI was hallucinating instead of
returning empty/null.

These tests pin down the contract: when the description does NOT contain
evidence for a feature, the pipeline (router + judge + fast-reject) MUST
return None / empty for that feature, never a hallucinated number or word.

All tests in this module are integration tests against the live worker on
port 8001 and consume real OpenAI tokens.
Run only this file with real AI:
    python -m pytest tests/test_anti_hallucination.py -v -m real_ai

To skip in CI / dev without keys:
    python -m pytest -m "not real_ai"
"""

import pytest

from tests.conftest import extract_debug, extract_value


pytestmark = [pytest.mark.integration, pytest.mark.real_ai]


# ---------- shared helpers ----------

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


def _text_feature(suffix: str = "") -> dict:
    return {"type": "text", "options": [], "suffix": suffix, "prefix": ""}


def _hallucinated_message(product_id, feature_name, value, debug):
    return (
        f"\nHALLUCINATION DETECTED:\n"
        f"  product_id={product_id}\n"
        f"  feature='{feature_name}'\n"
        f"  returned value={value!r}\n"
        f"  source={debug.get('source')!r}\n"
        f"  router={debug.get('router')}\n"
        f"  extraction_reasoning={debug.get('extraction_reasoning')}\n"
    )


# ---------- negative cases: AI must return empty ----------

def test_vague_feature_name_returns_empty(call_worker):
    """
    Reproduces operator's product-403 bug.

    The feature name is intentionally meaningless ("NewFeature"). The
    description does describe the product concretely, but nothing in it
    answers the question "what is its NewFeature?". The pipeline must
    return None — NOT pick up '240' or any other unrelated number.
    """
    product = _product(
        pid=403,
        name="Dyson V15 Detect cordless vacuum",
        description=(
            "Dyson V15 Detect cordless vacuum cleaner with 240AW suction "
            "power, 60-minute runtime, and laser dust detection. "
            "Weight: 3.0 kg. Color: yellow."
        ),
    )
    schema = {"NewFeature": _text_feature()}

    body = call_worker(product, schema)
    value = extract_value(body, 403, "NewFeature")
    debug = extract_debug(body, 403, "NewFeature")

    assert value is None or value == "" or str(value).lower() == "none", (
        _hallucinated_message(403, "NewFeature", value, debug)
    )


def test_unrelated_feature_returns_empty(call_worker):
    """
    The product is a vacuum cleaner; the requested feature is 'Screen Resolution'.
    Vacuums don't have screen resolution, and the description doesn't mention
    one. The pipeline must return None.
    """
    product = _product(
        pid=501,
        name="Dyson V15 Detect cordless vacuum",
        description=(
            "Dyson V15 Detect cordless vacuum cleaner with 240AW suction "
            "power and 60-minute runtime."
        ),
    )
    schema = {"Screen Resolution": _text_feature()}

    body = call_worker(product, schema)
    value = extract_value(body, 501, "Screen Resolution")
    debug = extract_debug(body, 501, "Screen Resolution")

    assert value is None or value == "" or str(value).lower() == "none", (
        _hallucinated_message(501, "Screen Resolution", value, debug)
    )


def test_numeric_feature_with_only_unrelated_numbers(call_worker):
    """
    The description contains dimension numbers (240 / 160 / 90) but no battery
    capacity. The pipeline must NOT return any of those numbers for
    'Battery Capacity'. This is the exact pattern of the bug: a stray number
    in the description being grabbed as the answer to an unrelated numeric
    feature.
    """
    product = _product(
        pid=502,
        name="Office desk OD-2400",
        description=(
            "Solid wood office desk, dimensions 240x160x90 cm. "
            "Includes cable management tray and two drawers."
        ),
    )
    schema = {"Battery Capacity": _text_feature(suffix="mAh")}

    body = call_worker(product, schema)
    value = extract_value(body, 502, "Battery Capacity")
    debug = extract_debug(body, 502, "Battery Capacity")

    # Must be empty.
    assert value is None or value == "" or str(value).lower() == "none", (
        _hallucinated_message(502, "Battery Capacity", value, debug)
    )

    # And specifically not any of the dimension numbers.
    if value is not None:
        stringified = str(value)
        for forbidden in ("240", "160", "90"):
            assert forbidden not in stringified, (
                f"Hallucinated a dimension as Battery Capacity: {value!r}"
            )


# ---------- positive case: AI must extract when evidence is present ----------

def test_feature_present_in_description_returns_value(call_worker):
    """
    Sanity check: without this, the negative tests above could pass by an
    over-strict pipeline that just returns None for everything. Here the
    feature IS explicitly stated in the description — the pipeline must
    extract it.
    """
    product = _product(
        pid=601,
        name="Dyson V15 Detect cordless vacuum",
        description=(
            "Dyson V15 Detect cordless vacuum cleaner. Suction power: 240AW. "
            "Runtime: 60 minutes. Weight: 3.0 kg."
        ),
    )
    schema = {"Suction Power": _text_feature(suffix="AW")}

    body = call_worker(product, schema)
    value = extract_value(body, 601, "Suction Power")
    debug = extract_debug(body, 601, "Suction Power")

    assert value is not None and str(value).lower() != "none" and value != "", (
        f"\nExpected pipeline to extract Suction Power=240 from a "
        f"description that literally says '240AW'.\n"
        f"  got value={value!r}\n"
        f"  debug={debug}\n"
    )
    # The number 240 should appear somewhere in the returned value.
    assert "240" in str(value), (
        f"Expected '240' in extracted Suction Power, got {value!r}"
    )
