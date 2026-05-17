"""
Tests for prompt_hint injection into the LLM prompt.

TC-PH-001  test_hint_passes_through_to_llm_prompt
    Patches the pipeline.extract_feature call, captures the product_text argument,
    asserts the hint appears verbatim in it.

TC-PH-002  test_hint_with_brand_constraint (real_ai)
    Sends a real request with hint = "Extract only the manufacturer brand, not product
    model name" for a Brand feature. Product: "Apple MacBook Pro 16 - M3 Max".
    Expects result == "Apple", not "MacBook Pro".
"""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from app.models import FeatureOption, ProductData, ProductContext
from app.services.job_processor import JobProcessor
from tests.conftest import extract_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_product(pid: int = 1, name: str = "Apple MacBook Pro 16 — M3 Max",
                  description: str = "Apple MacBook Pro 16-inch laptop with M3 Max chip.",
                  category_id: int = 999) -> ProductData:
    return ProductData(
        id=pid,
        category_id=category_id,
        id_path="",
        name=name,
        description=description,
        price=0.0,
        context=ProductContext(existing_features={}, company_id=0),
        languages=["en"],
    )


def _make_schema_with_hint(hint: str | None = None) -> dict:
    return {
        "Brand": FeatureOption(
            type="text",
            options=[],
            prompt_hint=hint,
        )
    }


# ---------------------------------------------------------------------------
# TC-PH-001 — unit: hint reaches product_text sent to pipeline
# ---------------------------------------------------------------------------

def test_hint_passes_through_to_llm_prompt():
    """
    Patch AiFeaturePipeline.extract_feature; verify the product_text argument
    the JobProcessor passes to it contains the prompt_hint verbatim when set.
    """
    hint_text = "Extract only the manufacturer brand, not product model name"
    product = _make_product()
    schema = _make_schema_with_hint(hint=hint_text)

    captured_calls = []

    async def fake_extract_feature(product_text, feature_name, **kwargs):
        captured_calls.append({"product_text": product_text, "feature_name": feature_name})
        return {
            "value": "Apple",
            "tokens": 10,
            "router_debug": {},
            "extraction_reasoning": "test",
            "deduced_context": None,
            "source": "description",
            "source_urls": None,
        }

    mock_pipeline = MagicMock()
    mock_pipeline.extract_feature = fake_extract_feature

    mock_db_cache = MagicMock()
    mock_db_cache.get_cached_value = AsyncMock(return_value=None)
    mock_db_cache.set_cached_value = AsyncMock()

    mock_matcher = MagicMock()
    mock_matcher.find_best_match = MagicMock(return_value=None)

    semaphore = asyncio.Semaphore(10)
    processor = JobProcessor(mock_pipeline, mock_db_cache, mock_matcher, semaphore)

    asyncio.run(processor.process_product(product, schema, client_id=0, use_cache=False))

    assert captured_calls, "extract_feature must have been called"
    product_text_used = captured_calls[0]["product_text"]

    assert hint_text in product_text_used, (
        f"prompt_hint must appear verbatim in the product_text passed to the pipeline.\n"
        f"Hint: {hint_text!r}\n"
        f"Got product_text: {product_text_used!r}"
    )
    assert "OPERATOR CONSTRAINT" in product_text_used, (
        "The constraint section header must be present when a hint is injected"
    )


# ---------------------------------------------------------------------------
# TC-PH-002 — real AI: hint steers result toward correct brand
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TC-PH-003 — hint with special chars does not break prompt assembly
# ---------------------------------------------------------------------------

def test_hint_with_special_chars_does_not_break_prompt():
    """
    prompt_hint containing backticks, newlines, curly braces, and double-quotes
    must not cause a JSON parse error or template injection in the product_text
    assembled by JobProcessor. The pipeline call must succeed and the raw hint
    text must appear verbatim in the captured product_text string.
    """
    tricky_hint = 'Use `exact` match.\nDo NOT use {} braces.\nAvoid "quoted" values.'
    product = _make_product()
    schema = _make_schema_with_hint(hint=tricky_hint)

    captured_calls = []

    async def fake_extract_feature(product_text, feature_name, **kwargs):
        captured_calls.append({"product_text": product_text})
        return {
            "value": "Apple",
            "tokens": 10,
            "router_debug": {},
            "extraction_reasoning": "test",
            "deduced_context": None,
            "source": "description",
            "source_urls": None,
        }

    mock_pipeline = MagicMock()
    mock_pipeline.extract_feature = fake_extract_feature

    mock_db_cache = MagicMock()
    mock_db_cache.get_cached_value = AsyncMock(return_value=None)
    mock_db_cache.set_cached_value = AsyncMock()

    mock_matcher = MagicMock()
    mock_matcher.find_best_match = MagicMock(return_value=None)

    semaphore = asyncio.Semaphore(10)
    processor = JobProcessor(mock_pipeline, mock_db_cache, mock_matcher, semaphore)

    # Must not raise (no JSON parse error, no template exception).
    asyncio.run(processor.process_product(product, schema, client_id=0, use_cache=False))

    assert captured_calls, "extract_feature must have been called even with a tricky hint"
    product_text_used = captured_calls[0]["product_text"]

    # The full hint string must appear verbatim — no escaping, truncation, or injection.
    assert tricky_hint in product_text_used, (
        f"Tricky prompt_hint must appear verbatim in product_text.\n"
        f"Hint: {tricky_hint!r}\nGot product_text: {product_text_used!r}"
    )


# ---------------------------------------------------------------------------
# TC-PH-004 — empty prompt_hint does not inject constraint block into prompt
# ---------------------------------------------------------------------------

def test_empty_prompt_hint_field_does_not_inject_constraint_block():
    """
    When prompt_hint is '' (empty string), the OPERATOR CONSTRAINT block must NOT
    appear in the product_text passed to the pipeline. Empty-string hints should
    be filtered out, not formatted as an empty constraint section.
    """
    product = _make_product()
    schema = _make_schema_with_hint(hint='')  # explicit empty string

    captured_calls = []

    async def fake_extract_feature(product_text, feature_name, **kwargs):
        captured_calls.append({"product_text": product_text})
        return {
            "value": "Apple",
            "tokens": 10,
            "router_debug": {},
            "extraction_reasoning": "test",
            "deduced_context": None,
            "source": "description",
            "source_urls": None,
        }

    mock_pipeline = MagicMock()
    mock_pipeline.extract_feature = fake_extract_feature

    mock_db_cache = MagicMock()
    mock_db_cache.get_cached_value = AsyncMock(return_value=None)
    mock_db_cache.set_cached_value = AsyncMock()

    mock_matcher = MagicMock()
    mock_matcher.find_best_match = MagicMock(return_value=None)

    semaphore = asyncio.Semaphore(10)
    processor = JobProcessor(mock_pipeline, mock_db_cache, mock_matcher, semaphore)

    asyncio.run(processor.process_product(product, schema, client_id=0, use_cache=False))

    assert captured_calls, "extract_feature must have been called"
    product_text_used = captured_calls[0]["product_text"]

    assert "OPERATOR CONSTRAINT" not in product_text_used, (
        "An empty prompt_hint must NOT inject the OPERATOR CONSTRAINT block into the prompt. "
        f"Got product_text: {product_text_used!r}"
    )


# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.real_ai
def test_hint_with_brand_constraint(call_worker):
    """
    Real OpenAI call. Without hint, AI might return "MacBook Pro" for Brand.
    With hint = 'Extract only the manufacturer brand, not product model name',
    it must return "Apple".
    """
    product = {
        "id": 9001,
        "category_id": 777,
        "id_path": "",
        "name": "Apple MacBook Pro 16 — M3 Max",
        "description": "Apple MacBook Pro 16-inch laptop with M3 Max chip, 36 GB RAM.",
        "price": 3499,
        "context": {"existing_features": {}, "company_id": 0},
        "languages": ["en"],
    }
    schema = {
        "Brand": {
            "type": "text",
            "options": [],
            "prompt_hint": "Extract only the manufacturer brand, not product model name",
        }
    }

    body = call_worker(product, schema)
    value = extract_value(body, product_id=9001, feature_name="Brand")

    assert value is not None, "Brand must be extracted (not None)"
    # extract_value can return either a plain string OR a per-language dict
    # ({"en": "Apple", "ru": "Apple"}) depending on the worker's multi-lang mode.
    extracted = value
    if isinstance(extracted, dict):
        extracted = extracted.get("en") or next(iter(extracted.values()), "")
    assert str(extracted).strip().lower() == "apple", (
        f"With the manufacturer-only hint, Brand must resolve to 'Apple', got: {value!r}"
    )
