"""
Pydantic schema validation tests for /process-batch.

Malformed payloads must come back as 422 (FastAPI/pydantic validation error),
never 500 (uncaught exception inside the handler).

No real_ai marker — validation must happen before any LLM call.
"""

import pytest
import requests


pytestmark = [pytest.mark.integration]


def _post(worker_url: str, secret: str, payload: dict, timeout: int = 10):
    return requests.post(
        f"{worker_url}/process-batch",
        json=payload,
        headers={"X-Internal-Secret": secret},
        timeout=timeout,
    )


def test_empty_products_array(worker_url, internal_secret, worker_alive):
    """
    `products: []` is structurally valid pydantic (List can be empty), so the
    worker should accept it and return 200 with no data — NOT 500. If a future
    schema change makes empty lists invalid (min_items=1), 422 is also OK.
    """
    r = _post(worker_url, internal_secret, {
        "client_id": 9999,
        "products": [],
        "schemas": {},
        "use_cache": False,
        "research_mode": "off",
    })
    assert r.status_code != 500, f"Worker 500'd on empty products: {r.text[:300]}"
    assert r.status_code in (200, 422), (
        f"Expected 200 or 422, got {r.status_code}: {r.text[:300]}"
    )
    if r.status_code == 200:
        body = r.json()
        assert body.get("status") == "success"
        assert body.get("data") == [] or body.get("data") is None


def test_missing_products_field(worker_url, internal_secret, worker_alive):
    """`products` is required — omitting it must 422, never 500."""
    r = _post(worker_url, internal_secret, {
        "client_id": 9999,
        "schemas": {},
        "use_cache": False,
        "research_mode": "off",
    })
    assert r.status_code == 422, (
        f"Expected 422 for missing 'products', got {r.status_code}: {r.text[:300]}"
    )


def test_missing_schemas(worker_url, internal_secret, worker_alive):
    """
    The Python model field is `schemas` (not `languages` — `languages` lives
    on each product). `schemas` is required at the batch root; omitting must
    422, never 500.
    """
    r = _post(worker_url, internal_secret, {
        "client_id": 9999,
        "products": [{
            "id": 1,
            "category_id": 1,
            "name": "x",
            "description": "x",
            "context": {"existing_features": {}, "company_id": 0},
            "languages": ["en"],
        }],
        "use_cache": False,
        "research_mode": "off",
    })
    assert r.status_code == 422, (
        f"Expected 422 for missing 'schemas', got {r.status_code}: {r.text[:300]}"
    )


def test_product_missing_required_field(worker_url, internal_secret, worker_alive):
    """A product without required `id` / `name` must 422, never 500."""
    r = _post(worker_url, internal_secret, {
        "client_id": 9999,
        "products": [{
            # 'id' intentionally missing
            "category_id": 1,
            "name": "x",
            "description": "x",
            "context": {"existing_features": {}, "company_id": 0},
            "languages": ["en"],
        }],
        "schemas": {"1": {"X": {"type": "text", "options": []}}},
        "use_cache": False,
        "research_mode": "off",
    })
    assert r.status_code == 422, (
        f"Expected 422 for product missing 'id', got {r.status_code}: {r.text[:300]}"
    )
