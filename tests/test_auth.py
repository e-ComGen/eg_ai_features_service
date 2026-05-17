"""
Auth tests for the worker's /process-batch endpoint.

The worker MUST reject requests without (or with a wrong) X-Internal-Secret
header BEFORE invoking the AI pipeline. No real_ai marker — these should
never reach OpenAI.
"""

import pytest
import requests


pytestmark = [pytest.mark.integration]


def _minimal_payload() -> dict:
    return {
        "client_id": 9999,
        "products": [{
            "id": 8001,
            "category_id": 999,
            "id_path": "",
            "name": "Test",
            "description": "Test product",
            "price": 0,
            "context": {"existing_features": {}, "company_id": 0},
            "languages": ["en"],
        }],
        "schemas": {"999": {"X": {"type": "text", "options": [], "suffix": "", "prefix": ""}}},
        "use_cache": False,
        "research_mode": "off",
    }


def test_missing_internal_secret_returns_401(worker_url, worker_alive):
    """No header at all -> rejected, never 500."""
    r = requests.post(
        f"{worker_url}/process-batch",
        json=_minimal_payload(),
        timeout=10,
    )
    # FastAPI Header(...) with no value yields 422 for missing-required-header;
    # 401 / 403 are also acceptable (explicit auth rejection).
    # The contract: NOT 500, NOT 200.
    assert r.status_code != 500, f"Worker 500'd on missing header: {r.text[:300]}"
    assert r.status_code != 200, f"Worker accepted request with no auth: {r.text[:300]}"
    assert r.status_code in (401, 403, 422), (
        f"Expected 401/403/422 for missing auth, got {r.status_code}: {r.text[:300]}"
    )


def test_wrong_internal_secret_returns_401(worker_url, worker_alive):
    """Wrong header value -> rejected, never 500."""
    r = requests.post(
        f"{worker_url}/process-batch",
        json=_minimal_payload(),
        headers={"X-Internal-Secret": "WRONG_VALUE"},
        timeout=10,
    )
    assert r.status_code != 500, f"Worker 500'd on wrong header: {r.text[:300]}"
    assert r.status_code != 200, f"Worker accepted wrong secret: {r.text[:300]}"
    assert r.status_code in (401, 403), (
        f"Expected 401/403 for wrong auth, got {r.status_code}: {r.text[:300]}"
    )


def test_valid_internal_secret_returns_200(worker_url, internal_secret, worker_alive):
    """Sanity: a valid secret is accepted (not rejected at the auth layer)."""
    r = requests.post(
        f"{worker_url}/process-batch",
        json=_minimal_payload(),
        headers={"X-Internal-Secret": internal_secret},
        timeout=120,
    )
    assert r.status_code == 200, (
        f"Valid secret rejected with {r.status_code}: {r.text[:300]}"
    )
