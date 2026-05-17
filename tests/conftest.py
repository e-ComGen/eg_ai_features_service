"""
Shared pytest fixtures for CpAiFeatures integration tests.

These tests hit the running AI-worker service on port 8001 directly
(bypassing the ai_orchestrator on port 8000 to skip auth/billing).

Prereqs:
    1. uvicorn app.main:app --port 8001 must be running.
    2. .env must contain INTERNAL_SERVICE_SECRET and OPENAI_API_KEY.

Tests that consume real OpenAI tokens are marked @pytest.mark.real_ai
so they can be deselected in CI:  pytest -m "not real_ai"
"""

import os
import pytest
import requests
from dotenv import load_dotenv

# Load .env from project root so INTERNAL_SERVICE_SECRET is available.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

WORKER_URL = os.getenv("AI_WORKER_URL", "http://127.0.0.1:8001")
INTERNAL_SECRET = os.getenv("INTERNAL_SERVICE_SECRET", "")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_ai: tests that make real OpenAI API calls (cost tokens)",
    )
    config.addinivalue_line(
        "markers",
        "integration: tests that hit the live worker service on port 8001",
    )


@pytest.fixture(scope="session")
def worker_url():
    return WORKER_URL


@pytest.fixture(scope="session")
def internal_secret():
    if not INTERNAL_SECRET:
        pytest.skip("INTERNAL_SERVICE_SECRET not set in .env")
    return INTERNAL_SECRET


@pytest.fixture(scope="session")
def worker_alive(worker_url):
    """Skip integration tests if the worker isn't responding."""
    try:
        r = requests.get(f"{worker_url}/docs", timeout=3)
        if r.status_code >= 500:
            pytest.skip(f"Worker at {worker_url} returned {r.status_code}")
    except requests.RequestException as e:
        pytest.skip(f"Worker not reachable at {worker_url}: {e}")
    return True


@pytest.fixture
def call_worker(worker_url, internal_secret, worker_alive):
    """
    Returns a function (product, schema, **kwargs) -> response JSON.

    Sends a single-product batch to /process-batch with auth header.
    use_cache defaults to False to ensure the AI pipeline actually runs.
    """

    def _call(product: dict, schema: dict, *, client_id: int = 9999,
              use_cache: bool = False, research_mode: str = "off",
              timeout: int = 120) -> dict:
        category_id = product["category_id"]
        payload = {
            "client_id": client_id,
            "products": [product],
            "schemas": {str(category_id): schema},
            "use_cache": use_cache,
            "research_mode": research_mode,
        }
        resp = requests.post(
            f"{worker_url}/process-batch",
            json=payload,
            headers={"X-Internal-Secret": internal_secret},
            timeout=timeout,
        )
        assert resp.status_code == 200, (
            f"Worker returned {resp.status_code}: {resp.text[:500]}"
        )
        body = resp.json()
        assert body.get("status") == "success", f"Unexpected body: {body}"
        return body

    return _call


def extract_value(response_body: dict, product_id: int, feature_name: str):
    """Pull the filled value (or None) for a given feature out of the response."""
    for item in response_body.get("data", []):
        if item.get("product_id") == product_id:
            return item.get("filled_features", {}).get(feature_name)
    return None


def extract_debug(response_body: dict, product_id: int, feature_name: str) -> dict:
    """Pull the debug_info entry for a given feature (used for diagnostic logs)."""
    for item in response_body.get("data", []):
        if item.get("product_id") == product_id:
            return item.get("debug_info", {}).get(feature_name, {})
    return {}
