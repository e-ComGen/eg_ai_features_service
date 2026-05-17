"""
Phase 1 async API contract tests between PHP and Python.

These pin the contract for the upcoming Phase 2 async endpoints on the
Python worker side:
    POST   /api/v1/process-batch        -> {worker_job_id, status}
    GET    /api/v1/jobs/{id}/status     -> {status, data?}

As of Phase 1, the Python worker only exposes a synchronous /process-batch
endpoint. These tests auto-skip with a clear TODO until the async endpoints
land. When they do, just remove the skip guard and they should run.
"""

import time

import pytest
import requests


pytestmark = [pytest.mark.integration]


# Probe once per session. If the v1 async route is missing, skip the whole
# module — we don't want red CI just because Phase 2 hasn't shipped.
@pytest.fixture(scope="module")
def async_endpoints_available(worker_url, internal_secret, worker_alive):
    """Skip module if async v1 endpoints don't exist yet."""
    try:
        r = requests.post(
            f"{worker_url}/api/v1/process-batch",
            json={"client_id": 0, "products": [], "schemas": {}},
            headers={"X-Internal-Secret": internal_secret},
            timeout=5,
        )
    except requests.RequestException as e:
        pytest.skip(f"Async endpoint probe failed: {e}")
    if r.status_code == 404:
        pytest.skip(
            # TODO: enable when Phase 2 lands Python async endpoints
            "POST /api/v1/process-batch returns 404 — Phase 2 async API "
            "not yet implemented on the Python worker."
        )
    return True


def _minimal_product(pid: int = 7001) -> dict:
    return {
        "id": pid,
        "category_id": 999,
        "id_path": "",
        "name": "Dyson V15 Detect cordless vacuum",
        "description": (
            "Dyson V15 Detect cordless vacuum cleaner. Suction power: 240AW. "
            "Runtime: 60 minutes."
        ),
        "price": 0,
        "context": {"existing_features": {}, "company_id": 0},
        "languages": ["en"],
    }


def _minimal_schema() -> dict:
    return {"Suction Power": {"type": "text", "options": [], "suffix": "AW", "prefix": ""}}


def _payload(pid: int = 7001) -> dict:
    return {
        "client_id": 9999,
        "products": [_minimal_product(pid)],
        "schemas": {"999": _minimal_schema()},
        "use_cache": False,
        "research_mode": "off",
    }


def test_process_batch_returns_worker_job_id(
    worker_url, internal_secret, async_endpoints_available
):
    """POST /api/v1/process-batch returns a job handle, not the result inline."""
    resp = requests.post(
        f"{worker_url}/api/v1/process-batch",
        json=_payload(),
        headers={"X-Internal-Secret": internal_secret},
        timeout=15,
    )
    assert resp.status_code in (200, 202), (
        f"Expected 200 or 202, got {resp.status_code}: {resp.text[:300]}"
    )
    body = resp.json()
    assert "worker_job_id" in body, f"Missing worker_job_id in {body}"
    assert isinstance(body["worker_job_id"], str) and body["worker_job_id"]
    assert body.get("status") in {"processing", "ready"}, (
        f"Unexpected status: {body.get('status')!r}"
    )


def test_jobs_status_endpoint_returns_processing_then_ready(
    worker_url, internal_secret, async_endpoints_available
):
    """Submit a job, poll until ready. Must transition processing -> ready."""
    submit = requests.post(
        f"{worker_url}/api/v1/process-batch",
        json=_payload(pid=7002),
        headers={"X-Internal-Secret": internal_secret},
        timeout=15,
    )
    assert submit.status_code in (200, 202), submit.text[:300]
    job_id = submit.json()["worker_job_id"]

    # First poll (immediate) — expect processing OR ready if very fast.
    first = requests.get(
        f"{worker_url}/api/v1/jobs/{job_id}/status",
        headers={"X-Internal-Secret": internal_secret},
        timeout=10,
    )
    assert first.status_code == 200, first.text[:300]
    assert first.json().get("status") in {"processing", "ready"}

    # Poll loop until ready or 60s elapsed.
    deadline = time.time() + 60
    final = None
    while time.time() < deadline:
        r = requests.get(
            f"{worker_url}/api/v1/jobs/{job_id}/status",
            headers={"X-Internal-Secret": internal_secret},
            timeout=10,
        )
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        if body.get("status") == "ready":
            final = body
            break
        time.sleep(2)

    assert final is not None, "Job never transitioned to 'ready' within 60s"
    assert isinstance(final.get("data"), list), (
        f"Expected 'data' array in ready response, got: {final}"
    )


def test_jobs_status_unknown_id_returns_404_or_error(
    worker_url, internal_secret, async_endpoints_available
):
    """Unknown job id must not crash the worker."""
    r = requests.get(
        f"{worker_url}/api/v1/jobs/nonexistent-id-xxxx/status",
        headers={"X-Internal-Secret": internal_secret},
        timeout=10,
    )
    assert r.status_code != 500, f"Worker 500'd on unknown id: {r.text[:300]}"
    if r.status_code == 200:
        body = r.json()
        assert body.get("status") == "error", (
            f"200 response for unknown id must carry status=error: {body}"
        )
        assert body.get("error") or body.get("message"), (
            "Error response should include an error/message field"
        )
    else:
        assert r.status_code == 404, (
            f"Expected 404 or 200-with-error, got {r.status_code}: {r.text[:300]}"
        )
