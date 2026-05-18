"""Tests for Excel upload endpoints and /fill-single endpoint.

Uses TestClient (no live server needed) with mocked PipelineAdapter.
"""
import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Build a minimal in-memory WB Excel for upload tests
# ---------------------------------------------------------------------------

def _make_wb_bytes() -> bytes:
    """Return a bytes buffer containing a minimal WB Excel file."""
    rows = [{"Артикул продавца": "SKU-001", "Наименование": "Товар", "Материал": ""}]
    buf = io.BytesIO()
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Товары", index=False)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """TestClient with mocked heavy dependencies so app imports don't fail."""
    # Patch everything that tries to connect to external services at import time
    patches = [
        patch("app.services.llm_manager.OpenAIManager.__init__", return_value=None),
        patch("app.services.db_cache.DatabaseCacheManager.__init__", return_value=None),
        patch("app.services.matcher.MatcherService.__init__", return_value=None),
        patch("app.services.job_processor.JobProcessor.__init__", return_value=None),
        patch("app.services.ai_pipeline.AiFeaturePipeline.__init__", return_value=None),
        patch("app.services.providers.factory.get_main_manager", return_value=MagicMock()),
        patch("app.services.providers.factory.get_vision_provider", return_value=MagicMock()),
        patch("app.services.providers.factory.get_web_search_client", return_value=MagicMock()),
        patch("app.services.enrichment.VisionProducer.__init__", return_value=None),
        patch("app.services.enrichment.WebSearchProducer.__init__", return_value=None),
        patch("app.database.init_db", new_callable=AsyncMock),
    ]
    with patch("app.config.OPENAI_API_KEY", "test-key"), \
         patch("app.config.WEB_SEARCH_MODEL", "gpt-4o"), \
         patch("app.config.WEB_SEARCH_MAX_CONCURRENT", 1), \
         patch("app.config.PROVIDER_MAIN", "openai"):
        started = [p.start() for p in patches]
        try:
            from app.main import app
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# 1. /fill-single returns attributes list (mocked adapter)
# ---------------------------------------------------------------------------

def test_fill_single_returns_attributes(client):
    from app.services.enrichment.base import AttributeValue, Source

    fake_value = AttributeValue(
        attribute_id=0,
        value="Хлопок",
        confidence=0.9,
        source=Source.DESCRIPTION,
        evidence="Из описания: хлопок 100%",
        judge_validated=False,
    )

    with patch(
        "app.main.PipelineAdapter.run",
        new_callable=AsyncMock,
        return_value=[fake_value],
    ):
        resp = client.post(
            "/fill-single",
            params={"name": "Футболка", "description": "100% хлопок", "marketplace": "wb"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "filled_attributes" in body
    assert len(body["filled_attributes"]) == 1
    attr = body["filled_attributes"][0]
    assert attr["value"] == "Хлопок"
    assert attr["confidence"] == 0.9
    assert attr["source"] == "description"


# ---------------------------------------------------------------------------
# 2. POST /excel/upload returns job_id
# ---------------------------------------------------------------------------

def test_upload_excel_returns_job_id(client):
    excel_bytes = _make_wb_bytes()
    resp = client.post(
        "/excel/upload",
        files={"file": ("template.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        params={"marketplace": "wb"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "job_id" in body
    assert "status_url" in body
    assert "download_url" in body
    assert len(body["job_id"]) == 32  # uuid4 hex


# ---------------------------------------------------------------------------
# 3. GET /excel/status/{job_id} returns 404 for unknown job
# ---------------------------------------------------------------------------

def test_status_404_for_unknown_job(client):
    resp = client.get("/excel/status/nonexistent_job_id_12345")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 4. GET /excel/download/{job_id} returns 404 for unknown job
# ---------------------------------------------------------------------------

def test_download_404_for_unknown_job(client):
    resp = client.get("/excel/download/nonexistent_job_id_12345")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 5. GET /excel/status after upload shows queued or processing status
# ---------------------------------------------------------------------------

def test_status_returns_job_info_after_upload(client):
    excel_bytes = _make_wb_bytes()
    upload_resp = client.post(
        "/excel/upload",
        files={"file": ("template.xlsx", excel_bytes, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        params={"marketplace": "wb"},
    )
    assert upload_resp.status_code == 200
    job_id = upload_resp.json()["job_id"]

    status_resp = client.get(f"/excel/status/{job_id}")
    assert status_resp.status_code == 200
    status_body = status_resp.json()
    assert "status" in status_body
    assert status_body["status"] in ("queued", "processing", "done", "error")
    assert status_body.get("marketplace") == "wb"
