"""Unit tests for LLM response cache (llm_cache.py + StructuredLlmManager integration).

Run: pytest tests/test_llm_cache.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from app.services.providers.base import LlmResponse
from app.services.providers.llm_cache import LlmCache, make_cache_key
from app.services.providers.structured_adapter import StructuredLlmManager


# ---------------------------------------------------------------------------
# Simple Pydantic model used as response_model in all tests
# ---------------------------------------------------------------------------

class _SimpleModel(BaseModel):
    name: str
    value: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_provider(content: str, input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    resp = LlmResponse(
        content=content,
        model="test-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=0.0,
        raw={},
    )
    provider = MagicMock()
    provider.complete = AsyncMock(return_value=resp)
    return provider


def _make_manager(provider) -> StructuredLlmManager:
    return StructuredLlmManager(provider=provider, model="test-model")


# ---------------------------------------------------------------------------
# Test 1: Cache hit returns same content + tokens=0
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_hit_returns_model_and_zero_tokens():
    """When a matching entry exists in the cache, no LLM call is made and tokens=0."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    env_overrides = {
        "LLM_CACHE_ENABLED": "1",
        "LLM_CACHE_DB": db_path,
    }

    # Pre-populate the cache manually
    cache = LlmCache.__new__(LlmCache)
    from app.services.providers import llm_cache as _mod
    original_path_fn = _mod._db_path
    _mod._db_path = lambda: __import__("pathlib").Path(db_path)

    # Reset singleton so our new env takes effect
    _mod._cache = None
    cache = LlmCache()

    key = make_cache_key("sys", "user", _SimpleModel)
    expected = _SimpleModel(name="CachedItem", value=42)
    cache.set(key, expected.model_dump_json())

    provider = _mock_provider('{"name": "LiveItem", "value": 99}')
    manager = _make_manager(provider)

    with patch.dict(os.environ, env_overrides):
        _mod._cache = cache  # inject pre-seeded cache instance
        result, tokens = await manager.structured_request("sys", "user", _SimpleModel)

    # Restore
    _mod._db_path = original_path_fn
    _mod._cache = None

    assert result is not None
    assert result.name == "CachedItem"
    assert result.value == 42
    assert tokens == 0
    provider.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: Cache miss calls the underlying provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_miss_calls_provider():
    """On a cache miss the provider is called exactly once and the result is stored."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name

    from app.services.providers import llm_cache as _mod
    original_path_fn = _mod._db_path
    _mod._db_path = lambda: __import__("pathlib").Path(db_path)
    _mod._cache = None  # reset singleton

    live_payload = json.dumps({"name": "LiveItem", "value": 7})
    provider = _mock_provider(live_payload)
    manager = _make_manager(provider)

    env_overrides = {"LLM_CACHE_ENABLED": "1", "LLM_CACHE_DB": db_path}

    with patch.dict(os.environ, env_overrides):
        result, tokens = await manager.structured_request("sys_miss", "user_miss", _SimpleModel)

    # Restore
    _mod._db_path = original_path_fn
    _mod._cache = None

    assert result is not None
    assert result.name == "LiveItem"
    assert result.value == 7
    assert tokens == 150  # 100 + 50
    provider.complete.assert_called_once()

    # Verify the entry was written to the DB
    cache = LlmCache.__new__(LlmCache)
    from app.services.providers import llm_cache as _mod2
    _mod2._db_path = lambda: __import__("pathlib").Path(db_path)
    cache2 = LlmCache()
    key = make_cache_key("sys_miss", "user_miss", _SimpleModel)
    stored = cache2.get(key)
    assert stored is not None
    data = json.loads(stored)
    assert data["name"] == "LiveItem"
    _mod2._db_path = original_path_fn


# ---------------------------------------------------------------------------
# Test 3: Cache disabled (env not set) skips cache entirely
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_disabled_skips_cache():
    """When LLM_CACHE_ENABLED is not set (or '0'), the cache is never consulted."""
    from app.services.providers import llm_cache as _mod
    _mod._cache = None

    live_payload = json.dumps({"name": "AlwaysLive", "value": 1})
    provider = _mock_provider(live_payload)
    manager = _make_manager(provider)

    # Ensure cache is OFF
    env_override = {"LLM_CACHE_ENABLED": "0"}
    with patch.dict(os.environ, env_override):
        result, tokens = await manager.structured_request("sys_no_cache", "user_no_cache", _SimpleModel)

    assert result is not None
    assert result.name == "AlwaysLive"
    # Provider MUST have been called (no cache shortcut)
    provider.complete.assert_called_once()
    # Tokens should be non-zero (real call happened)
    assert tokens == 150
