"""Tests for TnvedSource cache-key fix.

Bug: cache was keyed by category_id alone, so different product types
(type_id) sharing the same Ozon category_id would all receive the first
type's cached ТН ВЭД code.

Fix: cache key is now (category_id, type_key) where type_key =
ozon_type_id when set, else the leaf of category_path.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.services.enrichment.base import ExtractionContext
from app.services.enrichment.sources.tnved_source import TnvedSource


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHARED_CAT_ID = 200000933


def _ctx(type_id: int | None, *, path_leaf: str = "Одежда") -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name="Test product",
        category_id=SHARED_CAT_ID,
        category_path=["Одежда и обувь", path_leaf],
        ozon_type_id=type_id,
    )


# ---------------------------------------------------------------------------
# Unit tests for _make_cache_key (pure / synchronous)
# ---------------------------------------------------------------------------

class TestMakeCacheKey:
    def test_uses_ozon_type_id_when_present(self):
        ctx = _ctx(type_id=101)
        key = TnvedSource._make_cache_key(SHARED_CAT_ID, ctx)
        assert key == (SHARED_CAT_ID, 101)

    def test_different_type_ids_produce_different_keys(self):
        key_a = TnvedSource._make_cache_key(SHARED_CAT_ID, _ctx(type_id=101))
        key_b = TnvedSource._make_cache_key(SHARED_CAT_ID, _ctx(type_id=202))
        assert key_a != key_b

    def test_same_type_id_produces_same_key(self):
        key_a = TnvedSource._make_cache_key(SHARED_CAT_ID, _ctx(type_id=101))
        key_b = TnvedSource._make_cache_key(SHARED_CAT_ID, _ctx(type_id=101))
        assert key_a == key_b

    def test_fallback_to_category_path_leaf_when_no_type_id(self):
        ctx = _ctx(type_id=None, path_leaf="Футболки")
        key = TnvedSource._make_cache_key(SHARED_CAT_ID, ctx)
        assert key == (SHARED_CAT_ID, "Футболки")

    def test_different_path_leaves_produce_different_keys(self):
        key_a = TnvedSource._make_cache_key(SHARED_CAT_ID, _ctx(type_id=None, path_leaf="Футболки"))
        key_b = TnvedSource._make_cache_key(SHARED_CAT_ID, _ctx(type_id=None, path_leaf="Джинсы"))
        assert key_a != key_b

    def test_no_type_id_no_path_gives_none_type_key(self):
        ctx = ExtractionContext(
            product_id=1,
            product_name="X",
            category_id=SHARED_CAT_ID,
            category_path=[],
            ozon_type_id=None,
        )
        key = TnvedSource._make_cache_key(SHARED_CAT_ID, ctx)
        assert key == (SHARED_CAT_ID, None)


# ---------------------------------------------------------------------------
# Integration tests: resolve independence + cache-hit behaviour
# ---------------------------------------------------------------------------

def _make_source_with_mock_llm(side_effects: list[str]) -> TnvedSource:
    """Return a TnvedSource whose _call_llm returns side_effects in order."""
    source = TnvedSource.__new__(TnvedSource)
    source._judge = TnvedSource.__new__(TnvedSource)  # dummy
    source._cache = {}
    source._locks = {}
    source._locks_lock = asyncio.Lock()
    source._call_llm = AsyncMock(side_effect=side_effects)  # type: ignore[method-assign]
    return source


class TestResolveIndependence:
    """Two different type_ids must resolve independently — no cross-contamination."""

    def test_different_types_get_independent_codes(self):
        async def _run() -> None:
            source = _make_source_with_mock_llm(["6203423100", "6109100000"])

            ctx_trousers = _ctx(type_id=1001)   # first call → 6203423100
            ctx_tshirt = _ctx(type_id=1002)     # second call → 6109100000

            code_a = await source._resolve_for_category(SHARED_CAT_ID, ctx_trousers)
            code_b = await source._resolve_for_category(SHARED_CAT_ID, ctx_tshirt)

            assert code_a == "6203423100", f"Expected trousers code, got {code_a}"
            assert code_b == "6109100000", f"Expected t-shirt code, got {code_b}"
            # Critical: t-shirt must NOT return trousers code
            assert code_b != code_a, "Cross-contamination: second type got first type's code"
            assert source._call_llm.call_count == 2, (
                f"Expected 2 LLM calls (one per type), got {source._call_llm.call_count}"
            )

        asyncio.run(_run())

    def test_same_category_same_type_hits_cache(self):
        """Same (category, type) must call LLM only once."""
        async def _run() -> None:
            source = _make_source_with_mock_llm(["6203423100"])

            ctx_first = _ctx(type_id=1001)
            ctx_second = _ctx(type_id=1001)  # same type_id → must hit cache

            code_a = await source._resolve_for_category(SHARED_CAT_ID, ctx_first)
            code_b = await source._resolve_for_category(SHARED_CAT_ID, ctx_second)

            assert code_a == "6203423100"
            assert code_b == "6203423100"
            assert source._call_llm.call_count == 1, (
                f"Expected 1 LLM call (cache hit for second), got {source._call_llm.call_count}"
            )

        asyncio.run(_run())

    def test_cross_contamination_regression(self):
        """Regression: category_id=200000933 trousers code must NOT bleed into jacket."""
        async def _run() -> None:
            # Three distinct type_ids under the same category (real-world scenario)
            source = _make_source_with_mock_llm([
                "6203423100",  # trousers
                "6201920000",  # jacket
                "6109100000",  # t-shirt
            ])

            ctx_trousers = _ctx(type_id=301)
            ctx_jacket = _ctx(type_id=302)
            ctx_tshirt = _ctx(type_id=303)

            code_tr = await source._resolve_for_category(SHARED_CAT_ID, ctx_trousers)
            code_jk = await source._resolve_for_category(SHARED_CAT_ID, ctx_jacket)
            code_ts = await source._resolve_for_category(SHARED_CAT_ID, ctx_tshirt)

            assert code_tr == "6203423100"
            assert code_jk == "6201920000"
            assert code_ts == "6109100000"
            # None of the later types should be contaminated by the first
            assert code_jk != code_tr, "jacket got trousers code"
            assert code_ts != code_tr, "t-shirt got trousers code"

        asyncio.run(_run())
