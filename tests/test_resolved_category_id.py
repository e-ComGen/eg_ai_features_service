"""Tests for resolved_category_id threading through Ozon value_id resolution.

Bug: main value_id resolution used the STALE template context.category_id instead of
the live Ozon description-category-id (dcid) resolved from ozon_type_id. The local
dict ozon_dictionary.json.gz CONTAINS the live dcid — so threading it is the whole fix.

Oracle (ground truth, verified live):
    toaster ozon_type_id=96031 -> resolve_description_category_id -> 17039630
    resolve_value_id(17039630, 96031, 10400, '1 год') -> 970716397
    resolve_value_id(47156221, 96031, 10400, '1 год') -> None  (stale cat)

Invariants:
    I1: live dcid present -> main resolver uses it.
    I2: ozon_type_id=None -> falls back to template category_id, no regression.
    I3: context.category_id is NEVER mutated (only resolved_category_id is set).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.enrichment.base import AttributeValue, ExtractionContext, Source
from app.services.enrichment.strategies.ozon_strategy import OzonStrategy

# ---------------------------------------------------------------------------
# Constants (oracle-derived, do not change)
# ---------------------------------------------------------------------------
_STALE_CAT = 47156221       # шаблонный category_id тостера — values-API отвергает
_LIVE_DCID = 17039630       # живой dcid для type_id=96031 (Тостер, Тепловая обработка)
_TYPE_ID = 96031             # Ozon type_id: Тостер
_ATTR_GARANTIA = 10400       # Гарантийный срок
_VALUE_1_GOD = "1 год"
_EXPECTED_VID = 970716397    # живой value_id для (17039630, 96031, 10400, '1 год')

_RESOLVE_DCID_PATH = (
    "app.services.enrichment.strategies.ozon_strategy.resolve_description_category_id"
)
_IS_TRUNCATED_PATH = (
    "app.services.enrichment.strategies.ozon_strategy.is_truncated"
)


def _ctx_toaster(*, resolved: int | None = None) -> ExtractionContext:
    """ExtractionContext with stale template category_id, live ozon_type_id."""
    return ExtractionContext(
        product_id=1,
        product_name="Тостер Philips HD2581/90",
        category_id=_STALE_CAT,
        ozon_type_id=_TYPE_ID,
        resolved_category_id=resolved,
    )


def _av_garantia() -> AttributeValue:
    """AttributeValue: attr 10400 'Гарантийный срок', value '1 год'."""
    return AttributeValue(
        attribute_id=_ATTR_GARANTIA,
        value=_VALUE_1_GOD,
        confidence=0.9,
        source=Source.DESCRIPTION,
    )


# ---------------------------------------------------------------------------
# T1 — oracle/threading: live dcid resolves correct value_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t1_live_dcid_resolves_correct_value_id():
    """I1: with live dcid populated, resolver uses it and finds 970716397.

    resolve_description_category_id is mocked (no network); resolve_value_id
    hits the REAL local dict at dcid 17039630 -> should return 970716397.
    """
    ctx = _ctx_toaster()
    av = _av_garantia()
    strategy = OzonStrategy()

    with patch(_RESOLVE_DCID_PATH, new_callable=AsyncMock, return_value=_LIVE_DCID), \
         patch(_IS_TRUNCATED_PATH, return_value=False):
        result = await strategy.resolve_value_ids_async(av, ctx)

    assert result.value_id == _EXPECTED_VID, (
        f"Expected value_id={_EXPECTED_VID} via live dcid {_LIVE_DCID}, "
        f"got {result.value_id!r}. Bug: stale cat {_STALE_CAT} returns None."
    )


# ---------------------------------------------------------------------------
# T2 — no-regression: ozon_type_id=None falls back gracefully
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t2_no_type_id_fallback_no_crash():
    """I2: ozon_type_id=None -> early return, no crash, value_id stays None."""
    ctx = ExtractionContext(
        product_id=2,
        product_name="Товар без type_id",
        category_id=_STALE_CAT,
        ozon_type_id=None,
    )
    av = _av_garantia()
    strategy = OzonStrategy()

    result = await strategy.resolve_value_ids_async(av, ctx)

    # No crash; value_id should stay None (can't resolve without type_id)
    assert result.value_id is None
    # template category_id unchanged
    assert ctx.category_id == _STALE_CAT


# ---------------------------------------------------------------------------
# T3 — no-mutation: template category_id is NEVER changed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_t3_template_category_id_not_mutated():
    """I3: after resolve, context.category_id is still the original stale value."""
    ctx = _ctx_toaster()
    av = _av_garantia()
    strategy = OzonStrategy()

    with patch(_RESOLVE_DCID_PATH, new_callable=AsyncMock, return_value=_LIVE_DCID), \
         patch(_IS_TRUNCATED_PATH, return_value=False):
        await strategy.resolve_value_ids_async(av, ctx)

    assert ctx.category_id == _STALE_CAT, (
        f"category_id must not be mutated; expected {_STALE_CAT}, got {ctx.category_id}"
    )
    # resolved_category_id should now hold the live dcid
    assert ctx.resolved_category_id == _LIVE_DCID
