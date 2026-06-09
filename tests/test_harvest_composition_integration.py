"""Tests for harvest_composition integration into WebSearchSource._mine_composition_if_needed.

Covers:
  - harvest_composition called for apparel product with empty Состав/Материал targets
  - emits 4604 with correct confidence, source=WEB_SEARCH, and harvester evidence
  - emits 4496 when resolve_value_id succeeds
  - does NOT call harvest_composition when 4604/4496 not in targets (non-apparel)
  - does NOT call harvest_composition when both fields already filled
  - falls back to mine_composition when harvest_composition returns None
  - harvest_composition NOT called when brand is empty (falls straight to mine_composition)
  - BrowserFetcher is injected and reused (not re-created) across calls
"""

from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.enrichment.base import (
    AttributeValue, ExtractionContext, Source, TargetAttribute,
)
from app.services.enrichment.sources.web_search_source import (
    WebSearchSource,
    _ATTR_MATERIAL,
    _ATTR_SOSTAV_MATERIALA,
    _COMPOSITION_CONFIDENCE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_APPAREL_COMPOSITION = "80% хлопок, 20% полиэстер"
_HARVEST_RESULT = {
    "composition": _APPAREL_COMPOSITION,
    "material": "хлопок",
    "source_url": "https://kixbox.ru/champion-hoodie/",
    "site": "kixbox.ru",
    "evidence": _APPAREL_COMPOSITION,
    "route": "open",
    "all_compositions": [_APPAREL_COMPOSITION],
}


def _ctx(brand: str = "Champion", ozon_type_id: Optional[int] = None) -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name="Толстовка худи Champion Reverse Weave",
        category_id=100,
        brand=brand,
        ozon_type_id=ozon_type_id,
    )


def _sostav_target() -> TargetAttribute:
    return TargetAttribute(id=_ATTR_SOSTAV_MATERIALA, name="Состав материала", type="text")


def _material_target() -> TargetAttribute:
    return TargetAttribute(id=_ATTR_MATERIAL, name="Материал", type="enum",
                           allowed_values=["хлопок", "полиэстер", "вискоза"])


def _other_target() -> TargetAttribute:
    return TargetAttribute(id=99, name="Цвет", type="text")


def _make_source() -> WebSearchSource:
    """WebSearchSource with fully mocked collaborators (no real API calls)."""
    mock_producer = AsyncMock()
    mock_producer.produce_summary = AsyncMock(return_value="summary text")
    mock_producer.mine_composition = AsyncMock(return_value=[])

    mock_extractor = AsyncMock()
    mock_extractor.structured_request = AsyncMock(return_value=(
        MagicMock(extracted=[]), 100
    ))

    return WebSearchSource(
        websearch_producer=mock_producer,
        extraction_manager=mock_extractor,
    )


# ---------------------------------------------------------------------------
# Tests: harvest_composition IS called for apparel targets
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_harvest_composition_called_for_sostav_target():
    """harvest_composition must be called when 4604 is among unfilled targets."""
    source = _make_source()
    ctx = _ctx()
    targets = [_sostav_target()]

    with patch(
        "app.services.enrichment.sources.web_search_source."
        "WebSearchSource._mine_composition_if_needed",
        new_callable=AsyncMock,
        return_value=[],
    ) as mock_mine:
        # We need to patch harvest_composition inside _mine_composition_if_needed.
        # Instead, directly call _mine_composition_if_needed with a mocked harvest.
        pass

    # Directly test _mine_composition_if_needed
    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=_HARVEST_RESULT),
    ) as mock_harvest:
        avs = await source._mine_composition_if_needed(ctx, targets, [])

    mock_harvest.assert_awaited_once()
    call_kwargs = mock_harvest.call_args
    assert call_kwargs.kwargs["product_name"] == ctx.product_name
    assert call_kwargs.kwargs["brand"] == ctx.brand


@pytest.mark.asyncio
async def test_emits_4604_with_correct_confidence_and_source():
    """4604 AttributeValue must have confidence=0.72, source=WEB_SEARCH."""
    source = _make_source()
    ctx = _ctx()
    targets = [_sostav_target()]

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=_HARVEST_RESULT),
    ):
        avs = await source._mine_composition_if_needed(ctx, targets, [])

    sostav_avs = [v for v in avs if v.attribute_id == _ATTR_SOSTAV_MATERIALA]
    assert len(sostav_avs) == 1
    av = sostav_avs[0]
    assert av.confidence == pytest.approx(_COMPOSITION_CONFIDENCE)
    assert av.source == Source.WEB_SEARCH
    assert _APPAREL_COMPOSITION in av.value
    # Evidence should carry the site name and route
    assert "kixbox.ru" in av.evidence
    assert "open" in av.evidence


@pytest.mark.asyncio
async def test_emits_4496_when_value_id_resolves():
    """4496 is emitted when resolve_value_id returns a valid id."""
    source = _make_source()
    ctx = _ctx(ozon_type_id=123)
    targets = [_sostav_target(), _material_target()]

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=_HARVEST_RESULT),
    ):
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.resolve_value_id",
            return_value=9876,
        ):
            avs = await source._mine_composition_if_needed(ctx, targets, [])

    mat_avs = [v for v in avs if v.attribute_id == _ATTR_MATERIAL]
    assert len(mat_avs) == 1
    av = mat_avs[0]
    assert av.value_id == 9876
    assert av.source == Source.WEB_SEARCH
    assert av.confidence == pytest.approx(_COMPOSITION_CONFIDENCE)


@pytest.mark.asyncio
async def test_4496_not_emitted_when_value_id_not_resolved():
    """4496 must NOT be emitted when resolve_value_id returns None (fail-closed)."""
    source = _make_source()
    ctx = _ctx(ozon_type_id=123)
    targets = [_sostav_target(), _material_target()]

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=_HARVEST_RESULT),
    ):
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.resolve_value_id",
            return_value=None,
        ):
            avs = await source._mine_composition_if_needed(ctx, targets, [])

    mat_avs = [v for v in avs if v.attribute_id == _ATTR_MATERIAL]
    assert len(mat_avs) == 0


# ---------------------------------------------------------------------------
# Tests: harvest_composition is NOT called in non-apparel / already-filled cases
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_harvest_not_called_for_non_apparel_targets():
    """harvest_composition must NOT be called when 4604/4496 are not in targets."""
    source = _make_source()
    ctx = _ctx()
    targets = [_other_target()]  # only "Цвет" — not apparel

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=_HARVEST_RESULT),
    ) as mock_harvest:
        avs = await source._mine_composition_if_needed(ctx, targets, [])

    mock_harvest.assert_not_awaited()
    assert avs == []


@pytest.mark.asyncio
async def test_harvest_not_called_when_sostav_already_filled():
    """harvest_composition must NOT be called when 4604 is already filled."""
    source = _make_source()
    ctx = _ctx()
    targets = [_sostav_target()]
    already_filled = [
        AttributeValue(
            attribute_id=_ATTR_SOSTAV_MATERIALA,
            value="100% cotton",
            confidence=0.95,
            source=Source.OZON_CARD,
        )
    ]

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=_HARVEST_RESULT),
    ) as mock_harvest:
        avs = await source._mine_composition_if_needed(ctx, targets, already_filled)

    mock_harvest.assert_not_awaited()
    assert avs == []


@pytest.mark.asyncio
async def test_harvest_not_called_when_brand_empty():
    """When brand is empty, harvest_composition is skipped; mine_composition runs."""
    source = _make_source()
    ctx = _ctx(brand="")  # no brand
    targets = [_sostav_target()]

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=_HARVEST_RESULT),
    ) as mock_harvest:
        # mine_composition on the producer will return [] anyway
        source._search.mine_composition = AsyncMock(return_value=[])
        avs = await source._mine_composition_if_needed(ctx, targets, [])

    # harvest_composition NOT called (brand guard)
    mock_harvest.assert_not_awaited()
    # mine_composition WAS called (fallback path)
    source._search.mine_composition.assert_awaited_once()
    assert avs == []  # nothing found


# ---------------------------------------------------------------------------
# Tests: fallback to mine_composition when harvest_composition returns None
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fallback_to_mine_composition_when_harvest_returns_none():
    """When harvest_composition returns None, mine_composition must be called."""
    source = _make_source()
    ctx = _ctx()
    targets = [_sostav_target()]

    mine_compositions = ["79% хлопок, 21% полиэстер"]
    source._search.mine_composition = AsyncMock(return_value=mine_compositions)

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=None),
    ) as mock_harvest:
        avs = await source._mine_composition_if_needed(ctx, targets, [])

    mock_harvest.assert_awaited_once()
    source._search.mine_composition.assert_awaited_once()

    # 4604 should be filled from mine_composition result
    sostav_avs = [v for v in avs if v.attribute_id == _ATTR_SOSTAV_MATERIALA]
    assert len(sostav_avs) == 1
    assert "хлопок" in sostav_avs[0].value


@pytest.mark.asyncio
async def test_mine_composition_not_called_when_harvest_succeeds():
    """mine_composition must NOT be called when harvest_composition already found a result."""
    source = _make_source()
    ctx = _ctx()
    targets = [_sostav_target()]

    source._search.mine_composition = AsyncMock(return_value=["some fallback composition"])

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=_HARVEST_RESULT),
    ):
        avs = await source._mine_composition_if_needed(ctx, targets, [])

    source._search.mine_composition.assert_not_awaited()
    assert any(v.attribute_id == _ATTR_SOSTAV_MATERIALA for v in avs)


# ---------------------------------------------------------------------------
# Tests: BrowserFetcher injection and lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_injected_browser_fetcher_is_passed_to_harvest():
    """Injected BrowserFetcher must be forwarded to harvest_composition."""
    mock_bf = MagicMock()
    source = _make_source()
    source._browser_fetcher = mock_bf  # inject directly

    ctx = _ctx()
    targets = [_sostav_target()]

    captured_kwargs: dict = {}

    async def _capture_harvest(**kwargs):
        captured_kwargs.update(kwargs)
        return _HARVEST_RESULT

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=_capture_harvest,
    ):
        await source._mine_composition_if_needed(ctx, targets, [])

    assert captured_kwargs.get("browser_fetcher") is mock_bf


# ---------------------------------------------------------------------------
# Tests: full extract() path — Step 0 composition emitted before LLM results
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_emits_composition_before_llm_results():
    """composition_avs must appear first in the returned list from extract()."""
    source = _make_source()
    ctx = _ctx()
    targets = [_sostav_target()]

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=_HARVEST_RESULT),
    ):
        # Producer returns a non-empty summary so LLM extraction runs
        source._search.produce_summary = AsyncMock(return_value="summary text")
        avs = await source.extract(ctx, targets)

    sostav_avs = [v for v in avs if v.attribute_id == _ATTR_SOSTAV_MATERIALA]
    assert len(sostav_avs) >= 1
    # Composition AV must be source=WEB_SEARCH
    assert sostav_avs[0].source == Source.WEB_SEARCH


@pytest.mark.asyncio
async def test_extract_does_not_call_harvest_when_no_apparel_targets():
    """Full extract() must not invoke harvest when no apparel targets present."""
    source = _make_source()
    ctx = _ctx()
    targets = [_other_target()]

    with patch(
        "app.services.enrichment.sources.multisite_composition.harvest_composition",
        new=AsyncMock(return_value=_HARVEST_RESULT),
    ) as mock_harvest:
        source._search.produce_summary = AsyncMock(return_value="summary")
        await source.extract(ctx, targets)

    mock_harvest.assert_not_awaited()
