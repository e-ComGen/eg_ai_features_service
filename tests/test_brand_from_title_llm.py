"""Unit tests for _apply_brand_from_title_llm — LLM-based brand extraction from title.

Tests cover:
  (a) Correct in-title brand → filled
  (b) Out-of-enum / hallucinated brand → dropped (empty > wrong)
  (c) LLM returns null → field stays empty
  (d) Brand not in title tokens → dropped (title-anchor guard)
  (e) Category-noun brand → dropped
  (f) The 4 two-word-category cases resolve given title + enum

All LLM calls are mocked — no network access.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    _apply_brand_from_title_llm,
    _BRAND_FROM_TITLE_LLM_EVIDENCE,
    _BRAND_TARGET_ATTR_ID,
)

_BRAND_ID = _BRAND_TARGET_ATTR_ID  # 31


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(name: str, category_path=None, brand=None) -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name=name,
        category_id=1,
        category_path=category_path or [],
        brand=brand,
    )


def _brand_target(allowed, *, attr_id=_BRAND_ID) -> TargetAttribute:
    return TargetAttribute(id=attr_id, name="Бренд", type="enum", allowed_values=allowed)


def _val_for(values, attr_id=_BRAND_ID):
    return next((v for v in values if v.attribute_id == attr_id), None)


def _mock_llm_returning(brand_value):
    """Build a StructuredLlmManager mock that returns the given brand string or None."""
    response = MagicMock()
    response.brand = brand_value  # None or str

    manager = MagicMock()
    manager.structured_request = AsyncMock(return_value=(response, 0))
    return manager


# ---------------------------------------------------------------------------
# (a) Correct in-title brand → filled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_correct_brand_in_title_filled():
    """(a) LLM returns valid in-title brand → filled with source=DESCRIPTION."""
    ctx = _ctx("Умная колонка Яндекс Станция Мини 2",
               category_path=["Электроника", "Умная колонка"])
    target = _brand_target(["Яндекс", "Mail.Ru", "Сбер"])
    mock_llm = _mock_llm_returning("Яндекс")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: ["Яндекс", "Mail.Ru", "Сбер"],
        )

    v = _val_for(out)
    assert v is not None
    assert v.value == "Яндекс"
    assert v.source == Source.DESCRIPTION
    assert v.evidence == _BRAND_FROM_TITLE_LLM_EVIDENCE
    assert v.attribute_id == _BRAND_ID


@pytest.mark.asyncio
async def test_pocketbook_in_title_filled():
    """(a2) 'Электронная книга PocketBook 629 Verse' → PocketBook filled."""
    ctx = _ctx("Электронная книга PocketBook 629 Verse",
               category_path=["Электроника", "Электронная книга"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("PocketBook")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: ["PocketBook", "Kindle", "Onyx"],
        )

    v = _val_for(out)
    assert v is not None and v.value == "PocketBook"
    assert v.source == Source.DESCRIPTION


@pytest.mark.asyncio
async def test_bosch_in_title_filled():
    """(a3) 'Стиральная машина Bosch WGG2540MOE' → Bosch filled."""
    ctx = _ctx("Стиральная машина Bosch WGG2540MOE",
               category_path=["Бытовая техника", "Стиральная машина"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("Bosch")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: ["Bosch", "Samsung", "LG", "Indesit"],
        )

    v = _val_for(out)
    assert v is not None and v.value == "Bosch"


@pytest.mark.asyncio
async def test_samsung_microwave_in_title_filled():
    """(a4) 'Микроволновая печь Samsung MS23K3513AK' → Samsung filled."""
    ctx = _ctx("Микроволновая печь Samsung MS23K3513AK",
               category_path=["Бытовая техника", "Микроволновая печь"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("Samsung")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: ["Samsung", "LG", "Bosch", "Panasonic"],
        )

    v = _val_for(out)
    assert v is not None and v.value == "Samsung"


# ---------------------------------------------------------------------------
# (b) Out-of-enum / hallucinated brand → dropped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_hallucinated_brand_not_in_enum_dropped():
    """(b) LLM returns 'HUGO' but it's NOT in the allowed list → dropped."""
    ctx = _ctx("Кроссовки Nike Air мужские",
               category_path=["Обувь", "Кроссовки"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("HUGO")  # hallucinated, not in enum

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: ["Nike", "Adidas", "Puma"],
        )

    assert _val_for(out) is None  # dropped: not in enum


@pytest.mark.asyncio
async def test_hallucinated_brand_not_in_title_dropped():
    """(b2) LLM returns valid enum brand but it's NOT in the title → dropped."""
    ctx = _ctx("Кроссовки мужские спортивные",
               category_path=["Обувь", "Кроссовки"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("Nike")  # 'Nike' in enum but NOT in title

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: ["Nike", "Adidas"],
        )

    assert _val_for(out) is None  # dropped: title-anchor guard


# ---------------------------------------------------------------------------
# (c) LLM returns null → field stays empty
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_null_field_stays_empty():
    """(c) LLM returns null → brand field stays empty, no crash."""
    ctx = _ctx("Наушники TWS беспроводные",
               category_path=["Электроника", "Наушники"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning(None)

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: ["Sony", "Bose", "JBL"],
        )

    assert _val_for(out) is None  # null returned → empty


# ---------------------------------------------------------------------------
# (d) Already filled → NOT overwritten (deterministic path wins first)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_already_filled_not_overwritten():
    """(d) Brand already filled → LLM path skips (filled_attr_ids check)."""
    ctx = _ctx("Кроссовки Nike Air мужские",
               category_path=["Обувь", "Кроссовки"])
    target = _brand_target(["Nike", "Adidas"])
    existing = AttributeValue(
        attribute_id=_BRAND_ID, value="Nike", confidence=0.95,
        source=Source.DESCRIPTION, evidence="brand_from_name",
    )
    mock_llm = _mock_llm_returning("Adidas")  # should NOT be called

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [existing], [target], ctx,
            brand_options_fn=lambda aid: ["Nike", "Adidas"],
        )

    # existing value is preserved; LLM mock is never invoked for this attr
    assert mock_llm.structured_request.call_count == 0
    v = _val_for(out)
    assert v is not None and v.value == "Nike"


# ---------------------------------------------------------------------------
# (e) Category-noun returned by LLM → dropped
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_category_noun_returned_by_llm_dropped():
    """(e) LLM returns 'колонка' (category noun) even though it's in options → dropped."""
    ctx = _ctx("Умная колонка Яндекс Станция Мини 2",
               category_path=["Электроника", "Умная колонка"])
    target = _brand_target([])
    # 'колонка' is both in options and in title tokens but is a category noun
    mock_llm = _mock_llm_returning("колонка")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: ["Яндекс", "колонка", "Сбер"],
        )

    assert _val_for(out) is None  # category-noun guard


# ---------------------------------------------------------------------------
# (f) No options available (both t.allowed_values and brand_options_fn empty)
#     → free-text path: LLM IS called with unconstrained prompt; result validated
#     by title-anchor + category-noun guards (enum-match guard skipped).
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_enum_free_brand_fills_yandex():
    """(f1) Умная колонка Яндекс — free-text brand, no options → Яндекс filled."""
    ctx = _ctx("Умная колонка Яндекс Станция Мини 2",
               category_path=["Электроника", "Умная колонка"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("Яндекс")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: [],  # no dict → enum_free=True
        )

    v = _val_for(out)
    assert v is not None and v.value == "Яндекс"
    assert v.source == Source.DESCRIPTION
    assert v.evidence == _BRAND_FROM_TITLE_LLM_EVIDENCE
    assert v.value_id is None  # free-text: no enum id


@pytest.mark.asyncio
async def test_enum_free_brand_fills_pocketbook():
    """(f2) Электронная книга PocketBook — free-text brand → PocketBook filled."""
    ctx = _ctx("Электронная книга PocketBook 629 Verse",
               category_path=["Электроника", "Электронная книга"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("PocketBook")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: [],
        )

    v = _val_for(out)
    assert v is not None and v.value == "PocketBook"


@pytest.mark.asyncio
async def test_enum_free_brand_fills_bosch():
    """(f3) Стиральная машина Bosch — free-text brand → Bosch filled."""
    ctx = _ctx("Стиральная машина Bosch WGG2540MOE",
               category_path=["Бытовая техника", "Стиральная машина"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("Bosch")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: [],
        )

    v = _val_for(out)
    assert v is not None and v.value == "Bosch"


@pytest.mark.asyncio
async def test_enum_free_brand_fills_samsung():
    """(f4) Микроволновая печь Samsung — free-text brand → Samsung filled."""
    ctx = _ctx("Микроволновая печь Samsung MS23K3513AK",
               category_path=["Бытовая техника", "Микроволновая печь"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("Samsung")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: [],
        )

    v = _val_for(out)
    assert v is not None and v.value == "Samsung"


@pytest.mark.asyncio
async def test_enum_free_mud_title_stays_empty():
    """(f5) Mud title with no brand → LLM returns null → field stays empty."""
    ctx = _ctx("Умная колонка Мини 2",
               category_path=["Электроника", "Умная колонка"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning(None)  # no brand identifiable

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: [],
        )

    assert _val_for(out) is None


@pytest.mark.asyncio
async def test_enum_free_hallucinated_brand_not_in_title_dropped():
    """(f6) LLM hallucinates a brand not in the mud title → title-anchor drops it."""
    ctx = _ctx("Умная колонка Мини 2",
               category_path=["Электроника", "Умная колонка"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("Яндекс")  # "Яндекс" NOT in this title

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: [],
        )

    assert _val_for(out) is None  # title-anchor guard rejects it


# ---------------------------------------------------------------------------
# (g) value_id resolution from brand_id_fn
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_value_id_attached_from_brand_id_fn():
    """(g) brand_id_fn supplies {brand:id} → value_id attached on fill."""
    ctx = _ctx("Умная колонка Яндекс Станция Мини 2",
               category_path=["Электроника", "Умная колонка"])
    target = _brand_target([])
    mock_llm = _mock_llm_returning("Яндекс")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: ["Яндекс", "Сбер"],
            brand_id_fn=lambda aid: {"Яндекс": 99999, "Сбер": 11111},
        )

    v = _val_for(out)
    assert v is not None and v.value == "Яндекс"
    assert v.value_id == 99999


# ---------------------------------------------------------------------------
# (h) LLM failure → no crash, field stays empty
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_failure_no_crash():
    """(h) LLM call raises → exception swallowed, field stays empty."""
    ctx = _ctx("Умная колонка Яндекс Станция Мини 2",
               category_path=["Электроника", "Умная колонка"])
    target = _brand_target([])

    manager = MagicMock()
    manager.structured_request = AsyncMock(side_effect=RuntimeError("network error"))

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=manager):
        out = await _apply_brand_from_title_llm(
            [], [target], ctx,
            brand_options_fn=lambda aid: ["Яндекс", "Сбер"],
        )

    assert _val_for(out) is None  # fail-safe: empty on error


# ---------------------------------------------------------------------------
# (i) Non-brand targets not touched
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_non_brand_targets_untouched():
    """(i) Function only acts on brand targets; non-brand attrs pass through."""
    ctx = _ctx("Кроссовки Nike Air", category_path=["Обувь", "Кроссовки"])
    color_target = TargetAttribute(id=10, name="Цвет", type="enum", allowed_values=["Чёрный"])
    existing_color = AttributeValue(
        attribute_id=10, value="Чёрный", confidence=0.9, source=Source.WEB_SEARCH,
    )
    mock_llm = _mock_llm_returning("Nike")

    with patch("app.services.enrichment.pipeline.get_main_manager", return_value=mock_llm):
        out = await _apply_brand_from_title_llm(
            [existing_color], [color_target], ctx,
        )

    # No brand targets → LLM never called, color value passes through
    assert mock_llm.structured_request.call_count == 0
    assert len(out) == 1 and out[0].value == "Чёрный"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
