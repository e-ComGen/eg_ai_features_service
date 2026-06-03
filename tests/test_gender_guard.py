"""Unit tests for the GENERAL gender-consistency guard at the merge layer.

Covers _apply_gender_guard in app/services/enrichment/pipeline.py — the guard
runs over ALL sources (web_search/llm/vision/cards), not just card sources, and
drops gendered «Пол» values that conflict with the product name or are supported
only by external-guess sources when the name is gender-neutral.
"""
import pytest

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    _apply_gender_guard,
    _GENDER_EXTERNAL_SOURCES,
)

# Пол — is_collection field на Ozon (value хранится списком), attribute_id 9163.
_POL_ID = 9163


def _ctx(name: str) -> ExtractionContext:
    return ExtractionContext(product_id=1, product_name=name, category_id=1)


def _pol_target() -> TargetAttribute:
    return TargetAttribute(id=_POL_ID, name="Пол", type="text", is_collection=True)


def _pol_value(value, source: Source) -> AttributeValue:
    return AttributeValue(
        attribute_id=_POL_ID,
        value=value,
        confidence=0.88,
        source=source,
        is_collection=True,
    )


def _ids(values: list[AttributeValue]) -> set:
    return {v.attribute_id for v in values}


def test_constant_uses_real_source_strings():
    assert _GENDER_EXTERNAL_SOURCES == {
        Source.WEB_SEARCH,
        Source.LLM_KNOWLEDGE,
        Source.COMPETITOR_RAG,
    }


def test_neutral_name_female_from_web_search_only_dropped():
    """(i) NEUTRAL name + Женский from web_search only → DROPPED."""
    ctx = _ctx("Кроссовки Adidas Ultraboost 22")
    cand = [_pol_value(["Женский"], Source.WEB_SEARCH)]
    out = _apply_gender_guard(cand, [_pol_target()], ctx)
    assert _POL_ID not in _ids(out), "external-only gendered value must be dropped"


def test_explicit_male_name_female_dropped():
    """(ii) explicit «мужская» name + Женский → DROPPED (conflict)."""
    ctx = _ctx("Куртка мужская The North Face Resolve")
    # даже из товар-специфичного источника: явный конфликт имени побеждает
    cand = [_pol_value(["Женский"], Source.OZON_CARD)]
    out = _apply_gender_guard(cand, [_pol_target()], ctx)
    assert _POL_ID not in _ids(out), "value conflicting with explicit name gender must be dropped"


def test_explicit_female_name_female_kept():
    """(iii) explicit «женское» name (Платье befree) + Женский → KEPT."""
    ctx = _ctx("Платье befree летнее женское")
    cand = [_pol_value(["Женский"], Source.WEB_SEARCH)]
    out = _apply_gender_guard(cand, [_pol_target()], ctx)
    assert _POL_ID in _ids(out), "value matching explicit name gender must be kept"
    assert out[0].value == ["Женский"]


def test_neutral_name_female_from_ozon_card_kept():
    """(iv) NEUTRAL name + Женский from ozon_card → KEPT (product-specific source)."""
    ctx = _ctx("Кроссовки Adidas Ultraboost 22")
    cand = [_pol_value(["Женский"], Source.OZON_CARD)]
    out = _apply_gender_guard(cand, [_pol_target()], ctx)
    assert _POL_ID in _ids(out), "product-specific source value must be kept on neutral name"


def test_neutral_name_supported_by_product_source_keeps_web_too():
    """Если тот же элемент несёт и product-source — external-кандидат тоже остаётся."""
    ctx = _ctx("Кроссовки Adidas Ultraboost 22")
    cand = [
        _pol_value(["Женский"], Source.WEB_SEARCH),
        _pol_value(["Женский"], Source.OZON_CARD),
    ]
    out = _apply_gender_guard(cand, [_pol_target()], ctx)
    # оба кандидата на _POL_ID остаются (значение подтверждено product-source)
    assert len([v for v in out if v.attribute_id == _POL_ID]) == 2


def test_scalar_value_neutral_name_web_only_dropped():
    """Скалярный (не-list) gender value тоже отсекается."""
    ctx = _ctx("Кроссовки Adidas Ultraboost 22")
    cand = [
        AttributeValue(
            attribute_id=_POL_ID, value="Женский", confidence=0.88,
            source=Source.WEB_SEARCH, is_collection=False,
        )
    ]
    out = _apply_gender_guard(cand, [_pol_target()], ctx)
    assert _POL_ID not in _ids(out)


def test_non_gender_attribute_untouched():
    """Не-gender атрибут проходит сквозь без изменений."""
    ctx = _ctx("Кроссовки Adidas Ultraboost 22")
    color = TargetAttribute(id=10, name="Цвет", type="text")
    cand = [AttributeValue(attribute_id=10, value="Чёрный", confidence=0.9, source=Source.WEB_SEARCH)]
    out = _apply_gender_guard(cand, [color], ctx)
    assert _ids(out) == {10}


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
