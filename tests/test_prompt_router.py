"""Unit tests for app/services/enrichment/prompt_router.py."""
import pytest

from app.services.enrichment.base import TargetAttribute
from app.services.enrichment.prompt_router import (
    classify_target,
    extract_unit,
    format_target_line,
    build_meta_guidance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(
    name: str,
    attr_type: str = "text",
    allowed_values: list[str] | None = None,
    is_collection: bool = False,
) -> TargetAttribute:
    return TargetAttribute(
        id=1,
        name=name,
        type=attr_type,
        allowed_values=allowed_values,
        is_collection=is_collection,
    )


# ---------------------------------------------------------------------------
# classify_target tests
# ---------------------------------------------------------------------------

def test_classify_enum_color():
    """Атрибут 'Цвет' с allowed_values → enum."""
    t = _make_target("Цвет товара", allowed_values=["Черный", "Белый", "Красный"])
    assert classify_target(t) == "enum"


def test_classify_dimensions_sm():
    """'Длина, см' с numeric-типом → dimensions."""
    t = _make_target("Длина, см", attr_type="Integer")
    assert classify_target(t) == "dimensions"


def test_classify_model_name_full():
    """'Название модели для шаблона наименования' → model_name."""
    t = _make_target("Название модели для шаблона наименования")
    assert classify_target(t) == "model_name"


def test_classify_model_name_partnumber():
    """'Партномер' → model_name (приоритет выше enum)."""
    t = _make_target("Партномер", allowed_values=None)
    assert classify_target(t) == "model_name"


def test_classify_numeric_with_unit():
    """'Мощность блока питания, Вт' (нет allowed_values, не габарит) → numeric."""
    t = _make_target("Мощность блока питания, Вт", attr_type="Integer")
    assert classify_target(t) == "numeric"


def test_classify_text_brand_no_allowed():
    """'Бренд' без allowed_values → text."""
    t = _make_target("Бренд")
    assert classify_target(t) == "text"


def test_classify_model_name_wins_over_enum():
    """Если имя содержит 'артикул' И есть allowed_values — model_name должен выиграть."""
    t = _make_target("Артикул производителя", allowed_values=["A1", "B2"])
    assert classify_target(t) == "model_name"


def test_classify_dimensions_height():
    """'Высота, мм' → dimensions."""
    t = _make_target("Высота, мм", attr_type="Decimal")
    assert classify_target(t) == "dimensions"


# ---------------------------------------------------------------------------
# extract_unit tests
# ---------------------------------------------------------------------------

def test_extract_unit_vt():
    assert extract_unit("Мощность блока питания, Вт") == "Вт"


def test_extract_unit_sm():
    assert extract_unit("Длина, см") == "см"


def test_extract_unit_no_unit():
    assert extract_unit("Цвет товара") is None


def test_extract_unit_slash_unit():
    """Единица вида 'об/мин'."""
    assert extract_unit("Скорость вращения, об/мин") == "об/мин"


def test_extract_unit_empty_string():
    assert extract_unit("") is None


# ---------------------------------------------------------------------------
# format_target_line tests
# ---------------------------------------------------------------------------

def test_format_target_line_enum():
    t = _make_target("Цвет товара", allowed_values=["Черный", "Белый"])
    line = format_target_line(t)
    assert "kind=enum" in line
    assert "allowed=" in line
    assert "Черный" in line


def test_format_target_line_model_name():
    t = _make_target("Название модели для шаблона наименования")
    line = format_target_line(t)
    assert "kind=model_name" in line
    assert "allowed=" not in line


def test_format_target_line_numeric_has_unit():
    t = _make_target("Мощность блока питания, Вт", attr_type="Integer")
    line = format_target_line(t)
    assert "kind=numeric" in line
    assert "unit=Вт" in line


def test_format_target_line_is_collection():
    t = _make_target("Теги", is_collection=True)
    line = format_target_line(t)
    assert "is_collection=true" in line


# ---------------------------------------------------------------------------
# build_meta_guidance tests
# ---------------------------------------------------------------------------

def test_build_meta_guidance_contains_enum_rule():
    guidance = build_meta_guidance()
    assert "kind=enum" in guidance
    assert "MUST" in guidance


def test_build_meta_guidance_contains_model_name_rule():
    guidance = build_meta_guidance()
    assert "kind=model_name" in guidance
    assert "brand" in guidance.lower() or "бренд" in guidance.lower()


def test_build_meta_guidance_contains_all_kinds():
    guidance = build_meta_guidance()
    for kind in ("kind=enum", "kind=numeric", "kind=dimensions", "kind=model_name", "kind=text"):
        assert kind in guidance, f"Missing rule for {kind}"
