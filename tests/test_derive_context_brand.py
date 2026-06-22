"""Тест _derive_context_brand — бэкфилл context.brand из названия (Баг 3 eg_importer).

Продавец оставляет колонку «Бренд» пустой, но бренд в начале имени («Nike Air Max 90»).
Без context.brand brand-gated источники (ozon_card/regard/onliner/IceCat) не находят
donor-карточку → Пол/Цвет/Размер каскадят в no_data. Бэкфиллим ВЫСОКОТОЧНО: ровно один
словарный бренд из имени; иначе None (пусто честнее мусорной донор-карточки).
"""
from __future__ import annotations

from app.services.enrichment.base import ExtractionContext, TargetAttribute
from app.services.enrichment.pipeline import _derive_context_brand


BRAND_ID = 31
_SNEAKER_BRANDS = ["Nike", "Adidas", "PUMA", "New Balance", "Reebok"]


def _ctx(name: str, brand=None):
    return ExtractionContext(
        product_id=1, product_name=name, category_id=15621048,
        category_path=["Обувь", "Кроссовки"], brand=brand,
    )


def _brand_target(allowed=None):
    return TargetAttribute(id=BRAND_ID, name="Бренд", type="enum",
                           allowed_values=allowed)


def test_single_brand_in_name_resolved():
    """«Nike Air Max 90» + словарь брендов → 'Nike'."""
    out = _derive_context_brand(_ctx("Nike Air Max 90"), [_brand_target(_SNEAKER_BRANDS)])
    assert out == "Nike"


def test_multiword_brand_in_name_resolved():
    """Бренд из двух слов («New Balance 574») матчится целиком."""
    out = _derive_context_brand(_ctx("New Balance 574 серые"), [_brand_target(_SNEAKER_BRANDS)])
    assert out == "New Balance"


def test_existing_brand_not_overwritten():
    """context.brand уже задан продавцом → None (не трогаем)."""
    out = _derive_context_brand(_ctx("Nike Air Max 90", brand="Adidas"),
                                [_brand_target(_SNEAKER_BRANDS)])
    assert out is None


def test_no_brand_target_returns_none():
    """Нет brand-таргета в схеме → None."""
    color = TargetAttribute(id=10096, name="Цвет товара", type="enum",
                            allowed_values=["черный", "белый"])
    out = _derive_context_brand(_ctx("Nike Air Max 90"), [color])
    assert out is None


def test_brand_absent_from_name_returns_none():
    """Имя без словарного бренда → None (пусто честнее мусора)."""
    out = _derive_context_brand(_ctx("Кроссовки спортивные мужские"),
                                [_brand_target(_SNEAKER_BRANDS)])
    assert out is None


def test_two_brands_leftmost_wins():
    """Два бренда в имени → leftmost-tiebreak (как у самого поля «Бренд»): первый.

    RU-заголовок маркетплейса кладёт настоящий бренд первым в описательной части;
    _disambiguate_brand_matches резолвит в один по самой ранней позиции. context.brand
    должен совпадать с тем, что движок проставит в поле «Бренд» — консистентность.
    """
    out = _derive_context_brand(_ctx("Nike Adidas коллаб кроссовки"),
                                [_brand_target(_SNEAKER_BRANDS)])
    assert out == "Nike"


def test_full_dict_via_options_fn_when_allowed_empty():
    """allowed_values пуст (truncated enum) → берём ПОЛНЫЙ словарь через brand_options_fn."""
    out = _derive_context_brand(
        _ctx("PUMA Flyer Runner"),
        [_brand_target(allowed=None)],
        brand_options_fn=lambda attr_id: _SNEAKER_BRANDS,
    )
    assert out == "PUMA"


def test_options_fn_failure_is_non_fatal():
    """brand_options_fn падает + allowed_values пуст → None, без исключения."""
    def _boom(attr_id):
        raise RuntimeError("словарь недоступен")
    out = _derive_context_brand(
        _ctx("Nike Air Max 90"), [_brand_target(allowed=None)],
        brand_options_fn=_boom,
    )
    assert out is None
