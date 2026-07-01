"""Variant-leak differential oracle for FIX-12: ensures fix does NOT boost score of wrong product/brand/variant."""
from __future__ import annotations

from app.services.enrichment.sources.ozon_card_source import OzonCardSource, _BRAND_LINE_THRESHOLD, _EXACT_THRESHOLD


def test_variant_leak_poco_vs_samsung():
    query = "POCO X6 5G"
    category_leaf = "Смартфоны"
    query_brand = "POCO"
    query_name = query

    # правильная плитка
    correct_tile, correct_score = OzonCardSource._pick_best_match(
        query,
        [{"title": "Смартфон POCO X6 5G 8/256 ГБ"}],
        category_leaf=category_leaf,
        query_brand=query_brand,
        query_name=query_name
    )
    assert correct_score >= _BRAND_LINE_THRESHOLD, f"Correct tile score {correct_score} < {_BRAND_LINE_THRESHOLD}"

    # чужая плитка (другой бренд)
    wrong_tile, wrong_score = OzonCardSource._pick_best_match(
        query,
        [{"title": "Смартфон Samsung Galaxy A54 5G"}],
        category_leaf=category_leaf,
        query_brand=query_brand,
        query_name=query_name
    )
    assert wrong_score < _BRAND_LINE_THRESHOLD, f"Wrong tile score {wrong_score} >= {_BRAND_LINE_THRESHOLD}"


def test_variant_leak_bosch_vs_makita():
    query = "Bosch GSB 13 RE"
    category_leaf = "Дрели ударные"
    query_brand = "Bosch"
    query_name = query

    # правильная плитка
    correct_tile, correct_score = OzonCardSource._pick_best_match(
        query,
        [{"title": "Дрель ударная Bosch GSB 13 RE 600 Вт"}],
        category_leaf=category_leaf,
        query_brand=query_brand,
        query_name=query_name
    )
    assert correct_score >= _EXACT_THRESHOLD, f"Correct tile score {correct_score} < {_EXACT_THRESHOLD}"

    # чужая плитка (другой бренд)
    wrong_tile, wrong_score = OzonCardSource._pick_best_match(
        query,
        [{"title": "Дрель ударная Makita HP1631 710 Вт"}],
        category_leaf=category_leaf,
        query_brand=query_brand,
        query_name=query_name
    )
    assert wrong_score < _BRAND_LINE_THRESHOLD, f"Wrong tile score {wrong_score} >= {_BRAND_LINE_THRESHOLD}"


def test_clothing_guard_still_skips_wrong_type():
    query = "Куртка The North Face"
    category_leaf = "Куртки"
    query_brand = None
    query_name = query

    # шорты того же бренда — не тот тип товара, гард должен сработать
    tile, score = OzonCardSource._pick_best_match(
        query,
        [{"title": "Шорты The North Face мужские"}],
        category_leaf=category_leaf,
        query_brand=query_brand,
        query_name=query_name
    )
    assert score < _BRAND_LINE_THRESHOLD, f"Wrong type tile score {score} >= {_BRAND_LINE_THRESHOLD}"
