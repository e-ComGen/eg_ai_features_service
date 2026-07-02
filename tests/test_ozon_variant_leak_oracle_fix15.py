"""FIX-15 расширенный variant-leak differential-оракул: сосед-однобрендник vs реальный.

Manifest: docs/MANIFEST_ozon_model_conflict_abstain_fix15.md, Acceptance §2.

К FIX-12 набору (чужой бренд <65, см. test_ozon_variant_leak_oracle_fix12.py — НЕ трогаем)
добавлен сосед-однобрендник: тот же бренд, ДРУГАЯ конкретная модель/вариант должен
получать штраф ниже _BRAND_LINE_THRESHOLD, а реальный товар — выигрывать (>=65).

Пары из манифеста (Acceptance §2 + oracle M2/M3/M8/M1):
  - POCO X6 5G -> POCO M8 Pro 5G       : СОСЕД, < 65 (M2, правило A)
  - POCO X6 5G -> POCO X6 Pro 5G       : СОСЕД, < 65 (M3, правило B)
  - Samsung Galaxy S24 -> Galaxy S23   : СОСЕД, < 65 (M8, правило A)
  - POCO X6 5G -> POCO X6 5G (real)    : РЕАЛЬНЫЙ, >= 65, выигрывает над соседом (M1)

Плюс FIX-12 регресс-инвариант (чужой бренд остаётся <65, гард FIX-15 его не задевает).
"""
from __future__ import annotations

from app.services.enrichment.sources.ozon_card_source import (
    OzonCardSource,
    _BRAND_LINE_THRESHOLD,
)


def _tile(title: str) -> dict:
    return {"title": title}


def _score(query: str, title: str, category_leaf: str, query_brand: str) -> float:
    _, score = OzonCardSource._pick_best_match(
        query, [_tile(title)], category_leaf=category_leaf,
        query_brand=query_brand, query_name=query,
    )
    return score


# ---------------------------------------------------------------------------
# Сосед-однобрендник: должен уйти НИЖЕ порога (per-pair)
# ---------------------------------------------------------------------------

def test_pair_poco_x6_vs_m8_pro_sibling_below_threshold():
    score = _score("POCO X6 5G", "Смартфон POCO M8 Pro 5G 8/256 ГБ",
                    "Смартфоны", "POCO")
    assert score < _BRAND_LINE_THRESHOLD, f"score={score} — сосед M8 Pro прошёл порог"


def test_pair_poco_x6_vs_x6_pro_sibling_below_threshold():
    score = _score("POCO X6 5G", "Смартфон POCO X6 Pro 5G 12/512",
                    "Смартфоны", "POCO")
    assert score < _BRAND_LINE_THRESHOLD, f"score={score} — сосед X6 Pro прошёл порог"


def test_pair_samsung_s24_vs_s23_sibling_below_threshold():
    score = _score("Samsung Galaxy S24", "Смартфон Samsung Galaxy S23",
                    "Смартфоны", "Samsung")
    assert score < _BRAND_LINE_THRESHOLD, f"score={score} — сосед S23 прошёл порог"


# ---------------------------------------------------------------------------
# Реальный товар: выигрывает (>= порога), даже когда сосед тоже в кандидатах
# ---------------------------------------------------------------------------

def test_pair_poco_x6_real_wins_over_sibling():
    query = "POCO X6 5G"
    tiles = [
        _tile("Смартфон POCO M8 Pro 5G 8/256 ГБ"),  # сосед
        _tile("Смартфон POCO X6 5G 8/256 ГБ"),       # реальный товар
    ]
    tile, score = OzonCardSource._pick_best_match(
        query, tiles, category_leaf="Смартфоны", query_brand="POCO", query_name=query,
    )
    assert tile is not None
    assert (tile.get("title") or "").find("M8") == -1, "выбран сосед вместо реального X6"
    assert score >= _BRAND_LINE_THRESHOLD, f"реальный X6 не прошёл порог, score={score}"


def test_pair_poco_x6_real_alone_scores_above_threshold():
    score = _score("POCO X6 5G", "Смартфон POCO X6 5G 8/256 ГБ", "Смартфоны", "POCO")
    assert score >= _BRAND_LINE_THRESHOLD, f"score={score}"


# ---------------------------------------------------------------------------
# FIX-12 регресс: чужой бренд остаётся <65 (гард FIX-15 не задевает brand-путь)
# ---------------------------------------------------------------------------

def test_fix12_regression_wrong_brand_still_below_threshold():
    score = _score("POCO X6 5G", "Смартфон Samsung Galaxy A54 5G", "Смартфоны", "POCO")
    assert score < _BRAND_LINE_THRESHOLD, f"score={score} — FIX-12 бренд-гард ослаблен?"


def test_fix12_regression_clothing_guard_still_below_threshold():
    score = _score("Куртка The North Face", "Шорты The North Face мужские", "Куртки", None)
    assert score < _BRAND_LINE_THRESHOLD, f"score={score} — FIX-12 clothing-гард ослаблен?"


# ---------------------------------------------------------------------------
# Summary — печатает per-pair баллы для отчёта GATE/ORACLE (видно в -s выводе)
# ---------------------------------------------------------------------------

def test_summary_print_all_pair_scores(capsys):
    pairs = [
        ("POCO X6 5G", "Смартфон POCO M8 Pro 5G 8/256 ГБ", "Смартфоны", "POCO", "sibling(M2)"),
        ("POCO X6 5G", "Смартфон POCO X6 Pro 5G 12/512", "Смартфоны", "POCO", "sibling(M3)"),
        ("Samsung Galaxy S24", "Смартфон Samsung Galaxy S23", "Смартфоны", "Samsung", "sibling(M8)"),
        ("POCO X6 5G", "Смартфон POCO X6 5G 8/256 ГБ", "Смартфоны", "POCO", "real(M1)"),
        ("POCO X6 5G", "Смартфон Samsung Galaxy A54 5G", "Смартфоны", "POCO", "wrong-brand(FIX-12)"),
    ]
    for query, title, cat, brand, label in pairs:
        score = _score(query, title, cat, brand)
        verdict = "MATCH" if score >= _BRAND_LINE_THRESHOLD else "abstain/skip"
        print(f"ORACLE-PAIR [{label}] score={score:.1f} -> {verdict}")
