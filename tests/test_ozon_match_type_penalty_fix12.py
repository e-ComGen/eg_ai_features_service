"""FIX-12: type-mismatch штраф не должен бить по правильному товару.

Root cause (rooted 2026-07-02): `_pick_best_match` штрафовал -30 всякий раз,
когда `category_leaf` (таксономия CS-Cart, часто мн.ч./составное — «Смартфоны»,
«Дрели ударные») не входит буквальной подстрокой в заголовок карточки Ozon
(«Смартфон POCO X6 5G...», «Дрель ударная Bosch GSB 13 RE...»). Разная
словоформа ложно засчитывалась как type-mismatch -> Bosch 57.1, POCO 28.8,
оба ниже _BRAND_LINE_THRESHOLD=65 -> карточка отбрасывалась.

Фикс вводит `_category_present_in_title` (stem/prefix-aware, общий префикс
>= 4 симв. между токенами категории и токенами заголовка) и гейтит штраф по
ней вместо буквальной подстроки. Гард на настоящий type-mismatch (одежда:
«Куртка» vs «Шорты») остаётся жив.

INV-12a: `_category_present_in_title` — префикс-матч >=4 симв, per O1/O2/O4.
INV-12b: штраф ВСЁ ЕЩЁ срабатывает на настоящем type-mismatch (O3, O5).
INV-12c: штраф НЕ срабатывает при stem-присутствии категории (O1, O2, O4).
INV-12d: прочие пути скоринга (brand/gender/model-bonus/threshold) не тронуты.
INV-12e: хелпер чистый/идемпотентный, без I/O, пустые строки -> False.
"""
from __future__ import annotations

import pytest

from app.services.enrichment.sources.ozon_card_source import (
    OzonCardSource,
    _BRAND_LINE_THRESHOLD,
    _EXACT_THRESHOLD,
    _TYPE_MISMATCH_PENALTY,
    _category_present_in_title,
    _common_prefix_len,
    _extract_model_tokens,
)


# ---------------------------------------------------------------------------
# INV-12a / INV-12c: _category_present_in_title — oracle O1, O2, O4 (True)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cat_leaf_low,title_low",
    [
        ("дрели ударные", "дрель ударная bosch gsb 13 re 600 вт"),   # O1
        ("смартфоны", "смартфон poco x6 5g 8/256 гб"),                # O2
        ("куртки", "куртка the north face зимняя"),                   # O4
    ],
)
def test_inv12a_category_present_true(cat_leaf_low, title_low):
    assert _category_present_in_title(cat_leaf_low, title_low) is True


# ---------------------------------------------------------------------------
# INV-12b: _category_present_in_title — oracle O3, O5 (False, гард жив)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "cat_leaf_low,title_low",
    [
        ("куртки", "шорты the north face мужские"),  # O3
        ("футболки", "шорты nike 44"),                # O5
    ],
)
def test_inv12b_category_absent_false(cat_leaf_low, title_low):
    assert _category_present_in_title(cat_leaf_low, title_low) is False


# ---------------------------------------------------------------------------
# INV-12e: чистота / edge-кейсы хелпера
# ---------------------------------------------------------------------------

def test_inv12e_empty_strings_do_not_raise():
    assert _category_present_in_title("", "дрель ударная bosch") is False
    assert _category_present_in_title("дрели", "") is False
    assert _category_present_in_title("", "") is False
    assert _category_present_in_title(None, "дрель") is False  # type: ignore[arg-type]
    assert _category_present_in_title("дрели", None) is False  # type: ignore[arg-type]


def test_inv12e_short_tokens_ignored():
    # оба токена короче 4 символов -> отфильтрованы -> совпадать нечему
    assert _category_present_in_title("топ", "топ") is False  # "топ" len=3 < 4
    assert _category_present_in_title("usb хаб", "usb-хаб") is False  # оба токена <4 симв с обеих сторон
    assert _common_prefix_len("usb", "usc") == 2  # < 4, но это только внутренняя утилита, не публичный контракт


def test_inv12e_common_prefix_len_basic():
    assert _common_prefix_len("дрели", "дрель") == 4  # "дрел"
    assert _common_prefix_len("куртки", "куртка") == 5  # "курт" + общий 'к' -> "куртк"
    assert _common_prefix_len("куртки", "шорты") == 0
    assert _common_prefix_len("", "abc") == 0
    assert _common_prefix_len("abc", "") == 0


def test_inv12e_idempotent_and_pure():
    args = ("смартфоны", "смартфон poco x6 5g")
    r1 = _category_present_in_title(*args)
    r2 = _category_present_in_title(*args)
    assert r1 == r2 is True


def test_inv12e_case_insensitive():
    assert _category_present_in_title("СМАРТФОНЫ", "Смартфон POCO X6 5G") is True


def test_inv12e_various_separators():
    # дефис / длинное тире / слэш все распознаются как разделители
    assert _category_present_in_title("дрели-ударные", "дрель/ударная bosch") is True


# ---------------------------------------------------------------------------
# Integration: _pick_best_match — oracle O1-O5 end-to-end (score/class)
# ---------------------------------------------------------------------------

def _tile(title: str) -> dict:
    return {"title": title}


def test_o1_bosch_no_penalty_reaches_exact_or_close():
    query = "Bosch GSB 13 RE"
    title = "Дрель ударная Bosch GSB 13 RE 600 Вт"
    tile, score = OzonCardSource._pick_best_match(
        query, [_tile(title)], category_leaf="Дрели ударные",
        query_brand="Bosch", query_name=query,
    )
    assert tile is not None
    # без -30 штрафа high partial/token_sort фьюжн должен пройти exact-порог
    assert score >= _BRAND_LINE_THRESHOLD, f"score={score} (штраф ложно применился?)"


def test_o2_poco_no_penalty_score_not_deflated_by_30():
    query = "POCO X6 5G"
    title = "Смартфон POCO X6 5G 8/256 ГБ"
    tile, score_fixed = OzonCardSource._pick_best_match(
        query, [_tile(title)], category_leaf="Смартфоны",
        query_brand="POCO", query_name=query,
    )
    assert tile is not None
    # сравниваем с тем же скорингом, но с категорией, которая ТОЧНО не
    # присутствует (латиница-заглушка) -> штраф обязан сработать -> ниже на 30
    tile2, score_penalized = OzonCardSource._pick_best_match(
        query, [_tile(title)], category_leaf="zzzzнесуществующая",
        query_brand="POCO", query_name=query,
    )
    assert score_fixed - score_penalized == pytest.approx(_TYPE_MISMATCH_PENALTY, abs=0.01)


def test_o3_jacket_vs_shorts_penalty_still_applies():
    query = "Куртка The North Face"
    title = "Шорты The North Face мужские"
    tile, score = OzonCardSource._pick_best_match(
        query, [_tile(title)], category_leaf="Куртки",
        query_brand=None, query_name=query,
    )
    assert tile is not None
    assert score < _BRAND_LINE_THRESHOLD, f"score={score} (гард type-mismatch ослаблен!)"


def test_o4_jacket_vs_jacket_no_penalty():
    query = "Куртка The North Face"
    title = "Куртка The North Face зимняя"
    tile_a, score_with_cat = OzonCardSource._pick_best_match(
        query, [_tile(title)], category_leaf="Куртки",
        query_brand=None, query_name=query,
    )
    tile_b, score_no_cat = OzonCardSource._pick_best_match(
        query, [_tile(title)], category_leaf=None,
        query_brand=None, query_name=query,
    )
    # категория присутствует по смыслу (курт-) -> штраф не должен был примениться;
    # скор с категорией == скор без категории (никакого -30 незаметно не влетело)
    assert score_with_cat == pytest.approx(score_no_cat, abs=0.01)


def test_o5_tshirt_vs_shorts_penalty_still_applies():
    query = "Футболка Nike"
    title = "Шорты Nike 44"
    tile, score = OzonCardSource._pick_best_match(
        query, [_tile(title)], category_leaf="Футболки",
        query_brand="Nike", query_name=query,
    )
    assert tile is not None
    assert score < _BRAND_LINE_THRESHOLD, f"score={score} (гард type-mismatch ослаблен!)"


# ---------------------------------------------------------------------------
# INV-12d: соседние пути скоринга не тронуты (spot checks)
# ---------------------------------------------------------------------------

def test_inv12d_extract_model_tokens_unchanged_behavior():
    # short numeric-ish tokens (len<3 after digit-run) остаются исключены —
    # ключевое поведение, от которого зависит clothing-гард (O5: "44" не модель).
    assert _extract_model_tokens("Шорты Nike 44") == set()
    assert "gsb13re" not in _extract_model_tokens("Bosch GSB 13 RE")  # раздельные токены короче 3


def test_inv12d_brand_mismatch_penalty_path_untouched():
    # чужой бренд в title при явном query_brand — должен уйти ниже порога
    # независимо от FIX-12 (тип штраф тут не должен маскировать brand-guard).
    query = "Samsung Galaxy A54 5G"
    title = "Смартфон Xiaomi Redmi Note 13"
    tile, score = OzonCardSource._pick_best_match(
        query, [_tile(title)], category_leaf="Смартфоны",
        query_brand="Samsung", query_name=query,
    )
    assert score < _BRAND_LINE_THRESHOLD


def test_inv12d_thresholds_constants_unchanged():
    assert _BRAND_LINE_THRESHOLD == 65.0
    assert _TYPE_MISMATCH_PENALTY == 30.0
    assert _EXACT_THRESHOLD >= _BRAND_LINE_THRESHOLD
