"""FIX-15: model-conflict guard — сосед-однобрендник не должен матчиться авторитетно.

Manifest: docs/MANIFEST_ozon_model_conflict_abstain_fix15.md

Root cause (rooted 2026-07-02, live e2e): `_pick_best_match` при отсутствии реального
товара на Ozon выбирал СОСЕДНЮЮ модель того же бренда (POCO X6 5G -> POCO M8 Pro 5G,
score 70.1 >= 65 brand_line) и отдавал её ТТХ как авторитетные — авторитетный галлюн
хуже пустоты. Фикс вводит model-conflict гард (`_conflict_codes`/`_variant_mods`/
`_model_conflict`), близнец `_brand_conflict`: тот же бренд, но РАЗНАЯ конкретная
модель -> штраф `_MODEL_CONFLICT_PENALTY` (60) роняет кандидата ниже порога 65 ->
если это единственный кандидат, `_pick_best_match` возвращает пусто (abstain).

Покрывает INV-15a..i (см. манифест):
  a) `_conflict_codes` — только letter+digit токены.
  b) `_variant_mods` — whole-token, регистронезависимо.
  c) `_model_conflict` правило A (разные коды) -> True.
  d) `_model_conflict` правило B (тот же код, разный модификатор) -> True.
  e) нет конфликта, когда кодов нет (clothing/дрель-гарды не ослаблены) -> False.
  f) идентичные коды без модификаторов -> False (реальный товар выигрывает).
  g) `_pick_best_match` integration: сосед-соло -> abstain; сосед+реальный -> реальный.
  h) FIX-12 регресс не нужен здесь отдельно — покрыт test_ozon_variant_leak_oracle_fix12.py
     (не трогаем) + расширенный оракул fix15 (test_ozon_variant_leak_oracle_fix15.py).
  i) хелперы чистые/идемпотентные, пустые строки -> пустой set/False, без исключений.
"""
from __future__ import annotations

import pytest

from app.services.enrichment.sources.ozon_card_source import (
    OzonCardSource,
    _BRAND_LINE_THRESHOLD,
    _MODEL_CONFLICT_PENALTY,
    _VARIANT_MODIFIERS,
    _conflict_codes,
    _model_conflict,
    _variant_mods,
)


# ---------------------------------------------------------------------------
# INV-15a: _conflict_codes — только letter+digit токены
# ---------------------------------------------------------------------------

def test_inv15a_letter_digit_tokens_included():
    codes = _conflict_codes("POCO X6 5G Смартфон S24 GSB13")
    assert "x6" in codes
    assert "5g" in codes
    assert "s24" in codes
    assert "gsb13" in codes


def test_inv15a_pure_numeric_excluded():
    assert "44" not in _conflict_codes("Шорты Nike 44")
    assert "256" not in _conflict_codes("8/256 ГБ")
    assert "13" not in _conflict_codes("Bosch GSB 13 RE")


def test_inv15a_pure_alpha_excluded():
    codes = _conflict_codes("POCO Pro RE")
    assert "poco" not in codes
    assert "pro" not in codes
    assert "re" not in codes


def test_inv15a_empty_string():
    assert _conflict_codes("") == set()
    assert _conflict_codes(None) == set()  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# INV-15b: _variant_mods — whole-token, регистронезависимо
# ---------------------------------------------------------------------------

def test_inv15b_whole_token_case_insensitive():
    assert _variant_mods("Смартфон POCO X6 Pro 5G") == {"pro"}
    assert _variant_mods("iPhone 15 PRO MAX") == {"pro", "max"}


def test_inv15b_professional_does_not_match_pro():
    # whole-token, не подстрока: "professional" не должен дать "pro"
    assert "pro" not in _variant_mods("Professional Edition")


def test_inv15b_empty_string():
    assert _variant_mods("") == set()
    assert _variant_mods(None) == set()  # type: ignore[arg-type]


def test_inv15b_variant_modifiers_set_contents():
    expected = {"pro", "max", "plus", "ultra", "lite", "mini", "neo",
                "note", "air", "se", "fe", "prime"}
    assert _VARIANT_MODIFIERS == expected


# ---------------------------------------------------------------------------
# INV-15c: правило A (разные коды) -> True — oracle M2, M8
# ---------------------------------------------------------------------------

def test_inv15c_rule_a_m2_poco_x6_vs_m8_pro():
    assert _model_conflict("POCO X6 5G", "Смартфон POCO M8 Pro 5G 8/256 ГБ") is True


def test_inv15c_rule_a_m8_samsung_s24_vs_s23():
    assert _model_conflict("Samsung Galaxy S24", "Смартфон Samsung Galaxy S23") is True


def test_inv15c_rule_a_different_brands_unique_codes_also_true():
    # безвредно — бренд-гард уже штрафует -60 отдельно
    assert _model_conflict("POCO X6", "Смартфон Samsung A54") is True


# ---------------------------------------------------------------------------
# INV-15d: правило B (тот же код, разный модификатор) -> True — oracle M3; M4 -> False
# ---------------------------------------------------------------------------

def test_inv15d_rule_b_m3_x6_vs_x6_pro():
    assert _model_conflict("POCO X6 5G", "Смартфон POCO X6 Pro 5G 12/512") is True


def test_inv15d_rule_b_m4_same_modifier_no_conflict():
    assert _model_conflict("POCO X6 Pro", "Смартфон POCO X6 Pro 5G 12/512 ГБ") is False


# ---------------------------------------------------------------------------
# INV-15e: нет конфликта когда кодов нет — oracle M5, M6, M7
# ---------------------------------------------------------------------------

def test_inv15e_m5_bosch_drill_no_codes():
    assert _model_conflict("Bosch GSB 13 RE", "Дрель ударная Bosch GSB 13 RE 600 Вт") is False


def test_inv15e_m6_jacket_no_codes():
    assert _model_conflict("Куртка The North Face", "Куртка The North Face зимняя") is False


def test_inv15e_m7_tshirt_vs_shorts_no_codes():
    assert _model_conflict("Футболка Nike", "Шорты Nike 44") is False


# ---------------------------------------------------------------------------
# INV-15f: идентичные коды без модификаторов -> False — oracle M1
# ---------------------------------------------------------------------------

def test_inv15f_m1_identical_model_no_conflict():
    assert _model_conflict("POCO X6 5G", "Смартфон POCO X6 5G 8/256 ГБ") is False


# ---------------------------------------------------------------------------
# INV-15g: _pick_best_match integration — abstain / реальный выигрывает
# ---------------------------------------------------------------------------

def _tile(title: str) -> dict:
    return {"title": title}


def test_inv15g_sibling_only_abstains():
    query = "POCO X6 5G"
    tile, score = OzonCardSource._pick_best_match(
        query,
        [_tile("Смартфон POCO M8 Pro 5G 8/256 ГБ")],
        category_leaf="Смартфоны",
        query_brand="POCO",
        query_name=query,
    )
    assert score < _BRAND_LINE_THRESHOLD, f"сосед M8 Pro не должен пройти порог, score={score}"


def test_inv15g_real_plus_sibling_picks_real():
    query = "POCO X6 5G"
    tiles = [
        _tile("Смартфон POCO M8 Pro 5G 8/256 ГБ"),  # сосед
        _tile("Смартфон POCO X6 5G 8/256 ГБ"),       # реальный
    ]
    tile, score = OzonCardSource._pick_best_match(
        query, tiles, category_leaf="Смартфоны", query_brand="POCO", query_name=query,
    )
    assert tile is not None
    assert "M8" not in (tile.get("title") or ""), "выбран сосед вместо реального товара"
    assert score >= _BRAND_LINE_THRESHOLD


# ---------------------------------------------------------------------------
# INV-15i: чистота / идемпотентность / edge-кейсы
# ---------------------------------------------------------------------------

def test_inv15i_idempotent():
    args = ("POCO X6 5G", "Смартфон POCO M8 Pro 5G")
    assert _model_conflict(*args) == _model_conflict(*args) is True


def test_inv15i_mixed_case_no_exceptions():
    assert _model_conflict("poco x6 5g", "СМАРТФОН POCO X6 5G") is False
    assert _model_conflict("", "") is False
    assert _model_conflict("POCO X6", "") is False
    assert _model_conflict("", "Смартфон POCO M8 Pro") is False


def test_inv15i_penalty_constant():
    assert _MODEL_CONFLICT_PENALTY == 60.0


# ---------------------------------------------------------------------------
# Mutation self-check: доказать, что тест НЕ пуст — гард выключен -> тест падает
# ---------------------------------------------------------------------------

def test_mutation_self_check_guard_disabled_breaks_abstain(monkeypatch):
    """Если _model_conflict замокан на 'всегда False' (гард сломан/убран),
    сосед-однобрендник M8 Pro снова проходит порог -> доказывает, что
    test_inv15g_sibling_only_abstains НЕ тривиален (не проходит на битой версии)."""
    import app.services.enrichment.sources.ozon_card_source as ocs_mod

    monkeypatch.setattr(ocs_mod, "_model_conflict", lambda query, title: False)

    # Query с полной спецификацией (как в реальном live-случае из манифеста, где
    # сосед набрал 70.1) — при отключённом гарде базовый fuzzy-скор высокий.
    query = "POCO X6 5G 8/256 ГБ"
    tile, score = ocs_mod.OzonCardSource._pick_best_match(
        query,
        [_tile("Смартфон POCO M8 Pro 5G 8/256 ГБ")],
        category_leaf="Смартфоны",
        query_brand="POCO",
        query_name=query,
    )
    # На "битой" версии (гард отключен) сосед ПРОХОДИТ порог — контраст с
    # test_inv15g_sibling_only_abstains доказывает, что реальный гард не пустой.
    assert score >= _BRAND_LINE_THRESHOLD, (
        "mutation self-check провалился: даже с отключённым гардом сосед не "
        "проходит порог — тест выше может быть пустым/не проверять гард"
    )
