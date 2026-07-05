"""Оракул-таблица для _category_present_in_title (recall-fix узких категорий).

Старый матчер (токены len>=4, префикс>=4) давал false-negative на легит-матчах, где
корень категории в заголовке ровно 3 символа («Соки↔сок», «Ножи↔нож») → ложный штраф
_TYPE_MISMATCH_PENALTY=-30 → валидная Ozon-карта уходила в abstain (просадка recall).

Новый матчер добавляет правило B (короткий токен — полный префикс длинного, остаток
<= 2 симв. ∈ RU-флексий), сохраняя правило A (общий префикс >= 4). Этот файл фиксит
поведение построчно, чтобы будущие правки не вернули ни false-negative, ни false-positive.
Решение архитектуры: docs/adr (fable). Не удалять без пере-обоснования.
"""
import pytest

from app.services.enrichment.sources.ozon_card_source import _category_present_in_title as f

# (category_leaf, card_title, expected)
ORACLE = [
    # --- rule B: 3-символьные корни (то, что старый матчер терял) ---
    ("соки", "сок добрый яблочный", True),
    ("ножи", "нож кухонный tramontina", True),
    # --- rule B precision: НЕ ловить ложное ---
    ("соки", "сокол охотничий", False),      # остаток «ол» не флексия
    ("соки", "соска детская", False),        # «сок» не полный префикс «соска»
    ("ножи", "ножницы канцелярские", False), # остаток «ницы» > 2 симв.
    # --- rule A: старое поведение сохранено ---
    ("блендеры", "блендер philips погружной", True),
    ("часы", "часы casio наручные", True),
    ("ручки", "ручка parker шариковая", True),
    ("диски", "диск отрезной bosch", True),
    ("лампы", "лампа led gauss e27", True),
    ("мыло", "мыло dove кремовое", True),
    ("часы", "часовщик мастерская", False),  # префикс «час»=3 < 4, rule B не срабатывает
    # --- слишком короткие токены отфильтрованы (нужна эскалация identity, не матчер) ---
    ("тв", "телевизор samsung qled", False),
    # --- пустые входы ---
    ("", "любой заголовок", False),
    ("категория", "", False),
]


@pytest.mark.parametrize("cat,title,expected", ORACLE)
def test_category_present_in_title_oracle(cat, title, expected):
    assert f(cat.lower(), title.lower()) is expected
