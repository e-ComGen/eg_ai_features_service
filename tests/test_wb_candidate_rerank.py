"""Оракул для #3: локальный рерэнк кандидатов search.wb.ru по релевантности name.

search.wb.ru отдаёт products в popular-порядке; точный артикул часто вне топ-10
(живой факт: «Тостер Bosch TAT3A011» → popular-топ = ДРУГИЕ модели Bosch, целевой
артикул вне топа → терялся при обрезке на 10 → карта не находилась). _rank_candidates
скорит name+brand fuzzy к запросу и берёт top-N по релевантности (0 доп. кредитов).

Токенизация Cyrillic-aware (баг latin-only regex поймали на ревью tier-0). Решение:
docs/adr (fable). Не удалять без пере-обоснования.
"""
import pytest

from app.services.enrichment.sources.wb_card_source import (
    _candidate_relevance as rel,
    _rank_candidates as rank,
)

Q = "Тостер Bosch TAT3A011"


def test_exact_article_scores_max():
    assert rel(Q, "Тостер TAT3A011 нержавейка", "BOSCH") == 1.0


def test_wrong_article_scores_partial():
    s = rel(Q, "Тостер TAT4P429", "BOSCH")
    assert 0.0 < s < 1.0


def test_exact_beats_wrong_cyrillic_preserved():
    assert rel(Q, "Тостер TAT3A011", "BOSCH") > rel(Q, "Тостер TAT4P429", "BOSCH")


def test_cyrillic_discriminator_works():
    assert rel("сок яблочный", "Сок яблочный Добрый", "") > \
        rel("сок яблочный", "Сок апельсиновый", "")


def test_low_popular_target_surfaced_into_top10():
    prods = [{"id": 1000 + i, "name": f"Тостер Bosch TAT{i}X", "brand": "BOSCH"}
             for i in range(30)]
    prods[25] = {"id": 9999, "name": "Тостер TAT3A011 стальной", "brand": "BOSCH"}
    top = rank(prods, Q)
    assert 9999 in top and len(top) == 10 and top[0] == 9999


def test_already_top_target_not_regressed():
    prods = [{"id": 9999, "name": "Тостер TAT3A011 стальной", "brand": "BOSCH"}]
    prods += [{"id": 2000 + i, "name": f"Тостер X{i}", "brand": "BOSCH"}
              for i in range(20)]
    assert 9999 in rank(prods, Q)


def test_no_names_graceful_popular_fallback():
    prods = [{"id": 700 + i} for i in range(20)]
    assert rank(prods, Q) == list(range(700, 710))


def test_malformed_entries_skipped():
    prods = [None, {"id": "x"}, {"noid": 1}, {"id": 5, "name": "Тостер TAT3A011"}]
    assert rank(prods, Q) == [5]
