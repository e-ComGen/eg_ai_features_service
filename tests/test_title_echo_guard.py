# -*- coding: utf-8 -*-
"""Tests for the title-echo guard (UniversalGate Step 1d): a sub-name /
descriptive field whose value equals the WHOLE product title is a lazy
placeholder echo and is dropped. The main name/merge fields legitimately hold
the title and must NOT match the denylist."""
import pytest

from app.services.enrichment.pipeline import PipelineOrchestrator as P

RE = P._TITLE_ECHO_FIELD_RE
norm = P._norm_for_title_echo


# fields where value == full title is always wrong → must match denylist
DENYLIST_FIELDS = [
    "Аннотация",
    "Название вкуса",
    "Название цвета",
    "Название аромата",
    "Название запаха",
    "Модель ТС",
]

# main name / merge fields that legitimately hold the title → must NOT match
LEGIT_NAME_FIELDS = [
    "Название",
    "Наименование",
    "Название файла PDF",
    "Название модели (для объединения в одну карточку)",
    "Название модели для шаблона наименования",
    "Объединить в похожие товары",
    "Озон.Видео: название",
]


@pytest.mark.parametrize("field", DENYLIST_FIELDS)
def test_denylist_fields_match(field):
    assert RE.search(field) is not None


@pytest.mark.parametrize("field", LEGIT_NAME_FIELDS)
def test_legit_name_fields_do_not_match(field):
    assert RE.search(field) is None


def test_norm_equality_ignores_punctuation_and_case():
    assert norm("Автомагнитола Pioneer AVH-X7800BT") == \
        norm("автомагнитола pioneer  avh x7800bt")


def test_norm_handles_real_pairs():
    # value echoes the title (punctuation/case differences) → equal
    assert norm("Чай листовой Ahmad Tea English Breakfast 200г") == \
        norm("Чай листовой Ahmad Tea English Breakfast 200г")
    # a real descriptive annotation does NOT equal the title
    assert norm("Покорите зимние дороги с шинами Nokian") != \
        norm("Шины Nokian Tyres Nordman 8 205/55 R16")
