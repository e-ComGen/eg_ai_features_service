"""Enum-membership гард для МНОГОЗНАЧНЫХ (Ⓜ️) словарных полей.

Баг (тостер «Функциональные особенности», cat тостера, attr словарь из 7 значений):
поле — Множественный выбор, движок выдал список
['выдвижной (съемный) лоток для крошек', 'экстра-подъем', 'автоцентрирование тостов']
— ноль точных членов словаря Ozon, мусор от не-LLM источника (IceCat/donor) протёк
мимо LLM-гейта _enforce_allowed_values И мимо скалярного finalize-гарда (тот
исключал списки). Фикс: _apply_enum_membership_guard фильтрует каждый член списка.
"""
from app.services.enrichment.base import AttributeValue, Source, TargetAttribute
from app.services.enrichment.pipeline import _apply_enum_membership_guard

_TOASTER_FEAT = [
    "Автозагрузка тостов",
    "Автоматическое поднятие",
    "Автоматическое центрирование тостов",
    "Кнопка отмены",
    "Регулировка степени обжаривания",
    "Рисунок на тосте",
    "Экстра-подъем тостов",
]


def _target(attr_id, *, name="Attr", allowed=None, is_collection=False):
    return TargetAttribute(
        id=attr_id,
        name=name,
        type="enum" if allowed else "text",
        allowed_values=allowed,
        is_collection=is_collection,
    )


def _value(attr_id, value, *, source=Source.ICECAT, value_ids=None,
           is_collection=False, evidence=None):
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=0.8,
        source=source,
        value_ids=value_ids,
        is_collection=is_collection,
        evidence=evidence,
    )


def _by_id(targets):
    return {t.id: t for t in targets}


# ── основной кейс: мусор дропнут, канон оставлен/каноникализирован, дедуп ──────

def test_collection_drops_garbage_keeps_canonical():
    targets = [_target(52, name="Функциональные особенности",
                       allowed=_TOASTER_FEAT, is_collection=True)]
    merged = [_value(52, [
        "выдвижной (съемный) лоток для крошек",   # мусор → дроп
        "автоцентрирование тостов",                # вариант → канон
        "Автоматическое центрирование тостов",     # точное (дубль варианта)
        "Регулировка степени обжаривания",         # точное
    ], value_ids=[111, 222, 333, 444], is_collection=True)]
    out = _apply_enum_membership_guard(merged, _by_id(targets))
    assert len(out) == 1
    vals = out[0].value
    # мусор ушёл
    assert "выдвижной (съемный) лоток для крошек" not in vals
    # все оставшиеся — из словаря
    assert all(x in _TOASTER_FEAT for x in vals)
    # «Автоматическое центрирование тостов» присутствует ровно один раз (дедуп
    # варианта+точного)
    assert vals.count("Автоматическое центрирование тостов") == 1
    assert "Регулировка степени обжаривания" in vals
    # состав изменился → value_ids сброшены под пере-резолв
    assert out[0].value_ids is None


# ── все члены — мусор → поле дропнуто целиком ─────────────────────────────────

def test_collection_all_garbage_drops_field():
    targets = [_target(52, name="Функциональные особенности",
                       allowed=_TOASTER_FEAT, is_collection=True)]
    merged = [_value(52, ["выдвижной лоток", "подогрев готовых тостов"],
                     is_collection=True)]
    out = _apply_enum_membership_guard(merged, _by_id(targets))
    assert out == []


# ── чистый список не трогаем, value_ids сохраняются ──────────────────────────

def test_collection_all_valid_unchanged_keeps_value_ids():
    targets = [_target(52, allowed=_TOASTER_FEAT, is_collection=True)]
    merged = [_value(52, ["Кнопка отмены", "Рисунок на тосте"],
                     value_ids=[10, 20], is_collection=True)]
    out = _apply_enum_membership_guard(merged, _by_id(targets))
    assert len(out) == 1
    assert out[0].value == ["Кнопка отмены", "Рисунок на тосте"]
    assert out[0].value_ids == [10, 20]


# ── скаляр-регрессия: мусор дроп, точное оставлено ───────────────────────────

def test_scalar_out_of_list_dropped():
    targets = [_target(33, name="Количество отделений", allowed=["1", "2", "3", "4"])]
    merged = [_value(33, "8")]
    assert _apply_enum_membership_guard(merged, _by_id(targets)) == []


def test_scalar_in_list_kept():
    targets = [_target(33, name="Количество отделений", allowed=["1", "2", "3", "4"])]
    merged = [_value(33, "2")]
    out = _apply_enum_membership_guard(merged, _by_id(targets))
    assert len(out) == 1 and out[0].value == "2"


# ── исключения: ТН ВЭД и поля без словаря не трогаем ─────────────────────────

def test_tnved_list_exempt():
    targets = [_target(21, name="ТН ВЭД коды ЕАЭС", allowed=["8516720000 - Тостеры"])]
    merged = [_value(21, "8516720000 - Прочее", evidence="tnved_resolver:live")]
    out = _apply_enum_membership_guard(merged, _by_id(targets))
    assert len(out) == 1 and out[0].value == "8516720000 - Прочее"


def test_no_allowed_values_passthrough():
    targets = [_target(99, name="Комплектация", allowed=None)]
    merged = [_value(99, ["тостер", "инструкция"], is_collection=True)]
    out = _apply_enum_membership_guard(merged, _by_id(targets))
    assert len(out) == 1 and out[0].value == ["тостер", "инструкция"]
