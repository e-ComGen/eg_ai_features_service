"""Tests for gender fixes: synonym→value_id normalizer + required-field fallback.

Два генеральных (без хардкода категорий) фикса для REQUIRED поля «Пол» (Ozon
attr 9163, is_required, is_collection):

  FIX 1 (ozon_loader.normalize_gender_value): свободные гендер-фразы карточки
         («Мужчинам», «Для мужчин», «men's») мапятся на канон Ozon («Мужской»)
         ДО резолва value_id, чтобы 22880/22881/22882/22883 не терялись.
  FIX 2 (pipeline._apply_gender_guard): когда гард дропает значение, конфликтую-
         щее с явным полом ИМЕНИ, и REQUIRED поле «Пол» опустело бы — подставляем
         канон пола из имени («Футболка мужская» → «Мужской»). Детей не выдумываем.

Все тесты — чистые unit-тесты, без LLM/сети.
"""
import json
from unittest.mock import patch

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import _apply_gender_guard


# Канонические значения «Пол» Ozon (attr 9163) с реальными value_id.
_GENDER_VALUES = [
    {"id": 22880, "value": "Мужской"},
    {"id": 22881, "value": "Женский"},
    {"id": 22882, "value": "Девочки"},
    {"id": 22883, "value": "Мальчики"},
]

_OZON_GENDER_DICT = {
    "schema_version": 2,
    "source": "ozon_seller_api",
    "generated_at": "2026-06-01",
    "categories": {
        "100:200": {
            "description_category_id": 100,
            "type_id": 200,
            "name": "Одежда",
            "path": ["Одежда"],
            "characteristics": [
                {"id": 9163, "name": "Пол", "type": "Option",
                 "is_required": True, "is_collection": True,
                 "description": "Пол", "values": _GENDER_VALUES},
            ],
        }
    },
}


def _reset_ozon_loader_cache() -> None:
    from app.services.enrichment.strategies.dictionaries.ozon_loader import (
        load_ozon_dictionary,
    )
    load_ozon_dictionary.cache_clear()


def _gender_target(is_required: bool = True) -> TargetAttribute:
    return TargetAttribute(
        id=9163, name="Пол", type="enum",
        is_collection=True, is_required=is_required,
    )


def _ctx(product_name: str) -> ExtractionContext:
    return ExtractionContext(
        product_id=1, product_name=product_name,
        category_id=100, ozon_type_id=200,
    )


# ---------------------------------------------------------------------------
# normalize_gender_value — прямые unit-проверки маппинга вариант→канон
# ---------------------------------------------------------------------------

def test_normalize_gender_value_adult_variants():
    from app.services.enrichment.strategies.dictionaries.ozon_loader import (
        normalize_gender_value,
    )
    for variant in ("Мужчинам", "Для мужчин", "мужская", "men's", "MALE"):
        assert normalize_gender_value(variant) == "Мужской", variant
    for variant in ("Женщинам", "для женщин", "женское", "women's", "Female"):
        assert normalize_gender_value(variant) == "Женский", variant


def test_normalize_gender_value_kids_distinct_from_adults():
    """КРИТИЧНО: дети НЕ схлопываются во взрослых."""
    from app.services.enrichment.strategies.dictionaries.ozon_loader import (
        normalize_gender_value,
    )
    assert normalize_gender_value("для девочек") == "Девочки"
    assert normalize_gender_value("девочка") == "Девочки"
    assert normalize_gender_value("girls") == "Девочки"
    assert normalize_gender_value("для мальчиков") == "Мальчики"
    assert normalize_gender_value("boy") == "Мальчики"
    # И именно НЕ взрослый канон:
    assert normalize_gender_value("для девочек") != "Женский"
    assert normalize_gender_value("для мальчиков") != "Мужской"


def test_normalize_gender_value_unknown_returns_none():
    from app.services.enrichment.strategies.dictionaries.ozon_loader import (
        normalize_gender_value,
    )
    assert normalize_gender_value("Унисекс") is None
    assert normalize_gender_value("красный") is None
    assert normalize_gender_value("") is None
    assert normalize_gender_value(None) is None


# ---------------------------------------------------------------------------
# FIX 1 — variant phrase resolves to canonical value_id
# ---------------------------------------------------------------------------

def test_fix1_muzhchinam_resolves_to_22880(tmp_path):
    """«Мужчинам»/«Для мужчин» → нормализатор → «Мужской» → value_id 22880."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_GENDER_DICT), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ), patch(
        # matcher отключён: покрываем только synonym→exact путь, без семантики
        "app.services.enrichment.strategies.dictionaries.ozon_loader._get_matcher",
        return_value=None,
    ):
        _reset_ozon_loader_cache()
        from app.services.enrichment.strategies.dictionaries.ozon_loader import (
            resolve_value_id,
        )
        assert resolve_value_id(100, 200, 9163, "Мужчинам") == 22880
        assert resolve_value_id(100, 200, 9163, "Для мужчин") == 22880
        assert resolve_value_id(100, 200, 9163, "Женщинам") == 22881
        # дети — корректные отдельные id
        assert resolve_value_id(100, 200, 9163, "для девочек") == 22882
        assert resolve_value_id(100, 200, 9163, "для мальчиков") == 22883


def test_fix1_regression_canonical_unchanged(tmp_path):
    """Регрессия: уже канонический «Мужской» по-прежнему даёт 22880."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_GENDER_DICT), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ), patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader._get_matcher",
        return_value=None,
    ):
        _reset_ozon_loader_cache()
        from app.services.enrichment.strategies.dictionaries.ozon_loader import (
            resolve_value_id,
        )
        assert resolve_value_id(100, 200, 9163, "Мужской") == 22880
        assert resolve_value_id(100, 200, 9163, "Женский") == 22881


def test_fix1_normalizer_neutral_for_nongender_attr(tmp_path):
    """Нормализатор нейтрален для не-гендерных атрибутов (канона в словаре нет)."""
    dict_with_color = json.loads(json.dumps(_OZON_GENDER_DICT))
    dict_with_color["categories"]["100:200"]["characteristics"].append(
        {"id": 1003, "name": "Цвет", "type": "Option",
         "is_required": False, "is_collection": False,
         "values": [{"id": 501, "value": "Красный"}]}
    )
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(dict_with_color), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ), patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader._get_matcher",
        return_value=None,
    ):
        _reset_ozon_loader_cache()
        from app.services.enrichment.strategies.dictionaries.ozon_loader import (
            resolve_value_id,
        )
        # «мужская» как значение цвета — канона «Мужской» в values нет → None,
        # а реальный цвет резолвится как обычно.
        assert resolve_value_id(100, 200, 1003, "мужская") is None
        assert resolve_value_id(100, 200, 1003, "Красный") == 501


# ---------------------------------------------------------------------------
# FIX 2 — required field stays filled after guard drop
# ---------------------------------------------------------------------------

def test_fix2_required_filled_after_dropping_conflicting():
    """Футболка «мужская» + карточка «Женский» → дроп female, REQUIRED Пол
    заполнен «Мужской» (а не пусто)."""
    ctx = _ctx("Футболка мужская")
    targets = [_gender_target(is_required=True)]
    values = [
        AttributeValue(
            attribute_id=9163, value="Женский", confidence=0.8,
            source=Source.OZON_CARD,
        ),
    ]
    out = _apply_gender_guard(values, targets, ctx)
    gender_vals = [v for v in out if v.attribute_id == 9163]
    assert gender_vals, "REQUIRED Пол не должен опустеть"
    assert all(str(v.value) == "Мужской" for v in gender_vals)


def test_fix2_unknown_name_gender_stays_empty():
    """Нейтральное имя + единственный конфликт нечем заменить → НЕ выдумываем.

    Имя нейтрально (None) → правило 2 не срабатывает, дроп идёт только если все
    источники external. Карточный источник → значение остаётся. Здесь проверяем,
    что fallback НЕ инжектится при name_gender=None даже для required."""
    ctx = _ctx("Кроссовки Ultraboost 22")
    targets = [_gender_target(is_required=True)]
    # Единственный кандидат — external-guess (web_search) → дропнется правилом 3.
    values = [
        AttributeValue(
            attribute_id=9163, value="Женский", confidence=0.7,
            source=Source.WEB_SEARCH,
        ),
    ]
    out = _apply_gender_guard(values, targets, ctx)
    gender_vals = [v for v in out if v.attribute_id == 9163]
    # Имя нейтрально → пол неизвестен → fallback НЕ инжектится, поле пусто.
    assert not gender_vals, "Нейтральное имя: пол не выдумываем"


def test_fix2_no_fallback_for_optional_field():
    """Для OPTIONAL поля fallback НЕ инжектится (поведение опционалок не меняем)."""
    ctx = _ctx("Футболка мужская")
    targets = [_gender_target(is_required=False)]
    values = [
        AttributeValue(
            attribute_id=9163, value="Женский", confidence=0.8,
            source=Source.OZON_CARD,
        ),
    ]
    out = _apply_gender_guard(values, targets, ctx)
    gender_vals = [v for v in out if v.attribute_id == 9163]
    assert not gender_vals, "OPTIONAL: опустевшее поле остаётся пустым"


def test_fix2_fallback_never_invents_kids_canon():
    """Fallback подставляет ТОЛЬКО взрослый канон (Мужской/Женский), никогда
    детский — детский подтип из имени не выводим. Имя женское + конфликтная
    мужская карточка → REQUIRED заполнен «Женский» (не «Девочки»/«Мальчики»)."""
    ctx = _ctx("Платье женское")
    targets = [_gender_target(is_required=True)]
    values = [
        AttributeValue(
            attribute_id=9163, value="Мужской", confidence=0.8,
            source=Source.OZON_CARD,
        ),
    ]
    out = _apply_gender_guard(values, targets, ctx)
    gender_vals = [v for v in out if v.attribute_id == 9163]
    assert gender_vals
    assert all(str(v.value) == "Женский" for v in gender_vals)
    assert all(str(v.value) not in ("Девочки", "Мальчики") for v in gender_vals)


def test_fix2_no_drop_no_fallback_regression():
    """Регрессия: Джинсы «мужская» + карточка «Мужской» — ничего не дропается,
    fallback не дублирует, значение проходит как есть."""
    ctx = _ctx("Джинсы мужская")
    targets = [_gender_target(is_required=True)]
    values = [
        AttributeValue(
            attribute_id=9163, value="Мужской", confidence=0.9,
            source=Source.OZON_CARD,
        ),
    ]
    out = _apply_gender_guard(values, targets, ctx)
    gender_vals = [v for v in out if v.attribute_id == 9163]
    assert len(gender_vals) == 1
    assert str(gender_vals[0].value) == "Мужской"
    assert gender_vals[0].source == Source.OZON_CARD  # не подменён fallback-ом
