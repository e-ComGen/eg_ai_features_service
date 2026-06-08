"""Tests for MarketplaceStrategy pattern — WB, Ozon, Default, Factory.

All tests are pure unit tests — no LLM calls, no external I/O.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.strategies.base import MarketplaceStrategy, ValidationResult
from app.services.enrichment.strategies.default_strategy import DefaultStrategy
from app.services.enrichment.strategies.wildberries_strategy import (
    WildberriesStrategy,
    WB_SKIP_SEMANTIC_TYPES,
    WB_BANNED_PHRASES,
)
from app.services.enrichment.strategies.ozon_strategy import OzonStrategy, OZON_SKIP_SEMANTIC_TYPES
from app.services.enrichment.strategies.factory import get_strategy
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.strategies.base import MarketplaceStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(
    id: int = 1,
    name: str = "Цвет",
    type: str = "text",
    semantic_type: str | None = None,
) -> TargetAttribute:
    return TargetAttribute(id=id, name=name, type=type, semantic_type=semantic_type)


def _make_ctx() -> ExtractionContext:
    return ExtractionContext(product_id=1, product_name="Test Product", category_id=1)


def _make_value(attr_id: int = 1, value: str = "red", source: Source = Source.DESCRIPTION) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=0.9,
        source=source,
    )


def _mock_source(source_type: Source, extract_return=None):
    from app.services.enrichment.base import AttributeSource, LlmJudge
    s = MagicMock(spec=AttributeSource)
    s.source_type = source_type
    s.is_applicable = MagicMock(return_value=True)
    s.extract = AsyncMock(return_value=extract_return or [])
    judge = MagicMock(spec=LlmJudge)
    judge.source = source_type
    judge.validate = AsyncMock(return_value=True)
    s.get_judge = MagicMock(return_value=judge)
    return s


def _build_pipeline(strategy=None, desc_values=None):
    desc_src = _mock_source(Source.DESCRIPTION, desc_values or [])
    know_src = _mock_source(Source.LLM_KNOWLEDGE, [])
    vis_src = _mock_source(Source.VISION, [])
    web_src = _mock_source(Source.WEB_SEARCH, [])

    classifier = MagicMock()
    classifier.classify = AsyncMock(return_value={})
    cost_pred = MagicMock()
    cost_pred.is_web_search_worth = AsyncMock(return_value=False)

    orch = PipelineOrchestrator(
        description_source=desc_src,
        knowledge_source=know_src,
        vision_source=vis_src,
        websearch_source=web_src,
        classifier=classifier,
        cost_predictor=cost_pred,
        strategy=strategy,
    )
    return orch


# ---------------------------------------------------------------------------
# 1. DefaultStrategy — filters nothing
# ---------------------------------------------------------------------------

def test_default_strategy_noop_filters_nothing():
    strategy = DefaultStrategy()
    targets = [
        _make_target(1, semantic_type="ean"),
        _make_target(2, semantic_type="article"),
        _make_target(3, semantic_type="color"),
    ]
    result = strategy.filter_unsupported_attributes(targets)
    assert result == targets  # no-op: all kept


# ---------------------------------------------------------------------------
# 2. DefaultStrategy — validates everything as valid
# ---------------------------------------------------------------------------

def test_default_strategy_validates_everything_as_valid():
    strategy = DefaultStrategy()
    target = _make_target(1, type="text")
    ctx = _make_ctx()

    for value in ["лучший", "№1", "some random text", "", 42, None]:
        result = strategy.validate_value(target, value, ctx)
        assert result.is_valid, f"DefaultStrategy should accept value: {value!r}"


# ---------------------------------------------------------------------------
# 3. WB — skips EAN attributes
# ---------------------------------------------------------------------------

def test_wb_strategy_skips_ean_attributes():
    strategy = WildberriesStrategy()
    targets = [
        _make_target(1, semantic_type="ean"),
        _make_target(2, semantic_type="color"),
    ]
    result = strategy.filter_unsupported_attributes(targets)
    assert len(result) == 1
    assert result[0].id == 2


# ---------------------------------------------------------------------------
# 4. WB — skips article/sku attributes
# ---------------------------------------------------------------------------

def test_wb_strategy_skips_article_attributes():
    strategy = WildberriesStrategy()
    targets = [
        _make_target(1, semantic_type="article"),
        _make_target(2, semantic_type="sku"),
        _make_target(3, semantic_type="material"),
    ]
    result = strategy.filter_unsupported_attributes(targets)
    ids = [t.id for t in result]
    assert 1 not in ids
    assert 2 not in ids
    assert 3 in ids


# ---------------------------------------------------------------------------
# 5. WB — rejects banned phrases
# ---------------------------------------------------------------------------

def test_wb_strategy_rejects_banned_phrases():
    strategy = WildberriesStrategy()
    target = _make_target(1, type="text")
    ctx = _make_ctx()

    for phrase in WB_BANNED_PHRASES:
        text = f"Это {phrase} товар на рынке"
        result = strategy.validate_value(target, text, ctx)
        assert not result.is_valid, f"WB should reject text containing '{phrase}'"
        assert phrase in (result.reason or "")


# ---------------------------------------------------------------------------
# 6. WB — accepts normal text
# ---------------------------------------------------------------------------

def test_wb_strategy_accepts_normal_text():
    strategy = WildberriesStrategy()
    target = _make_target(1, type="text")
    ctx = _make_ctx()

    result = strategy.validate_value(target, "Хлопковая футболка синего цвета", ctx)
    assert result.is_valid
    assert result.normalized_value == "Хлопковая футболка синего цвета"


# ---------------------------------------------------------------------------
# 7. OzonStrategy — basic init and name
# ---------------------------------------------------------------------------

def test_ozon_strategy_basic_init():
    strategy = OzonStrategy()
    assert strategy.name == "ozon"
    # filter_unsupported_attributes is callable
    targets = [_make_target(1, semantic_type="color"), _make_target(2, semantic_type="ean")]
    result = strategy.filter_unsupported_attributes(targets)
    # EAN should be filtered
    ids = [t.id for t in result]
    assert 2 not in ids
    assert 1 in ids


# ---------------------------------------------------------------------------
# 8. Factory — returns DefaultStrategy for None
# ---------------------------------------------------------------------------

def test_factory_returns_default_for_none():
    strategy = get_strategy(None)
    assert isinstance(strategy, DefaultStrategy)
    assert strategy.name == "default"


# ---------------------------------------------------------------------------
# 9. Factory — returns DefaultStrategy for unknown name
# ---------------------------------------------------------------------------

def test_factory_returns_default_for_unknown_name():
    strategy = get_strategy("unknown_marketplace_xyz")
    assert isinstance(strategy, DefaultStrategy)
    assert strategy.name == "default"


# ---------------------------------------------------------------------------
# 10. Factory — returns WbStrategy for "wb" and "wildberries"
# ---------------------------------------------------------------------------

def test_factory_returns_wb_strategy_for_wb_or_wildberries():
    wb1 = get_strategy("wb")
    assert isinstance(wb1, WildberriesStrategy)
    assert wb1.name == "wb"

    wb2 = get_strategy("wildberries")
    assert isinstance(wb2, WildberriesStrategy)

    # Case-insensitive
    wb3 = get_strategy("WB")
    assert isinstance(wb3, WildberriesStrategy)


# ---------------------------------------------------------------------------
# 11. Factory — returns OzonStrategy for "ozon"
# ---------------------------------------------------------------------------

def test_factory_returns_ozon_strategy():
    strategy = get_strategy("ozon")
    assert isinstance(strategy, OzonStrategy)
    assert strategy.name == "ozon"

    # Case-insensitive
    strategy2 = get_strategy("OZON")
    assert isinstance(strategy2, OzonStrategy)


# ---------------------------------------------------------------------------
# 12. PipelineOrchestrator — uses DefaultStrategy when not specified
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_uses_default_strategy_when_not_specified():
    orch = _build_pipeline(strategy=None)
    assert isinstance(orch._strategy, DefaultStrategy)
    # Ensure enrich runs without error
    ctx = _make_ctx()
    result = await orch.enrich(ctx, [_make_target(1)])
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# 13. PipelineOrchestrator — uses strategy.filter_unsupported_attributes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_uses_strategy_filter_unsupported():
    # WB strategy filters out ean semantic_type
    strategy = WildberriesStrategy()
    orch = _build_pipeline(strategy=strategy)

    targets = [
        _make_target(1, semantic_type="ean"),   # should be filtered
        _make_target(2, semantic_type="color"),  # should remain
    ]
    ctx = _make_ctx()

    # The description source mock returns value for attr 2
    desc_values = [_make_value(attr_id=2, value="blue")]
    desc_src = _mock_source(Source.DESCRIPTION, desc_values)
    orch._sources[Source.DESCRIPTION] = desc_src
    from app.services.enrichment.confidence_aware_judge import ConfidenceAwareJudgeWrapper
    orch._judges[Source.DESCRIPTION] = ConfidenceAwareJudgeWrapper(desc_src.get_judge())

    result = await orch.enrich(ctx, targets)
    returned_ids = {v.attribute_id for v in result}
    # attr 1 (ean) should never appear — WB filters it before pipeline
    assert 1 not in returned_ids


# ---------------------------------------------------------------------------
# 14. PipelineOrchestrator — strategy.validate_value returning False drops value
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_uses_strategy_validate_value():
    """Mock strategy that rejects everything; pipeline should return empty list."""

    class RejectAllStrategy(MarketplaceStrategy):
        @property
        def name(self) -> str:
            return "reject_all"

        def validate_value(self, target, value, context) -> ValidationResult:
            return ValidationResult(is_valid=False, reason="test rejection")

    strategy = RejectAllStrategy()
    desc_values = [_make_value(attr_id=1, value="red")]
    orch = _build_pipeline(strategy=strategy, desc_values=desc_values)

    ctx = _make_ctx()
    result = await orch.enrich(ctx, [_make_target(1)])
    # All values should be dropped by validation
    assert result == []


# ---------------------------------------------------------------------------
# 15. OzonStrategy — normalize_target_with_context использует словарь v2
# ---------------------------------------------------------------------------

# Тестовый словарь в формате v2 (compound keys)
_OZON_TEST_DICT_V2 = {
    "schema_version": 2,
    "source": "ozon_seller_api",
    "generated_at": "2026-05-20",
    "categories": {
        "100:200": {
            "description_category_id": 100,
            "type_id": 200,
            "name": "Тестовая категория",
            "path": ["Тест"],
            "characteristics": [
                {"id": 9048, "name": "Бренд", "type": "String",
                 "is_required": True, "is_collection": False, "description": "Бренд товара"},
                {"id": 4180, "name": "Цвет товара", "type": "Option",
                 "is_required": False, "is_collection": False, "description": "Цвет"},
                {"id": 7777, "name": "Объём, мл", "type": "Integer",
                 "is_required": False, "is_collection": False, "description": "Объём"},
            ],
        }
    },
}


def _reset_ozon_loader_cache() -> None:
    from app.services.enrichment.strategies.dictionaries.ozon_loader import load_ozon_dictionary
    load_ozon_dictionary.cache_clear()


def test_ozon_normalize_target_with_context_enriches_metadata(tmp_path):
    """normalize_target_with_context подставляет name и type из словаря v2."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_TEST_DICT_V2), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        _reset_ozon_loader_cache()
        strategy = OzonStrategy()
        # ExtractionContext без ozon_type_id — fallback на category_id
        ctx = ExtractionContext(product_id=1, product_name="Тест", category_id=100)
        # target с id=4180 (Цвет товара, type=Option → enum)
        target = TargetAttribute(id=4180, name="Старое название", type="text")
        result = strategy.normalize_target_with_context(target, ctx)

    assert result.id == 4180
    assert result.name == "Цвет товара"
    assert result.type == "enum"  # Option → enum через _OZON_TYPE_MAP
    assert result.description == "Цвет"


def test_ozon_normalize_target_with_context_unknown_id_passthrough(tmp_path):
    """normalize_target_with_context не меняет target если id отсутствует в словаре."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_TEST_DICT_V2), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        _reset_ozon_loader_cache()
        strategy = OzonStrategy()
        ctx = ExtractionContext(product_id=1, product_name="Тест", category_id=100)
        target = TargetAttribute(id=99999, name="Неизвестный атрибут", type="text")
        result = strategy.normalize_target_with_context(target, ctx)

    # Без изменений — id нет в словаре
    assert result.id == 99999
    assert result.name == "Неизвестный атрибут"
    assert result.type == "text"


# ---------------------------------------------------------------------------
# 16. OzonStrategy — filter_by_dictionary оставляет только словарные атрибуты
# ---------------------------------------------------------------------------


def test_ozon_filter_by_dictionary_keeps_only_known_attributes(tmp_path):
    """filter_by_dictionary оставляет targets чьи id есть в словаре категории."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_TEST_DICT_V2), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        _reset_ozon_loader_cache()
        strategy = OzonStrategy()
        ctx = ExtractionContext(product_id=1, product_name="Тест", category_id=100)
        targets = [
            TargetAttribute(id=9048, name="Бренд", type="text"),    # есть в словаре
            TargetAttribute(id=4180, name="Цвет", type="text"),     # есть в словаре
            TargetAttribute(id=55555, name="Чужой атрибут", type="text"),  # нет
        ]
        result = strategy.filter_by_dictionary(targets, ctx)

    kept_ids = [t.id for t in result]
    assert 9048 in kept_ids
    assert 4180 in kept_ids
    assert 55555 not in kept_ids


def test_ozon_filter_by_dictionary_graceful_on_missing_category(tmp_path):
    """filter_by_dictionary возвращает все targets если категория не найдена в словаре."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_TEST_DICT_V2), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        _reset_ozon_loader_cache()
        strategy = OzonStrategy()
        # category_id=999 нет в словаре
        ctx = ExtractionContext(product_id=1, product_name="Тест", category_id=999)
        targets = [
            TargetAttribute(id=1, name="А", type="text"),
            TargetAttribute(id=2, name="Б", type="text"),
        ]
        result = strategy.filter_by_dictionary(targets, ctx)

    # Graceful degradation: возвращаем все
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Task A: is_collection propagation
# ---------------------------------------------------------------------------

_OZON_TEST_DICT_WITH_COLLECTION = {
    "schema_version": 2,
    "source": "ozon_seller_api",
    "generated_at": "2026-05-20",
    "categories": {
        "100:200": {
            "description_category_id": 100,
            "type_id": 200,
            "name": "Тест",
            "path": ["Тест"],
            "characteristics": [
                {"id": 1001, "name": "Материал", "type": "String",
                 "is_required": False, "is_collection": True, "description": "Список материалов"},
                {"id": 1002, "name": "Бренд", "type": "String",
                 "is_required": True, "is_collection": False, "description": "Бренд"},
                {"id": 1003, "name": "Цвет", "type": "Option",
                 "is_required": False, "is_collection": False,
                 "description": "Цвет",
                 "values": [
                     {"id": 501, "value": "Красный"},
                     {"id": 502, "value": "Синий"},
                     {"id": 503, "value": "Зелёный"},
                 ]},
                {"id": 1004, "name": "Теги", "type": "String",
                 "is_required": False, "is_collection": True,
                 "description": "Теги",
                 "values": [
                     {"id": 601, "value": "Хит"},
                     {"id": 602, "value": "Новинка"},
                 ]},
            ],
        }
    },
}


def test_is_collection_propagates_to_target(tmp_path):
    """normalize_target_with_context переносит is_collection=True из словаря."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_TEST_DICT_WITH_COLLECTION), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        _reset_ozon_loader_cache()
        strategy = OzonStrategy()
        ctx = ExtractionContext(product_id=1, product_name="Тест", category_id=100, ozon_type_id=200)
        target_collection = TargetAttribute(id=1001, name="Материал", type="text")
        target_scalar = TargetAttribute(id=1002, name="Бренд", type="text")
        result_collection = strategy.normalize_target_with_context(target_collection, ctx)
        result_scalar = strategy.normalize_target_with_context(target_scalar, ctx)

    assert result_collection.is_collection is True
    assert result_scalar.is_collection is False


def test_attribute_value_accepts_scalar():
    """AttributeValue принимает скалярное value без ошибок."""
    av = AttributeValue(attribute_id=1, value="красный", confidence=0.9, source=Source.DESCRIPTION)
    assert av.value == "красный"
    assert av.is_collection is False


def test_attribute_value_accepts_list():
    """AttributeValue принимает list value без ошибок."""
    av = AttributeValue(
        attribute_id=1, value=["хлопок", "полиэстер"], confidence=0.9,
        source=Source.DESCRIPTION, is_collection=True,
    )
    assert av.value == ["хлопок", "полиэстер"]
    assert av.is_collection is True


def test_attribute_value_accepts_mixed_list():
    """AttributeValue принимает list смешанных скалярных типов."""
    av = AttributeValue(
        attribute_id=2, value=["A", 1, True], confidence=0.8,
        source=Source.LLM_KNOWLEDGE, is_collection=True,
    )
    assert len(av.value) == 3


# ---------------------------------------------------------------------------
# Task B: resolve_value_id helper and resolve_value_ids method
# ---------------------------------------------------------------------------

def test_resolve_value_id_finds_matching_id(tmp_path):
    """resolve_value_id возвращает id для known value (case-insensitive); matcher отключён."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_TEST_DICT_WITH_COLLECTION), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ), patch(
        # Disable matcher so this test covers only exact/case-insensitive path
        "app.services.enrichment.strategies.dictionaries.ozon_loader._get_matcher",
        return_value=None,
    ):
        _reset_ozon_loader_cache()
        from app.services.enrichment.strategies.dictionaries.ozon_loader import resolve_value_id
        # Прямое совпадение
        assert resolve_value_id(100, 200, 1003, "Красный") == 501
        # Case-insensitive
        assert resolve_value_id(100, 200, 1003, "синий") == 502
        # Нет совпадения (matcher disabled — exact miss)
        assert resolve_value_id(100, 200, 1003, "Фиолетовый") is None
        # Характеристика без values
        assert resolve_value_id(100, 200, 1002, "Apple") is None
        # Неизвестная характеристика
        assert resolve_value_id(100, 200, 9999, "X") is None


def test_resolve_value_ids_scalar(tmp_path):
    """resolve_value_ids привязывает value_id для одиночного значения."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_TEST_DICT_WITH_COLLECTION), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        _reset_ozon_loader_cache()
        strategy = OzonStrategy()
        ctx = ExtractionContext(
            product_id=1, product_name="Тест", category_id=100, ozon_type_id=200
        )
        av = AttributeValue(
            attribute_id=1003, value="Синий", confidence=0.9,
            source=Source.DESCRIPTION, is_collection=False,
        )
        result = strategy.resolve_value_ids(av, ctx)

    assert result.value_id == 502
    assert result.value_ids is None


def test_resolve_value_ids_collection(tmp_path):
    """resolve_value_ids привязывает value_ids для массива значений; matcher отключён."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_TEST_DICT_WITH_COLLECTION), encoding="utf-8"
    )
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ), patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader._get_matcher",
        return_value=None,
    ):
        _reset_ozon_loader_cache()
        strategy = OzonStrategy()
        ctx = ExtractionContext(
            product_id=1, product_name="Тест", category_id=100, ozon_type_id=200
        )
        av = AttributeValue(
            attribute_id=1004, value=["Хит", "Новинка", "Несуществующий"],
            confidence=0.9, source=Source.DESCRIPTION, is_collection=True,
        )
        result = strategy.resolve_value_ids(av, ctx)

    assert result.value_ids == [601, 602]  # "Несуществующий" пропускается
    assert result.value_id is None


def test_resolve_value_ids_no_type_id_passthrough():
    """resolve_value_ids — no-op если ozon_type_id не задан в контексте."""
    strategy = OzonStrategy()
    ctx = ExtractionContext(product_id=1, product_name="X", category_id=100)  # без ozon_type_id
    av = AttributeValue(attribute_id=1003, value="Синий", confidence=0.9, source=Source.DESCRIPTION)
    result = strategy.resolve_value_ids(av, ctx)
    assert result.value_id is None
    assert result.value_ids is None


def test_default_strategy_resolve_value_ids_passthrough():
    """DefaultStrategy.resolve_value_ids — no-op."""
    from app.services.enrichment.strategies.default_strategy import DefaultStrategy
    strategy = DefaultStrategy()
    ctx = ExtractionContext(product_id=1, product_name="X", category_id=1)
    av = AttributeValue(attribute_id=1, value="test", confidence=0.8, source=Source.DESCRIPTION)
    result = strategy.resolve_value_ids(av, ctx)
    assert result is av  # возвращает тот же объект без изменений


# ---------------------------------------------------------------------------
# llm_resolve_tail — graceful skip on LLM timeout / network error
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_resolve_tail_graceful_on_timeout(tmp_path):
    """llm_resolve_tail returns input unchanged when the LLM call raises an exception.

    This covers the production hang scenario: a stalled/timed-out DeepSeek
    response must NOT block the product — the resolver must skip tail resolution
    and leave value_id=None as-is.
    """
    import json
    from unittest.mock import patch, AsyncMock
    from openai import APITimeoutError

    dict_with_options = {
        "schema_version": 2,
        "source": "ozon_seller_api",
        "generated_at": "2026-06-08",
        "categories": {
            "10:20": {
                "description_category_id": 10,
                "type_id": 20,
                "name": "Тест",
                "path": ["Тест"],
                "characteristics": [
                    {
                        "id": 55,
                        "name": "Сезон",
                        "type": "Option",
                        "is_required": False,
                        "is_collection": False,
                        "description": "Сезон носки",
                        "values": [
                            {"id": 101, "value": "Лето"},
                            {"id": 102, "value": "Зима"},
                            {"id": 103, "value": "На любой сезон"},
                        ],
                    }
                ],
            }
        },
    }
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(dict_with_options), encoding="utf-8"
    )

    # Clear module-level LLM resolve cache to avoid cross-test pollution.
    import app.services.enrichment.strategies.ozon_strategy as _ozon_mod
    _ozon_mod._LLM_RESOLVE_CACHE.clear()

    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        from app.services.enrichment.strategies.dictionaries.ozon_loader import (
            load_ozon_dictionary,
        )
        load_ozon_dictionary.cache_clear()

        strategy = OzonStrategy()
        ctx = ExtractionContext(
            product_id=1, product_name="Тест", category_id=10, ozon_type_id=20
        )
        # Unmatched value (value_id still None) — tail resolver should try it.
        av = AttributeValue(
            attribute_id=55,
            value="Круглогодичный",
            confidence=0.8,
            source=Source.DESCRIPTION,
            is_collection=False,
        )
        av.value_id = None  # not yet resolved

        targets = [
            TargetAttribute(id=55, name="Сезон", type="enum"),
        ]

        # Simulate LLM timeout raised from _llm_batch_choose → get_main_manager().structured_request
        mock_llm = AsyncMock()
        mock_llm.structured_request = AsyncMock(
            side_effect=APITimeoutError(request=None)
        )

        with patch(
            "app.services.providers.factory.get_main_manager",
            return_value=mock_llm,
        ):
            result = await strategy.llm_resolve_tail([av], targets, ctx)

    # Product must complete — value returned unchanged, value_id stays None.
    assert len(result) == 1
    assert result[0].value_id is None
    assert result[0].value == "Круглогодичный"
