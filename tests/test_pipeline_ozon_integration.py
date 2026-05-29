"""Pipeline integration test: OzonStrategy filter_by_dictionary + normalize.

Проверяет что targets неизвестные словарю отфильтровываются до вызова sources.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.base import (
    Source,
    AttributeValue,
    AttributeSource,
    LlmJudge,
    TargetAttribute,
    ExtractionContext,
)
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.strategies.ozon_strategy import OzonStrategy


# Минимальный тестовый словарь v2 — только char_id 9048 и 4180 известны
_OZON_TEST_DICT = {
    "schema_version": 2,
    "source": "ozon_seller_api",
    "generated_at": "2026-05-20",
    "categories": {
        "500:700": {
            "description_category_id": 500,
            "type_id": 700,
            "name": "Тест",
            "path": ["Тест"],
            "characteristics": [
                {"id": 9048, "name": "Бренд", "type": "String",
                 "is_required": True, "is_collection": False, "description": "Бренд товара"},
                {"id": 4180, "name": "Цвет товара", "type": "Option",
                 "is_required": False, "is_collection": False, "description": "Цвет"},
            ],
        }
    },
}


def _mock_source(source_type: Source, extract_return=None):
    s = MagicMock(spec=AttributeSource)
    s.source_type = source_type
    s.is_applicable = MagicMock(return_value=True)
    s.extract = AsyncMock(return_value=extract_return or [])
    judge = MagicMock(spec=LlmJudge)
    judge.source = source_type
    judge.validate = AsyncMock(return_value=True)
    s.get_judge = MagicMock(return_value=judge)
    return s


def _reset_ozon_loader_cache():
    from app.services.enrichment.strategies.dictionaries.ozon_loader import load_ozon_dictionary
    load_ozon_dictionary.cache_clear()


@pytest.mark.asyncio
async def test_ozon_dictionary_filter_removes_unknown_targets_before_sources(tmp_path):
    """Targets с id вне словаря должны отфильтровываться до вызова любого source."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_TEST_DICT), encoding="utf-8"
    )

    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        _reset_ozon_loader_cache()

        strategy = OzonStrategy()

        desc_src = _mock_source(Source.DESCRIPTION, [])
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

        # category_id=500, ozon_type_id=700 → словарь найдёт пару 500:700
        ctx = ExtractionContext(
            product_id=42,
            product_name="Тестовый товар",
            category_id=500,
            marketplace="ozon",
            ozon_type_id=700,
        )

        targets = [
            TargetAttribute(id=9048, name="Бренд", type="text"),      # есть в словаре
            TargetAttribute(id=4180, name="Цвет", type="text"),        # есть в словаре
            TargetAttribute(id=99999, name="Неизвестный", type="text"), # нет в словаре
        ]

        await orch.enrich(ctx, targets)

        # source должен был получить только 2 известных target-а (без id=99999) — проверяем первый вызов
        assert desc_src.extract.called
        # call_args_list[0] — первый вызов в основном pipeline (до finishing pass)
        first_call_targets = desc_src.extract.call_args_list[0][0][1]
        called_ids = {t.id for t in first_call_targets}
        assert 99999 not in called_ids, "Неизвестный target не должен попасть в sources"
        assert 9048 in called_ids
        assert 4180 in called_ids


@pytest.mark.asyncio
async def test_ozon_normalize_enriches_target_type_before_sources(tmp_path):
    """normalize_target_with_context должен поменять type='text' → 'enum' для Цвет товара."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_OZON_TEST_DICT), encoding="utf-8"
    )

    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        _reset_ozon_loader_cache()

        strategy = OzonStrategy()

        captured_targets: list[TargetAttribute] = []

        async def _capture_extract(ctx, targets, already_filled=None):
            captured_targets.extend(targets)
            return []

        desc_src = _mock_source(Source.DESCRIPTION, [])
        desc_src.extract = _capture_extract
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

        ctx = ExtractionContext(
            product_id=1,
            product_name="Тест",
            category_id=500,
            marketplace="ozon",
            ozon_type_id=700,
        )

        targets = [
            TargetAttribute(id=4180, name="Цвет", type="text"),  # normalize → enum
        ]

        await orch.enrich(ctx, targets)

        # После normalize type должен стать "enum" (Option → enum)
        color_target = next((t for t in captured_targets if t.id == 4180), None)
        assert color_target is not None
        assert color_target.type == "enum"
        assert color_target.name == "Цвет товара"  # name из словаря
