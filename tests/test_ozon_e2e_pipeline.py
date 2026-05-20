"""E2E integration test: pipeline → build_ozon_import_payload, no live API calls."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

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
from app.services.enrichment.marketplaces.ozon_payload import (
    OzonDimensions,
    OzonProductInput,
    OzonImportPayload,
    build_ozon_import_payload,
)

# Минимальный тестовый словарь — реальная пара (cat_id=500, type_id=700)
_TEST_DICT = {
    "schema_version": 2,
    "source": "ozon_seller_api",
    "generated_at": "2026-05-20",
    "categories": {
        "500:700": {
            "description_category_id": 500,
            "type_id": 700,
            "name": "Одежда",
            "path": ["Одежда"],
            "characteristics": [
                {
                    "id": 9048,
                    "name": "Бренд",
                    "type": "String",
                    "is_required": True,
                    "is_collection": False,
                    "description": "Бренд товара",
                    "values": [
                        {"id": 971042156, "value": "Adidas"},
                        {"id": 971042157, "value": "Nike"},
                    ],
                },
                {
                    "id": 4180,
                    "name": "Цвет товара",
                    "type": "Option",
                    "is_required": False,
                    "is_collection": True,
                    "description": "Цвет",
                    "values": [
                        {"id": 61576, "value": "Синий"},
                        {"id": 61577, "value": "Красный"},
                    ],
                },
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
async def test_e2e_pipeline_to_ozon_payload(tmp_path):
    """Pipeline enrich → build_ozon_import_payload → valid payload с dictionary_value_id."""
    (tmp_path / "ozon_dictionary.json").write_text(
        json.dumps(_TEST_DICT), encoding="utf-8"
    )

    from unittest.mock import patch
    with patch(
        "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
        tmp_path,
    ):
        _reset_ozon_loader_cache()

        strategy = OzonStrategy()

        # LLM source возвращает Бренд=Adidas и Цвет=[Синий, Красный]
        desc_src = _mock_source(
            Source.DESCRIPTION,
            [
                AttributeValue(
                    attribute_id=9048,
                    value="Adidas",
                    confidence=0.95,
                    source=Source.DESCRIPTION,
                    is_collection=False,
                ),
                AttributeValue(
                    attribute_id=4180,
                    value=["Синий", "Красный"],
                    confidence=0.90,
                    source=Source.DESCRIPTION,
                    is_collection=True,
                ),
            ],
        )
        know_src = _mock_source(Source.LLM_KNOWLEDGE)
        vis_src = _mock_source(Source.VISION)
        web_src = _mock_source(Source.WEB_SEARCH)

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
            product_name="Кроссовки Adidas синие",
            product_description="Спортивные кроссовки для бега.",
            category_id=500,
            marketplace="ozon",
            ozon_type_id=700,
        )

        targets = [
            TargetAttribute(id=9048, name="Бренд", type="text"),
            TargetAttribute(id=4180, name="Цвет", type="text"),
        ]

        result_attrs = await orch.enrich(ctx, targets)

        # Проверяем что pipeline вернул что-то
        assert len(result_attrs) >= 1

        # Собираем payload
        product = OzonProductInput(
            offer_id="OZON-SKU-001",
            name="Кроссовки Adidas синие",
            price="3500",
            vat="0.20",
            currency_code="RUB",
            images=["https://cdn.example.com/shoe1.jpg"],
            weight=600,
            weight_unit="g",
            dimensions=OzonDimensions(depth=300, width=150, height=100, dimension_unit="mm"),
        )

        payload = build_ozon_import_payload(product, 500, 700, result_attrs)

        # --- Утверждения ---
        assert isinstance(payload, OzonImportPayload)

        item = payload.to_api_dict()["items"][0]

        # offer_id присутствует
        assert item["offer_id"] == "OZON-SKU-001"

        # Структура валидна по Pydantic
        reparsed = OzonImportPayload.model_validate({"items": [item]})
        assert reparsed.items[0].offer_id == "OZON-SKU-001"

        # Хотя бы один атрибут с dictionary_value_id
        all_values = [
            v
            for a in item["attributes"]
            for v in a["values"]
        ]
        has_dict_id = any("dictionary_value_id" in v for v in all_values)
        assert has_dict_id, "Ожидался хотя бы один dictionary_value_id от resolve_value_ids"

        # to_api_dict() JSON-serializable
        json_str = json.dumps(item)
        assert json.loads(json_str)["offer_id"] == "OZON-SKU-001"
