"""End-to-end live test для нового PipelineOrchestrator.

Использует реальные API (DeepSeek + Gemini + Serper) на одном тестовом товаре.
Cost ~$0.02-0.10. Запускается только при RUN_LIVE_TESTS=1.
"""
import os
import pytest
from tests.integration.conftest import skip_unless_live, MAX_COST_PER_TEST

from app.services.enrichment.base import (
    Source, ExtractionContext, TargetAttribute,
)
from app.services.enrichment.pipeline import PipelineOrchestrator


@skip_unless_live
@pytest.mark.asyncio
async def test_pipeline_e2e_iphone_real_apis():
    """Реальный товар → real DeepSeek/Gemini/Serper → assert flow worked."""
    orch = PipelineOrchestrator()

    context = ExtractionContext(
        product_id=999,
        product_name="Apple iPhone 15 Pro 256GB Titanium",
        product_description="Смартфон Apple iPhone 15 Pro 256GB. Цвет: Titanium Black. Новый, в упаковке.",
        category_id=1,
        category_path=["Электроника", "Смартфоны"],
        brand="Apple",
        image_urls=[],  # пусто, vision-stage будет skip
        source_urls=[],
        max_cost_usd=0.20,  # cap чтобы не разорить
    )

    targets = [
        TargetAttribute(id=1, name="Бренд", type="text"),
        TargetAttribute(id=2, name="Модель", type="text"),
        TargetAttribute(id=3, name="Цвет", type="text", semantic_type="color"),
        TargetAttribute(id=4, name="Объём памяти", type="text"),
        TargetAttribute(id=5, name="Вес", type="text", semantic_type="weight"),
    ]

    result = await orch.enrich(context, targets)

    # Assertions — лояльные потому что LLM может варьировать
    assert len(result) >= 2, f"Expected at least 2 attributes filled, got {len(result)}: {[v.attribute_id for v in result]}"

    # Хотя бы один attribute должен быть из DESCRIPTION (легко из описания)
    desc_values = [v for v in result if v.source == Source.DESCRIPTION]
    assert len(desc_values) >= 1, f"Expected ≥1 DESCRIPTION-sourced value, got: {[(v.attribute_id, v.source) for v in result]}"

    # Brand "Apple" должно найтись (это очень очевидно)
    brand_values = [v for v in result if v.attribute_id == 1]
    if brand_values:
        assert "apple" in str(brand_values[0].value).lower()

    # Cost cap соблюден
    # (Note: context.cost_so_far_usd может быть 0 если cost tracking ещё не реализован — это TODO в spec)
    assert context.llm_calls_so_far >= 1, "Expected ≥1 LLM call"
    assert context.llm_calls_so_far <= 20, f"Too many LLM calls: {context.llm_calls_so_far}"

    print(f"\n[E2E v2] Filled: {len(result)} attrs, LLM calls: {context.llm_calls_so_far}")
    for v in result:
        print(f"  [{v.source.value}] id={v.attribute_id} = {v.value!r} (conf={v.confidence:.2f}, judge={v.judge_validated})")


@skip_unless_live
@pytest.mark.asyncio
async def test_pipeline_e2e_with_image_vision_stage():
    """Товар с реальным image_url → vision stage активируется."""
    orch = PipelineOrchestrator()

    # Используем picsum для теста vision (стабильный image CDN)
    context = ExtractionContext(
        product_id=998,
        product_name="Тестовая красная футболка",
        product_description="Хлопковая футболка, размер M.",
        category_id=2,
        category_path=["Одежда", "Футболки"],
        image_urls=["https://picsum.photos/seed/redshirt/400/400.jpg"],
        max_cost_usd=0.20,
    )

    targets = [
        TargetAttribute(id=10, name="Цвет", type="text", semantic_type="color"),
        TargetAttribute(id=11, name="Материал", type="text", semantic_type="material_visual"),
        TargetAttribute(id=12, name="Размер", type="text"),
    ]

    result = await orch.enrich(context, targets)

    # Хотя бы что-то нашли
    assert len(result) >= 1, f"Expected ≥1 attribute, got {len(result)}"

    print(f"\n[E2E vision] Filled {len(result)} attrs, LLM calls: {context.llm_calls_so_far}")
    for v in result:
        print(f"  [{v.source.value}] id={v.attribute_id} = {v.value!r}")


@skip_unless_live
@pytest.mark.asyncio
async def test_pipeline_e2e_no_description_forces_other_stages():
    """Товар без description → DescriptionSource не должен возвращать,
    Classifier + Knowledge/WebSearch должны попытаться."""
    orch = PipelineOrchestrator()

    context = ExtractionContext(
        product_id=997,
        product_name="Sony WH-1000XM5",  # известный товар
        product_description="",  # ПУСТО
        category_id=3,
        category_path=["Электроника", "Наушники"],
        brand="Sony",
        max_cost_usd=0.20,
    )

    targets = [
        TargetAttribute(id=20, name="Бренд", type="text"),
        TargetAttribute(id=21, name="Тип", type="text"),
    ]

    result = await orch.enrich(context, targets)

    # Description вернёт пусто (is_applicable=False)
    desc_values = [v for v in result if v.source == Source.DESCRIPTION]
    assert len(desc_values) == 0, "Description should be skipped when description is empty"

    # Должны быть values из других sources (knowledge/web_search)
    other_values = [v for v in result if v.source != Source.DESCRIPTION]
    print(f"\n[E2E no-desc] Got {len(other_values)} from non-description sources: {[v.source.value for v in other_values]}")
    # Lenient — пусть хоть что-то найдётся через other sources
