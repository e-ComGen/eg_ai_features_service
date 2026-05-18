"""LlmClassifier — routing intelligence для cost-aware pipeline.

После DescriptionSource (всегда первый), если остались ненайденные атрибуты,
Classifier решает где их искать. 1 LLM call возвращает решение для всех
ненайденных attrs сразу.

Spec: docs/architecture/pipeline.md, section "Stage 1 / LlmClassifier".
"""
from typing import Optional
from pydantic import BaseModel, Field, AliasChoices
from app.services.enrichment.base import (
    Source, TargetAttribute, ExtractionContext,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager


class ClassifierDecision(BaseModel):
    """Решение Classifier для одной характеристики."""
    attribute_id: int
    suggested_sources: list[Source] = Field(
        ..., min_length=1, max_length=3,
        description="Источники по приоритету (cheapest first). Может быть пустой для give-up.",
        validation_alias=AliasChoices("suggested_sources", "sources"),
    )
    reasoning: str = Field(default="", max_length=200)

    model_config = {"populate_by_name": True}


class _ClassifierResponse(BaseModel):
    """Top-level LLM response — для всех unfilled attrs."""
    decisions: list[ClassifierDecision]


class LlmClassifier:
    """1 LLM call: какие источники использовать для каждой ненайденной характеристики."""

    def __init__(self, llm_manager: Optional[StructuredLlmManager] = None):
        self._llm = llm_manager or get_main_manager()

    async def classify(
        self,
        context: ExtractionContext,
        unfilled_attributes: list[TargetAttribute],
    ) -> dict[int, list[Source]]:
        """Returns: {attribute_id: ordered list of sources to try}."""
        if not unfilled_attributes:
            return {}

        attrs_block = "\n".join([
            f"- id={a.id}, name={a.name!r}, type={a.type}" +
            (f", semantic_type={a.semantic_type!r}" if a.semantic_type else "")
            for a in unfilled_attributes
        ])

        system_prompt = (
            "You are a routing classifier for a product attribute extraction pipeline.\n"
            "For each unfilled attribute, choose 1-3 sources to try, ordered cheapest-first.\n\n"
            "Available sources:\n"
            "- 'llm_knowledge': cheapest, good for well-known products (Apple, Nike, etc) and standard specs\n"
            "- 'vision': for visually-determinable attributes (color, material appearance, shape, visible labels)\n"
            "- 'web_search': most expensive, best for precise specs (weight, dimensions) of known products\n\n"
            "Rules (apply in order — first matching rule wins):\n"
            "1. Strictly visual attributes (color, material appearance, shape, packaging design): ['vision'].\n"
            "2. NEVER assign numeric/measurable attributes (weight, volume, calories, proteins, fats, "
            "carbohydrates, vitamins, shelf life, dimensions, battery capacity) to 'vision' alone.\n"
            "3. Product composition/ingredient facts (flavor, taste, ingredients, sugar content, "
            "allergens, fat percentage): ['llm_knowledge'] — these are product definition facts.\n"
            "4. Nutrition/food numeric facts (proteins g, fats g, carbohydrates g, calories kcal, shelf life): "
            "['llm_knowledge', 'web_search'] — always include web_search as fallback.\n"
            "5. Brand/manufacturer/country/article/SKU of well-known products: ['llm_knowledge'].\n"
            "6. Precise numeric product specs (exact weight, dimensions, battery) of well-known products: "
            "['llm_knowledge', 'web_search'].\n"
            "7. Precise specs of obscure/niche products: ['web_search'].\n"
            "8. Nothing fits: [] (give up).\n"
            "Return short reasoning (max 200 chars)."
        )
        user_text = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n"
            f"Category: {' / '.join(context.category_path) or 'n/a'}\n"
            f"Has photos: {bool(context.image_urls)}\n\n"
            f"Unfilled attributes:\n{attrs_block}\n\n"
            f"Return 'decisions' list."
        )

        parsed, _ = await self._llm.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=_ClassifierResponse,
        )
        if parsed is None:
            # Fallback: try description+knowledge for everything
            return {a.id: [Source.LLM_KNOWLEDGE] for a in unfilled_attributes}

        context.llm_calls_so_far += 1
        result = {d.attribute_id: d.suggested_sources for d in parsed.decisions}

        # Safety net: numeric attrs routed to llm_knowledge-only get web_search appended.
        # LLM classifiers tend to over-trust knowledge for numeric food/product specs;
        # web_search is cheap insurance for any numeric that knowledge misses.
        numeric_ids = {a.id for a in unfilled_attributes if a.type == "numeric"}
        for attr_id in numeric_ids:
            sources = result.get(attr_id, [])
            if sources and Source.WEB_SEARCH not in sources:
                result[attr_id] = sources + [Source.WEB_SEARCH]

        return result
