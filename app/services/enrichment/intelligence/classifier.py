"""LlmClassifier — routing intelligence для cost-aware pipeline.

После DescriptionSource (всегда первый), если остались ненайденные атрибуты,
Classifier решает где их искать. 1 LLM call возвращает решение для всех
ненайденных attrs сразу.

Spec: docs/architecture/pipeline.md, section "Stage 1 / LlmClassifier".
"""
from typing import Optional
from pydantic import BaseModel, Field
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
        description="Источники по приоритету (cheapest first). Может быть пустой для give-up."
    )
    reasoning: str = Field(max_length=200)


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
            "Rules:\n"
            "- For visual attributes (color, material_visual, shape): start with 'vision'\n"
            "- For brand/model facts: start with 'llm_knowledge'\n"
            "- For precise numeric specs (weight, exact dimensions): start with 'web_search'\n"
            "- If nothing seems to fit: empty list = give up on this attribute\n"
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
        return {d.attribute_id: d.suggested_sources for d in parsed.decisions}
