"""VisionSource — извлекает visual characteristics с фото товара.

2 LLM calls:
1. VisionProducer (Gemini 2.5 Flash) → текстовое описание видимого на фото
2. Extraction LLM (DeepSeek) → AttributeValue list из этого текста

Применим для visual attributes (цвет, материал по виду, форма). Не применим
для невидимых свойств (вес, состав, мощность).

Spec: docs/architecture/pipeline.md, section "Stage 3 / VisionSource".
"""
from typing import Optional
from pydantic import BaseModel, Field, AliasChoices
from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager
from app.services.enrichment.vision_producer import VisionProducer
from app.services.enrichment.judges.vision_judge import VisionJudge


# Semantic types которые можно извлечь визуально.
VISUAL_SEMANTIC_TYPES = {
    "color", "material_visual", "shape", "form_factor",
    "visible_size", "visible_label", "pattern", "texture",
}


class _VisionExtractedAttr(BaseModel):
    model_config = {"populate_by_name": True}
    attribute_id: int = Field(..., validation_alias=AliasChoices("attribute_id", "id"))
    value: str | int | float | bool | list[str | int | float | bool] = Field(..., validation_alias=AliasChoices("value", "attribute_value", "extracted_value"))
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: Optional[str] = Field(None, description="что на фото подтверждает")


class _VisionExtractionResponse(BaseModel):
    extracted: list[_VisionExtractedAttr]


class VisionSource(AttributeSource):
    def __init__(
        self,
        vision_producer: Optional[VisionProducer] = None,
        extraction_manager: Optional[StructuredLlmManager] = None,
    ):
        self._vision = vision_producer or VisionProducer()
        self._extractor = extraction_manager or get_main_manager()
        self._judge = VisionJudge()
        # Cache vision text per (context.product_id) чтобы не делать vision call 2 раза для одного товара
        self._vision_cache: dict[int, Optional[str]] = {}

    @property
    def source_type(self) -> Source:
        return Source.VISION

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим если есть image_urls И целевой attribute визуально определяем."""
        if not context.image_urls:
            return False
        # Filter by semantic_type если задан
        if target.semantic_type and target.semantic_type not in VISUAL_SEMANTIC_TYPES:
            return False
        return True

    async def extract(self, context: ExtractionContext, targets: list[TargetAttribute]) -> list[AttributeValue]:
        if not context.image_urls or not targets:
            return []

        # Step 1: vision call (cached per product_id)
        if context.product_id not in self._vision_cache:
            vision_text = await self._vision.produce_description(
                image_urls=context.image_urls,
                product_name=context.product_name,
            )
            self._vision_cache[context.product_id] = vision_text
            context.llm_calls_so_far += 1
        else:
            vision_text = self._vision_cache[context.product_id]

        if not vision_text:
            return []

        # Step 2: extraction from vision text
        targets_block = "\n".join([
            f"- id={t.id}, name={t.name!r}, type={t.type}"
            + (f", allowed={t.allowed_values}" if t.allowed_values else "")
            + (", is_collection=true" if t.is_collection else "")
            for t in targets
        ])

        system_prompt = (
            "You extract visual attributes from a description of what's visible on product photos. "
            "Only include attributes that are CLEARLY visible. If unsure, skip. "
            "When an attribute has 'allowed' values listed, you MUST choose your answer from that list "
            "(use the closest matching option). Do not invent values outside the allowed list. "
            "Evidence should quote the relevant phrase from the vision description. "
            "If the target has is_collection=true, return a JSON array of values; otherwise a single scalar."
        )
        user_text = (
            f"Vision description (from product photos):\n{vision_text}\n\n"
            f"Target attributes (visual):\n{targets_block}\n\n"
            f"Return JSON with 'extracted' list of {{attribute_id, value, confidence, evidence}}."
        )

        parsed, tokens = await self._extractor.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=_VisionExtractionResponse,
        )
        if parsed is None:
            return []
        context.llm_calls_so_far += 1

        target_by_id = {t.id: t for t in targets}
        return [
            AttributeValue(
                attribute_id=a.attribute_id,
                value=a.value,
                confidence=a.confidence,
                source=Source.VISION,
                evidence=a.evidence,
                semantic_type=target_by_id[a.attribute_id].semantic_type
                              if a.attribute_id in target_by_id else None,
                is_collection=target_by_id[a.attribute_id].is_collection
                              if a.attribute_id in target_by_id else False,
            )
            for a in parsed.extracted
        ]

    def get_judge(self) -> LlmJudge:
        return self._judge
