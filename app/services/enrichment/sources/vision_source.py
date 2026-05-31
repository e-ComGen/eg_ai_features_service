"""VisionSource — извлекает visual characteristics с фото товара.

2 LLM calls:
1. VisionProducer (Gemini 2.5 Flash) → текстовое описание видимого на фото
2. Extraction LLM (DeepSeek) → AttributeValue list + identifiers (MPN/EAN/article) из этого текста

Применим для visual attributes (цвет, материал по виду, форма). Не применим
для невидимых свойств (вес, состав, мощность).

Identifier enrichment: если на фото виден MPN/EAN/article на коробке/наклейке —
извлекаем и обогащаем ExtractionContext, чтобы downstream sources (IceCat / PDF /
WebSearch) могли использовать точный код вместо угадывания через LLM.

Spec: docs/architecture/pipeline.md, section "Stage 3 / VisionSource".
"""
import logging
from typing import Optional
from pydantic import BaseModel, Field, AliasChoices, model_validator
from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)

logger = logging.getLogger(__name__)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager
from app.services.enrichment.vision_producer import VisionProducer
from app.services.enrichment.judges.vision_judge import VisionJudge
from app.services.enrichment.prompt_router import (
    format_target_line, build_meta_guidance,
    build_already_filled_block, filter_already_filled_targets,
)
from app.services.enrichment.strategies.base import MarketplaceStrategy
from app.services.enrichment.strategies.default_strategy import DefaultStrategy


# Semantic types которые можно извлечь визуально — ОРИГИНАЛЬНЫЙ v8 whitelist.
# История:
#   v8 (8 типов): coverage required 88.8%, Vision 4 fills.
#   v10b (18 типов): Vision дал 67 fills но они НЕТОЧНЫЕ (видит фото чужой
#     brand_line модели от OzonCard) → coverage drop до 81.2%.
#   v12 (7 типов + indicator/lighting): Vision 20 fills, coverage required 87.5%
#     (-1.3 pp от v8). Vision просто перетирал более точные fills других sources.
#   Возвращаем к v8: Vision только на color/material/shape/form_factor/...
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


class _VisionIdentifiers(BaseModel):
    """Идентификаторы прочитанные с этикетки/коробки на фото.

    Используются для обогащения ExtractionContext: даёт downstream sources
    (IceCat, PDF datasheet, WebSearch) точный код вместо guess через LLM.
    Все поля опциональны — заполняются только если ЯВНО видны на фото.
    """
    mpn: Optional[str] = Field(None, description="Manufacturer Part Number с наклейки/коробки")
    ean: Optional[str] = Field(None, description="EAN/UPC barcode digits с упаковки")
    article: Optional[str] = Field(None, description="Артикул производителя")


class _VisionExtractionResponse(BaseModel):
    extracted: list[_VisionExtractedAttr]
    identifiers: Optional[_VisionIdentifiers] = Field(
        None,
        description="Идентификаторы прочитанные с фото для обогащения context (MPN/EAN/article)"
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_null_values(cls, data):
        if isinstance(data, dict) and isinstance(data.get("extracted"), list):
            data["extracted"] = [
                e for e in data["extracted"]
                if isinstance(e, dict) and e.get("value") is not None
            ]
        return data


class VisionSource(AttributeSource):
    def __init__(
        self,
        vision_producer: Optional[VisionProducer] = None,
        extraction_manager: Optional[StructuredLlmManager] = None,
        strategy: Optional[MarketplaceStrategy] = None,
    ):
        self._vision = vision_producer or VisionProducer()
        self._extractor = extraction_manager or get_main_manager()
        self._judge = VisionJudge()
        self._strategy: MarketplaceStrategy = strategy or DefaultStrategy()
        # Cache vision text per (context.product_id) чтобы не делать vision call 2 раза для одного товара
        self._vision_cache: dict[int, Optional[str]] = {}

    @property
    def source_type(self) -> Source:
        return Source.VISION

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим если есть image_urls И целевой attribute не numeric.

        Semantic types это hint а не whitelist: пробуем все non-numeric attrs
        когда есть фото, judge отфильтрует мусор. Это даёт vision доступ к
        package labels (80 PLUS, бренд, страна, артикул, гарантия).
        """
        if not context.image_urls:
            return False
        # Numeric targets — vision plохо измеряет числа без референса.
        if target.type == "numeric":
            return False
        # Semantic type — hint: если задан и явно визуальный, ok; если задан
        # но не визуальный, всё равно пробуем (judge решит).
        return True

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: list[AttributeValue] | None = None,
    ) -> list[AttributeValue]:
        if not context.image_urls or not targets:
            return []

        # Убираем уже заполненные attrs из targets чтобы не тратить токены
        effective_targets = filter_already_filled_targets(targets, already_filled or [])
        if not effective_targets:
            return []

        # Регистрируем allowed_values enum-targets у judge, чтобы он мог
        # fast-accept enum values matching allowed list verbatim (skip strict LLM judging).
        enum_allowed = {
            t.id: t.allowed_values
            for t in effective_targets
            if t.allowed_values
        }
        if enum_allowed:
            self._judge.register_allowed_values(enum_allowed)

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

        # Step 2: extraction from vision text с type-aware подсказками
        targets_block = "\n".join([format_target_line(t) for t in effective_targets])
        already_preamble, already_rule = build_already_filled_block(already_filled or [])

        system_prompt = (
            "You extract visual attributes from a description of what's visible on product photos. "
            "Only include attributes that are CLEARLY visible. If unsure, skip. "
            "Evidence should quote the relevant phrase from the vision description. "
            "If the target has is_collection=true, return a JSON array of values; otherwise a single scalar.\n\n"
            "FOR ENUM TARGETS (with allowed= list): output MUST be one of the listed allowed values "
            "VERBATIM (no paraphrasing, no translation, no case changes). If the vision description "
            "uses different wording (e.g. 'Фронтальный' when allowed=['ARGB','RGB','Отсутствует',"
            "'Одноцветная']) — map it to the closest allowed value if the mapping is unambiguous; "
            "otherwise omit the attribute. Do not guess.\n\n"
            "IDENTIFIERS: if the vision description explicitly mentions a Manufacturer Part Number "
            "(MPN, format like MPE-7501-AFAAG / R-PK650D-FA0B-EU / 90YE00A4-B0NA00), an EAN/UPC "
            "barcode (12-13 digits), or an article number (артикул) read from a product label / "
            "sticker / box, populate the top-level 'identifiers' object: "
            "{mpn: '...', ean: '...', article: '...'}. Only include identifiers clearly transcribed "
            "from the photo (mentioned in the vision description). Otherwise omit them or set to null."
            + build_meta_guidance()
            + already_rule
        )
        user_text = (
            f"Vision description (from product photos):\n{vision_text}\n\n"
            + already_preamble
            + f"Target attributes (visual):\n{targets_block}\n\n"
            "Return JSON with two fields: "
            "'extracted' — list of {attribute_id, value, confidence, evidence}; "
            "'identifiers' — optional object {mpn, ean, article} with codes read from photo (may be null)."
        )

        response_model = self._strategy.build_response_model(_VisionExtractionResponse, targets)
        parsed, tokens = await self._extractor.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=response_model,
        )
        if parsed is None:
            return []
        context.llm_calls_so_far += 1

        # Обогащаем ExtractionContext идентификаторами прочитанными с фото.
        # Не перезаписываем уже заданные значения (вышестоящие sources имеют приоритет).
        identifiers = getattr(parsed, "identifiers", None)
        if identifiers is not None:
            enriched: list[str] = []
            mpn_val = (identifiers.mpn or "").strip() if identifiers.mpn else ""
            if mpn_val and not context.mpn:
                context.mpn = mpn_val
                enriched.append(f"mpn={mpn_val}")
            ean_val = (identifiers.ean or "").strip() if identifiers.ean else ""
            if ean_val and not context.ean:
                context.ean = ean_val
                enriched.append(f"ean={ean_val}")
            article_val = (identifiers.article or "").strip() if identifiers.article else ""
            if article_val and not context.article:
                context.article = article_val
                enriched.append(f"article={article_val}")
            if enriched:
                logger.info("[Vision] context enrich: %s", ", ".join(enriched))

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
