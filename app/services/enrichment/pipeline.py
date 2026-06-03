"""PipelineOrchestrator — sequential cost-aware extraction.

Главный entry-point для enrichment. Объединяет все sources, judges и intelligence
в один flow с early-exit (если все targets заполнены — останавливаемся) и
cost gating (CostPredictor перед expensive web search).

Spec: docs/architecture/pipeline.md, section "PipelineOrchestrator".
"""
import logging
from typing import Optional

from pydantic import BaseModel, Field

from app.services.enrichment.base import (
    Source,
    SOURCE_PRIORITY,
    AttributeValue,
    TargetAttribute,
    ExtractionContext,
    AttributeSource,
)
from app.services.enrichment.sources import (
    DescriptionSource,
    LlmKnowledgeSource,
    VisionSource,
    WebSearchSource,
    CompetitorRagSource,
    IceCatSource,
    PdfDatasheetSource,
    OzonCardSource,
    WbCardSource,
    UgcSource,
    TnvedSource,
)
from app.services.enrichment.intelligence import LlmClassifier, CostPredictor
from app.services.providers.factory import get_main_manager
from app.services.enrichment.confidence_aware_judge import ConfidenceAwareJudgeWrapper
from app.services.enrichment.strategies.base import MarketplaceStrategy
from app.services.enrichment.strategies.default_strategy import DefaultStrategy
from app.services.enrichment.finishing import FinishingExtractor

logger = logging.getLogger(__name__)

# Card-protection в финальном _merge: карточные источники (копия из live-карточки
# того же товара) не должны перетираться инференсом (LLM-знания / web-поиск), если
# их confidence лишь незначительно ниже. Band = допустимый зазор.
_CARD_SOURCES = {Source.WB_CARD, Source.OZON_CARD}
_INFERENCE_SOURCES = {Source.LLM_KNOWLEDGE, Source.WEB_SEARCH}
_CARD_PROTECTION_BAND = 0.10


def _merge_winner(
    challenger: AttributeValue, incumbent: AttributeValue
) -> AttributeValue:
    """Выбирает победителя для одного attribute_id между двумя кандидатами.

    Спец-правило ТОЛЬКО для пары карточка-vs-инференс: карточный источник
    удерживает атрибут, если card_conf >= other_conf - _CARD_PROTECTION_BAND.
    Для всех прочих пар — обычное правило: highest-conf, tie-break по SOURCE_PRIORITY.
    Поведение симметрично (order-independent).
    """
    # Card-protection band: только карточка-vs-инференс
    card, inference = None, None
    if challenger.source in _CARD_SOURCES and incumbent.source in _INFERENCE_SOURCES:
        card, inference = challenger, incumbent
    elif incumbent.source in _CARD_SOURCES and challenger.source in _INFERENCE_SOURCES:
        card, inference = incumbent, challenger
    if card is not None:
        if card.confidence >= inference.confidence - _CARD_PROTECTION_BAND:
            return card
        return inference

    # Обычное правило для всех прочих пар (без изменений).
    if challenger.confidence > incumbent.confidence:
        return challenger
    if (
        challenger.confidence == incumbent.confidence
        and SOURCE_PRIORITY[challenger.source] > SOURCE_PRIORITY[incumbent.source]
    ):
        return challenger
    return incumbent


def _is_collection_value(v: AttributeValue) -> bool:
    """True если значение надо мерджить как коллекцию (union), а не winner-takes-all."""
    return bool(v.is_collection) or isinstance(v.value, list)


def _norm_elements(value) -> list[str]:
    """Нормализованные (strip+lower) элементы значения.

    Скаляр → один элемент; список → поэлементно. Используется для consensus-подсчёта
    и поэлементного дедупа коллекций (вместо str(list) по всей строке).
    """
    if isinstance(value, list):
        out: list[str] = []
        for el in value:
            s = str(el).strip().lower()
            if s:
                out.append(s)
        return out
    s = str(value).strip().lower()
    return [s] if s else []


def _merge_collection(
    a: AttributeValue, b: AttributeValue
) -> AttributeValue:
    """Объединяет два коллекционных кандидата на один attribute_id.

    UNION дедуплицированных (регистронезависимо) элементов обоих источников.
    Порядок: первое вхождение сохраняется. value_ids объединяются параллельно
    значениям (best-effort: если оба источника несут ids — мерджим, иначе сбрасываем,
    чтобы их корректно дорезолвил resolve_value_ids в _finalize). confidence = max.
    """
    base = a if a.confidence >= b.confidence else b

    merged_values: list = []
    merged_ids: list = []
    seen: set[str] = set()
    have_ids = True  # ids валидны только если ОБА источника дали ids на все элементы

    for src in (a, b):
        vals = src.value if isinstance(src.value, list) else [src.value]
        ids = src.value_ids if isinstance(src.value_ids, list) else None
        if ids is None or len(ids) != len(vals):
            have_ids = False
        for i, el in enumerate(vals):
            norm = str(el).strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged_values.append(el)
            merged_ids.append(ids[i] if (ids is not None and i < len(ids)) else None)

    if not have_ids or any(x is None for x in merged_ids):
        merged_ids = None  # дорезолвит resolve_value_ids в _finalize

    return base.model_copy(update={
        "value": merged_values,
        "value_ids": merged_ids,
        "value_id": None,
        "is_collection": True,
        "confidence": max(a.confidence, b.confidence),
    })


class PipelineOrchestrator:
    """Sequential cost-aware pipeline.

    Order: Description → IceCat → CompetitorRAG (fallback) → Classifier → (per-attr routing) → Knowledge → Vision → CostPredictor → WebSearch
    Early-exit at each stage if all targets filled with high confidence.
    IceCat вставлен сразу после Description: brand-verified спеки без LLM, самый авторитетный источник.
    CompetitorRAG запускается ТОЛЬКО как фолбэк когда IceCat вернул < 5 атрибутов (403/404 бренд).
    Если IceCat закрыл ≥ 5 атрибутов — RAG пропускается (экономия Qdrant I/O и compute).
    """

    def __init__(
        self,
        description_source: Optional[DescriptionSource] = None,
        knowledge_source: Optional[LlmKnowledgeSource] = None,
        vision_source: Optional[VisionSource] = None,
        websearch_source: Optional[WebSearchSource] = None,
        competitor_rag_source: Optional[CompetitorRagSource] = None,
        icecat_source: Optional[IceCatSource] = None,
        pdf_datasheet_source: Optional[PdfDatasheetSource] = None,
        ozon_card_source: Optional[OzonCardSource] = None,
        wb_card_source: Optional[WbCardSource] = None,
        ugc_source: Optional[UgcSource] = None,
        tnved_source: Optional[TnvedSource] = None,
        classifier: Optional[LlmClassifier] = None,
        cost_predictor: Optional[CostPredictor] = None,
        strategy: Optional[MarketplaceStrategy] = None,
    ):
        self._strategy: MarketplaceStrategy = strategy or DefaultStrategy()
        _strat = self._strategy  # передаём стратегию в sources для build_response_model
        self._sources: dict[Source, AttributeSource] = {
            Source.DESCRIPTION: description_source or DescriptionSource(strategy=_strat),
            Source.LLM_KNOWLEDGE: knowledge_source or LlmKnowledgeSource(strategy=_strat),
            Source.VISION: vision_source or VisionSource(strategy=_strat),
            Source.WEB_SEARCH: websearch_source or WebSearchSource(strategy=_strat),
        }
        # CompetitorRagSource хранится отдельно (не в _sources) — у него особый порядок вызова
        # и нет смысла включать в classifier routing (он не LLM-based).
        # Передаём None → source не создаётся автоматически (нет готового индекса по умолчанию).
        self._competitor_rag: Optional[CompetitorRagSource] = competitor_rag_source
        # IceCatSource хранится отдельно: brand-verified, HTTP, без LLM.
        # None → IceCat stage пропускается.
        self._icecat: Optional[IceCatSource] = icecat_source
        # PdfDatasheetSource: datasheet PDF от производителя через Gemini native PDF.
        # None → PDF stage пропускается.
        self._pdf_datasheet: Optional[PdfDatasheetSource] = pdf_datasheet_source
        # OzonCardSource: копия характеристик из живой Ozon-карточки через Scrappey.
        # None → Ozon card stage пропускается.
        self._ozon_card: Optional[OzonCardSource] = ozon_card_source
        # WbCardSource: копия характеристик из WB basket-API (бесплатно, без anti-bot).
        # None → WB card stage пропускается.
        self._wb_card: Optional[WbCardSource] = wb_card_source
        # UgcSource: отзывы и Q&A с Ozon/WB для compat/physical attrs.
        # None → UGC stage пропускается.
        self._ugc: Optional[UgcSource] = ugc_source
        # TnvedSource: per-category резолвер ТН ВЭД ЕАЭС с кэшем.
        # Создаётся ОДИН раз → кэш переживает все товары батча.
        # None → по умолчанию создаём инстанс (всегда нужен для Ozon).
        self._tnved: TnvedSource = tnved_source or TnvedSource()
        self._judges: dict[Source, ConfidenceAwareJudgeWrapper] = {
            src: ConfidenceAwareJudgeWrapper(s.get_judge())
            for src, s in self._sources.items()
        }
        # Judge для CompetitorRag (если source передан)
        if self._competitor_rag is not None:
            self._judges[Source.COMPETITOR_RAG] = ConfidenceAwareJudgeWrapper(
                self._competitor_rag.get_judge()
            )
        # Judge для IceCat (если source передан)
        if self._icecat is not None:
            self._judges[Source.ICECAT] = ConfidenceAwareJudgeWrapper(
                self._icecat.get_judge()
            )
        # Judge для PdfDatasheet (если source передан)
        if self._pdf_datasheet is not None:
            self._judges[Source.PDF_DATASHEET] = ConfidenceAwareJudgeWrapper(
                self._pdf_datasheet.get_judge()
            )
        # Judge для OzonCard (если source передан)
        if self._ozon_card is not None:
            self._judges[Source.OZON_CARD] = ConfidenceAwareJudgeWrapper(
                self._ozon_card.get_judge()
            )
        # Judge для WbCard (если source передан)
        if self._wb_card is not None:
            self._judges[Source.WB_CARD] = ConfidenceAwareJudgeWrapper(
                self._wb_card.get_judge()
            )
        # Judge для UGC (если source передан)
        if self._ugc is not None:
            self._judges[Source.UGC] = ConfidenceAwareJudgeWrapper(
                self._ugc.get_judge()
            )
        self._classifier = classifier or LlmClassifier()
        self._cost_predictor = cost_predictor or CostPredictor()
        self._finisher = FinishingExtractor(sources=list(self._sources.values()))

    async def enrich(
        self, context: ExtractionContext, targets: list[TargetAttribute]
    ) -> list[AttributeValue]:
        """Run full pipeline. Returns merged final attributes (one per attribute_id)."""
        # Шаг 0a: убираем атрибуты которые marketplace не поддерживает
        targets = self._strategy.filter_unsupported_attributes(targets)
        # Шаг 0b: убираем атрибуты неизвестные словарю (Ozon: только словарные char_id)
        targets = self._strategy.filter_by_dictionary(targets, context)
        # Шаг 0c: обогащаем оставшиеся targets метаданными из словаря (name, type, description)
        targets = [
            self._strategy.normalize_target_with_context(t, context)
            for t in targets
        ]

        all_values: list[AttributeValue] = []
        # filled_so_far — накапливаем high-confidence AVs для skip-filled кооперации
        filled_so_far: list[AttributeValue] = []

        # Stage 0: DescriptionSource (always first, cheapest)
        new_avs = await self._run_stage(Source.DESCRIPTION, context, targets)
        all_values += new_avs
        filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
        remaining = self._remaining_targets(targets, all_values)
        if not remaining:
            all_values += await self._run_finishing(context, targets, all_values)
            all_values += await self._generate_annotation(context, targets, all_values)
            return await self._finalize_async(all_values, targets, context)

        # Stage 0.45: WbCardSource — копия характеристик с похожего WB-товара
        # через бесплатный basket-API (БЕЗ Scrappey credits). Запускаем ПЕРВЫМ
        # — нулевая стоимость, ~300ms latency. WB-чары мапятся на тот же
        # Ozon-словарь (мы заполняем для Ozon, но WB — отличный источник).
        if self._wb_card is not None and remaining:
            new_avs = await self._run_wb_card_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.5: OzonCardSource — копия характеристик с похожего Ozon-товара.
        # Запускаем ПЕРВЫМ (до IceCat, PDF, LLM) — на eval-аудите парсер достаёт
        # 95.5% chars (21/22) из /features/ страницы и mapping на Ozon dict
        # gives direct attr_id resolution. Конкурирующие sources при таком
        # порядке дополняют OzonCard на attrs которые тот пропустил (outlier
        # товары, OzonCard match=skip), а не наоборот — OzonCard «съел» 14
        # потенциальных fills у других sources в предыдущей версии порядка.
        if self._ozon_card is not None and remaining:
            new_avs = await self._run_ozon_card_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.55: IceCatSource — brand-verified спеки без LLM (IceCat Open API).
        # Дополняет attrs которые OzonCard не закрыл (brand_line skip, outlier товары).
        # При 403/404 (неизвестный бренд) возвращает [].
        # Vision Stage 0.52 (mpn/ean enrich) был отключён в v10b — Vision видит фото
        # чужой brand_line карточки от OzonCard, MPN с неё неточен. Vision запускается
        # только Stage 3 (classifier-routed) для реально визуальных attrs.
        icecat_filled_count = 0
        if self._icecat is not None:
            new_avs = await self._run_icecat_stage(context, remaining, already_filled=filled_so_far)
            icecat_filled_count = len([v for v in new_avs if v.is_confident()])
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.6: PdfDatasheetSource — official manufacturer datasheet PDF (Gemini native).
        # Запускается ПОСЛЕ IceCat: дополняет / перекрывает atрибуты не найденные через IceCat.
        # Skip-guard внутри source: если ≥80% targets уже filled, source сам возвращает [].
        if self._pdf_datasheet is not None and remaining:
            new_avs = await self._run_pdf_datasheet_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.7: CompetitorRagSource — дешёвый RAG без LLM (0 API calls)
        # Запускается всегда когда source задан — Qdrant query ~100ms на товар.
        # Раньше был skip-guard "< 5 IceCat fills" но это мешало измерять реальный
        # эффект RAG (icecat avg = 5.35 на БП → RAG никогда не вызывался).
        if self._competitor_rag is not None:
            new_avs = await self._run_rag_stage(context, remaining, already_filled=filled_so_far)
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 1: Classifier — 1 LLM call for routing decisions
        routing = await self._classifier.classify(context, remaining)

        # Stage 2: LlmKnowledgeSource — детерминированный добор на ОСТАТОЧНЫХ таргетах.
        # Запускаем LK, если LLM_KNOWLEDGE присутствует в routing ВООБЩЕ (не только [0]).
        # Раньше условие было routing[t.id][0] == LLM_KNOWLEDGE — недетерминированный
        # классификатор то ставил LK первым, то нет → apparel-поля (Стиль/Назначение/
        # Особенности/Рисунок) непостоянно доходили до LK между товарами.
        # Безопасность: knowledge_targets берётся из remaining (незаполненные), и сам
        # source ещё раз фильтрует already_filled. Карточные значения (WB/Ozon, conf≥0.90)
        # уже исключены из remaining → LK только ДОБИРАЕТ пустые, не перетирает.
        knowledge_targets = [
            t for t in remaining
            if Source.LLM_KNOWLEDGE in routing.get(t.id, [])
            and self._sources[Source.LLM_KNOWLEDGE].is_applicable(context, t)
        ]
        if knowledge_targets:
            new_avs = await self._run_stage(
                Source.LLM_KNOWLEDGE, context, knowledge_targets,
                already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 3: VisionSource — attrs with VISION in suggested AND image_urls present
        vision_targets = [
            t for t in remaining
            if Source.VISION in routing.get(t.id, [])
            and self._sources[Source.VISION].is_applicable(context, t)
        ]
        if vision_targets:
            new_avs = await self._run_stage(
                Source.VISION, context, vision_targets,
                already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 4 gate: CostPredictor — check if web search is worth running
        #
        # Force-websearch attrs bypass CostPredictor AND the classifier's routing decision.
        # They are sent to WebSearch even if:
        #   - classifier didn't suggest WEB_SEARCH for them, or
        #   - they were already filled by LLM_KNOWLEDGE (web is more authoritative for these).
        #
        # Groups:
        #  - force_ws: targets whose id is in strategy.force_websearch_targets(targets)
        #              → always run WebSearch, regardless of routing / prior fill
        #  - optional_ws: remaining targets that classifier routed to WEB_SEARCH
        #                 → CostPredictor decides as before
        force_attr_ids = self._strategy.force_websearch_targets(targets)
        ws_applicable = self._sources[Source.WEB_SEARCH].is_applicable

        # Force targets: all targets in the force list (not just unfilled ones — we want
        # the web value to potentially override the LLM-knowledge value in _merge if
        # it has higher confidence).
        force_ws = [
            t for t in targets  # from ALL original targets, not just remaining
            if t.id in force_attr_ids
            and ws_applicable(context, t)
        ]

        # Optional targets: unfilled, routed to WEB_SEARCH, not in force list
        optional_ws = [
            t for t in remaining
            if t.id not in force_attr_ids
            and Source.WEB_SEARCH in routing.get(t.id, [])
            and ws_applicable(context, t)
        ]

        # Decide on optional targets via CostPredictor
        approved_optional: list = []
        if optional_ws:
            worth = await self._cost_predictor.is_web_search_worth(context, optional_ws)
            if worth:
                approved_optional = optional_ws

        # Deduplicate by id (force list may overlap with optional)
        seen_ids: set[int] = set()
        websearch_targets: list[TargetAttribute] = []
        for t in force_ws + approved_optional:
            if t.id not in seen_ids:
                seen_ids.add(t.id)
                websearch_targets.append(t)

        if not websearch_targets:
            all_values += await self._run_finishing(context, targets, all_values)
            all_values += await self._generate_annotation(context, targets, all_values)
            return await self._finalize_async(all_values, targets, context)

        if force_ws:
            logger.debug(
                "[Pipeline] force-websearch attrs: %s",
                [t.id for t in force_ws],
            )

        # Stage 4: WebSearchSource — передаём все накопленные high-conf AVs
        all_values += await self._run_stage(
            Source.WEB_SEARCH, context, websearch_targets,
            already_filled=filled_so_far,
        )
        filled_so_far = self._merge_high_conf(filled_so_far, all_values)

        # Stage 4.5: UgcSource — отзывы и Q&A с Ozon/WB для compat/physical attrs
        # (длина кабеля как у покупателя, шум, совместимость материнской платой).
        # Запускается последним перед finishing — после всех structured sources.
        if self._ugc is not None:
            remaining_for_ugc = self._remaining_targets(targets, all_values)
            if remaining_for_ugc:
                new_avs = await self._run_ugc_stage(
                    context, remaining_for_ugc, already_filled=filled_so_far,
                )
                all_values += new_avs

        # Stage 4.7: TnvedSource — per-category резолвер ТН ВЭД ЕАЭС.
        # Запускается после всех товарных sources: кэш по category_id уже тёплый
        # если несколько товаров одной категории обрабатываются параллельно.
        new_avs = await self._run_tnved_stage(context, targets, already_filled=filled_so_far)
        all_values += new_avs

        # Stage 5: Finishing pass — focused re-extraction for empty required attributes
        all_values += await self._run_finishing(context, targets, all_values)

        # Stage 5.5: Аннотация generation — generative step, not extraction.
        # Runs AFTER all sources and finishing so it can use the full set of filled attrs.
        all_values += await self._generate_annotation(context, targets, all_values)

        # Stage 6: детерминированный resolve_value_ids (в _finalize) + LLM-резолвер
        # ХВОСТА — батч-вызов на нерезолвнутые enum-value_id (семантика/перевод).
        return await self._finalize_async(all_values, targets, context)

    async def _generate_annotation(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        all_values: list[AttributeValue],
    ) -> list[AttributeValue]:
        """Stage 5.5: генерация поля «Аннотация» из уже собранных характеристик.

        Аннотация — генерируемое маркетинговое описание, не извлекаемое.
        Запускается ПОСЛЕ всех источников. Judge не нужен — это генерация, не извлечение.
        Пропускается если Аннотация уже заполнена или не входит в targets.
        """
        # Detect Аннотация target (case-insensitive)
        annotation_target = next(
            (t for t in targets if t.name.lower() == "аннотация"),
            None,
        )
        if annotation_target is None:
            return []

        # Skip if already filled with high confidence
        already_filled_ids = {v.attribute_id for v in all_values if v.is_confident()}
        if annotation_target.id in already_filled_ids:
            return []

        # Build characteristics summary from filled values
        filled_by_id: dict[int, AttributeValue] = {}
        for v in all_values:
            prev = filled_by_id.get(v.attribute_id)
            if prev is None or v.confidence > prev.confidence:
                filled_by_id[v.attribute_id] = v

        target_names: dict[int, str] = {t.id: t.name for t in targets}
        char_lines = []
        for attr_id, av in filled_by_id.items():
            if attr_id == annotation_target.id:
                continue
            name = target_names.get(attr_id, str(attr_id))
            char_lines.append(f"  {name}: {av.value}")

        chars_block = "\n".join(char_lines) if char_lines else "  (нет данных)"

        class _AnnotationResponse(BaseModel):
            annotation: str = Field(..., description="Маркетинговое описание товара 2-4 предложения")

        system_prompt = (
            "Ты маркетолог. Составь маркетинговое описание товара 2-4 предложения "
            "на основе предоставленных характеристик. Текст должен быть живым, "
            "продающим, без перечислений через запятую. Только текст описания, без заголовков."
        )
        user_text = (
            f"Товар: {context.product_name}\n"
            f"Бренд: {context.brand or 'неизвестен'}\n"
            f"Категория: {' / '.join(context.category_path) or 'н/д'}\n\n"
            f"Характеристики:\n{chars_block}\n\n"
            f"Составь маркетинговое описание товара 2-4 предложения."
        )

        try:
            llm = get_main_manager()
            parsed, _ = await llm.structured_request(
                system_prompt=system_prompt,
                user_text=user_text,
                response_model=_AnnotationResponse,
            )
        except Exception as e:
            logger.warning("[Pipeline] annotation generation failed: %s", e, exc_info=True)
            return []

        if parsed is None or not parsed.annotation.strip():
            return []

        context.llm_calls_so_far += 1
        return [
            AttributeValue(
                attribute_id=annotation_target.id,
                value=parsed.annotation.strip(),
                confidence=0.9,
                source=Source.LLM_KNOWLEDGE,
                evidence="generated from collected attributes",
                is_collection=False,
            )
        ]

    async def _finalize_async(
        self,
        all_values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Async финализация: детерминированный _finalize + LLM-резолвер ХВОСТА value_id.

        _finalize остаётся синхронным и неизменным (merge + post-process +
        детерминированный resolve_value_ids + validation). Затем один батч-LLM-вызов
        добивает нерезолвнутые enum-value_id (семантика/перевод). Для не-Ozon
        стратегий llm_resolve_tail — no-op. Используется во ВСЕХ точках выхода enrich.
        """
        finalized = self._finalize(all_values, targets, context)
        return await self._strategy.llm_resolve_tail(finalized, targets, context)

    def _finalize(
        self,
        all_values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Merge + strategy post-process + strategy validation. Used at every early-exit point."""
        merged = self._merge(all_values)

        # Strategy post-processing FIRST (добавляет CategoryDefaults + cross-fills с source=DESCRIPTION).
        # Должно идти ДО resolve_value_ids, иначе свежедобавленные AVs не получат value_id.
        merged = self._strategy.post_process_values(merged, targets, context)

        # Привязываем словарные value_id(s) (Ozon) или no-op для других стратегий.
        # Прогоняем ВСЕ AVs включая только что добавленные CategoryDefaults — иначе они
        # уходят в БД с value_id=NULL и проваливают value_id_resolution метрику.
        merged = [self._strategy.resolve_value_ids(v, context) for v in merged]

        # Strategy validation — drop or normalise individual values
        filtered: list[AttributeValue] = []
        for v in merged:
            target = next((t for t in targets if t.id == v.attribute_id), None)
            if target:
                result = self._strategy.validate_value(target, v.value, context)
                if result.is_valid:
                    if result.normalized_value is not None:
                        v.value = result.normalized_value
                    filtered.append(v)
                # else: drop value (validation failed)
            else:
                filtered.append(v)
        return filtered

    async def _run_rag_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить CompetitorRagSource + его судью. Ошибки не прерывают pipeline."""
        if self._competitor_rag is None:
            return []
        judge_wrapper = self._judges.get(Source.COMPETITOR_RAG)
        try:
            extracted = await self._competitor_rag.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] competitor_rag source failed: %s", e, exc_info=True)
            return []

        if judge_wrapper is None:
            return extracted

        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] competitor_rag judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_pdf_datasheet_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить PdfDatasheetSource + judge. Ошибки не прерывают pipeline."""
        if self._pdf_datasheet is None:
            return []
        judge_wrapper = self._judges.get(Source.PDF_DATASHEET)
        try:
            extracted = await self._pdf_datasheet.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] pdf_datasheet source failed: %s", e, exc_info=True)
            return []

        if judge_wrapper is None:
            return extracted

        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] pdf_datasheet judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_wb_card_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить WbCardSource + его судью. Ошибки не прерывают pipeline."""
        if self._wb_card is None:
            return []
        judge_wrapper = self._judges.get(Source.WB_CARD)
        try:
            extracted = await self._wb_card.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] wb_card source failed: %s", e, exc_info=True)
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] wb_card judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_ugc_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить UgcSource + его судью. Ошибки не прерывают pipeline."""
        if self._ugc is None:
            return []
        judge_wrapper = self._judges.get(Source.UGC)
        try:
            extracted = await self._ugc.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] ugc source failed: %s", e, exc_info=True)
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] ugc judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_tnved_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить TnvedSource (per-category кэш). Ошибки не прерывают pipeline."""
        try:
            extracted = await self._tnved.extract(context, targets, already_filled=already_filled)
        except Exception as e:
            logger.warning("[Pipeline] tnved source failed: %s", e, exc_info=True)
            return []
        # TnvedJudge — детерминированный (10 цифр), всегда применяем
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                valid = await self._tnved.get_judge().validate(value, context)
                if valid:
                    results.append(value)
            except Exception as e:
                logger.warning(
                    "[Pipeline] tnved judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_ozon_card_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить OzonCardSource + его судью. Ошибки не прерывают pipeline."""
        if self._ozon_card is None:
            return []
        judge_wrapper = self._judges.get(Source.OZON_CARD)
        try:
            extracted = await self._ozon_card.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] ozon_card source failed: %s", e, exc_info=True)
            return []

        if judge_wrapper is None:
            return extracted

        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] ozon_card judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_icecat_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить IceCatSource + его судью. Ошибки не прерывают pipeline."""
        if self._icecat is None:
            return []
        judge_wrapper = self._judges.get(Source.ICECAT)
        try:
            extracted = await self._icecat.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] icecat source failed: %s", e, exc_info=True)
            return []

        if judge_wrapper is None:
            return extracted

        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] icecat judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_stage(
        self,
        src: Source,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Run one source + judges, return validated values."""
        source = self._sources[src]
        judge_wrapper = self._judges[src]
        try:
            extracted = await source.extract(context, targets, already_filled=already_filled)
        except Exception as e:
            logger.warning("[Pipeline] %s source failed: %s", src.value, e, exc_info=True)
            return []

        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] %s judge failed for attr %s: %s",
                    src.value, value.attribute_id, e,
                )
        return results

    async def _run_finishing(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        all_values: list[AttributeValue],
    ) -> list[AttributeValue]:
        """Stage 5: focused re-extraction for empty required attributes."""
        return await self._finisher.extract_missing(context, targets, all_values)

    def _remaining_targets(
        self, all_targets: list[TargetAttribute], collected: list[AttributeValue]
    ) -> list[TargetAttribute]:
        """Targets which don't yet have a confident value."""
        confident_attr_ids = {v.attribute_id for v in collected if v.is_confident()}
        return [t for t in all_targets if t.id not in confident_attr_ids]

    def _merge_high_conf(
        self,
        existing: list[AttributeValue],
        new_avs: list[AttributeValue],
        threshold: float = 0.85,
    ) -> list[AttributeValue]:
        """Обновляет список already_filled: добавляет/заменяет AVs с confidence ≥ threshold.

        Используется для передачи skip-filled контекста следующему source в pipeline.
        """
        by_id: dict[int, AttributeValue] = {v.attribute_id: v for v in existing}
        for v in new_avs:
            if v.confidence < threshold:
                continue
            prev = by_id.get(v.attribute_id)
            if prev is None or v.confidence > prev.confidence:
                by_id[v.attribute_id] = v
        return list(by_id.values())

    def _merge(self, all_values: list[AttributeValue]) -> list[AttributeValue]:
        """Per attribute_id, pick highest confidence; tie-break by SOURCE_PRIORITY.

        Ensemble voting: если ≥2 разных source выдали одинаковое
        normalized value на тот же attribute_id — bump confidence
        +0.10 (cap 0.97). Cross-source agreement = сильный сигнал
        достоверности (LLM сказал, web search подтвердил, etc).
        """
        # Step 1: count unique sources per (attribute_id, normalized element).
        # Для коллекций ключуем ПОЭЛЕМЕНТНО (а не по str(list)), чтобы consensus
        # и дедуп работали по отдельным элементам, а не по строке всего списка.
        sources_per_value: dict[tuple[int, str], set] = {}
        for v in all_values:
            for norm_value in _norm_elements(v.value):
                key = (v.attribute_id, norm_value)
                sources_per_value.setdefault(key, set()).add(v.source)

        # Step 2: apply consensus bonus. Для скаляра — по значению; для коллекции
        # — если ХОТЯ БЫ один элемент подтверждён ≥2 источниками.
        boosted: list[AttributeValue] = []
        for v in all_values:
            n_sources = max(
                (len(sources_per_value[(v.attribute_id, ev)]) for ev in _norm_elements(v.value)),
                default=0,
            )
            if n_sources >= 2 and v.confidence < 0.97:
                new_conf = min(0.97, v.confidence + 0.10)
                boosted.append(v.model_copy(update={"confidence": new_conf}))
            else:
                boosted.append(v)

        # Step 3: per attribute_id.
        #  - Коллекционные (is_collection или value-список): UNION дедуплицированных
        #    элементов всех судьёй-прошедших/уверенных источников (multi-value
        #    значения теряться не должны — Особенности/Декор и т.п.).
        #  - Скалярные: highest-conf wins с card-protection band (_merge_winner),
        #    поведение БЕЗ изменений.
        by_id: dict[int, AttributeValue] = {}
        for v in boosted:
            existing = by_id.get(v.attribute_id)
            if existing is None:
                by_id[v.attribute_id] = v
                continue
            if _is_collection_value(v) or _is_collection_value(existing):
                by_id[v.attribute_id] = _merge_collection(existing, v)
            else:
                by_id[v.attribute_id] = _merge_winner(v, existing)
        return list(by_id.values())
