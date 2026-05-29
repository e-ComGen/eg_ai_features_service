"""PipelineOrchestrator — sequential cost-aware extraction.

Главный entry-point для enrichment. Объединяет все sources, judges и intelligence
в один flow с early-exit (если все targets заполнены — останавливаемся) и
cost gating (CostPredictor перед expensive web search).

Spec: docs/architecture/pipeline.md, section "PipelineOrchestrator".
"""
import logging
from typing import Optional

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
)
from app.services.enrichment.intelligence import LlmClassifier, CostPredictor
from app.services.enrichment.confidence_aware_judge import ConfidenceAwareJudgeWrapper
from app.services.enrichment.strategies.base import MarketplaceStrategy
from app.services.enrichment.strategies.default_strategy import DefaultStrategy
from app.services.enrichment.finishing import FinishingExtractor

logger = logging.getLogger(__name__)


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
        # OzonCardSource: копия характеристик из живой Ozon-карточки через Apify.
        # None → Ozon card stage пропускается.
        self._ozon_card: Optional[OzonCardSource] = ozon_card_source
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
            return self._finalize(all_values, targets, context)

        # Stage 0.5: IceCatSource — brand-verified спеки без LLM (HTTP к IceCat Open API)
        # Запускаем ПЕРВЫМ (до RAG и Classifier) — самый авторитетный источник спецификаций.
        # При 403/404 (неизвестный бренд) возвращает [] — тогда RAG подхватывает как фолбэк.
        icecat_filled_count = 0
        if self._icecat is not None:
            new_avs = await self._run_icecat_stage(context, remaining, already_filled=filled_so_far)
            icecat_filled_count = len([v for v in new_avs if v.is_confident()])
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                return self._finalize(all_values, targets, context)

        # Stage 0.55: OzonCardSource — копия характеристик из живой Ozon-карточки (Apify).
        # Запускается ПОСЛЕ IceCat и ДО PDF: если IceCat закрыл атрибут (brand-verified),
        # OzonCard не перезаписывает; если нет — OzonCard может дать «exact» совпадение
        # (наивысший приоритет среди внешних источников) или brand-line частичное.
        # Skip-guard внутри source: ≥80% filled → пропуск.
        if self._ozon_card is not None and remaining:
            new_avs = await self._run_ozon_card_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                return self._finalize(all_values, targets, context)

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
                return self._finalize(all_values, targets, context)

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
                return self._finalize(all_values, targets, context)

        # Stage 1: Classifier — 1 LLM call for routing decisions
        routing = await self._classifier.classify(context, remaining)

        # Stage 2: LlmKnowledgeSource — attrs where first suggested source = LLM_KNOWLEDGE
        knowledge_targets = [
            t for t in remaining
            if routing.get(t.id) and routing[t.id][0] == Source.LLM_KNOWLEDGE
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
                return self._finalize(all_values, targets, context)

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
                return self._finalize(all_values, targets, context)

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
            return self._finalize(all_values, targets, context)

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

        # Stage 5: Finishing pass — focused re-extraction for empty required attributes
        all_values += await self._run_finishing(context, targets, all_values)

        return self._finalize(all_values, targets, context)

    def _finalize(
        self,
        all_values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Merge + strategy post-process + strategy validation. Used at every early-exit point."""
        merged = self._merge(all_values)

        # Привязываем словарные value_id(s) (Ozon) или no-op для других стратегий
        merged = [self._strategy.resolve_value_ids(v, context) for v in merged]

        # Strategy post-processing (e.g. casing normalisation for known enums)
        merged = self._strategy.post_process_values(merged, targets, context)

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
        """Per attribute_id, pick highest confidence; tie-break by SOURCE_PRIORITY."""
        by_id: dict[int, AttributeValue] = {}
        for v in all_values:
            existing = by_id.get(v.attribute_id)
            if existing is None:
                by_id[v.attribute_id] = v
                continue
            if v.confidence > existing.confidence:
                by_id[v.attribute_id] = v
            elif (
                v.confidence == existing.confidence
                and SOURCE_PRIORITY[v.source] > SOURCE_PRIORITY[existing.source]
            ):
                by_id[v.attribute_id] = v
        return list(by_id.values())
