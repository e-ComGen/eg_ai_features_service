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
)
from app.services.enrichment.intelligence import LlmClassifier, CostPredictor
from app.services.enrichment.confidence_aware_judge import ConfidenceAwareJudgeWrapper
from app.services.enrichment.strategies.base import MarketplaceStrategy
from app.services.enrichment.strategies.default_strategy import DefaultStrategy

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Sequential cost-aware pipeline.

    Order: Description → Classifier → (per-attr routing) → Knowledge → Vision → CostPredictor → WebSearch
    Early-exit at each stage if all targets filled with high confidence.
    """

    def __init__(
        self,
        description_source: Optional[DescriptionSource] = None,
        knowledge_source: Optional[LlmKnowledgeSource] = None,
        vision_source: Optional[VisionSource] = None,
        websearch_source: Optional[WebSearchSource] = None,
        classifier: Optional[LlmClassifier] = None,
        cost_predictor: Optional[CostPredictor] = None,
        strategy: Optional[MarketplaceStrategy] = None,
    ):
        self._sources: dict[Source, AttributeSource] = {
            Source.DESCRIPTION: description_source or DescriptionSource(),
            Source.LLM_KNOWLEDGE: knowledge_source or LlmKnowledgeSource(),
            Source.VISION: vision_source or VisionSource(),
            Source.WEB_SEARCH: websearch_source or WebSearchSource(),
        }
        self._judges: dict[Source, ConfidenceAwareJudgeWrapper] = {
            src: ConfidenceAwareJudgeWrapper(s.get_judge())
            for src, s in self._sources.items()
        }
        self._classifier = classifier or LlmClassifier()
        self._cost_predictor = cost_predictor or CostPredictor()
        self._strategy: MarketplaceStrategy = strategy or DefaultStrategy()

    async def enrich(
        self, context: ExtractionContext, targets: list[TargetAttribute]
    ) -> list[AttributeValue]:
        """Run full pipeline. Returns merged final attributes (one per attribute_id)."""
        # NEW: apply strategy.filter_unsupported_attributes before pipeline starts
        targets = self._strategy.filter_unsupported_attributes(targets)

        all_values: list[AttributeValue] = []

        # Stage 0: DescriptionSource (always first, cheapest)
        all_values += await self._run_stage(Source.DESCRIPTION, context, targets)
        remaining = self._remaining_targets(targets, all_values)
        if not remaining:
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
            all_values += await self._run_stage(Source.LLM_KNOWLEDGE, context, knowledge_targets)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                return self._finalize(all_values, targets, context)

        # Stage 3: VisionSource — attrs with VISION in suggested AND image_urls present
        vision_targets = [
            t for t in remaining
            if Source.VISION in routing.get(t.id, [])
            and self._sources[Source.VISION].is_applicable(context, t)
        ]
        if vision_targets:
            all_values += await self._run_stage(Source.VISION, context, vision_targets)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                return self._finalize(all_values, targets, context)

        # Stage 4 gate: CostPredictor — check if web search is worth running
        websearch_targets = [
            t for t in remaining
            if Source.WEB_SEARCH in routing.get(t.id, [])
            and self._sources[Source.WEB_SEARCH].is_applicable(context, t)
        ]
        if not websearch_targets:
            return self._finalize(all_values, targets, context)

        worth = await self._cost_predictor.is_web_search_worth(context, websearch_targets)
        if not worth:
            return self._finalize(all_values, targets, context)

        # Stage 4: WebSearchSource
        all_values += await self._run_stage(Source.WEB_SEARCH, context, websearch_targets)

        return self._finalize(all_values, targets, context)

    def _finalize(
        self,
        all_values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Merge + strategy post-process + strategy validation. Used at every early-exit point."""
        merged = self._merge(all_values)

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

    async def _run_stage(
        self, src: Source, context: ExtractionContext, targets: list[TargetAttribute]
    ) -> list[AttributeValue]:
        """Run one source + judges, return validated values."""
        source = self._sources[src]
        judge_wrapper = self._judges[src]
        try:
            extracted = await source.extract(context, targets)
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

    def _remaining_targets(
        self, all_targets: list[TargetAttribute], collected: list[AttributeValue]
    ) -> list[TargetAttribute]:
        """Targets which don't yet have a confident value."""
        confident_attr_ids = {v.attribute_id for v in collected if v.is_confident()}
        return [t for t in all_targets if t.id not in confident_attr_ids]

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
