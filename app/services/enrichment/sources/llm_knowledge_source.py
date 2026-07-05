"""LlmKnowledgeSource — извлекает характеристики из обучающей памяти LLM.

Полезно для известных брендов/моделей (iPhone, Nike, Samsung) — LLM знает
типичные характеристики без поиска. 1 LLM call. Confidence threshold 0.92
(строже чем для description, потому что LLM может галлюцинировать).

Spec: docs/architecture/pipeline.md, section "Stage 2 / LlmKnowledgeSource".
"""
import asyncio
from typing import Optional

from app import config
from pydantic import BaseModel, Field, AliasChoices, model_validator
from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager, get_openai_strict_manager, get_ensemble_managers, get_grounding_manager
from app.services.enrichment.judges.knowledge_judge import KnowledgeJudge
from app.services.enrichment.prompt_router import (
    format_target_line, build_meta_guidance,
    build_already_filled_block, filter_already_filled_targets,
)
from app.services.enrichment.strategies.base import MarketplaceStrategy
from app.services.enrichment.strategies.default_strategy import DefaultStrategy
from app.services.enrichment.ensemble.reconcile import reconcile as ensemble_reconcile


class _KnowledgeAttr(BaseModel):
    attribute_id: int = Field(..., validation_alias=AliasChoices("attribute_id", "id"))
    value: str | int | float | bool | list[str | int | float | bool] = Field(..., validation_alias=AliasChoices("value", "attribute_value", "extracted_value"))
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: Optional[str] = Field(None, max_length=200, description="откуда LLM знает")

    model_config = {"populate_by_name": True}


class _KnowledgeResponse(BaseModel):
    known_attributes: list[_KnowledgeAttr]

    @model_validator(mode="before")
    @classmethod
    def _drop_null_values(cls, data):
        if isinstance(data, dict) and isinstance(data.get("known_attributes"), list):
            def _value_of(e):
                if isinstance(e, dict):
                    return e.get("value")
                # Also accept already-constructed _KnowledgeAttr instances
                return getattr(e, "value", None)

            data["known_attributes"] = [
                e for e in data["known_attributes"]
                if _value_of(e) is not None
            ]
        return data


class LlmKnowledgeSource(AttributeSource):
    """1 LLM call — извлечение из обучающей памяти LLM.

    Применим только для well-known товаров (есть бренд + конкретная модель).
    Для no-name товаров возвращает [] чтобы не тратить токены на галлюцинации.
    """

    def __init__(
        self,
        llm_manager: Optional[StructuredLlmManager] = None,
        strategy: Optional[MarketplaceStrategy] = None,
    ):
        self._llm = llm_manager or get_main_manager()
        self._judge = KnowledgeJudge()
        self._strategy: MarketplaceStrategy = strategy or DefaultStrategy()

    @property
    def source_type(self) -> Source:
        return Source.LLM_KNOWLEDGE

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим если есть бренд (известный товар) или конкретное модель в названии."""
        # Heuristic: длинное название с бренд-likely substring И не пустое
        if not context.product_name or len(context.product_name) < 5:
            return False
        # Если есть явный brand field — отлично
        if context.brand:
            return True
        # Иначе предполагаем что если название содержит признаки модели (число, версия) — known
        # Для MVP: возвращаем True для всех товаров с осмысленным name, judge отфильтрует галлюцинации
        return True

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: list[AttributeValue] | None = None,
    ) -> list[AttributeValue]:
        if not targets or not context.product_name:
            return []

        # Убираем уже заполненные attrs из targets чтобы не тратить токены
        effective_targets = filter_already_filled_targets(targets, already_filled or [])
        if not effective_targets:
            return []

        # Build prompt с type-aware подсказками
        # Батчинг: разбиваем targets на чанки по CHUNK_SIZE, context не дублируем
        CHUNK_SIZE = 30
        already_preamble, already_rule = build_already_filled_block(already_filled or [])

        system_prompt = (
            "You are a product knowledge expert. Given a product name (and optional brand), "
            "you provide attribute values that you confidently KNOW from your training data. "
            "If you are NOT sure about an attribute — DO NOT include it. Better to skip than guess. "
            "Confidence scale: 0.95-1.0 = industry-standard or official spec (e.g. Samsung S24 Ultra "
            "camera is 200MP, Adidas Superstar sole is rubber — these are well-known facts); "
            "0.92-0.94 = highly likely but minor variation possible; "
            "0.85-0.91 = you recall this but not fully certain — INCLUDE with this confidence, "
            "a second judge will verify; "
            "below 0.85 = DO NOT include. "
            "Set confidence=0.95 for facts you know with certainty from official specs or brand history. "
            "Brief reasoning helps audit (e.g., 'official Samsung spec', 'Adidas classic model'). "
            "If the target has is_collection=true, return a JSON array of values; otherwise a single scalar. "
            "NEVER output: country of origin / manufacturer country; material composition or content "
            "percentages; shelf-life / service-life / warranty periods; specific technical specs "
            "(chipset/SoC, GPU, battery capacity mAh, power W, IP rating, Bluetooth/Wi-Fi version, "
            "RPM, suction Pa, screen/camera resolution) — UNLESS you are certain of THIS exact branded "
            "product model AND that value is an official published spec. For generic, commodity, or "
            "unbranded items, SKIP these fields entirely. "
            "Do NOT fabricate a value to satisfy a required field. Leaving a field empty is correct "
            "and expected when the source does not support a value."
            + build_meta_guidance()
            + already_rule
        )

        context_prefix = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n"
            f"Category: {' / '.join(context.category_path) or 'n/a'}\n\n"
            + already_preamble
        )

        chunks = [
            effective_targets[i: i + CHUNK_SIZE]
            for i in range(0, len(effective_targets), CHUNK_SIZE)
        ]

        target_by_id = {t.id: t for t in targets}
        if config.LLM_ENSEMBLE_ENABLED:
            return await self._extract_ensemble(
                context=context,
                chunks=chunks,
                context_prefix=context_prefix,
                system_prompt=system_prompt,
                target_by_id=target_by_id,
                effective_targets=effective_targets,
            )

        all_extracted: list[_KnowledgeAttr] = []

        for chunk in chunks:
            targets_block = "\n".join([format_target_line(t) for t in chunk])
            user_text = (
                context_prefix
                + f"Target attributes:\n{targets_block}\n\n"
                f"Return only attributes you confidently know. Field name: 'known_attributes'."
            )

            response_model = self._strategy.build_response_model(_KnowledgeResponse, chunk)
            # Маршрутизация: enum-heavy модели → OpenAI strict mode для token-level enforcement
            llm = self._llm
            if getattr(response_model, "__has_enum_constraints__", False):
                strict = get_openai_strict_manager()
                if strict is not None:
                    llm = strict
            parsed, _tokens = await llm.structured_request(
                system_prompt=system_prompt,
                user_text=user_text,
                response_model=response_model,
            )
            context.llm_calls_so_far += 1
            if parsed is not None:
                all_extracted.extend(parsed.known_attributes)

        # Дедуп + защита от кросс-чанк галлюцинаций: только реально запрошенные id,
        # первое вхождение на id.
        _eff_ids = {t.id for t in effective_targets}
        _seen: set[int] = set()
        all_extracted = [
            a for a in all_extracted
            if a.attribute_id in _eff_ids
            and not (a.attribute_id in _seen or _seen.add(a.attribute_id))
        ]

        return [
            AttributeValue(
                attribute_id=a.attribute_id,
                value=a.value,
                confidence=a.confidence,
                source=Source.LLM_KNOWLEDGE,
                evidence=a.reasoning,
                semantic_type=target_by_id[a.attribute_id].semantic_type
                              if a.attribute_id in target_by_id else None,
                is_collection=target_by_id[a.attribute_id].is_collection
                              if a.attribute_id in target_by_id else False,
            )
            for a in all_extracted
        ]

    async def _extract_ensemble(
        self,
        context: ExtractionContext,
        chunks: list[list[TargetAttribute]],
        context_prefix: str,
        system_prompt: str,
        target_by_id: dict[int, TargetAttribute],
        effective_targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Phase 2a: 2-vendor ensemble path (DeepSeek + OpenAI gpt-4o-mini), strict consensus.

        Called only when config.LLM_ENSEMBLE_ENABLED is True (see extract() above).
        Calls model A and model B IN PARALLEL per chunk via asyncio.gather, then
        reconciles each chunk's pair of results via ensemble.reconcile() (strict
        2-model consensus: only-in-one-model values are abstained, never emitted).
        KnowledgeJudge is intentionally NOT invoked in this path -- consensus
        replaces it. Cross-chunk dedup mirrors the existing single-model path:
        first occurrence of a given attribute_id wins, and only ids that were
        actually requested (in effective_targets) are ever emitted.
        """
        manager_a, manager_b = get_ensemble_managers()
        grounding_manager = get_grounding_manager()
        arbiter = self._llm  # reuse the source's configured manager (DeepSeek by default) as the cheap arbiter

        _eff_ids = {t.id for t in effective_targets}
        all_values: list[AttributeValue] = []

        for chunk in chunks:
            targets_block = "\n".join([format_target_line(t) for t in chunk])
            user_text = (
                context_prefix
                + f"Target attributes:\n{targets_block}\n\n"
                f"Return only attributes you confidently know. Field name: 'known_attributes'."
            )
            response_model = self._strategy.build_response_model(_KnowledgeResponse, chunk)

            (parsed_a, _tokens_a), (parsed_b, _tokens_b) = await asyncio.gather(
                manager_a.structured_request(
                    system_prompt=system_prompt, user_text=user_text, response_model=response_model,
                ),
                manager_b.structured_request(
                    system_prompt=system_prompt, user_text=user_text, response_model=response_model,
                ),
            )
            context.llm_calls_so_far += 2

            chunk_ids = {t.id for t in chunk}
            list_a = [
                a for a in (parsed_a.known_attributes if parsed_a is not None else [])
                if a.attribute_id in chunk_ids and a.attribute_id in _eff_ids
            ]
            list_b = [
                b for b in (parsed_b.known_attributes if parsed_b is not None else [])
                if b.attribute_id in chunk_ids and b.attribute_id in _eff_ids
            ]

            chunk_values = await ensemble_reconcile(
                list_a, list_b, chunk, arbiter, judge=self._judge, context=context,
                manager_a=manager_a, manager_b=manager_b, grounding_manager=grounding_manager,
            )
            all_values.extend(chunk_values)

        # Cross-chunk dedup safety net (mirrors the single-model path): first
        # occurrence per attribute_id wins, only ids that were actually requested.
        seen: set[int] = set()
        deduped: list[AttributeValue] = []
        for v in all_values:
            if v.attribute_id in _eff_ids and v.attribute_id not in seen:
                seen.add(v.attribute_id)
                deduped.append(v)
        return deduped

    def get_judge(self) -> LlmJudge:
        return self._judge
