"""Marketplace strategies — изолируют marketplace-specific поведение.

Pipeline core не знает про WB/Ozon. Все различия (словари, лимиты, банлисты,
форматы значений) — в strategy. По умолчанию DefaultStrategy (no-op).
"""
from abc import ABC, abstractmethod
from typing import Optional, Any, Type
from pydantic import BaseModel

from app.services.enrichment.base import (
    AttributeValue, TargetAttribute, ExtractionContext,
)


class ValidationResult(BaseModel):
    is_valid: bool
    normalized_value: Optional[Any] = None  # marketplace-corrected value (например 'хлопок' → 'Хлопок')
    reason: Optional[str] = None             # если invalid


class MarketplaceStrategy(ABC):
    """Базовая абстракция marketplace-specific логики."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Имя стратегии: 'default' | 'wb' | 'ozon'."""

    def normalize_target(self, target: TargetAttribute) -> TargetAttribute:
        """Преобразовать target под marketplace conventions.

        Default: no-op. Marketplace strategies могут переопределить
        для подмешивания allowed_values из marketplace dictionary, изменения
        semantic_type и т.п.
        """
        return target

    def validate_value(
        self,
        target: TargetAttribute,
        value: Any,
        context: ExtractionContext,
    ) -> ValidationResult:
        """Marketplace-specific validation. Default: всё валидно.

        Например WB: проверка что value в категорийном словаре, длина в лимитах,
        нет запрещённых слов ('лучший', '№1' для рекламы).
        """
        return ValidationResult(is_valid=True, normalized_value=value)

    def filter_by_dictionary(
        self,
        targets: list[TargetAttribute],
        context: "ExtractionContext",
    ) -> list[TargetAttribute]:
        """Оставить только targets известных словарю. Default: no-op."""
        return targets

    def normalize_target_with_context(
        self,
        target: TargetAttribute,
        context: "ExtractionContext",
    ) -> TargetAttribute:
        """Обогатить target метаданными из словаря. Default: no-op."""
        return target

    def filter_unsupported_attributes(
        self, targets: list[TargetAttribute],
    ) -> list[TargetAttribute]:
        """Отфильтровать target'ы которые AI вообще не должен пытаться заполнить
        на этом marketplace. Default: ничего не фильтруется.

        Например: WB skip 'Артикул производителя' (селлер сам), Ozon — другой набор.
        """
        return targets

    def post_process_values(
        self,
        values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """После merge — финальная marketplace-specific корректировка.
        Default: no-op.

        Например WB: приведение casing 'хлопок' → 'Хлопок' для известных enum.
        """
        return values

    def resolve_value_ids(
        self,
        attribute_value: AttributeValue,
        context: "ExtractionContext",
    ) -> AttributeValue:
        """Привязать словарные value_id(s). Default: no-op (pass-through)."""
        return attribute_value

    async def llm_resolve_tail(
        self,
        values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: "ExtractionContext",
    ) -> list[AttributeValue]:
        """LLM-резолвер хвоста нерезолвнутых value_id. Default: no-op (pass-through)."""
        return values

    def force_websearch_targets(
        self,
        targets: list[TargetAttribute],
    ) -> set[int]:
        """Return attribute IDs to force-route through WebSearch regardless of CostPredictor.

        Default: empty (no force). Strategies override with universal logic
        (e.g. kind=dimensions, kind=numeric, large enum dictionaries).
        """
        return set()

    def build_response_model(
        self,
        base_model: Type[BaseModel],
        targets: list[TargetAttribute],
    ) -> Type[BaseModel]:
        """Default: return base_model unchanged. Strategies override to enforce constraints."""
        return base_model
