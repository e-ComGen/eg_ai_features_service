"""Marketplace strategies — изолируют marketplace-specific поведение.

Pipeline core не знает про WB/Ozon. Все различия (словари, лимиты, банлисты,
форматы значений) — в strategy. По умолчанию DefaultStrategy (no-op).
"""
from abc import ABC, abstractmethod
from typing import Optional, Any
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
