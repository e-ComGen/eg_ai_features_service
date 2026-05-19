"""OzonStrategy — STUB. Заполнить когда будут реальные Ozon API access.

Текущий behavior — почти то же что Default, но с Ozon-specific skips
для атрибутов которые AI не должен заполнять на Ozon.
"""
from typing import Any
from .base import MarketplaceStrategy, ValidationResult
from app.services.enrichment.base import (
    AttributeValue, TargetAttribute, ExtractionContext,
)
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_category,
    get_ozon_category_name,
)
# TODO (Tier 2): use get_ozon_characteristics_for_category in normalize_target
#   to inject Ozon allowed_values into TargetAttribute before enrichment.


# TODO (Tier 2): заполнить из реальных Ozon API данных
# Атрибуты которые Ozon генерит или требует от продавца напрямую
OZON_SKIP_SEMANTIC_TYPES = frozenset({
    "ean", "upc", "gtin", "barcode",  # Ozon требует от продавца в отдельном поле
    "imei", "serial",                  # specific для каждого экземпляра
    # TODO (Tier 2): расширить список на основе Ozon Content API docs
})

# TODO (Tier 2): заполнить реальным банлистом Ozon модерации
OZON_BANNED_PHRASES: frozenset[str] = frozenset({
    # Placeholder — реальный банлист подтянуть из Ozon модерации guidelines
    "лучший", "№1", "номер один",
})


class OzonStrategy(MarketplaceStrategy):
    """Ozon-specific overrides. STUB — заполнить когда будут реальные Ozon API access."""

    @property
    def name(self) -> str:
        return "ozon"

    def filter_unsupported_attributes(
        self, targets: list[TargetAttribute],
    ) -> list[TargetAttribute]:
        """Skip атрибуты которые Ozon требует от продавца напрямую или генерит сам.

        TODO (Tier 2): расширить на основе Ozon Content API категорийных словарей.
        """
        return [
            t for t in targets
            if not t.semantic_type or t.semantic_type not in OZON_SKIP_SEMANTIC_TYPES
        ]

    def validate_value(
        self,
        target: TargetAttribute,
        value: Any,
        context: ExtractionContext,
    ) -> ValidationResult:
        """Базовая проверка на банлист.

        TODO (Tier 2): проверка enum словарей категорий Ozon (Ozon Content API).
        TODO (Tier 2): проверка лимитов длины по типу атрибута.
        TODO (Tier 2): валидация форматов (цвет в RGB/hex, размеры в конкретных единицах).
        """
        if target.type == "text" and isinstance(value, str):
            v_lower = value.lower()
            for phrase in OZON_BANNED_PHRASES:
                if phrase and phrase in v_lower:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Contains banned phrase: '{phrase}'"
                    )
        return ValidationResult(is_valid=True, normalized_value=value)

    # normalize_target, post_process_values — TODO для будущих итераций
    # TODO (Tier 2): normalize_target для подмешивания Ozon category dictionary allowed_values
    # TODO (Tier 2): post_process_values для casing и форматирования enum значений Ozon
