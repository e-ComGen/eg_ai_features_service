"""WildberriesStrategy — placeholder с TODO для будущей интеграции с WB API.

Текущий behavior — почти то же что Default, но с marketplace-specific skips
для атрибутов которые AI не должен заполнять на WB.
"""
from typing import Any
from .base import MarketplaceStrategy, ValidationResult
from app.services.enrichment.base import (
    AttributeValue, TargetAttribute, ExtractionContext,
)


# Атрибуты которые AI не должен заполнять на WB — селлер сам или WB генерит
WB_SKIP_SEMANTIC_TYPES = frozenset({
    "ean", "upc", "gtin", "barcode",  # юридическая ответственность селлера
    "article", "sku",                  # WB генерит nm_id автоматически
    "imei", "serial",                  # specific для каждого экземпляра
})

# Запрещённые слова в описаниях (банлист WB модерации)
WB_BANNED_PHRASES = frozenset({
    "лучший", "№1", "номер один", "лидер рынка",
    "уникальный", "эксклюзив",
})


class WildberriesStrategy(MarketplaceStrategy):
    """WB-specific overrides."""

    @property
    def name(self) -> str:
        return "wb"

    def filter_unsupported_attributes(
        self, targets: list[TargetAttribute],
    ) -> list[TargetAttribute]:
        """Skip атрибуты которые WB-селлер заполняет сам."""
        return [
            t for t in targets
            if not t.semantic_type or t.semantic_type not in WB_SKIP_SEMANTIC_TYPES
        ]

    def validate_value(
        self,
        target: TargetAttribute,
        value: Any,
        context: ExtractionContext,
    ) -> ValidationResult:
        """Проверка на банлист (для текстовых описаний)."""
        if target.type == "text" and isinstance(value, str):
            v_lower = value.lower()
            for phrase in WB_BANNED_PHRASES:
                if phrase and phrase in v_lower:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Contains banned phrase: '{phrase}'"
                    )
        # TODO: проверка enum словарей категорий WB (Tier 2 — подтянуть из WB Content API)
        # TODO: проверка лимитов длины (Tier 2)
        return ValidationResult(is_valid=True, normalized_value=value)

    # post_process_values, normalize_target — TODO для будущих итераций
