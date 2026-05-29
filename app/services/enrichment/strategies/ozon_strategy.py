"""OzonStrategy — marketplace-specific логика для Ozon.

Использует ozon_dictionary.json (203MB, 9227 leaf type_id, 343356 характеристик)
для авторитетной схемы характеристик: normalize_target подмешивает метаданные
из словаря, validate_value проверяет банлист и (будущее) enum-значения.
"""
from typing import Any, Optional, Type
from .base import MarketplaceStrategy, ValidationResult
from app.services.enrichment.base import (
    AttributeValue, TargetAttribute, ExtractionContext,
)
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    get_ozon_characteristics_for_category,
    get_ozon_category_name,
    resolve_value_id,
    is_truncated,
)
from app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup import (
    search_value as _runtime_search_value,
)
from pydantic import BaseModel, model_validator
from app.services.enrichment.prompt_router import classify_target


# Атрибуты которые Ozon требует от продавца напрямую или генерит сам
OZON_SKIP_SEMANTIC_TYPES = frozenset({
    "ean", "upc", "gtin", "barcode",  # Ozon требует от продавца в отдельном поле
    "imei", "serial",                  # специфично для каждого экземпляра
})

# Банлист фраз (placeholder — расширить из модерации Ozon)
OZON_BANNED_PHRASES: frozenset[str] = frozenset({
    "лучший", "№1", "номер один",
})

# Маппинг Ozon type → TargetAttribute.type
_OZON_TYPE_MAP: dict[str, str] = {
    "String": "text",
    "Integer": "numeric",
    "Decimal": "numeric",
    "Boolean": "bool",
    "Option": "enum",
    "MultiOption": "enum",
    "ImageUrl": "text",
    "Url": "text",
}


def _ozon_type_to_attr_type(ozon_type: str) -> str:
    """Перевести Ozon API type string в TargetAttribute.type."""
    return _OZON_TYPE_MAP.get(ozon_type, "text")


def _build_char_index(characteristics: list[dict]) -> dict[int, dict]:
    """Построить индекс {char_id: char_dict} из списка характеристик."""
    return {c["id"]: c for c in characteristics if "id" in c}


class OzonStrategy(MarketplaceStrategy):
    """Ozon-specific overrides с поддержкой ozon_dictionary.json."""

    @property
    def name(self) -> str:
        return "ozon"

    def force_websearch_targets(self, targets: list[TargetAttribute]) -> set[int]:
        """Force WebSearch для kinds где сайты производителей/ритейлеров точнее LLM-знаний.

        Универсальные правила (работают для всех 9227 категорий Ozon):
        - dimensions: физические габариты — LxWxH ищутся на странице товара
        - numeric: числовые спецификации (кол-во разъёмов, мощность, вес и т.п.)
        - enum с большим словарём (>100 значений): бренд, страна, цвет — длинный хвост,
          LLM делает ошибки написания; web-поиск точнее находит официальные значения
        """
        # Kinds для которых web-поиск обычно точнее LLM-знаний
        force_kinds = {"dimensions", "numeric"}
        ids: set[int] = set()
        for t in targets:
            k = classify_target(t)
            if k in force_kinds:
                ids.add(t.id)
            # Enum со словарём > 100 значений (бренд, страна, город, цвет):
            # LLM часто ошибается в написании на длинном хвосте — web точнее
            elif k == "enum" and t.allowed_values and len(t.allowed_values) > 100:
                ids.add(t.id)
        return ids

    def build_response_model(
        self,
        base_model: Type[BaseModel],
        targets: list[TargetAttribute],
    ) -> Type[BaseModel]:
        """Строим constrained response model с model_validator для enum-targets.

        Для каждого enum-target с непустым allowed_values добавляем проверку:
        если извлечённый value не входит в allowed list (case-sensitive) — ValueError,
        structured_adapter повторит запрос.
        """
        # Собираем словарь {attribute_id: frozenset(allowed_values)} для enum-targets
        enum_constraints: dict[int, frozenset[str]] = {
            t.id: frozenset(str(v) for v in t.allowed_values)
            for t in targets
            if classify_target(t) == "enum" and t.allowed_values
        }
        if not enum_constraints:
            return base_model

        _constraints = enum_constraints  # замыкание в validator

        class _ConstrainedModel(base_model):  # type: ignore[valid-type]
            @model_validator(mode="after")
            def _enforce_allowed_values(self):
                # Извлекаем список _ExtractedAttr из поля (поддерживаем разные имена полей)
                items = (
                    getattr(self, "extracted", None)
                    or getattr(self, "known_attributes", None)
                    or []
                )
                for item in items:
                    attr_id = getattr(item, "attribute_id", None)
                    if attr_id not in _constraints:
                        continue
                    allowed = _constraints[attr_id]
                    raw_value = getattr(item, "value", None)
                    # Проверяем скаляр или список (is_collection)
                    if isinstance(raw_value, list):
                        bad = [str(v) for v in raw_value if str(v) not in allowed]
                        if bad:
                            raise ValueError(
                                f"attr_id={attr_id}: values {bad} not in allowed list {sorted(allowed)}"
                            )
                    else:
                        if raw_value is not None and str(raw_value) not in allowed:
                            raise ValueError(
                                f"attr_id={attr_id}: value {raw_value!r} not in allowed list {sorted(allowed)}"
                            )
                return self

        _ConstrainedModel.__name__ = f"Constrained_{base_model.__name__}"
        _ConstrainedModel.__qualname__ = _ConstrainedModel.__name__
        # Флаг: sources увидят этот маркер и направят вызов в OpenAI strict provider
        _ConstrainedModel.__has_enum_constraints__ = True  # type: ignore[attr-defined]
        return _ConstrainedModel

    def _get_char_index(
        self,
        context: ExtractionContext,
    ) -> dict[int, dict]:
        """Получить индекс характеристик для категории из контекста.

        Использует (description_category_id, type_id) если оба доступны,
        иначе fallback на один category_id (перебор по первому type).
        """
        if context.ozon_type_id is not None:
            chars = get_ozon_characteristics_for_type(context.category_id, context.ozon_type_id)
        else:
            # Fallback: берём первый подходящий type для category_id
            chars = get_ozon_characteristics_for_category(context.category_id)
        return _build_char_index(chars)

    def normalize_target(self, target: TargetAttribute) -> TargetAttribute:
        """normalize_target без контекста — no-op (контекст нужен для словаря).

        Для обогащения метаданными используй normalize_target_with_context.
        """
        return target

    def normalize_target_with_context(
        self,
        target: TargetAttribute,
        context: ExtractionContext,
    ) -> TargetAttribute:
        """Обогатить TargetAttribute метаданными из Ozon dictionary.

        Если характеристика с таким id есть в словаре:
        - подставляем официальное name из словаря
        - уточняем type через _ozon_type_to_attr_type
        - подставляем description если у target его нет
        Если характеристика не найдена — возвращаем target без изменений.
        """
        char_index = self._get_char_index(context)
        char_meta = char_index.get(target.id)
        if not char_meta:
            return target

        # Строим обновлённый target с метаданными из словаря
        ozon_type = char_meta.get("type", "")
        return TargetAttribute(
            id=target.id,
            name=char_meta.get("name", target.name),
            type=_ozon_type_to_attr_type(ozon_type) if ozon_type else target.type,
            allowed_values=target.allowed_values,
            semantic_type=target.semantic_type,
            description=target.description or char_meta.get("description"),
            is_collection=bool(char_meta.get("is_collection", False)),
            is_required=bool(char_meta.get("is_required", target.is_required)),
        )

    def filter_unsupported_attributes(
        self, targets: list[TargetAttribute],
    ) -> list[TargetAttribute]:
        """Skip атрибуты которые Ozon требует от продавца напрямую или генерит сам."""
        return [
            t for t in targets
            if not t.semantic_type or t.semantic_type not in OZON_SKIP_SEMANTIC_TYPES
        ]

    def filter_by_dictionary(
        self,
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[TargetAttribute]:
        """Оставить только те targets, чей id есть в словаре для данной категории.

        Если словарь не содержит категорию — возвращаем все targets без фильтрации
        (graceful degradation: словарь может быть неполным).
        """
        char_index = self._get_char_index(context)
        if not char_index:
            # Словарь пуст или категория не найдена — не фильтруем
            return targets
        return [t for t in targets if t.id in char_index]

    def validate_value(
        self,
        target: TargetAttribute,
        value: Any,
        context: ExtractionContext,
    ) -> ValidationResult:
        """Проверка банлиста. В будущем: enum-словари категорий и лимиты длины."""
        if target.type == "text" and isinstance(value, str):
            v_lower = value.lower()
            for phrase in OZON_BANNED_PHRASES:
                if phrase and phrase in v_lower:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Contains banned phrase: '{phrase}'"
                    )
        return ValidationResult(is_valid=True, normalized_value=value)

    def resolve_value_ids(
        self,
        attribute_value: AttributeValue,
        context: ExtractionContext,
    ) -> AttributeValue:
        """Привязать словарные value_id(s) к AttributeValue через ozon_loader.

        Если у характеристики нет values-списка — возвращает без изменений.
        Для атрибутов с values_truncated=True используй resolve_value_ids_async.
        """
        cat_id = context.category_id
        type_id = context.ozon_type_id
        attr_id = attribute_value.attribute_id

        if type_id is None:
            return attribute_value

        if attribute_value.is_collection and isinstance(attribute_value.value, list):
            ids = [
                resolve_value_id(cat_id, type_id, attr_id, str(v))
                for v in attribute_value.value
            ]
            resolved = [i for i in ids if i is not None]
            if resolved:
                attribute_value.value_ids = resolved
        else:
            vid = resolve_value_id(cat_id, type_id, attr_id, str(attribute_value.value))
            if vid is not None:
                attribute_value.value_id = vid

        return attribute_value

    async def resolve_value_ids_async(
        self,
        attribute_value: AttributeValue,
        context: ExtractionContext,
        *,
        client_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> AttributeValue:
        """Async variant: resolve value ids with runtime API fallback for truncated dicts.

        Algorithm per value:
        1. Try static dictionary lookup (resolve_value_id). Fast, no network.
        2. If not found AND is_truncated(attr) → call search_value API.
        3. If still not found → leave value_id/value_ids unset.

        Args:
            attribute_value: The extracted value to annotate with Ozon dict ids.
            context: Extraction context with category_id and ozon_type_id.
            client_id: Ozon Client-Id for runtime lookups (falls back to env var).
            api_key: Ozon Api-Key for runtime lookups (falls back to env var).

        Returns:
            The same AttributeValue object, mutated in-place with value_id / value_ids.
        """
        cat_id = context.category_id
        type_id = context.ozon_type_id
        attr_id = attribute_value.attribute_id

        if type_id is None:
            return attribute_value

        truncated = is_truncated(cat_id, type_id, attr_id)

        async def _resolve_one(raw_value: str) -> Optional[int]:
            # 1. Static lookup
            vid = resolve_value_id(cat_id, type_id, attr_id, raw_value)
            if vid is not None:
                return vid
            # 2. Runtime fallback for truncated dictionaries
            if truncated:
                hit = await _runtime_search_value(
                    cat_id, type_id, attr_id, raw_value,
                    client_id=client_id, api_key=api_key,
                )
                if hit:
                    return hit["id"]
            return None

        if attribute_value.is_collection and isinstance(attribute_value.value, list):
            ids = [await _resolve_one(str(v)) for v in attribute_value.value]
            resolved = [i for i in ids if i is not None]
            if resolved:
                attribute_value.value_ids = resolved
        else:
            vid = await _resolve_one(str(attribute_value.value))
            if vid is not None:
                attribute_value.value_id = vid

        return attribute_value
