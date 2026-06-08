"""OzonStrategy — marketplace-specific логика для Ozon.

Использует ozon_dictionary.json (203MB, 9227 leaf type_id, 343356 характеристик)
для авторитетной схемы характеристик: normalize_target подмешивает метаданные
из словаря, validate_value проверяет банлист и (будущее) enum-значения.
"""
import logging
from typing import Any, Optional, Type
from .base import MarketplaceStrategy, ValidationResult
from app.services.enrichment.base import (
    AttributeValue, TargetAttribute, ExtractionContext, Source,
)
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    get_ozon_characteristics_for_category,
    get_ozon_category_name,
    get_attr_value_options,
    get_attr_value_pairs,
    get_attr_value_options_any_type,
    get_attr_value_pairs_any_type,
    resolve_value_id,
    is_truncated,
)
from app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup import (
    search_value as _runtime_search_value,
)
from app.services.enrichment.strategies.dictionaries.category_defaults import (
    CATEGORY_DEFAULTS,
    CATEGORY_CONDITIONAL_DEFAULTS,
)
from pydantic import BaseModel, Field, model_validator
from app.services.enrichment.prompt_router import classify_target

logger = logging.getLogger(__name__)

# Кэш LLM-резолвера хвоста value_id: переживает в рамках процесса.
# Ключ: (value_lower, hash(tuple(options))). Значение: выбранная allowed-строка
# ДОСЛОВНО или None (NONE). Один и тот же (value, options) повторяется десятки
# раз (Fully-Modular ×77) → уникальных пар мало, почти бесплатно.
_LLM_RESOLVE_CACHE: dict[tuple[str, int], Optional[str]] = {}


# Cross-fill rules: пары (id_a, id_b) — одно и то же значение под двумя именами.
# post_process_values копирует в ОБОИХ направлениях: если одна сторона заполнена,
# а другая пуста — заполняем пустую. Не перетирает уже заполненное.
_OZON_CROSSFILL_PAIRS: tuple[tuple[int, int], ...] = (
    (4381, 9024),   # Партномер ↔ Код продавца (MPN)
    (9048, 12141),  # Название модели (для объединения в одну карточку) ↔
                    # Название модели для шаблона наименования
    (9048, 9336),   # Название модели (для объединения) ↔ Модель/Марка
)

# R1: Атрибут «Объединить на одной карточке» (8292) — строится из бренд+модель.
# Если поле пусто ИЛИ содержит буквальный label поля (невалидно), деривируем.
_CARD_GROUP_ATTR_ID = 8292
# Стоп-слова категории/гендера, которые нужно срезать из product_name при деривации 8292
_CARD_GROUP_STOPWORDS: frozenset[str] = frozenset({
    "мужской", "мужская", "мужское", "мужские",
    "женский", "женская", "женское", "женские",
    "детский", "детская", "детское", "детские",
    "унисекс",
})

# Confidence для значений, проставленных детерминированно (CategoryDefaults / cross-fill).
_DEFAULT_CONFIDENCE = 0.85
# Confidence для conditional-defaults (стандарт формы-фактора — вероятный, но не гарантия).
_CONDITIONAL_DEFAULT_CONFIDENCE = 0.70


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


def _derive_card_group_value(
    product_name: str,
    brand: Optional[str],
    category_leaf: Optional[str],
) -> str:
    """Деривировать значение «Объединить на одной карточке» (8292): бренд + модельная строка.

    Алгоритм (generic — без хардкода категорий):
      1. Срезаем категорийный префикс (leaf name) — как в ozon_card_source._strip_category_prefix.
      2. Срезаем ведущие гендерные стоп-слова (_CARD_GROUP_STOPWORDS).
      3. Берём первые 5 токенов (бренд + модель, без спек-хвоста).

    Пример: Футболка мужская Nike Sportswear Club + leaf Футболка
      -> срезаем Футболка -> мужская Nike Sportswear Club
      -> срезаем мужская  -> Nike Sportswear Club
      -> первые 5 токенов -> Nike Sportswear Club
    """
    result = product_name.strip()

    # Шаг 1: срезаем категорийный префикс
    if category_leaf:
        low = result.lower()
        cat_low = category_leaf.strip().lower()
        if low.startswith(cat_low):
            result = result[len(cat_low):].strip()
        else:
            cat_words = {w for w in cat_low.split() if len(w) > 2}
            if cat_words:
                words = result.split()
                i = 0
                while i < len(words) and words[i].lower() in cat_words:
                    i += 1
                if i > 0:
                    result = " ".join(words[i:]).strip()

    # Шаг 2: срезаем ведущие гендерные стоп-слова
    tokens = result.split()
    i = 0
    while i < len(tokens) and tokens[i].lower() in _CARD_GROUP_STOPWORDS:
        i += 1
    if i > 0 and i < len(tokens):
        tokens = tokens[i:]

    # Шаг 3: первые 5 токенов = бренд + модельная строка
    result = " ".join(tokens[:5]).strip()

    # Если brand задан и не попал — prepend (защита от случаев где он выпал при срезе)
    if brand and brand.strip():
        b = brand.strip()
        if b.lower() not in result.lower():
            result = f"{b} {result}".strip()

    return result or product_name.strip()


# Порог fuzzy-схожести для анти-галлюцинационного гарда enum-маппинга.
# Та же логика, что в resolve_value_id (rapidfuzz WRatio ≥ 85): значение реально
# принадлежит списку только если лучший кандидат набирает >= порога. Ниже —
# значение НЕ из списка (напр. «хлопок» против [Акрил, Бязь]) → skip, не форс.
_ENUM_GUARD_FUZZY_THRESHOLD = 85


def _map_enum_value(
    raw_value: Any,
    canon_map: dict[str, str],
    allowed_list: list[str],
) -> Optional[str]:
    """Сопоставить извлечённое значение с каноническим allowed-значением.

    Возвращает каноническую строку из словаря ИЛИ None, если значение реально
    НЕ принадлежит списку (анти-галлюцинационный гард — не форсим ближайший enum).

    Стратегии (по убыванию строгости), та же логика что в resolve_value_id:
      1. Exact case-insensitive (через canon_map).
      2. Rapidfuzz WRatio на НОРМАЛИЗОВАННЫХ токенах (ё→е, латиница→кириллица,
         tech-aliases — как _normalize_token словаря). Принимаем лучший кандидат
         ТОЛЬКО при score >= порога.
           «Чёрный» vs «черный»            → norm-match 100 → маппится;
           «хлопковый» vs «Хлопок»          → высокий score → маппится;
           «хлопок» vs [Акрил, Бязь, ...]   → низкий score  → None (skip).
    """
    from app.services.enrichment.strategies.dictionaries.ozon_loader import _normalize_token

    if raw_value is None:
        return None
    key = str(raw_value).strip().lower()
    if not key:
        return None
    # Strategy 1: exact case-insensitive
    if key in canon_map:
        return canon_map[key]
    # Strategy 2: fuzzy на нормализованных токенах — только подлинного члена списка
    try:
        from rapidfuzz import fuzz, process
        norm_key = _normalize_token(key)
        # норм-индекс {normalized_allowed: canonical_allowed}
        norm_options = [(_normalize_token(a), a) for a in allowed_list]
        # exact-match после нормализации (ё→е и т.п.) — высшая уверенность
        for n, canon in norm_options:
            if n and n == norm_key:
                return canon
        best = process.extractOne(
            norm_key, [n for n, _ in norm_options], scorer=fuzz.WRatio
        )
        if best and best[1] >= _ENUM_GUARD_FUZZY_THRESHOLD:
            return norm_options[best[2]][1]
    except Exception as exc:
        logger.warning("enum-guard rapidfuzz failed: %s", exc)
    # Ниже порога → значение не из списка, не форсим ближайший → skip
    return None


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

        Для каждого enum-target с непустым allowed_values нормализуем извлечённое
        значение к канонической форме словаря. Анти-галлюцинационный гард: значение,
        которое реально НЕ принадлежит списку (низкая fuzzy-схожесть с лучшим
        кандидатом — напр. «хлопок» против [Акрил, Бязь, Полиэстер]) — НЕ форсится
        в ближайший enum, а ОТБРАСЫВАЕТСЯ (skip элемента / всего item). Это убирает
        retry-driven и token-level (strict) форс мусорного ближайшего значения.
        Гард общий для ВСЕХ источников (web_search/llm_knowledge/vision/description),
        т.к. они все строят модель через этот метод.
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

        # Индекс для case-insensitive нормализации: {attr_id: {lower_stripped_canonical: canonical}}
        # Позволяет найти точную каноническую строку из словаря по любому регистру/пробелу.
        _canonical_index: dict[int, dict[str, str]] = {
            attr_id: {v.strip().lower(): v for v in allowed}
            for attr_id, allowed in _constraints.items()
        }
        # Список allowed-строк на attr_id для fuzzy-гарда (порядок не важен).
        _allowed_list: dict[int, list[str]] = {
            attr_id: list(allowed) for attr_id, allowed in _constraints.items()
        }

        class _ConstrainedModel(base_model):  # type: ignore[valid-type]
            @model_validator(mode="after")
            def _enforce_allowed_values(self):
                # Извлекаем список _ExtractedAttr из поля (поддерживаем разные имена полей)
                field_name = (
                    "extracted" if getattr(self, "extracted", None) is not None
                    else "known_attributes" if getattr(self, "known_attributes", None) is not None
                    else None
                )
                if field_name is None:
                    return self
                items = getattr(self, field_name) or []

                kept_items = []
                for item in items:
                    attr_id = getattr(item, "attribute_id", None)
                    if attr_id not in _constraints:
                        kept_items.append(item)
                        continue
                    canon_map = _canonical_index[attr_id]
                    allowed_list = _allowed_list[attr_id]
                    raw_value = getattr(item, "value", None)
                    # Проверяем скаляр или список (is_collection)
                    if isinstance(raw_value, list):
                        normalized: list[str] = []
                        for v in raw_value:
                            canon = _map_enum_value(v, canon_map, allowed_list)
                            if canon is not None:
                                normalized.append(canon)
                            # else: значение не принадлежит списку — skip элемента
                            #       (не форсим ближайший enum = анти-галлюцинация)
                        if not normalized:
                            # все элементы отброшены → item целиком убираем
                            continue
                        item.value = normalized
                        kept_items.append(item)
                    else:
                        if raw_value is None:
                            kept_items.append(item)
                            continue
                        canon = _map_enum_value(raw_value, canon_map, allowed_list)
                        if canon is None:
                            # скаляр не принадлежит списку → отбрасываем весь item
                            # (не форсим ближайший enum = анти-галлюцинация)
                            continue
                        item.value = canon
                        kept_items.append(item)

                if len(kept_items) != len(items):
                    setattr(self, field_name, kept_items)
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

    def post_process_values(
        self,
        values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: Optional[ExtractionContext],
    ) -> list[AttributeValue]:
        """Zero-LLM дозаливка: category defaults + conditional defaults + cross-fill.

        Шаги:
        1. CATEGORY_DEFAULTS[context.category_id]: добавляем для каждого attr_id,
           который есть в targets и НЕ присутствует в values.
        2. CATEGORY_CONDITIONAL_DEFAULTS: если в товаре найден триггер-токен (через
           значение атрибута form_factor или product_name), проставляем стандартные
           значения для зависимых атрибутов (напр., габариты ATX-БП = 150×140×86 мм).
           Confidence ниже (0.70) — это «вероятный стандарт», judge/merge могут
           перекрыть лучшим источником.
        3. Cross-fill: для каждой пары (src, dst) из _OZON_CROSSFILL — если src
           заполнен и dst пуст, копируем value.

        Статичные правки идут с confidence=0.85, conditional — 0.70, всё с
        source=DESCRIPTION, чтобы pipeline мог их перетрясти judge при необходимости.
        """
        if context is None:
            return values

        target_ids: set[int] = {t.id for t in targets}
        existing_ids: set[int] = {v.attribute_id for v in values}
        extras: list[AttributeValue] = []

        # (A) CategoryDefaults — статичные значения категории.
        # Структура: {attr_id: (value_text, value_id_or_None)}.
        # Если value_id задан — пишем напрямую (минимум одна гарантия резолва);
        # иначе вызываем resolve_value_ids чтобы найти id через ozon_loader.
        cat_defaults = CATEGORY_DEFAULTS.get(context.category_id, {})
        for attr_id, default_payload in cat_defaults.items():
            if attr_id not in target_ids or attr_id in existing_ids:
                continue
            value_text, default_vid = default_payload
            av = AttributeValue(
                attribute_id=attr_id,
                value=value_text,
                confidence=_DEFAULT_CONFIDENCE,
                source=Source.DESCRIPTION,
                evidence="category default",
            )
            if default_vid is not None:
                av.value_id = default_vid
            else:
                # Fallback: пробуем найти value_id через словарь.
                self.resolve_value_ids(av, context)
            extras.append(av)
            existing_ids.add(attr_id)

        # (B) Conditional defaults — зависимые от другого атрибута / названия товара.
        # Пример: для блока питания форм-фактор ATX подразумевает стандартизированные
        # габариты 150×140×86 мм; если sources не извлекли длину/ширину/высоту, ставим.
        cond_rules = CATEGORY_CONDITIONAL_DEFAULTS.get(context.category_id, [])
        if cond_rules:
            # Собираем «поисковую строку» — значения form_factor атрибутов + product_name.
            # form_factor определяем по semantic_type у target — это работает для всех
            # категорий, не привязано к конкретному attr_id.
            form_factor_target_ids: set[int] = {
                t.id for t in targets if t.semantic_type == "form_factor"
            }
            search_tokens: list[str] = []
            for v in values:
                if v.attribute_id in form_factor_target_ids and v.value is not None:
                    if isinstance(v.value, list):
                        search_tokens.extend(str(x) for x in v.value)
                    else:
                        search_tokens.append(str(v.value))
            if context.product_name:
                search_tokens.append(context.product_name)
            search_blob = " ".join(search_tokens).upper()

            for kind, token, attr_defaults in cond_rules:
                if kind != "form_factor_contains":
                    continue
                if token.upper() not in search_blob:
                    continue
                # Первое совпавшее правило выигрывает — применяем и выходим.
                for attr_id, default_value in attr_defaults.items():
                    if attr_id not in target_ids or attr_id in existing_ids:
                        continue
                    extras.append(AttributeValue(
                        attribute_id=attr_id,
                        value=default_value,
                        confidence=_CONDITIONAL_DEFAULT_CONFIDENCE,
                        source=Source.DESCRIPTION,
                        evidence=f"conditional default for form_factor={token}",
                    ))
                    existing_ids.add(attr_id)
                break

        # (C) Cross-fill: для каждой пары (id_a, id_b) копируем в обе стороны.
        # values_by_id включает и оригинальные values, и extras добавленные на шагах A/B,
        # чтобы cross-fill видел атрибуты проставленные defaults.
        values_by_id: dict[int, AttributeValue] = {
            v.attribute_id: v for v in list(values) + extras
        }
        for id_a, id_b in _OZON_CROSSFILL_PAIRS:
            for src_id, dst_id in ((id_a, id_b), (id_b, id_a)):
                if dst_id in existing_ids or dst_id not in target_ids:
                    continue
                src_av = values_by_id.get(src_id)
                if src_av is None or src_av.value is None:
                    continue
                av = AttributeValue(
                    attribute_id=dst_id,
                    value=src_av.value,
                    confidence=src_av.confidence,
                    source=src_av.source,
                    evidence="cross-fill",
                )
                # Резолвим value_id для cross-fill тоже (если дубликат-атрибут enum).
                self.resolve_value_ids(av, context)
                extras.append(av)
                existing_ids.add(dst_id)
                # Обновляем индекс чтобы не заполнять dst ещё раз если пара встретится снова
                values_by_id[dst_id] = av

        # (D) Детерминированный дериватив для атрибута 8292
        # «Объединить на одной карточке» — уникальный ключ карточки = бренд + модель.
        # Заполняем если:
        #   - 8292 входит в targets
        #   - 8292 ещё не заполнен (existing_ids) ИЛИ заполнен невалидным placeholder-ом
        #     (label поля, например буквальный текст «Объединить на одной карточке»).
        if _CARD_GROUP_ATTR_ID in target_ids and context.product_name:
            # Находим target для 8292 чтобы получить name (label)
            card_group_target = next(
                (t for t in targets if t.id == _CARD_GROUP_ATTR_ID), None
            )
            field_label = (card_group_target.name if card_group_target else "").lower()
            existing_val: Optional[str] = None
            if _CARD_GROUP_ATTR_ID in existing_ids:
                existing_av = values_by_id.get(_CARD_GROUP_ATTR_ID)
                if existing_av and existing_av.value is not None:
                    existing_val = str(existing_av.value).strip()
            # Считаем невалидным: пусто ИЛИ значение совпадает с именем поля/label
            is_invalid = (
                existing_val is None
                or existing_val == ""
                or (field_label and existing_val.lower() == field_label)
            )
            if is_invalid:
                cat_leaf = context.category_path[-1] if context.category_path else None
                derived = _derive_card_group_value(
                    context.product_name, context.brand, cat_leaf
                )
                if derived:
                    av_8292 = AttributeValue(
                        attribute_id=_CARD_GROUP_ATTR_ID,
                        value=derived,
                        confidence=_DEFAULT_CONFIDENCE,
                        source=Source.DESCRIPTION,
                        evidence="deterministic: brand+model",
                    )
                    if _CARD_GROUP_ATTR_ID in existing_ids:
                        # Заменяем невалидный placeholder: убираем старый из extras и values
                        extras = [e for e in extras if e.attribute_id != _CARD_GROUP_ATTR_ID]
                        values = [v for v in values if v.attribute_id != _CARD_GROUP_ATTR_ID]
                    extras.append(av_8292)
                    existing_ids.add(_CARD_GROUP_ATTR_ID)
                    values_by_id[_CARD_GROUP_ATTR_ID] = av_8292

        return list(values) + extras

    def brand_value_options(
        self,
        attribute_id: int,
        context: ExtractionContext,
    ) -> list[str]:
        """Полный словарный список брендов для (cat_id, type_id, attribute_id).

        Бренд — огромный (часто truncated) enum, его allowed_values НЕ попадают в
        target.allowed_values. brand-from-name резолверу нужен ПОЛНЫЙ список — его
        и отдаёт get_attr_value_options (читает char['values'] из словаря).

        Когда ozon_type_id недоступен в контексте — перебирает ВСЕ type-записи
        категории через get_attr_value_options_any_type и возвращает значения из
        первой, у которой values непусты. get_ozon_characteristics_for_category
        НЕ подходит: она берёт первый type-entry, у которого values для «Бренд»
        могут быть пусты (values_truncated=True, значения не загружены).
        Без корректного fallback brand-from-name получает пустой список и не
        заполняет «Бренд» ни для одного реального продукта (drain B).
        """
        type_id = context.ozon_type_id
        if type_id is not None:
            return get_attr_value_options(context.category_id, type_id, attribute_id)
        # Fallback: ozon_type_id неизвестен — перебираем все type-entries.
        return get_attr_value_options_any_type(context.category_id, attribute_id)

    def brand_value_id_options(
        self,
        attribute_id: int,
        context: ExtractionContext,
    ) -> dict[str, int]:
        """Карта {бренд: словарный value_id} для (cat_id, type_id, attribute_id).

        brand-from-name резолвер выбрал бренд из имени, сматчив его против СТРОК
        brand_value_options; здесь даём ему те же словарные пары с id, чтобы
        привязать value_id точно (exact). Бренд — часто truncated enum, поэтому
        sync resolve_value_id мог не найти id в статическом словаре — а здесь id
        берётся ровно для того entry, который уже в словаре есть.

        Когда ozon_type_id недоступен — перебирает ВСЕ type-записи через
        get_attr_value_pairs_any_type (аналогично brand_value_options).
        Без корректного fallback value_id остаётся None → drain C.
        """
        type_id = context.ozon_type_id
        if type_id is not None:
            return get_attr_value_pairs(context.category_id, type_id, attribute_id)
        # Fallback: ozon_type_id неизвестен — перебираем все type-entries.
        return get_attr_value_pairs_any_type(context.category_id, attribute_id)

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

    async def llm_resolve_tail(
        self,
        values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """LLM-резолвер ХВОСТА нерезолвнутых value_id (семантика/перевод).

        Запускается ПОСЛЕ детерминированного resolve_value_ids (он остаётся первым
        и не меняется). Закрывает то, что не добил exact→fuzzy→vector→лемма:
        Fully-Modular→Модульный, EVA→ЭВА, Круглогодичный→На любой сезон.

        Алгоритм (один батч-вызов на товар, НЕ per-value):
        1. Собираем enum-значения, у которых value_id/value_ids ещё None, вместе
           с их allowed-списком из словаря (полным, не truncated target.allowed_values).
        2. Кэш по (value_lower, hash(options)) — повторы бесплатны, второго вызова нет.
        3. Один LLM-вызов: для каждого (value, allowed) вернуть индекс выбранного
           allowed ДОСЛОВНО либо -1 (NONE). temperature 0, дешёвая модель.
        4. Замапить выбранную строку обратно в value_id через resolve_value_id
           (exact-путь словаря). NONE/ошибка/таймаут → оставляем None как сейчас.

        Только ДОБАВЛЯЕТ резолв — хуже не делает (value-строка уже есть).
        """
        cat_id = context.category_id
        type_id = context.ozon_type_id
        if type_id is None:
            return values

        # enum-targets с непустым словарём allowed
        enum_target_ids = {
            t.id for t in targets if classify_target(t) == "enum"
        }

        # Собираем задачи: (av, raw_value, options, list_index_or_None, cache_key)
        # list_index_or_None: индекс элемента в массиве (is_collection) или None для скаляра.
        tasks: list[tuple] = []
        # Уникальные (value, options) для батча — дедуп через кэш + within-batch.
        to_ask: list[tuple[str, list[str]]] = []
        seen_keys: set[tuple[str, int]] = set()

        for av in values:
            if av.attribute_id not in enum_target_ids:
                continue
            options = get_attr_value_options(cat_id, type_id, av.attribute_id)
            if not options:
                continue
            opt_hash = hash(tuple(options))

            def _collect(raw_value: str, list_idx: Optional[int]):
                key = (raw_value.lower(), opt_hash)
                tasks.append((av, raw_value, options, list_idx, key))
                if key not in _LLM_RESOLVE_CACHE and key not in seen_keys:
                    seen_keys.add(key)
                    to_ask.append((raw_value, options))

            if av.is_collection and isinstance(av.value, list):
                # резолвим только хвост: элементы без покрытия в value_ids
                resolved_count = len(av.value_ids or [])
                if resolved_count >= len([v for v in av.value if v]):
                    continue
                for i, v in enumerate(av.value):
                    if v is None or str(v) == "":
                        continue
                    _collect(str(v), i)
            else:
                if av.value_id is not None:
                    continue
                if av.value is None or str(av.value) == "":
                    continue
                _collect(str(av.value), None)

        if not tasks:
            return values

        # Один батч-LLM-вызов на уникальные (value, options), которых нет в кэше.
        if to_ask:
            await self._llm_batch_choose(to_ask)

        # Маппинг результатов обратно в value_id / value_ids.
        for av, raw_value, options, list_idx, key in tasks:
            chosen = _LLM_RESOLVE_CACHE.get(key)
            if not chosen:
                continue
            vid = resolve_value_id(cat_id, type_id, av.attribute_id, chosen)
            if vid is None:
                continue
            if list_idx is None:
                av.value_id = vid
            else:
                existing = list(av.value_ids or [])
                if vid not in existing:
                    existing.append(vid)
                av.value_ids = existing

        return values

    async def _llm_batch_choose(
        self,
        to_ask: list[tuple[str, list[str]]],
    ) -> None:
        """Один батч-LLM-вызов: для каждого (value, allowed) выбрать индекс или -1.

        Записывает результат в _LLM_RESOLVE_CACHE по ключу (value_lower, hash(opts)).
        Graceful: любая ошибка/таймаут/None → оставляем None в кэше (как [MATCHING FAILED]).
        """
        from app.services.providers.factory import get_main_manager

        # Строим компактный нумерованный список заданий для промпта.
        lines: list[str] = []
        for i, (value, options) in enumerate(to_ask):
            opts_block = "; ".join(f"{j}={o}" for j, o in enumerate(options))
            lines.append(f"#{i} value={value!r} | allowed: {opts_block}")
        tasks_text = "\n".join(lines)

        class _Choice(BaseModel):
            task: int = Field(..., description="Номер задания (#N)")
            index: int = Field(
                ...,
                description="Индекс выбранного allowed-значения, или -1 если ни одно не подходит",
            )

        class _BatchResponse(BaseModel):
            choices: list[_Choice] = Field(default_factory=list)

        system_prompt = (
            "Ты сопоставляешь значение характеристики товара со словарём допустимых "
            "значений маркетплейса. Для КАЖДОГО задания верни индекс РОВНО одного "
            "значения из его списка allowed, которое по смыслу соответствует value "
            "(учитывай переводы и синонимы: Fully-Modular=Модульный, EVA=ЭВА, "
            "Круглогодичный=На любой сезон). Если НИ ОДНО значение не подходит — "
            "верни index=-1. При любом сомнении возвращай -1: лучше -1, чем неверный "
            "выбор. Не придумывай значений вне списка allowed."
        )
        user_text = (
            "Задания (для каждого выбери index из его allowed ИЛИ -1):\n"
            f"{tasks_text}\n\n"
            "Верни choices: по одному объекту {task, index} на каждое задание."
        )

        # Предзаполняем кэш None — graceful default, если LLM ничего не вернёт.
        for value, options in to_ask:
            _LLM_RESOLVE_CACHE.setdefault((value.lower(), hash(tuple(options))), None)

        try:
            llm = get_main_manager()
            parsed, _ = await llm.structured_request(
                system_prompt=system_prompt,
                user_text=user_text,
                response_model=_BatchResponse,
            )
        except Exception as e:
            logger.warning("[Ozon] LLM tail value_id resolver failed: %s", e)
            return

        if parsed is None:
            return

        for ch in parsed.choices:
            if ch.task < 0 or ch.task >= len(to_ask):
                continue
            value, options = to_ask[ch.task]
            key = (value.lower(), hash(tuple(options)))
            if 0 <= ch.index < len(options):
                # Записываем ДОСЛОВНУЮ allowed-строку (гард: индекс валиден).
                _LLM_RESOLVE_CACHE[key] = options[ch.index]
            else:
                _LLM_RESOLVE_CACHE[key] = None
