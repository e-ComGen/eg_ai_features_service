"""TnvedSource — per-(category, type) резолвер ТН ВЭД кода ЕАЭС.

ТН ВЭД код зависит от типа товара (type_id), а не только от категории.
Например, футболка (6109) и джинсы (6203) могут жить в одной category_id
Ozon, но иметь разные 10-значные коды ТН ВЭД.

Кэш ключуется по (category_id, type_id) — если type_id отсутствует,
фоллбэк к последнему элементу category_path (имя листовой категории).

Паттерн: double-checked locking (per-key asyncio.Lock) гарантирует,
что при 8 параллельных товарах одного (cat, type) LLM вызывается ровно 1 раз.

Source: LLM_KNOWLEDGE — ТН ВЭД — знание о категории товара, которое LLM
хорошо знает из обучающих данных (таможенные классификаторы публичны и
хорошо представлены в обучающем корпусе). Web_search здесь избыточен:
ТН ВЭД коды стабильны и не требуют актуальных данных.
"""
from __future__ import annotations

import asyncio
import os
import re
import logging
from typing import Optional, Tuple

from pydantic import BaseModel, Field

from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)
from app.services.providers.factory import get_main_manager

logger = logging.getLogger(__name__)


# Заголовок ТАМОЖЕННОЙ группы (4 значащих + 6 нулей, напр. 9206000000) — это НЕ
# валидный код для декларирования (Ozon отклоняет). LLM часто отдаёт именно его,
# когда «угадывает» по группе. Детерминированно отбрасываем.
_GROUP_HEADER_RE = re.compile(r"^\d{4}0{6}$")


def _is_group_header_code(code: str) -> bool:
    """True если код — заголовок группы (NNNN000000), не валидный для декларации."""
    return bool(_GROUP_HEADER_RE.match(code))


# Sentinel: «справочник Ozon искали (есть креды+type_id), кода НЕТ» → abstain.
# Отличается от None («не смогли проверить — отдать как есть»).
_ABSTAIN = object()


# ---------------------------------------------------------------------------
# Structured response model для LLM-запроса
# ---------------------------------------------------------------------------

class _TnvedResponse(BaseModel):
    code: str = Field(..., description="10-значный код ТН ВЭД ЕАЭС, только цифры")


# ---------------------------------------------------------------------------
# Judge — принимает только валидный 10-значный код
# ---------------------------------------------------------------------------

class TnvedJudge(LlmJudge):
    """Простой детерминированный judge: принимает только ровно 10 цифр."""
    source = Source.LLM_KNOWLEDGE

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        digits = re.sub(r"\D", "", str(value.value))
        return len(digits) == 10


# ---------------------------------------------------------------------------
# TnvedSource
# ---------------------------------------------------------------------------

class TnvedSource(AttributeSource):
    """Per-category резолвер ТН ВЭД с кэшем и per-category asyncio.Lock.

    is_applicable: только для target с «ТН ВЭД» в имени (case-insensitive).
    extract: один LLM-вызов на категорию; результат кэшируется в self._cache.
    """

    def __init__(self) -> None:
        self._llm = get_main_manager()
        self._judge = TnvedJudge()
        # Кэш: (category_id, type_key) → код (str) или None (не удалось резолвить)
        # type_key = ozon_type_id если задан, иначе листовой элемент category_path
        self._cache: dict[Tuple[int, Optional[object]], Optional[str]] = {}
        # Per-key lock для double-checked locking
        self._locks: dict[Tuple[int, Optional[object]], asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # защищает создание self._locks[key]

    @property
    def source_type(self) -> Source:
        return Source.LLM_KNOWLEDGE

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """True только если в имени target есть 'ТН ВЭД' (case-insensitive)."""
        return "тн вэд" in target.name.lower()

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        # 1. Найти ТН ВЭД target
        tnved_target: Optional[TargetAttribute] = None
        for t in targets:
            if self.is_applicable(context, t):
                tnved_target = t
                break

        if tnved_target is None:
            return []

        # 2. Пропустить только если уже заполнен САМИМ TnvedSource (sentinel prefix).
        # Заполнения от LLM_KNOWLEDGE/WEB_SEARCH без "tnved_resolver:" в evidence —
        # мусор (угадывают абсурдные коды), они будут дропнуты в pipeline._finalize_async.
        # TnvedSource ДОЛЖЕН запуститься независимо от таких «уверенных» мусорных значений.
        if already_filled:
            for av in already_filled:
                if (
                    av.attribute_id == tnved_target.id
                    and (av.evidence or "").startswith("tnved_resolver:")
                ):
                    logger.debug(
                        "[TnvedSource] attr %s already filled by TnvedSource — skip",
                        tnved_target.id,
                    )
                    return []

        # 3. Резолв с double-checked locking по (category_id, type_key)
        cat_id = context.category_id
        code = await self._resolve_for_category(cat_id, context)

        if code is None:
            logger.debug("[TnvedSource] category %s: could not resolve ТН ВЭД", cat_id)
            return []

        # 3a. Гард заголовка группы: 9206000000 и подобные — Ozon отклоняет.
        if _is_group_header_code(code):
            logger.info(
                "[TnvedSource] code %s — заголовок группы, невалиден для декларации "
                "→ abstain (пусто честнее отклонёнки)", code,
            )
            return []

        # 3b. Сверка с справочником ТН ВЭД Ozon — ПОЗИТИВНАЯ (безопасная):
        # код найден → вешаем авторитетный value_id из справочника. Не найден /
        # справочник недоступен → отдаём код как есть (гард группы уже прошёл).
        # Strict-abstain «нет в справочнике → не заполнять» НЕ включаем здесь: пока
        # не подтверждено живым вызовом, что ТН ВЭД у Ozon dict-backed (иначе 404
        # на не-словарном атрибуте убил бы ВСЕ коды). См. TODO strict-mode.
        validated_id = await self._validate_against_ozon_dict(
            code, tnved_target.id, context,
        )

        return [
            AttributeValue(
                attribute_id=tnved_target.id,
                value=code,
                value_id=validated_id if isinstance(validated_id, int) else None,
                confidence=0.90,
                source=Source.LLM_KNOWLEDGE,
                # "tnved_resolver:" prefix is a machine-readable sentinel used by
                # the TNVED_SOURCE_FIX_ENABLED filter in pipeline._finalize_async
                # to distinguish TnvedSource fills (trusted, validated 10-digit code)
                # from garbage LLM_KNOWLEDGE/WEB_SEARCH codes that must be dropped.
                evidence=(
                    "tnved_resolver: ТН ВЭД ЕАЭС, "
                    + ("подтверждён справочником Ozon" if isinstance(validated_id, int)
                       else "резолв по категории (справочник недоступен)")
                ),
            )
        ]

    async def _validate_against_ozon_dict(
        self,
        code: str,
        attr_id: int,
        context: ExtractionContext,
    ):
        """Сверить код с справочником ТН ВЭД Ozon.

        Returns:
          int   — код подтверждён справочником (авторитетный value_id);
          None  — не подтверждён (нет кред / type_id / API-сбой / не найден) →
                  отдать код как есть (гард группы уже отсёк заголовки).
        TODO strict-mode: когда живым вызовом подтвердим, что ТН ВЭД dict-backed,
        включить abstain при «creds+type есть, но кода нет» (вернуть _ABSTAIN).
        """
        type_id = context.ozon_type_id
        if type_id is None or not (os.getenv("OZON_CLIENT_ID") and os.getenv("OZON_API_KEY")):
            return None  # нечем валидировать — отдаём (гард уже прошёл)
        try:
            from app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup import (
                search_value,
            )
            hit = await search_value(context.category_id, type_id, attr_id, query=code)
        except Exception as exc:  # pragma: no cover — сетевой сбой не должен ронять стейдж
            logger.warning("[TnvedSource] dict-валидация ошибка для %s: %s", code, exc)
            return None
        if hit:
            hit_digits = re.sub(r"\D", "", str(hit.get("value", "")))
            if code == hit_digits or code in hit_digits:
                return hit.get("id")  # подтверждён + value_id из справочника
        return None  # не подтверждён — отдаём код как есть (safe; strict-mode TODO)

    @staticmethod
    def _make_cache_key(
        cat_id: int,
        context: ExtractionContext,
    ) -> Tuple[int, Optional[object]]:
        """Build a cache key that is unique per (category, product-type).

        Uses ozon_type_id when available; falls back to the leaf element of
        category_path so that garments that differ only by type_id (e.g. a
        t-shirt vs jeans sharing the same Ozon category_id) each get their
        own LLM call and their own cached result.
        """
        type_key: Optional[object] = context.ozon_type_id
        if type_key is None and context.category_path:
            type_key = context.category_path[-1]
        return (cat_id, type_key)

    async def _resolve_for_category(
        self,
        cat_id: int,
        context: ExtractionContext,
    ) -> Optional[str]:
        """Double-checked locking: резолвим ровно 1 раз на (category_id, type_key)."""
        cache_key = self._make_cache_key(cat_id, context)

        # Первая проверка кэша (без lock)
        if cache_key in self._cache:
            logger.debug("[TnvedSource] cache hit for key %s", cache_key)
            return self._cache[cache_key]

        # Получаем или создаём per-key lock (под общим locks_lock)
        async with self._locks_lock:
            if cache_key not in self._locks:
                self._locks[cache_key] = asyncio.Lock()
            cat_lock = self._locks[cache_key]

        # Захватываем per-key lock
        async with cat_lock:
            # Вторая проверка кэша (под lock) — классический double-checked locking
            if cache_key in self._cache:
                logger.debug("[TnvedSource] cache hit (after lock) for key %s", cache_key)
                return self._cache[cache_key]

            # Выполняем LLM-вызов
            code = await self._call_llm(context)
            self._cache[cache_key] = code
            if code:
                context.llm_calls_so_far += 1
                logger.info("[TnvedSource] key %s → ТН ВЭД %s", cache_key, code)
            return code

    async def _call_llm(self, context: ExtractionContext) -> Optional[str]:
        """Один сфокусированный LLM-вызов. Возвращает 10-значный код или None."""
        category_hint = " / ".join(context.category_path) if context.category_path else ""
        user_text = (
            f'Определи 10-значный код ТН ВЭД ЕАЭС для товара: "{context.product_name}".'
        )
        if category_hint:
            user_text += f'\nКатегория: {category_hint}.'
        user_text += "\nВерни ТОЛЬКО код (10 цифр), без пояснений."

        try:
            parsed, _ = await self._llm.structured_request(
                system_prompt=(
                    "Ты эксперт по таможенной классификации ЕАЭС. "
                    "Тебе дано название товара. Верни 10-значный код ТН ВЭД ЕАЭС "
                    "для этого товара. В поле code — только 10 цифр без пробелов и тире."
                ),
                user_text=user_text,
                response_model=_TnvedResponse,
            )
        except Exception as e:
            logger.warning("[TnvedSource] LLM call failed: %s", e)
            return None

        if parsed is None:
            return None

        # Валидация: ровно 10 цифр
        digits = re.sub(r"\D", "", parsed.code)
        if len(digits) == 10:
            return digits

        logger.warning("[TnvedSource] invalid ТН ВЭД response: %r (digits=%r)", parsed.code, digits)
        return None

    def get_judge(self) -> LlmJudge:
        return self._judge
