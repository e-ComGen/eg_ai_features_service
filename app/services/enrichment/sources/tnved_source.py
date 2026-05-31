"""TnvedSource — per-category резолвер ТН ВЭД кода ЕАЭС.

ТН ВЭД код — категорийная константа: у всех товаров одной категории Ozon код одинаков.
Поэтому один раз резолвим через LLM и кэшируем по category_id для всего батча.

Паттерн: double-checked locking (per-category asyncio.Lock) гарантирует,
что при 8 параллельных товарах одной категории LLM вызывается ровно 1 раз.

Source: LLM_KNOWLEDGE — ТН ВЭД — знание о категории товара, которое LLM
хорошо знает из обучающих данных (таможенные классификаторы публичны и
хорошо представлены в обучающем корпусе). Web_search здесь избыточен:
ТН ВЭД коды стабильны и не требуют актуальных данных.
"""
from __future__ import annotations

import asyncio
import re
import logging
from typing import Optional

from pydantic import BaseModel, Field

from app.services.enrichment.base import (
    AttributeSource, AttributeValue, TargetAttribute, ExtractionContext,
    Source, LlmJudge,
)
from app.services.providers.factory import get_main_manager

logger = logging.getLogger(__name__)


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
        # Кэш: category_id → код (str) или None (не удалось резолвить)
        self._cache: dict[int, Optional[str]] = {}
        # Per-category lock для double-checked locking
        self._locks: dict[int, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()  # защищает создание self._locks[id]

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

        # 2. Пропустить если уже заполнен с высокой уверенностью
        if already_filled:
            for av in already_filled:
                if av.attribute_id == tnved_target.id and av.is_confident():
                    logger.debug("[TnvedSource] attr %s already filled (confident)", tnved_target.id)
                    return []

        # 3. Резолв с double-checked locking по category_id
        cat_id = context.category_id
        code = await self._resolve_for_category(cat_id, context)

        if code is None:
            logger.debug("[TnvedSource] category %s: could not resolve ТН ВЭД", cat_id)
            return []

        return [
            AttributeValue(
                attribute_id=tnved_target.id,
                value=code,
                confidence=0.90,
                source=Source.LLM_KNOWLEDGE,
                evidence="ТН ВЭД ЕАЭС, резолв по категории (кэш)",
            )
        ]

    async def _resolve_for_category(
        self,
        cat_id: int,
        context: ExtractionContext,
    ) -> Optional[str]:
        """Double-checked locking: резолвим ровно 1 раз на category_id."""
        # Первая проверка кэша (без lock)
        if cat_id in self._cache:
            logger.debug("[TnvedSource] cache hit for category %s", cat_id)
            return self._cache[cat_id]

        # Получаем или создаём per-category lock (под общим locks_lock)
        async with self._locks_lock:
            if cat_id not in self._locks:
                self._locks[cat_id] = asyncio.Lock()
            cat_lock = self._locks[cat_id]

        # Захватываем per-category lock
        async with cat_lock:
            # Вторая проверка кэша (под lock) — классический double-checked locking
            if cat_id in self._cache:
                logger.debug("[TnvedSource] cache hit (after lock) for category %s", cat_id)
                return self._cache[cat_id]

            # Выполняем LLM-вызов
            code = await self._call_llm(context)
            self._cache[cat_id] = code
            if code:
                context.llm_calls_so_far += 1
                logger.info("[TnvedSource] category %s → ТН ВЭД %s", cat_id, code)
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
