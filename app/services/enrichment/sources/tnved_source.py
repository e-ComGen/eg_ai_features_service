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

        # 2. Пропустить если уже заполнен с высокой уверенностью
        if already_filled:
            for av in already_filled:
                if av.attribute_id == tnved_target.id and av.is_confident():
                    logger.debug("[TnvedSource] attr %s already filled (confident)", tnved_target.id)
                    return []

        # 3. Резолв с double-checked locking по (category_id, type_key)
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
