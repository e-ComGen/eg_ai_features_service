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


# Sentinel: «справочник Ozon искали (есть креды+type_id), кода НЕТ» → abstain.
# Отличается от None («не смогли проверить — отдать код как есть»).
# Источник истины для ТН ВЭД — per-category справочник Ozon (подтверждён
# dict-backed живым list_values), а НЕ эвристика по форме кода: 9206000000 —
# формально валидный заголовок 9206, но Ozon его не принимает для категорий, где
# его нет в справочнике. Только справочник решает.
_ABSTAIN = object()


# Маркеры снятых с действия / недействующих записей словаря ТН ВЭД Ozon.
# Ярлык вида "6203423100 - (Действие прекращено с 15.09.2024) … брюки … из
# денима" — код семантически верный, но запись депрекейтнута: Ozon её отвергнет
# (Баг 4 eg_importer). Отфильтровываем такие записи из constrained-pick и из
# dict-валидации, чтобы не отдать value_id недействующей записи.
_DEPRECATED_LABEL_RE = re.compile(
    r"действие\s+(?:прекращено|приостановлено)"
    r"|прекращ[еён]+о\s+(?:с\s+)?\d"
    r"|утратил[аои]?\s+силу"
    r"|исключ[еён]+",
    re.IGNORECASE,
)


def _is_deprecated_dict_label(value: str) -> bool:
    """True, если ярлык словаря помечен как снятый с действия/недействующий."""
    return bool(_DEPRECATED_LABEL_RE.search(value or ""))


# ---------------------------------------------------------------------------
# Structured response model для LLM-запроса
# ---------------------------------------------------------------------------

class _TnvedResponse(BaseModel):
    code: str = Field(..., description="10-значный код ТН ВЭД ЕАЭС, только цифры")


# ---------------------------------------------------------------------------
# Judge — принимает только валидный 10-значный код
# ---------------------------------------------------------------------------

class TnvedJudge(LlmJudge):
    """Детерминированный judge: ведущий код значения — ровно 10 цифр.

    Значение может быть голым кодом ("9206000000") ИЛИ полным ярлыком словаря
    Ozon ("9206000000 - Инструменты музыкальные ударные…"). Проверяем ВЕДУЩИЙ
    10-значный код, а не все цифры строки (в описании ярлыка тоже бывают цифры,
    напр. «…6103…» — иначе судья ложно отклонял бы валидный ярлык)."""
    source = Source.LLM_KNOWLEDGE

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        m = re.match(r"\s*(\d{10})(?:\D|$)", str(value.value))
        return bool(m)


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
        code = await self._resolve_for_category(cat_id, context, tnved_target.id)

        if code is None:
            logger.debug("[TnvedSource] category %s: could not resolve ТН ВЭД", cat_id)
            return []

        # 3a. СЛОВАРНАЯ валидация (источник истины — per-category справочник ТН ВЭД
        # Ozon, подтверждён dict-backed живым list_values). Ozon для словарного поля
        # принимает ПОЛНЫЙ ярлык словаря ("9206000000 - Инструменты музыкальные
        # ударные…") + value_id, а НЕ голый код — иначе «некорректное значение».
        #   (value, id) → код есть в справочнике → отдаём ПОЛНЫЙ ярлык + value_id;
        #   _ABSTAIN    → креды+type есть, кода НЕТ → не подставляем (no_data, честнее);
        #   None        → проверить нечем (нет кред/type/API-сбой) → отдаём код как есть.
        validated = await self._validate_against_ozon_dict(
            code, tnved_target.id, context,
        )
        if validated is _ABSTAIN:
            logger.info(
                "[TnvedSource] code %s НЕ в справочнике ТН ВЭД Ozon для cat=%s "
                "→ abstain (Ozon отклонил бы; no_data честнее)", code, cat_id,
            )
            return []

        if isinstance(validated, tuple):
            out_value, out_value_id = validated  # полный ярлык словаря + value_id
            confirmed = True
        else:
            out_value, out_value_id = code, None  # справочник недоступен — голый код
            confirmed = False

        return [
            AttributeValue(
                attribute_id=tnved_target.id,
                value=out_value,
                value_id=out_value_id,
                confidence=0.90,
                source=Source.LLM_KNOWLEDGE,
                # "tnved_resolver:" prefix is a machine-readable sentinel used by
                # the TNVED_SOURCE_FIX_ENABLED filter in pipeline._finalize_async
                # to distinguish TnvedSource fills from garbage LLM_KNOWLEDGE/
                # WEB_SEARCH codes that must be dropped.
                evidence=(
                    "tnved_resolver: ТН ВЭД ЕАЭС, "
                    + ("подтверждён справочником Ozon (полный ярлык+value_id)" if confirmed
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

        Справочник Ozon отдаёт значения в форме "КОД - описание" (напр.
        "3926200000 - Одежда..."), поэтому сверяем код с ВЕДУЩИМИ цифрами значения,
        а не со всеми (в описании тоже есть цифры). ТН ВЭД подтверждён dict-backed
        живым list_values, поэтому «не найдено» = реально нет в справочнике.

        Returns:
          (value, id) — код есть в справочнике: ПОЛНЫЙ ярлык словаря Ozon
                        ("9206000000 - Инструменты музыкальные ударные…") + value_id.
                        Ozon принимает именно ярлык, а не голый код;
          _ABSTAIN    — креды+type есть, поиск выполнен, кода НЕТ → не подставлять;
          None        — проверить нечем (нет кред / type_id / API-сбой) → отдать как есть.
        """
        type_id = context.ozon_type_id
        if type_id is None or not (os.getenv("OZON_CLIENT_ID") and os.getenv("OZON_API_KEY")):
            return None  # нечем валидировать — отдаём код как есть
        try:
            from app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup import (
                search_value,
            )
            hit = await search_value(context.category_id, type_id, attr_id, query=code)
        except Exception as exc:  # pragma: no cover — сетевой сбой не должен ронять стейдж
            logger.warning("[TnvedSource] dict-валидация ошибка для %s: %s", code, exc)
            return None
        if hit:
            hit_value = str(hit.get("value", ""))
            # Снятая с действия запись (Баг 4): код верный, но Ozon её отвергнет —
            # не отдаём value_id недействующей записи. Abstain честнее мусора.
            if _is_deprecated_dict_label(hit_value):
                logger.info(
                    "[TnvedSource] код %s резолвится в СНЯТУЮ с действия запись словаря "
                    "(%s) → abstain", code, hit_value[:80],
                )
                return _ABSTAIN
            # Ведущий код значения: "9206000000 - Инструменты…" → "9206000000".
            lead = re.match(r"\s*(\d{6,10})", hit_value)
            if lead and lead.group(1) == code:
                # ПОЛНЫЙ ярлык словаря + value_id — то, что принимает Ozon.
                return (hit_value, hit.get("id"))
        return _ABSTAIN  # dict-backed, поиск выполнен, точного кода нет → abstain

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
        attr_id: int = 0,
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
            code = await self._call_llm(context, attr_id)
            self._cache[cache_key] = code
            if code:
                context.llm_calls_so_far += 1
                logger.info("[TnvedSource] key %s → ТН ВЭД %s", cache_key, code)
            return code

    async def _call_llm(self, context: ExtractionContext, attr_id: int) -> Optional[str]:
        """Один сфокусированный LLM-вызов. Возвращает 10-значный код или None.

        Если доступен per-category справочник ТН ВЭД Ozon (есть type_id+креды) —
        делаем CONSTRAINED-PICK: даём LLM реальные ярлыки словаря и просим выбрать
        ОДИН наиболее точный. Это убирает класс ошибок «валидный, но не тот подкод»
        (напр. для кроссовок слепой guess давал 6404199000 «прочая обувь» вместо
        6404110000 «спортивная обувь» — оба в словаре, но второй правильный).
        Фоллбэк — слепой guess, когда справочник недоступен (offline/eval).
        """
        dict_labels = await self._fetch_dict_labels(context, attr_id)
        if dict_labels:
            picked = await self._constrained_pick(context, dict_labels)
            if picked:
                return picked
            logger.info(
                "[TnvedSource] constrained-pick не дал кода для cat=%s → fallback на слепой guess",
                context.category_id,
            )

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

    async def _fetch_dict_labels(
        self, context: ExtractionContext, attr_id: int
    ) -> list[dict]:
        """Per-category справочник ТН ВЭД Ozon (список {id, value}) или [].

        Возвращает [] когда нечем тянуть (нет type_id / кред / API-сбой) —
        тогда _call_llm уходит в слепой guess.
        """
        type_id = context.ozon_type_id
        if type_id is None or not (os.getenv("OZON_CLIENT_ID") and os.getenv("OZON_API_KEY")):
            return []
        try:
            from app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup import (
                list_values,
            )
            raw = await list_values(context.category_id, type_id, attr_id, max_values=300)
        except Exception as exc:  # pragma: no cover — сетевой сбой не должен ронять стейдж
            logger.warning("[TnvedSource] list_values ошибка для cat=%s: %s", context.category_id, exc)
            return []
        # Отсеять снятые с действия записи (Баг 4): чтобы constrained-pick не мог
        # выбрать депрекейтнутый код — Ozon отвергнет его value_id.
        active = [it for it in raw if not _is_deprecated_dict_label(str(it.get("value", "")))]
        dropped = len(raw) - len(active)
        if dropped:
            logger.info(
                "[TnvedSource] отфильтровано %d снятых с действия ярлыков ТН ВЭД для cat=%s",
                dropped, context.category_id,
            )
        return active

    async def _constrained_pick(
        self, context: ExtractionContext, labels: list[dict]
    ) -> Optional[str]:
        """Выбрать ОДИН код из реальных ярлыков словаря категории.

        labels: [{"id": int, "value": "КОД - описание"}]. Возвращает 10-значный
        ведущий код выбранного ярлыка (если он реально присутствует в списке),
        иначе None (→ фоллбэк на слепой guess).
        """
        # Множество допустимых ведущих кодов словаря (для верификации выбора).
        allowed_codes: set[str] = set()
        lines: list[str] = []
        for it in labels:
            val = str(it.get("value", ""))
            lead = re.match(r"\s*(\d{6,10})", val)
            if lead:
                allowed_codes.add(lead.group(1))
            lines.append(val)
        if not allowed_codes:
            return None

        category_hint = " / ".join(context.category_path) if context.category_path else ""
        user_text = (
            f'Товар: "{context.product_name}".'
            + (f'\nКатегория: {category_hint}.' if category_hint else "")
            + "\n\nДопустимые коды ТН ВЭД ЕАЭС для этой категории (выбери РОВНО ОДИН, "
              "наиболее точный для товара):\n"
            + "\n".join(lines)
            + "\n\nВерни ТОЛЬКО ведущий 10-значный код выбранной строки, без описания."
        )
        try:
            parsed, _ = await self._llm.structured_request(
                system_prompt=(
                    "Ты эксперт по таможенной классификации ЕАЭС. Тебе дан товар и "
                    "ЗАКРЫТЫЙ список допустимых кодов ТН ВЭД его категории. Выбери из "
                    "списка ОДИН код, максимально точно соответствующий товару, "
                    "предпочитая НАИБОЛЕЕ СПЕЦИФИЧНУЮ подкатегорию (например, для "
                    "спортивной обуви — код спортивной обуви, а не «прочая обувь»). "
                    "В поле code верни только 10 цифр выбранного кода."
                ),
                user_text=user_text,
                response_model=_TnvedResponse,
            )
        except Exception as e:
            logger.warning("[TnvedSource] constrained-pick LLM call failed: %s", e)
            return None

        if parsed is None:
            return None
        digits = re.sub(r"\D", "", parsed.code)
        if len(digits) == 10 and digits in allowed_codes:
            return digits
        logger.info(
            "[TnvedSource] constrained-pick вернул %r (digits=%r) — нет в словаре, отбрасываю",
            getattr(parsed, "code", None), digits,
        )
        return None

    def get_judge(self) -> LlmJudge:
        return self._judge
