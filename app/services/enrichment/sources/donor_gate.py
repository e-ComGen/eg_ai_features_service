"""donor_gate — LLM-шлагбаум «тот же товар или нет».

Проверяет, что карточка-донор описывает тот же товар что и цель, до того как
данные донора попадут в pipeline. Ловит ложные совпадения:

  Osprey Farpoint 40  → донор Osprey Daylite 13л  → DIFFERENT
  Tide 3кг            → донор 6кг                  → DIFFERENT
  70mai A500S         → донор A50                  → DIFFERENT
  Nike Футболка синяя → донор Nike Футболка красная → SAME (цвет ≠ разная модель)

Правила промпта:
  SAME:      разный цвет / разный год / разная комплектация /
             разный объём-упаковки в описании (Tide 3кг vs 6кг — DIFFERENT) /
             варианты с суффиксом (S/M/L) при одинаковой базе.
  DIFFERENT: другая модель / другой объём самого ТОВАРА /
             другая серия / другой числовой/буквенный индекс.

Использование в источниках:
    from app.services.enrichment.sources.donor_gate import is_same_product_cached

    if not await is_same_product_cached(gate, target_name, donor_title):
        return []   # fail-safe: при ошибке LLM gate пропускает (fail-open)
"""
from __future__ import annotations

import asyncio
import logging
import os
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Флаг включения гейта. По умолчанию ON.
_GATE_ENABLED: bool = os.getenv("DONOR_MATCH_GATE_ENABLED", "true").strip().lower() in (
    "1", "true", "yes", "on",
)

_SYSTEM_PROMPT = """\
Ты — строгий проверщик соответствия товаров. Решай ТОЛЬКО по приведённым названиям.

ПРАВИЛО SAME (одинаковые товары):
- другой цвет / другой год выпуска / другая комплектация (комплект vs без кабеля)
- суффикс варианта размера/цвета при одинаковой базовой модели (XS/S/M/L при одинаковом артикуле)

ПРАВИЛО DIFFERENT (разные товары):
- другой числовой или буквенный индекс модели (A500S ≠ A50; Farpoint 40 ≠ Daylite; 850G ≠ 650G)
- другой объём или вес самого ТОВАРА (рюкзак 40л ≠ 13л; порошок Tide 3кг ≠ 6кг)
- другая серия / линейка (Farpoint vs Daylite — разные серии Osprey)
- другое поколение, когда обозначается явно (Series 3 ≠ Series 4)

Когда сомневаешься — возвращай same=false (безопаснее отклонить чужой товар).

Верни JSON: {"same": true/false, "reason": "одна фраза"}
"""


class _SameProductResponse(BaseModel):
    same: bool
    reason: str = ""


class DonorVerdict(Enum):
    SAME = auto()
    DIFFERENT = auto()
    UNKNOWN = auto()


class DonorMatchGate:
    """Хранит кэш и lazy-инициализированный LLM manager.

    Один экземпляр создаётся на источник (WbCardSource, IceCatSource) и живёт
    столько же, сколько источник. Кэш: (target_lower, donor_lower) → bool.
    """

    def __init__(self) -> None:
        self._llm: Optional[object] = None  # StructuredLlmManager | OpenAIManager
        self._cache: dict[tuple[str, str], "DonorVerdict"] = {}

    def _get_llm(self):
        if self._llm is None:
            from app.services.providers.factory import get_main_manager
            self._llm = get_main_manager()
        return self._llm

    async def verdict(self, target_name: str, donor_title: str) -> tuple["DonorVerdict", bool]:
        """(вердикт, was_llm_call). Один LLM-вызов, без внутреннего ретрая. UNKNOWN не кэшируется."""
        if not _GATE_ENABLED:
            return (DonorVerdict.SAME, False)
        key = (target_name.strip().lower(), donor_title.strip().lower())
        if key in self._cache:
            return (self._cache[key], False)
        user_text = (f"Целевой товар: «{target_name}»\n"
                     f"Донор-карточка: «{donor_title}»\n\nЭто один и тот же товар?")
        try:
            parsed, _tokens = await asyncio.wait_for(
                self._get_llm().structured_request(
                    system_prompt=_SYSTEM_PROMPT, user_text=user_text,
                    response_model=_SameProductResponse), timeout=12)
            if parsed is None:
                return (DonorVerdict.UNKNOWN, True)
            verdict = DonorVerdict.SAME if parsed.same else DonorVerdict.DIFFERENT
            logger.info("[DonorGate] target='%s' donor='%s' → verdict=%s reason='%s'",
                        target_name[:60], donor_title[:60], verdict.name, parsed.reason[:80])
            self._cache[key] = verdict
            return (verdict, True)
        except Exception as exc:
            logger.warning("[DonorGate] LLM ошибка → UNKNOWN: %s", exc)
            return (DonorVerdict.UNKNOWN, True)

    async def is_same_product(self, target_name: str, donor_title: str) -> bool:
        """Fail-open bool-обёртка над verdict (IceCat). UNKNOWN/SAME→True, DIFFERENT→False."""
        v, _ = await self.verdict(target_name, donor_title)
        return v != DonorVerdict.DIFFERENT
