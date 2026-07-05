from __future__ import annotations
import logging
from typing import Literal, Optional
from app import config

logger = logging.getLogger(__name__)

_GROUNDING_SYSTEM = (
    "Ты проверяешь характеристику КОНКРЕТНОЙ модели товара по её официальной спецификации/надёжному источнику. "
    "Не рассуждай о том, что БЫВАЕТ в категории — проверяй ИМЕННО эту модель. "
    "Если источник не подтверждает точно — отвечай refute или unknown, не угадывай."
)

async def ground_value(
    product_name: str,
    attr_name: str,
    value,
    gm,
) -> Literal["confirm", "refute", "unknown"]:
    """
    Проверяет значение характеристики для КОНКРЕТНОЙ модели товара по внешнему источнику (веб-поиск).
    Возвращает "confirm", "refute" или "unknown".
    При отсутствии gm или любой ошибке — "unknown" (fail-safe).
    """
    if gm is None:
        return "unknown"
    user = (
        f'Товар (точная модель): "{product_name}"\n'
        f'Характеристика: "{attr_name}"\n'
        f'Проверяемое значение: "{value}"\n'
        "Верно ли это значение ИМЕННО для этой модели по её спецификации? "
        "Проверь по официальному источнику именно этой модели.\n"
        "Ответь СТРОГО одним словом: confirm (подтверждено источником), "
        "refute (источник противоречит), или unknown (источник не найден/неоднозначно). "
        "Затем с новой строки: Источник: <url или название спецификации>."
    )
    try:
        resp = await gm.complete(
            messages=[
                {"role": "system", "content": _GROUNDING_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=config.GROUNDING_MODEL,
            temperature=0.0,
            max_tokens=300,
        )
    except Exception:
        logger.warning("ground_value: grounding call failed for %s / %s", product_name, attr_name)
        return "unknown"
    if not resp or not resp.content:
        return "unknown"
    text = resp.content.lower()
    if "refute" in text:
        return "refute"
    if "confirm" in text:
        return "confirm"
    return "unknown"


async def ground_disagreement(
    product_name: str,
    attr_name: str,
    value_a,
    value_b,
    gm,
) -> Optional[str]:
    """
    Разрешает спор между двумя значениями характеристики для КОНКРЕТНОЙ модели товара
    по внешнему источнику. Возвращает одно из значений, или None при неопределённости/ошибке.
    """
    if gm is None:
        return None
    user = (
        f'Товар (точная модель): "{product_name}"\n'
        f'Характеристика: "{attr_name}"\n'
        f'Два кандидата: A = "{value_a}", B = "{value_b}".\n'
        "Какой из них верен ИМЕННО для этой модели по её официальной спецификации? "
        "Проверь по источнику именно этой модели.\n"
        "Ответь СТРОГО одним словом: A, B или neither (если ни один не подтверждён источником). "
        "Затем с новой строки: Источник: <url или название спецификации>."
    )
    try:
        resp = await gm.complete(
            messages=[
                {"role": "system", "content": _GROUNDING_SYSTEM},
                {"role": "user", "content": user},
            ],
            model=config.GROUNDING_MODEL,
            temperature=0.0,
            max_tokens=300,
        )
    except Exception:
        logger.warning("ground_disagreement: grounding call failed for %s / %s", product_name, attr_name)
        return None
    if not resp or not resp.content:
        return None
    text = resp.content.strip()
    if not text:
        return None
    first = text.splitlines()[0].strip().lower()
    if first.startswith("neither"):
        return None
    if first.startswith("a"):
        return value_a
    if first.startswith("b"):
        return value_b
    return None
