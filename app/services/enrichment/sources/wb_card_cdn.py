"""URL-билдинг, fetch и парсинг options для WB card.json с CDN.
Переиспользуется WbCardSource и будущим app/services/eg_seller_audit/card_fetcher.py.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Известный sharding table «vol → basket-NN». На 404 перебираем все NN (01..21).
# Формат: (верхняя_граница_vol_включительно, NN).
_BASKET_THRESHOLDS: list[tuple[int, str]] = [
    (143,  "01"),
    (287,  "02"),
    (431,  "03"),
    (719,  "04"),
    (1007, "05"),
    (1061, "06"),
    (1115, "07"),
    (1169, "08"),
    (1313, "09"),
    (1601, "10"),
    (1655, "11"),
    (1919, "12"),
    (2045, "13"),
    (2189, "14"),
    (2405, "15"),
    (2621, "16"),
    (2837, "17"),
    (3053, "18"),
    (3473, "19"),
    (3793, "20"),
    (4050, "21"),
    (4306, "22"),
    (4563, "23"),
    (4820, "24"),
    (5076, "25"),
    (5333, "26"),
    (5590, "27"),
    (5846, "28"),
    (6103, "29"),
    (6359, "30"),
    (6616, "31"),
    (6873, "32"),
    (7129, "33"),
    (7386, "34"),
    (7643, "35"),
    (7899, "36"),
    (8156, "37"),
]
# vol > 8156 → fallback basket-37 (последний живой якорь, проба 2026-07-02:
# nm_id 815621985/823775519/823776306, vol 8156, все на basket-37). Полный
# список NN для brute-force перебора на 404 расширен до 01..40 (каталог
# растёт быстрее таблицы — range покрывает корректность, таблица только
# сокращает латентность primary-попытки).
_BASKET_DEFAULT = "37"
_ALL_BASKET_NN: list[str] = [f"{n:02d}" for n in range(1, 41)]  # 01..40

# Chrome User-Agent для CDN GET (CDN не банит, но без UA иногда 403).
_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_HTTP_TIMEOUT = 15.0


def _basket_nn_from_table(nm_id: int) -> str:
    """Возвращает basket-NN из known таблицы по vol (fallback basket-21)."""
    vol = nm_id // 100_000
    for threshold, nn in _BASKET_THRESHOLDS:
        if vol <= threshold:
            return nn
    return _BASKET_DEFAULT


def _card_url(nn: str, nm_id: int) -> str:
    vol = nm_id // 100_000
    part = nm_id // 1000
    return (
        f"https://basket-{nn}.wbbasket.ru"
        f"/vol{vol}/part{part}/{nm_id}/info/ru/card.json"
    )


async def try_basket(
    client: httpx.AsyncClient,
    nn: str,
    nm_id: int,
) -> Optional[dict]:
    """Один CDN GET. None если non-200 / не JSON / network err."""
    url = _card_url(nn, nm_id)
    try:
        r = await client.get(url)
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        logger.debug("[WbCard] CDN network err %s: %s", url, exc)
        return None
    if r.status_code != 200:
        return None
    try:
        data = r.json()
    except ValueError:  # включает json.JSONDecodeError
        return None
    except Exception as exc:  # noqa: BLE001 — на всякий случай не падаем
        logger.debug("[WbCard] card.json parse err %s: %s", url, exc)
        return None
    return data if isinstance(data, dict) else None


async def fetch_card(
    client: httpx.AsyncClient,
    nm_id: int,
    try_basket_fn=try_basket,
) -> Optional[dict]:
    """Скачать card.json с CDN. Простой GET; на 404 перебор NN (01..21)."""
    primary_nn = _basket_nn_from_table(nm_id)
    # Сначала known NN, затем остальные (без повтора primary).
    order = [primary_nn] + [nn for nn in _ALL_BASKET_NN if nn != primary_nn]
    for nn in order:
        data = await try_basket_fn(client, nn, nm_id)
        if data is not None:
            if nn != primary_nn:
                logger.info("[WbCard] nm=%s найден на basket-%s (fallback)", nm_id, nn)
            data["_wb_basket_nn"] = nn
            return data
    logger.info("[WbCard] card.json не найден ни на одном basket для nm=%s", nm_id)
    return None


def extract_options(card: dict) -> list[dict]:
    """Извлечь characteristics из WB card.json.

    Структуры:
      - options: [{"name": "Цвет", "value": "Белый"}, ...]
      - grouped_options: [{"group_name": "...", "options": [...]}]
      - compositions: [{"name": "хлопок", "value": "100"}] ИЛИ ["хлопок 100%"]
        → Состав.
      - selling.brand_name → Бренд

    Возвращает [{name, value, value_ids=[]}], deduped по lowercase name.
    """
    out: list[dict] = []
    seen: set[str] = set()

    def _push(name: Any, value: Any) -> None:
        if not isinstance(name, str):
            return
        name = name.strip()
        if not name:
            return
        name_low = name.lower()
        if name_low in seen:
            return
        if isinstance(value, list):
            texts = [str(v).strip() for v in value if str(v).strip()]
            if not texts:
                return
            value_str = ", ".join(texts)
        elif isinstance(value, (str, int, float)):
            value_str = str(value).strip()
            if not value_str:
                return
        else:
            return
        seen.add(name_low)
        out.append({"name": name, "value": value_str, "value_ids": []})

    # 1. Плоские options
    for o in card.get("options") or []:
        if isinstance(o, dict):
            _push(o.get("name"), o.get("value"))

    # 2. grouped_options
    for grp in card.get("grouped_options") or []:
        if isinstance(grp, dict):
            for o in grp.get("options") or []:
                if isinstance(o, dict):
                    _push(o.get("name"), o.get("value"))

    # 3. compositions → Состав (+ typed variants → Материал подкладки / утеплителя).
    #
    # WB schema variants observed in the wild:
    #   a. [{"name": "хлопок", "value": "100"}]          — pct in value
    #   b. [{"name": "хлопок", "percentage": 80}]         — pct in percentage
    #   c. ["хлопок 80%"]                                  — plain string
    #   d. [{"name": "хлопок", "value": "100", "type": "подкладка"}]
    #      — typed sub-composition: group by type, emit as "Материал подкладки", etc.
    #
    # Typed variants map: type value → canonical field name.
    _COMP_TYPE_TO_FIELD: dict[str, str] = {
        "подкладка":  "Материал подкладки",
        "утеплитель": "Материал утеплителя",
        "верх":       "Материал верха",
        "подошва":    "Материал подошвы",
        "основной":   "Состав",  # explicit "основной" type → main composition
    }
    # Buckets: None-key = untyped (→ "Состав"), other keys → specific fields.
    comp_buckets: dict[str | None, list[str]] = {}
    for c in card.get("compositions") or []:
        if isinstance(c, dict):
            cname = str(c.get("name") or "").strip()
            # Accept value from "value" or "percentage" field.
            cval = c.get("value")
            if cval is None:
                cval = c.get("percentage")
            cval_str = str(cval).strip() if isinstance(cval, (str, int, float)) else ""
            ctype_raw = str(c.get("type") or "").strip().lower()
            ctype: str | None = ctype_raw if ctype_raw else None
            token = f"{cname} {cval_str}%" if (
                cname and cval_str and str(cval_str).isdigit()
            ) else (f"{cname} {cval_str}" if cname and cval_str else cname)
            if token:
                comp_buckets.setdefault(ctype, []).append(token)
        elif isinstance(c, str) and c.strip():
            comp_buckets.setdefault(None, []).append(c.strip())

    # Emit each bucket as a separate field.
    for ctype, parts in comp_buckets.items():
        if not parts:
            continue
        # Resolve field name: typed → mapped name; untyped → "Состав".
        if ctype is None:
            field_name = "Состав"
        else:
            field_name = _COMP_TYPE_TO_FIELD.get(ctype, f"Материал {ctype}")
        field_name_low = field_name.lower()
        if field_name_low not in seen:
            seen.add(field_name_low)
            out.append({"name": field_name, "value": ", ".join(parts), "value_ids": []})

    # 4. selling.brand_name → Бренд (seller-filled brand outside options[]/
    #    grouped_options[]/compositions[]; additive, dedups via _push's `seen`).
    selling = card.get("selling")
    if isinstance(selling, dict):
        brand_name = selling.get("brand_name")
        if isinstance(brand_name, str) and brand_name.strip():
            _push("Бренд", brand_name)

    return out
