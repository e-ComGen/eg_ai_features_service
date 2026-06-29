"""Живой probe цвета на воркере :8002 (color-from-name + WB, Scrappey off).

Шлёт 2 кроссовка (цвет в названии + без цвета) с таргетом «Цвет товара» и печатает,
что воркер вернул по цвету: filled vs skipped. Секрет читается из .env, НЕ печатается.
"""
from __future__ import annotations

import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import INTERNAL_SERVICE_SECRET  # грузит .env, секрет не печатаем

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

CAT = 15621048
COLOR_OPTS = ["черный", "белый", "серый", "синий", "красный", "зеленый",
              "коричневый", "бирюзовый", "бежевый", "розовый", "желтый", "оранжевый"]

SCHEMA = {
    "Цвет товара": {"id": 10096, "type": "enum", "options": COLOR_OPTS, "is_required": False},
}

PRODUCTS = [
    {"id": 1, "name": "Nike Air Max 90 черные", "brand": "Nike"},
    {"id": 2, "name": "Adidas Runfalcon 3.0 синие", "brand": "Adidas"},
    {"id": 3, "name": "Nike Air Max 90", "brand": "Nike"},
    {"id": 4, "name": "PUMA Flyer Runner", "brand": "PUMA"},  # eg_importer p3: web_search гадал «чёрный»
]


def _payload():
    return {
        "client_id": 1,
        "use_cache": False,
        "options": {"marketplace": "ozon", "enable_vision": True, "enable_web_search": True},
        "schemas": {str(CAT): SCHEMA},
        "products": [
            {
                "id": p["id"], "category_id": CAT, "name": p["name"], "description": "",
                "brand": p["brand"], "ozon_type_id": 91248,
                "context": {"existing_features": {}, "company_id": 0},
            }
            for p in PRODUCTS
        ],
    }


async def main():
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(
            "http://127.0.0.1:8002/process-batch",
            headers={"x-internal-secret": INTERNAL_SERVICE_SECRET},
            json=_payload(),
        )
    print("HTTP", r.status_code)
    if r.status_code != 200:
        print(r.text[:500]); return
    data = r.json()
    rows = data if isinstance(data, list) else data.get("results") or data.get("data") or []
    by_id = {p["id"]: p["name"] for p in PRODUCTS}
    for item in rows:
        pid = item.get("product_id")
        filled = item.get("filled_features", {}) or {}
        debug = item.get("debug_info", {}) or {}
        skipped = item.get("skipped", {}) or {}
        cval = filled.get("Цвет товара")
        cdbg = debug.get("Цвет товара", {})
        csk = skipped.get("Цвет товара", {})
        print(f"\n[{pid}] {by_id.get(pid, '?')}")
        print(f"    filled  Цвет = {cval!r}")
        if cdbg:
            print(f"    debug   src={cdbg.get('source')} value_id={cdbg.get('value_id')} "
                  f"evidence={str(cdbg.get('evidence'))[:60]}")
        if csk:
            print(f"    skipped reason={csk.get('reason')}")


if __name__ == "__main__":
    asyncio.run(main())
