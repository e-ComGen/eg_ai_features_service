"""E2E проба options-backfill из Ozon API на воркере (порт через --port).

«Количество отделений» (attr 4820) — закрытый словарь Ozon {1,2,3,4}, options
НЕ передаём (как eg_importer для числовых словарей). Движок должен сам дёрнуть
list_values и срезать verbatim «8» из описания.

Два кейса:
  correct   — cat 17039630 + type 96031 (values-API принимает) → ожидаем no_data
  template  — cat 47156221 + type 96031 (values-API: not found) → 8 протечёт
"""
from __future__ import annotations

import asyncio
import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import INTERNAL_SERVICE_SECRET  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

PORT = sys.argv[1] if len(sys.argv) > 1 else "8003"
DESC = ("Начните своё утро с идеально поджаренных тостов благодаря тостеру Philips "
        "HD2581/90. Компактный чёрный корпус с 8 отделениями и регулировкой степени "
        "обжаривания позволяет приготовить хрустящие тосты на любой вкус.")
# options НЕ передаём — провоцируем API-backfill
SCHEMA = {"Количество отделений": {"id": 4820, "type": "numeric", "is_required": False}}

CASES = [
    ("correct  cat=17039630", 17039630, 96031),
    ("template cat=47156221", 47156221, 96031),
]


async def _run(label, cat, type_id):
    payload = {
        "client_id": 1, "use_cache": False,
        "options": {"marketplace": "ozon", "enable_vision": False, "enable_web_search": False},
        "schemas": {str(cat): SCHEMA},
        "products": [{
            "id": 2, "category_id": cat, "name": "Philips HD2581/90",
            "description": DESC, "brand": "Philips", "ozon_type_id": type_id,
            "context": {"existing_features": {}, "company_id": 0},
        }],
    }
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"http://127.0.0.1:{PORT}/process-batch",
                         headers={"x-internal-secret": INTERNAL_SERVICE_SECRET}, json=payload)
    d = r.json()
    rows = d if isinstance(d, list) else d.get("results") or d.get("data") or []
    for it in rows:
        f = it.get("filled_features", {}).get("Количество отделений")
        sk = it.get("skipped", {}).get("Количество отделений", {})
        dbg = it.get("debug_info", {}).get("Количество отделений", {})
        print(f"{label}: filled={f!r} value_id={dbg.get('value_id')} "
              f"skip={sk.get('reason')} src={dbg.get('source')}")


async def main():
    for label, cat, tid in CASES:
        await _run(label, cat, tid)


if __name__ == "__main__":
    asyncio.run(main())
