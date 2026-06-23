"""Измерительный прогон: реальный hit-rate Scrappey по Ozon-карточкам × конфиг прокси.

Зачем: до 06-14 OzonCardSource слал голый datacenter-payload и «не бил RU». Мы добавили
антибот-рычаги (proxyCountry=Russia / residential proxy / browser / session). Этот скрипт
честно меряет, СКОЛЬКО карточек пробивается в каждом конфиге и СКОЛЬКО это стоит кредитов,
чтобы решать на цифрах, а не на ощущении.

Использует OzonCardSource.probe() — search→match→/features/→parse, БЕЗ LLM (тратятся
только Scrappey-кредиты). Каждый Scrappey-вызов считается = 1 кредит (точный счётчик).

ЗАПУСК (Scrappey должен быть включён — раскомментируй SCRAPPEY_KEY в .env):
    ./venv/Scripts/python.exe scripts/measure_scrappey_ozon.py            # A+B (дёшево)
    ./venv/Scripts/python.exe scripts/measure_scrappey_ozon.py --full     # A+B+C+D

Конфиги:
    A baseline-datacenter   proxyCountry=''      requestType=''        session=off  (старое поведение)
    B residential-RU-raw    proxyCountry=Russia  requestType=''        session=off  (гео-рычаг — гипотеза)
    C residential-RU-browser proxyCountry=Russia requestType='browser' session=off  (полный антибот)
    D + session-reuse       proxyCountry=Russia  requestType='browser' session=ON   (прогретая кука)

ВНИМАНИЕ: residential/browser-конфиги (B/C/D) могут стоить БОЛЬШЕ кредитов за вызов,
чем datacenter (A). Следи за балансом в дашборде Scrappey. Ключ НЕ печатается.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# UTF-8 stdout (Windows-консоль по умолчанию cp1251 — давится на ✅/❌/кириллице).
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

# Путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Грузим .env тем же механизмом, что и приложение (app/config.py) — чтобы скрипт
# увидел SCRAPPEY_KEY и OZON_* без ручной передачи в окружение.
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from app.services.enrichment.base import ExtractionContext  # noqa: E402
from app.services.enrichment.sources import ozon_card_source as ocs  # noqa: E402


# ── Тестовый набор: разнообразные RU-кроссовки (бренд в начале имени) ──────────
_PRODUCTS = [
    ("Nike Air Max 90", "Nike"),
    ("Adidas Runfalcon 3.0", "Adidas"),
    ("PUMA Flyer Runner", "PUMA"),
    ("New Balance 574 Core", "New Balance"),
    ("Reebok Classic Leather", "Reebok"),
    ("ASICS Gel-Contend 8", "ASICS"),
    ("Nike Revolution 6", "Nike"),
    ("Adidas Galaxy 6", "Adidas"),
    ("Demix Lite Run", "Demix"),
    ("Lacoste Carnaby Evo", "Lacoste"),
]
_CAT_PATH = ["Обувь", "Мужская обувь", "Кроссовки"]
_CAT_ID = 15621048


# ── Конфиги (имя → словарь модульных констант) ────────────────────────────────
_CONFIGS = {
    "A baseline-datacenter": dict(country="", req="", proxy="", session=False),
    "B residential-RU-raw": dict(country="Russia", req="", proxy="", session=False),
    "C residential-RU-browser": dict(country="Russia", req="browser", proxy="", session=False),
    "D RU-browser+session": dict(country="Russia", req="browser", proxy="", session=True),
}


def _apply_config(cfg: dict) -> None:
    """Перезаписать модульные Scrappey-константы под конфиг (читаются на каждый payload)."""
    ocs._SCRAPPEY_PROXY_COUNTRY = cfg["country"]
    ocs._SCRAPPEY_REQUEST_TYPE = cfg["req"]
    ocs._SCRAPPEY_PROXY = cfg["proxy"]
    ocs._SCRAPPEY_SESSION_REUSE = cfg["session"]


def _install_credit_counter() -> dict:
    """Обернуть _scrappey_fetch_once глобальным счётчиком вызовов (= кредитов)."""
    counter = {"calls": 0}
    orig = ocs.OzonCardSource._scrappey_fetch_once

    async def _counting(self, client, target_url, session=None):
        counter["calls"] += 1
        return await orig(self, client, target_url, session)

    ocs.OzonCardSource._scrappey_fetch_once = _counting
    return counter


async def _run_config(name: str, cfg: dict, products: list) -> None:
    _apply_config(cfg)
    key = os.environ.get("SCRAPPEY_KEY")
    source = ocs.OzonCardSource(scrappey_key=key)
    counter = _install_credit_counter()

    print(f"\n{'='*72}\n  КОНФИГ: {name}")
    print(f"  proxyCountry={cfg['country']!r} requestType={cfg['req']!r} "
          f"proxy={'set' if cfg['proxy'] else 'none'} session_reuse={cfg['session']}")
    print(f"{'='*72}")

    found = 0
    stages: dict[str, int] = {}
    chars_total = 0
    t0 = time.monotonic()
    for pid, (name_str, brand) in enumerate(products, 1):
        ctx = ExtractionContext(
            product_id=pid, product_name=name_str, category_id=_CAT_ID,
            category_path=_CAT_PATH, brand=brand,
        )
        ct0 = time.monotonic()
        res = await source.probe(ctx)
        dt = time.monotonic() - ct0
        st = res.get("stage", "?")
        stages[st] = stages.get(st, 0) + 1
        hit = res.get("found", False)
        if hit:
            found += 1
            chars_total += res.get("raw_chars", 0)
        print(f"  {'✅' if hit else '❌'} {name_str:32s} stage={st:11s} "
              f"chars={res.get('raw_chars', 0):3d} score={res.get('match_score')} {dt:4.1f}s")

    dt_all = time.monotonic() - t0
    n = len(products)
    print(f"\n  ИТОГ {name}: пробито {found}/{n} ({100*found//n}%), "
          f"кредитов(Scrappey-вызовов)={counter['calls']}, "
          f"avg_chars={chars_total // max(found,1)}, {dt_all:.0f}s")
    print(f"  stages: {stages}")


async def _main(full: bool, n: int) -> None:
    if not os.environ.get("SCRAPPEY_KEY"):
        print("❌ SCRAPPEY_KEY не задан в окружении. Раскомментируй SCRAPPEY_KEY в .env "
              "(строка ~34) и перезапусти. Без ключа probe() вернёт пусто.")
        sys.exit(2)

    products = _PRODUCTS[:n]
    order = ["A baseline-datacenter", "B residential-RU-raw"]
    if full:
        order += ["C residential-RU-browser", "D RU-browser+session"]
    print(f"Прогон конфигов: {order}  (по {len(products)} товаров)")
    for name in order:
        await _run_config(name, _CONFIGS[name], products)

    print(f"\n{'#'*72}\nГотово. Сравни hit-rate vs кредиты по конфигам выше и реши, "
          f"стоит ли держать Scrappey ON и в каком режиме.\n{'#'*72}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="включить дорогие конфиги C+D (browser)")
    ap.add_argument("--n", type=int, default=6, help="сколько товаров прогнать на конфиг (дефолт 6)")
    args = ap.parse_args()
    asyncio.run(_main(args.full, args.n))
