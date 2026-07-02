# -*- coding: utf-8 -*-
"""LIVE smoke test for FIX-13-WB (real network: scrape.do + search.wb.ru + CDN card.json).

Proves the chain: search.wb.ru (scrape.do) -> nm_id -> card.json on basket-37+ ->
>=5 characteristics of a real phone (type-gate picked a smartphone, not an accessory).

Manual/diagnostic script (not part of the pytest suite - makes real network calls,
consumes scrape.do credits). Run directly: python tests/manual_wb_scrapedo_fix13_smoke.py
"""
import asyncio
import os
import sys

sys.path.insert(0, r"C:\Users\Venya\PycharmProjects\CpAiFeatures-web-fetch")

from dotenv import load_dotenv  # noqa: E402
load_dotenv(r"C:\Users\Venya\PycharmProjects\CpAiFeatures-web-fetch\.env")

import httpx  # noqa: E402
import app.services.enrichment.sources.wb_card_source as mod  # noqa: E402


async def main() -> None:
    token = os.environ.get("SCRAPEDO_TOKEN")
    print("SCRAPEDO_TOKEN present:", bool(token), "len:", len(token) if token else 0)

    src = mod.WbCardSource()
    query = "poco x6 5g"

    print("\n--- STEP 1: _search (scrape.do + search.wb.ru) ---")
    nm_ids = await src._search(query)
    print("nm_ids:", nm_ids)
    if not nm_ids:
        print("FAIL: 0 nm_id returned - search transport did not work.")
        return

    print("\n--- STEP 2: _fetch_card per candidate (CDN, basket brute-force) ---")
    found = []
    async with httpx.AsyncClient(timeout=15.0) as client:
        for nm_id in nm_ids[: mod._MAX_FETCHED_CARDS]:
            primary_nn = mod._basket_nn_from_table(nm_id)
            card = await src._fetch_card(client, nm_id)
            if card is None:
                print(f"  nm_id={nm_id} primary_nn={primary_nn} -> 404 (no card)")
                continue
            options = src._extract_options(card)
            subj = card.get("subj_name") or card.get("subj_root_name") or "?"
            imt = card.get("imt_name") or "?"
            print(
                f"  nm_id={nm_id} primary_nn={primary_nn} subj='{subj}' "
                f"imt='{imt}' n_options={len(options)}"
            )
            found.append((nm_id, primary_nn, card, options, subj, imt))

    if not found:
        print("FAIL: 0 card.json fetched for any candidate nm_id.")
        return

    print("\n--- STEP 3: pick the best (type-gated) card via real extract() flow ---")
    from app.services.enrichment.base import ExtractionContext, TargetAttribute

    ctx = ExtractionContext(
        product_id=1,
        product_name="POCO X6 5G 12/256GB",
        category_id=1,
        category_path=["Электроника", "Смартфоны"],
        brand="Xiaomi",
    )
    targets = [
        TargetAttribute(id=1, name="Цвет", type="text"),
        TargetAttribute(id=2, name="Модель", type="text"),
        TargetAttribute(id=3, name="Объём встроенной памяти", type="text"),
        TargetAttribute(id=4, name="Оперативная память", type="text"),
        TargetAttribute(id=5, name="Диагональ экрана", type="text"),
        TargetAttribute(id=6, name="Тип процессора", type="text"),
        TargetAttribute(id=7, name="Емкость аккумулятора", type="text"),
        TargetAttribute(id=8, name="Вес", type="text"),
        TargetAttribute(id=9, name="Страна-изготовитель", type="text"),
        TargetAttribute(id=10, name="Гарантийный срок", type="text"),
        TargetAttribute(id=11, name="Артикул производителя", type="text"),
        TargetAttribute(id=12, name="Тип SIM-карты", type="text"),
        TargetAttribute(id=13, name="Разрешение камеры", type="text"),
        TargetAttribute(id=14, name="Материал корпуса", type="text"),
        TargetAttribute(id=15, name="Операционная система", type="text"),
    ]

    result = await src.extract(ctx, targets)
    print(f"mapped AttributeValues: {len(result)}")
    for av in result:
        tgt = next((t.name for t in targets if t.id == av.attribute_id), "?")
        print(f"  target='{tgt}' value={av.value!r} conf={av.confidence:.2f} src={av.source}")

    best = max(found, key=lambda t: len(t[3]))
    nm_id, primary_nn, card, options, subj, imt = best
    print("\n--- SUMMARY ---")
    print(f"nm_id={nm_id}")
    print(f"primary basket-NN (from table)={primary_nn}")
    print(f"subj_name='{subj}' imt_name='{imt}'")
    print(f"raw card options count={len(options)}")
    print(f"mapped AttributeValues count={len(result)}")
    print("sample raw characteristics:", options[:5])


asyncio.run(main())
