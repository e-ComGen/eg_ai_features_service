"""Smoke test для OzonCardSource (composer-api.bx mobile API).

Запуск:
  python scripts/test_ozon_card_source.py

Что делает:
  1. Создаёт OzonCardSource.
  2. Строит ExtractionContext для "Блок питания Cooler Master MWE Gold 750 V2".
  3. Берёт реальные PSU characteristics из Ozon dictionary (cat=17028612, type=91910).
  4. Вызывает extract().
  5. Печатает count + sample fill data (attr_name, value, value_id, source, confidence).
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)

from app.services.enrichment.base import ExtractionContext, TargetAttribute
from app.services.enrichment.sources.ozon_card_source import OzonCardSource
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
)

DESCRIPTION_CATEGORY_ID = 17028612
TYPE_ID = 91910
PRODUCT_NAME = "Блок питания Cooler Master MWE Gold 750 V2"
BRAND = "Cooler Master"


def build_targets(chars: list[dict]) -> list[TargetAttribute]:
    out = []
    for c in chars:
        allowed = None
        if c.get("values"):
            allowed = [v["value"] for v in c["values"][:50] if isinstance(v, dict) and "value" in v]
        out.append(TargetAttribute(
            id=c["id"],
            name=c["name"],
            type=c.get("type") or "text",
            allowed_values=allowed,
            is_collection=c.get("is_collection", False),
            is_required=c.get("is_required", False),
        ))
    return out


async def main() -> int:
    print(f"[Smoke] Loading Ozon dictionary for cat={DESCRIPTION_CATEGORY_ID}, type={TYPE_ID}")
    chars = get_ozon_characteristics_for_type(DESCRIPTION_CATEGORY_ID, TYPE_ID)
    if not chars:
        print("[ERR] no characteristics from Ozon dictionary — словарь не загружен?")
        return 1
    print(f"[Smoke] {len(chars)} characteristics ({sum(1 for c in chars if c.get('is_required'))} required)")

    targets = build_targets(chars)
    char_by_id = {c["id"]: c for c in chars}

    print(f"[Smoke] Instantiating OzonCardSource (composer-api.bx)")
    source = OzonCardSource()  # apify_token=None — ignored

    ctx = ExtractionContext(
        product_id=1,
        product_name=PRODUCT_NAME,
        product_description="",
        brand=BRAND,
        category_id=str(DESCRIPTION_CATEGORY_ID),
        category_path=["Электроника", "Блоки питания ПК", "Блок питания компьютера"],
        image_urls=[],
        marketplace="ozon",
        ozon_type_id=TYPE_ID,
    )

    print(f"[Smoke] Calling extract() for: {PRODUCT_NAME!r}")
    try:
        avs = await source.extract(ctx, targets, already_filled=[])
    except Exception as exc:
        print(f"[ERR] extract() raised {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        return 2

    print(f"\n[Smoke] RESULT: {len(avs)} attribute values returned")
    if not avs:
        print("[Smoke] no fills — likely 403/DataDome block from this IP, or no match found.")
        print("[Smoke] (See [OzonCard] log lines above for stage-by-stage trace.)")
        return 0

    print("\n[Smoke] Sample fills:")
    print("-" * 100)
    print(f"{'attr_id':>8}  {'attr_name':40s}  {'value':30s}  {'val_id':>8}  {'src':>10}  {'conf':>5}")
    print("-" * 100)
    for av in avs[:20]:
        name = char_by_id.get(av.attribute_id, {}).get("name", "?")[:38]
        val = str(av.value)[:28]
        vid = str(av.value_id) if av.value_id else "-"
        print(
            f"{av.attribute_id:>8}  {name:40s}  {val:30s}  {vid:>8}  "
            f"{str(av.source):>10}  {av.confidence:>5.2f}"
        )
    if len(avs) > 20:
        print(f"... ({len(avs) - 20} more)")
    return 0


if __name__ == "__main__":
    rc = asyncio.run(main())
    sys.exit(rc)
