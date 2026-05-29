"""Smoke test new OzonCardSource (Scrappey backend) end-to-end."""
import os, sys, asyncio
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")

from app.services.enrichment.sources.ozon_card_source import OzonCardSource
from app.services.enrichment.base import ExtractionContext, TargetAttribute


async def main():
    src = OzonCardSource()  # reads SCRAPPEY_KEY from env

    # Build a context for a power supply
    ctx = ExtractionContext(
        product_id=999999,
        product_name="Cooler Master MWE Gold 750 V2",
        brand="Cooler Master",
        category_id=15500,  # электроника, не важно для smoke
        ozon_type_id=None,
    )
    # Without real targets we can't map → but we can test the network flow
    # via _do_extract bypass. Instead instantiate fake targets:
    targets = [
        TargetAttribute(id=1, name="Мощность", type="text", is_collection=False),
        TargetAttribute(id=2, name="Форм-фактор", type="text", is_collection=False),
        TargetAttribute(id=3, name="Длина кабеля", type="text", is_collection=False),
        TargetAttribute(id=4, name="Бренд", type="text", is_collection=False),
        TargetAttribute(id=5, name="Модель", type="text", is_collection=False),
    ]

    print(f"[Smoke] Calling extract for '{ctx.product_name}'...")
    print(f"[Smoke] SCRAPPEY_KEY present: {bool(src._scrappey_key)}")
    print()
    import time
    t0 = time.time()
    result = await src.extract(ctx, targets, already_filled=[])
    print(f"[Smoke] extract returned {len(result)} AttributeValues in {time.time()-t0:.1f}s")
    for av in result:
        print(f"  [attr_id={av.attribute_id}] value={av.value[:60]!r} conf={av.confidence} value_id={av.value_id} ev={av.evidence}")


asyncio.run(main())
