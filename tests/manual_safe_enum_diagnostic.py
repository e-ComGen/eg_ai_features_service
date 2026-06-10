"""Targeted live diagnostic for SafeEnumFillSource — 3 apparel products only.

Conserves Serper: 3 products max, SKIP_RAG=1.
Run with:
    SAFE_LLM_ENUM_FILL_ENABLED=true DEBUG_LOG=1 python tests/manual_safe_enum_diagnostic.py
"""
from __future__ import annotations
import asyncio
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8")

os.environ["SAFE_LLM_ENUM_FILL_ENABLED"] = "true"
os.environ["LLM_CACHE_ENABLED"] = "1"
os.environ.setdefault("LLM_CACHE_DB", str(PROJECT_ROOT / ".llm_cache.sqlite"))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
)
from app.services.enrichment.base import ExtractionContext, TargetAttribute
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.strategies.factory import get_strategy
from app.services.enrichment.sources.icecat_source import IceCatSource
from app.services.enrichment.sources.pdf_datasheet_source import PdfDatasheetSource
from app.services.enrichment.sources.ozon_card_source import OzonCardSource
from app.services.enrichment.sources.wb_card_source import WbCardSource
from app.services.enrichment.sources.tnved_source import TnvedSource

# Exactly 3 apparel products with short-enum optional attrs
PRODUCTS = [
    (200000933, 93244, "Футболка мужская Nike Sportswear Club"),
    (200000933, 93182, "Платье женское befree летнее"),
    (15621048,  91248, "Кроссовки Adidas Ultraboost 22"),
]


def build_targets(chars: list[dict]) -> list[TargetAttribute]:
    out = []
    for c in chars:
        allowed = None
        if c.get("values"):
            allowed = [v["value"] for v in c["values"][:50]]
        out.append(TargetAttribute(
            id=c["id"],
            name=c["name"],
            type="text",
            allowed_values=allowed,
            is_collection=c.get("is_collection", False),
            is_required=c.get("is_required", False),
        ))
    return out


async def main():
    from app.config import SAFE_LLM_ENUM_FILL_ENABLED
    print(f"[Diagnostic] SAFE_LLM_ENUM_FILL_ENABLED={SAFE_LLM_ENUM_FILL_ENABLED}", flush=True)

    strategy = get_strategy("ozon")
    orchestrator = PipelineOrchestrator(
        strategy=strategy,
        competitor_rag_source=None,  # SKIP_RAG
        icecat_source=IceCatSource(),
        pdf_datasheet_source=PdfDatasheetSource(),
        ozon_card_source=OzonCardSource(),
        wb_card_source=WbCardSource(),
        tnved_source=TnvedSource(),
    )

    for cat_id, type_id, prod_name in PRODUCTS:
        print(f"\n{'='*70}", flush=True)
        print(f"[Product] {prod_name} (cat={cat_id}, type={type_id})", flush=True)

        chars = get_ozon_characteristics_for_type(cat_id, type_id)
        if not chars:
            print(f"  SKIP — no chars", flush=True)
            continue

        targets = build_targets(chars)
        # Log short-enum optional targets count
        from app.services.enrichment.sources.safe_enum_fill_source import _is_short_enum
        short_enum = [t for t in targets if _is_short_enum(t)]
        print(f"  total targets={len(targets)} short-enum-optional={len(short_enum)}", flush=True)
        if short_enum:
            print(f"  short-enum targets: {[(t.id, t.name, len(t.allowed_values or [])) for t in short_enum[:8]]}", flush=True)

        ctx = ExtractionContext(
            product_id=hash(prod_name) % 100000,
            product_name=prod_name,
            category_id=cat_id,
            category_path=[],
            marketplace="ozon",
            ozon_type_id=type_id,
        )

        result = await orchestrator.enrich(ctx, targets)
        safe_fills = [v for v in result if v.source == "safe_enum_fill"]
        print(f"  safe_enum_fill fills={len(safe_fills)}:", flush=True)
        for v in safe_fills:
            print(f"    attr_id={v.attribute_id} value={v.value!r} evidence={v.evidence!r}", flush=True)

        all_sources = sorted({v.source for v in result})
        print(f"  all sources in result: {all_sources}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
