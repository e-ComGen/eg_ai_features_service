"""Test if our 20 PSU products have EXACT MATCH cards in Qdrant Ozon dataset.

For each product:
  1. Embedding search top-5 with category filter
  2. Compute name similarity (rapidfuzz) between query and top-1
  3. Classify: EXACT_MATCH | BRAND_LINE | DIFFERENT
  4. If EXACT — show what characteristics we'd inherit
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# DLL load order matters on Windows — pyarrow/pandas BEFORE torch/sentence_transformers
try:
    import pyarrow  # noqa
    import pandas  # noqa
except ImportError:
    pass

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PRODUCTS = [
    "Блок питания Cooler Master MWE Gold 750 V2 Full Modular 750W ATX",
    "Блок питания Deepcool PM650D 650W 80+ Bronze",
    "Блок питания Zalman ZM600-XEII 600W 80+ Bronze",
    "Блок питания ASUS ROG STRIX 850G 850W 80+ Gold",
    "Блок питания AeroCool Cylon 600W ATX",
    "Блок питания be quiet! Pure Power 11 600W 80+ Gold",
    "Блок питания Corsair RM750x 750W 80+ Gold Fully Modular",
    "Блок питания Seasonic FOCUS GX-650 650W 80+ Gold",
    "Блок питания EVGA SuperNOVA 750 G6 750W 80+ Gold",
    "Блок питания Thermaltake Toughpower GF1 850W 80+ Gold",
    "Блок питания Cooler Master V850 SFX Gold 850W",
    "Блок питания FSP Hyper K 600W 80+ Bronze",
    "Блок питания Chieftec PolarFrost 650W 80+ Gold",
    "Блок питания NZXT C850 Gold 850W ATX 3.0",
    "Блок питания MSI MPG A850G PCIE5 850W 80+ Gold",
    "Блок питания Gigabyte UD850GM 850W 80+ Gold Modular",
    "Блок питания Lian Li SP750 SFX 750W 80+ Gold",
    "Блок питания SilverStone DA850 GOLD 850W ATX 3.0",
    "Блок питания XPG CYBERCORE 1000W 80+ Platinum",
    "Блок питания Deepcool PX1000G 1000W 80+ Gold Gen.5",
]

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

client = QdrantClient(url=os.environ["QDRANT_URL"], timeout=60)
print(f"Qdrant points: {client.get_collection('ozon_products').points_count:,}")

print("Loading model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

cat_filter = Filter(must=[FieldCondition(key="categories", match=MatchText(text="Блок питания компьютера"))])

stats = {"EXACT": 0, "STRONG": 0, "WEAK": 0, "NONE": 0}

for q in PRODUCTS:
    vec = model.encode(q, normalize_embeddings=True).tolist()
    resp = client.query_points(
        collection_name="ozon_products",
        query=vec, limit=3, with_payload=True, query_filter=cat_filter,
    )
    hits = resp.points
    if not hits:
        verdict = "NONE"
        info = "no hits"
    else:
        top = hits[0]
        top_name = (top.payload or {}).get("name", "")
        # Strip "Блок питания компьютера" / brand prefix shared by both
        # Use partial_ratio (rapidfuzz) — handles word-order + missing words better
        partial = fuzz.partial_ratio(q.lower(), top_name.lower())
        token_sort = fuzz.token_sort_ratio(q.lower(), top_name.lower())
        wratio = fuzz.WRatio(q.lower(), top_name.lower())
        if top.score >= 0.92 and partial >= 85:
            verdict = "EXACT"
        elif top.score >= 0.85 and partial >= 65:
            verdict = "STRONG"
        elif top.score >= 0.75:
            verdict = "WEAK"
        else:
            verdict = "NONE"
        info = f"score={top.score:.3f} partial={partial} token={token_sort} wRatio={wratio} | {top_name[:60]}"
    stats[verdict] += 1
    print(f"{verdict:6s} {q[:55]:55s}")
    print(f"       {info}")
    # If EXACT — show how many chars its competitor card has
    if verdict in ("EXACT", "STRONG") and hits:
        chars_raw = (hits[0].payload or {}).get("characteristics", "")
        if isinstance(chars_raw, str) and chars_raw:
            # rough count of attribute mappings
            try:
                parsed = eval(chars_raw) if chars_raw.startswith("{") else {}
                print(f"       → has {len(parsed)} attribute mappings ready to copy")
            except Exception:
                print(f"       → chars field exists (raw len {len(chars_raw)})")
    print()

print()
print("=" * 60)
print(f"SUMMARY across {len(PRODUCTS)} products:")
for k, v in stats.items():
    print(f"  {k:6s}: {v}/{len(PRODUCTS)} ({v/len(PRODUCTS)*100:.0f}%)")
