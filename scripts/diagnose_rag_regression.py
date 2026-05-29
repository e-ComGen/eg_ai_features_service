"""Investigate why PDF+RAG eval regressed Required from 73.8% → 67.5%.

Compares two eval JSON outputs product-by-product. Also queries Qdrant directly
to verify whether RAG retrieves relevant products for our PSU test set.
"""
from __future__ import annotations
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS = PROJECT_ROOT / "scripts" / "eval_results"
PDF_ONLY = RESULTS / "ps_by_name_20260527_215543.json"
PDF_RAG = RESULTS / "ps_by_name_20260528_154801.json"

# ---------- 1. Diff per-product fills ----------
print("=" * 70)
print("PART 1: Diff PDF-only vs PDF+RAG per product")
print("=" * 70)

a = json.loads(PDF_ONLY.read_text(encoding="utf-8"))
b = json.loads(PDF_RAG.read_text(encoding="utf-8"))

REQUIRED_ATTR_IDS = set()
# Inspect first product's filled list to detect required attrs (carry [REQ])
# Required attrs are marked in the eval log but not in JSON; let's match by name keywords known.
# Just compare each product's attr coverage of REQUIRED-named attrs.

for product_a, product_b in zip(a["products"], b["products"]):
    name = product_a["name"][:60]
    fills_a = {f["attribute_id"]: f for f in product_a["filled"]}
    fills_b = {f["attribute_id"]: f for f in product_b["filled"]}
    only_a = set(fills_a) - set(fills_b)
    only_b = set(fills_b) - set(fills_a)
    diff = set(fills_a) & set(fills_b)
    val_diff = [aid for aid in diff if fills_a[aid]["value"] != fills_b[aid]["value"]]
    if only_a or only_b or val_diff:
        print(f"\n--- {name}")
        print(f"  PDF-only filled: {len(fills_a)} | PDF+RAG filled: {len(fills_b)}")
        if only_a:
            print(f"  LOST in RAG run ({len(only_a)}):")
            for aid in list(only_a)[:6]:
                f = fills_a[aid]
                print(f"    id={aid} '{f.get('name','?')[:40]}' = {str(f['value'])[:60]} (src={f['source']})")
        if only_b:
            print(f"  NEW in RAG run ({len(only_b)}):")
            for aid in list(only_b)[:4]:
                f = fills_b[aid]
                print(f"    id={aid} '{f.get('name','?')[:40]}' = {str(f['value'])[:60]} (src={f['source']})")
        if val_diff:
            print(f"  VALUE CHANGED ({len(val_diff)}):")
            for aid in val_diff[:4]:
                fa, fb = fills_a[aid], fills_b[aid]
                print(f"    id={aid} '{fa.get('name','?')[:30]}'")
                print(f"      PDF-only: {str(fa['value'])[:55]} (src={fa['source']})")
                print(f"      PDF+RAG : {str(fb['value'])[:55]} (src={fb['source']})")


# ---------- 2. Query Qdrant directly with 3 sample PSU names ----------
print()
print("=" * 70)
print("PART 2: What does Qdrant actually return for PSU queries?")
print("=" * 70)

try:
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("missing deps")
    sys.exit(1)

client = QdrantClient(url=os.environ["QDRANT_URL"], timeout=60)
print(f"Connected to {os.environ['QDRANT_URL']}")
info = client.get_collection("ozon_products")
print(f"Total points: {info.points_count:,}")

print("\nLoading embed model (paraphrase-multilingual-MiniLM-L12-v2)...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

QUERIES = [
    "Блок питания Cooler Master MWE Gold 750 V2",
    "Блок питания Deepcool PM650D 650W 80+ Bronze",
    "Блок питания ASUS ROG STRIX 850G 850W 80+ Gold",
]

for q in QUERIES:
    print(f"\n--- Query: {q}")
    vec = model.encode(q, normalize_embeddings=True).tolist()
    hits = client.search(collection_name="ozon_products", query_vector=vec, limit=5, with_payload=True)
    for i, h in enumerate(hits, 1):
        name = (h.payload or {}).get("name", "?")[:80]
        cat = (h.payload or {}).get("categories", "?")[:60]
        print(f"  {i}. score={h.score:.3f}  name={name!r}")
        print(f"       cat={cat!r}")
