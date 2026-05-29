"""Direct RAG retrieval test — what does Qdrant return for our PSU queries?"""
from __future__ import annotations
import os
import sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import pyarrow  # noqa
    import pandas as pd  # noqa
except ImportError:
    pass

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

client = QdrantClient(url=os.environ["QDRANT_URL"], timeout=60)
info = client.get_collection("ozon_products")
print(f"Qdrant: {info.points_count:,} points")

print("Loading embed model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
print(f"Model dim={model.get_sentence_embedding_dimension()}")

QUERIES = [
    "Блок питания Cooler Master MWE Gold 750 V2 Full Modular 750W ATX",
    "Блок питания Deepcool PM650D 650W 80+ Bronze",
    "Блок питания ASUS ROG STRIX 850G 850W 80+ Gold",
    "Блок питания EVGA SuperNOVA 750 G6 750W 80+ Gold",
]

for q in QUERIES:
    print(f"\n=== {q}")
    vec = model.encode(q, normalize_embeddings=True).tolist()
    from qdrant_client.models import Filter, FieldCondition, MatchText
    flt = Filter(must=[FieldCondition(key="categories", match=MatchText(text="Блок питания компьютера"))])
    resp = client.query_points(collection_name="ozon_products", query=vec, limit=5, with_payload=True, query_filter=flt)
    hits = resp.points
    for i, h in enumerate(hits, 1):
        name = (h.payload or {}).get("name", "?")[:90]
        cat_raw = (h.payload or {}).get("categories", "?")
        cat = str(cat_raw)[:80]
        print(f"  {i}. score={h.score:.3f}  {name}")
        print(f"       cat: {cat}")
