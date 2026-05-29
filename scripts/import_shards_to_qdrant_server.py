"""Import shards to Qdrant SERVER (HTTP) — parallel to slow local-mode import.

Uses Qdrant running in WSL at localhost:6334. Much faster than local mode.
After completion, take Qdrant snapshot for portability.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "app" / "services" / "enrichment" / "strategies" / "dictionaries" / "data"
SHARDS_DIR = DATA_DIR / "from_pod"
QDRANT_URL = "http://localhost:6334"
COLLECTION = "ozon_products"
VECTOR_DIM = 384
UPSERT_BATCH = 1000  # 5000 → 49MB JSON > 32MB qdrant limit

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        OptimizersConfigDiff, HnswConfigDiff,
    )
except ImportError:
    sys.exit("qdrant-client not installed")

import numpy as np
import pandas as pd

print(f"[ImportSrv] Connecting to Qdrant Server at {QDRANT_URL}")
client = QdrantClient(url=QDRANT_URL, timeout=120)
info = client.get_collections()
print(f"[ImportSrv] Connected. Existing collections: {[c.name for c in info.collections]}")

if COLLECTION in [c.name for c in info.collections]:
    print(f"[ImportSrv] Deleting existing collection '{COLLECTION}'")
    client.delete_collection(COLLECTION)

print(f"[ImportSrv] Creating collection with HNSW disabled (bulk mode)")
client.create_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    optimizers_config=OptimizersConfigDiff(indexing_threshold=2_000_000_000),
    hnsw_config=HnswConfigDiff(m=0),
)

vector_shards = sorted(SHARDS_DIR.glob("vectors_*.npy"))
print(f"[ImportSrv] Found {len(vector_shards)} shards in {SHARDS_DIR}")

total = 0
t0 = time.time()

for vec_file in vector_shards:
    shard_idx = vec_file.stem.split("_")[1]
    payload_file = SHARDS_DIR / f"payloads_{shard_idx}.parquet"
    if not payload_file.exists():
        print(f"[ImportSrv] WARN: missing payload for {vec_file.name}")
        continue

    vecs = np.load(vec_file)
    df = pd.read_parquet(payload_file)
    n = min(len(vecs), len(df))

    for i in range(0, n, UPSERT_BATCH):
        batch_end = min(i + UPSERT_BATCH, n)
        points = []
        for j in range(i, batch_end):
            points.append(PointStruct(
                id=int(df.iloc[j]["variantid"]),
                vector=vecs[j].tolist(),
                payload={
                    "variantid": int(df.iloc[j]["variantid"]),
                    "name": str(df.iloc[j].get("name") or ""),
                    "description": str(df.iloc[j].get("description") or ""),
                    "categories": str(df.iloc[j].get("categories") or ""),
                    "characteristics": str(df.iloc[j].get("characteristics") or ""),
                },
            ))
        client.upsert(collection_name=COLLECTION, points=points)
        total += len(points)

    elapsed = time.time() - t0
    speed = total / elapsed if elapsed > 0 else 0
    print(f"[ImportSrv] Shard {shard_idx}: +{n:,} (total {total:,}) | {speed:.0f} rows/s | {elapsed:.0f}s", flush=True)

# Re-enable HNSW + trigger rebuild
print(f"\n[ImportSrv] Bulk upsert done. Triggering HNSW rebuild...")
client.update_collection(
    collection_name=COLLECTION,
    optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
    hnsw_config=HnswConfigDiff(m=16),
)

info = client.get_collection(COLLECTION)
elapsed = time.time() - t0
print(f"\n[ImportSrv] DONE")
print(f"  Total points: {info.points_count:,}")
print(f"  Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  Storage: WSL /tmp/qdrant_storage")
print(f"  Next: take snapshot via curl http://localhost:6334/collections/{COLLECTION}/snapshots")
