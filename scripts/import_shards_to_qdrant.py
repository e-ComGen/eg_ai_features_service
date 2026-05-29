"""Import pre-encoded vector shards from HF into local Qdrant file-based collection.

Workflow:
  1. Download all vectors_*.npy + payloads_*.parquet from HF dataset to /from_pod/.
  2. Create Qdrant collection with HNSW disabled (fast bulk insert).
  3. Insert all shards in batches.
  4. Re-enable HNSW + trigger one-time graph rebuild.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
import urllib.request
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN = os.environ["HF_TOKEN"]

state_path = PROJECT_ROOT / "scripts" / ".runpod_encode_state.json"
if state_path.exists():
    state = json.loads(state_path.read_text())
    HF_REPO_ID = state["hf_repo_id"]
else:
    HF_REPO_ID = sys.argv[1] if len(sys.argv) > 1 else sys.exit("no state, give hf_repo_id arg")

DATA_DIR = PROJECT_ROOT / "app" / "services" / "enrichment" / "strategies" / "dictionaries" / "data"
DOWNLOAD_DIR = DATA_DIR / "from_pod"
QDRANT_PATH = str(DATA_DIR / "ozon_rag.qdrant")
COLLECTION = "ozon_products"
VECTOR_DIM = 384
UPSERT_BATCH = 1000


def hf_list_files() -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{HF_REPO_ID}/tree/main"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def hf_download(remote_name: str, local_path: Path) -> None:
    url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{remote_name}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
    print(f"  Downloading {remote_name}", end=" ")
    with urllib.request.urlopen(req, timeout=300) as r:
        with open(local_path, "wb") as f:
            total = 0
            while True:
                chunk = r.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
    print(f"({total / 1024 / 1024:.1f} MB)")


def main():
    print(f"[Import] HF repo: {HF_REPO_ID}")
    print(f"[Import] Local Qdrant path: {QDRANT_PATH}")
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Discover and download all shards
    print(f"[Import] Listing files in {HF_REPO_ID}...")
    files = hf_list_files()
    shard_files = sorted([f["path"] for f in files if f["path"].startswith(("vectors_", "payloads_"))])
    print(f"[Import] Found {len(shard_files)} shard files")

    for fname in shard_files:
        local = DOWNLOAD_DIR / fname
        if local.exists() and local.stat().st_size > 0:
            print(f"  [skip] {fname} (already downloaded)")
            continue
        hf_download(fname, local)

    # 2. Create Qdrant collection
    print(f"\n[Import] Initializing Qdrant collection")
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance, VectorParams, PointStruct,
            OptimizersConfigDiff, HnswConfigDiff,
        )
    except ImportError:
        sys.exit("qdrant-client not installed. Run: pip install qdrant-client")

    client = QdrantClient(path=QDRANT_PATH)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        print(f"[Import] Deleting existing collection '{COLLECTION}'")
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=2_000_000_000),
        hnsw_config=HnswConfigDiff(m=0),
    )
    print(f"[Import] Collection created with HNSW disabled")

    # 3. Insert shards
    import numpy as np
    import pandas as pd

    vector_shards = sorted(DOWNLOAD_DIR.glob("vectors_*.npy"))
    total_inserted = 0
    t0 = time.time()

    for vec_file in vector_shards:
        shard_idx = vec_file.stem.split("_")[1]
        payload_file = DOWNLOAD_DIR / f"payloads_{shard_idx}.parquet"
        if not payload_file.exists():
            print(f"[Import] WARN: missing payload for {vec_file.name}, skip")
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
            total_inserted += len(points)

        elapsed = time.time() - t0
        speed = total_inserted / elapsed if elapsed > 0 else 0
        print(f"[Import] Shard {shard_idx}: +{n} (total {total_inserted:,}) | {speed:.0f} rows/s")

    # 4. Rebuild HNSW
    print(f"\n[Import] Rebuilding HNSW index (one-time, may take 10-30 min)")
    rebuild_start = time.time()
    client.update_collection(
        collection_name=COLLECTION,
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
        hnsw_config=HnswConfigDiff(m=16),
    )
    print(f"[Import] HNSW rebuild triggered in {time.time() - rebuild_start:.1f}s")

    info = client.get_collection(COLLECTION)
    print(f"\n[Import] DONE — {info.points_count:,} points in '{COLLECTION}'")
    print(f"[Import] Total elapsed: {(time.time() - t0):.0f}s")


if __name__ == "__main__":
    main()
