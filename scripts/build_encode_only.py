"""Encode 2.25M rows on GPU, save vectors + payloads as parquet shards. No Qdrant.

Output (in /workspace):
  vectors_0000.npy, vectors_0001.npy, ...      (200 MB each, 100k rows × 384 × float32)
  payloads_0000.parquet, payloads_0001.parquet (50-100 MB each)

After completion, run_encode.sh uploads all shards in a single HF commit.
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

try:
    import pyarrow  # noqa
    import pandas as pd  # noqa
except ImportError:
    pass

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

DATASET_NAME = "evgmaslov/ozon_ecup"
DATASET_CONFIG = "default"
DATASET_SPLIT = "train"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 256
SHARD_SIZE = 100_000
LIMIT = 3_000_000

OUT_DIR = Path("/workspace")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

print(f"[Encode] device={'cuda' if __import__('torch').cuda.is_available() else 'cpu'}")
print(f"[Encode] Loading model {EMBED_MODEL}")
model = SentenceTransformer(EMBED_MODEL, device="cuda")
print(f"[Encode] Model loaded dim={model.get_sentence_embedding_dimension()}")

print(f"[Encode] Downloading dataset {DATASET_NAME}/{DATASET_CONFIG}/{DATASET_SPLIT} (FULL, not streaming)")
t_dl = time.time()
dataset = load_dataset(
    DATASET_NAME, DATASET_CONFIG,
    split=DATASET_SPLIT, streaming=False,
    token=HF_TOKEN or None,
    cache_dir="/workspace/hf_cache",
)
print(f"[Encode] Dataset downloaded in {time.time()-t_dl:.0f}s — {len(dataset):,} rows local on disk")


def parse_field(raw):
    if raw is None:
        return ""
    if isinstance(raw, (dict, list)):
        return str(raw)[:1000]
    return str(raw)[:1000]


def safe_str(val, max_len: int = 500) -> str:
    if val is None:
        return ""
    s = str(val)
    return s[:max_len] if len(s) > max_len else s


shard_idx = 0
shard_vectors: list[np.ndarray] = []
shard_payloads: list[dict] = []
buffer_names: list[str] = []
buffer_rows: list[dict] = []
total_processed = 0


def flush_encode():
    global total_processed
    if not buffer_names:
        return
    vecs = model.encode(
        buffer_names,
        batch_size=len(buffer_names),
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)
    shard_vectors.append(vecs)
    for row in buffer_rows:
        try:
            vid = int(row.get("variantid"))
        except (TypeError, ValueError):
            continue
        shard_payloads.append({
            "variantid": vid,
            "name": safe_str(row.get("name"), 300),
            "description": safe_str(row.get("description"), 500),
            "categories": parse_field(row.get("categories")),
            "characteristics": parse_field(row.get("characteristic_attributes_mapping")),
        })
    total_processed += len(buffer_names)
    buffer_names.clear()
    buffer_rows.clear()


def save_shard():
    global shard_idx
    if not shard_vectors:
        return
    vecs = np.concatenate(shard_vectors)
    df = pd.DataFrame(shard_payloads)
    # Trim vectors to match payloads (drops rows with invalid variantid)
    if len(df) < len(vecs):
        vecs = vecs[:len(df)]
    np.save(OUT_DIR / f"vectors_{shard_idx:04d}.npy", vecs)
    df.to_parquet(OUT_DIR / f"payloads_{shard_idx:04d}.parquet", compression="zstd", index=False)
    print(f"[Encode] Shard {shard_idx:04d}: saved {len(vecs)} vectors, {len(df)} payloads", flush=True)
    shard_idx += 1
    shard_vectors.clear()
    shard_payloads.clear()


t0 = time.time()
last_log = 0
for row in dataset:
    if total_processed >= LIMIT:
        break
    vid = row.get("variantid")
    name = row.get("name") or ""
    if vid is None or not name.strip():
        continue
    buffer_names.append(name)
    buffer_rows.append(row)
    if len(buffer_names) >= BATCH_SIZE:
        flush_encode()
        if total_processed - last_log >= 10_000:
            elapsed = time.time() - t0
            speed = total_processed / elapsed if elapsed > 0 else 0
            print(f"[Encode] {total_processed:,} encoded | {speed:.0f} rows/s | {elapsed:.0f}s", flush=True)
            last_log = total_processed
        if len(shard_payloads) >= SHARD_SIZE:
            save_shard()

flush_encode()
save_shard()

elapsed = time.time() - t0
print(f"[Encode] DONE — {total_processed:,} rows in {elapsed:.0f}s = {total_processed/elapsed:.0f} rows/s", flush=True)
print(f"[Encode] Shards saved: {shard_idx}", flush=True)
