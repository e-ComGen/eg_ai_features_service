"""Restore datasets_rag collection from HF parquet shards to local Qdrant.

NEW APPROACH (no snapshot, no qdrant-on-pod):
  1. Download parquet_all/*.parquet from HF dataset repo.
     Files use unique names: <dataset>_file_NNNN.parquet
     (e.g. off_file_0000.parquet, abo_file_0000.parquet) so all 9 datasets
     coexist without name collisions — this was the root cause of the Phase-1/2
     overwrite bug where all datasets shared the same output dir.
  2. (Re)create local Qdrant collection `datasets_rag` (localhost:6333)
     with 384-d Cosine, on_disk vectors + HNSW + payload (low-RAM safeguard).
  3. Upsert all points in batches, reading vector+payload from parquet.
  4. Verify: print point count + run a sample vector query.

Idempotent: safe to re-run; existing points are overwritten by upsert-by-id.
Use --recreate to drop the collection before upsert (clean complete rebuild).

Usage (from WSL or Windows terminal):
    python scripts/restore_datasets_rag.py [--hf-repo USER/REPO] [--recreate] [--verify-only]

    # Full clean rebuild from all 9 datasets (recommended after runpod_spawn_datasets_all.py):
    python scripts/restore_datasets_rag.py --recreate

Reads HF_TOKEN + HF_REPO_ID from .env or environment.
Reads hf_repo_id from scripts/.runpod_datasets_all_state.json (or legacy
.runpod_datasets_state.json) if --hf-repo not given.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

HF_TOKEN: str = os.environ.get("HF_TOKEN", "")
QDRANT_URL: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "datasets_rag"
VECTOR_DIM = 384
UPSERT_BATCH = 1000
# Prefer the new all-datasets state file; fall back to legacy phase-1 state.
STATE_FILE = PROJECT_ROOT / "scripts" / ".runpod_datasets_all_state.json"
STATE_FILE_LEGACY = PROJECT_ROOT / "scripts" / ".runpod_datasets_state.json"


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download HF parquet shards and upsert into local Qdrant datasets_rag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--hf-repo", metavar="USER/REPO", default=None,
                   help="HF dataset repo id. Falls back to state file or HF_REPO_ID env var.")
    p.add_argument("--qdrant-url", default=QDRANT_URL,
                   help=f"Qdrant server URL. Default: {QDRANT_URL}")
    p.add_argument("--collection", default=COLLECTION,
                   help=f"Qdrant collection name. Default: {COLLECTION}")
    p.add_argument("--recreate", action="store_true",
                   help="Drop and recreate the collection before upsert (full rebuild).")
    p.add_argument("--verify-only", action="store_true",
                   help="Skip download+upsert; only run the verification query.")
    p.add_argument("--work-dir", default=None, metavar="DIR",
                   help="Local dir for downloaded parquet files. "
                        "Default: /tmp/datasets_rag_parquet (Linux) or %TEMP%\\datasets_rag_parquet.")
    p.add_argument("--batch", type=int, default=UPSERT_BATCH,
                   help=f"Qdrant upsert batch size. Default: {UPSERT_BATCH}")
    return p.parse_args()


# ── HF helpers ─────────────────────────────────────────────────────────────────

def resolve_hf_repo(args: argparse.Namespace) -> str:
    if args.hf_repo:
        return args.hf_repo
    env_val = os.environ.get("HF_REPO_ID", "")
    if env_val:
        return env_val
    for sf in (STATE_FILE, STATE_FILE_LEGACY):
        if sf.exists():
            d = json.loads(sf.read_text())
            repo = d.get("hf_repo_id", "")
            if repo:
                print(f"[restore] hf_repo_id from state ({sf.name}): {repo}")
                return repo
    sys.exit(
        "[restore] ERROR: cannot determine HF repo id.\n"
        "  Pass --hf-repo USER/REPO, or set HF_REPO_ID env, "
        "or run runpod_spawn_datasets.py first (creates state file)."
    )


def list_hf_parquet_files(repo_id: str) -> list[str]:
    """Return list of filenames under parquet_all/ in the HF dataset.

    parquet_all/ contains uniquely-named shards produced by
    runpod_pod_runner_datasets_all.sh, e.g.:
      off_file_0000.parquet, obf_file_0000.parquet, abo_file_0000.parquet …
    Unlike the old parquet/ folder (phase-1/2 runs), names never collide here.
    Falls back to legacy parquet/ prefix if parquet_all/ is empty (old runs).
    """
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError:
        sys.exit("[restore] huggingface_hub not installed: pip install huggingface_hub")
    api = HfApi(token=HF_TOKEN or None)
    files = list(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))

    # Primary: parquet_all/ (new all-in-one runner)
    parquet_all = sorted(f for f in files if f.startswith("parquet_all/") and f.endswith(".parquet"))
    if parquet_all:
        print(f"[restore] Using parquet_all/ ({len(parquet_all)} shards, unique per-dataset names).")
        return parquet_all

    # Legacy fallback: parquet/ (phase-1 / phase-2 old runs)
    legacy = sorted(f for f in files if f.startswith("parquet/") and f.endswith(".parquet"))
    if legacy:
        print(
            f"[restore] WARNING: parquet_all/ empty — falling back to legacy parquet/ "
            f"({len(legacy)} shards). These may be incomplete due to the overwrite bug. "
            "Run runpod_spawn_datasets_all.py for a complete rebuild."
        )
        return legacy

    return []


def download_parquet_file(repo_id: str, remote_path: str, local_path: Path) -> None:
    """Download a single file from HF dataset repo."""
    import urllib.request
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{remote_path}"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    print(f"[restore] GET {url}")
    with urllib.request.urlopen(req, timeout=300) as r:
        total = 0
        with open(local_path, "wb") as f:
            while True:
                chunk = r.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
                mb = total / (1024 * 1024)
                print(f"  {mb:.1f} MB", end="\r", flush=True)
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"\n  saved {local_path.name} ({size_mb:.1f} MB)")


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

def get_client(qdrant_url: str):
    try:
        from qdrant_client import QdrantClient  # type: ignore
    except ImportError:
        sys.exit("[restore] qdrant-client not installed: pip install qdrant-client")
    print(f"[restore] Connecting to Qdrant at {qdrant_url} ...")
    client = QdrantClient(url=qdrant_url, timeout=120)
    # Health check
    try:
        client.get_collections()
    except Exception as e:
        sys.exit(
            f"[restore] ERROR: cannot reach Qdrant at {qdrant_url}\n"
            f"  {e}\n"
            "  Start it first: bash ~/qdrant_start.sh"
        )
    print("[restore] Qdrant is healthy.")
    return client


def ensure_collection(client, name: str, recreate: bool) -> None:
    from qdrant_client.models import (  # type: ignore
        Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff,
    )
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        if recreate:
            print(f"[restore] Dropping existing collection '{name}' (--recreate) ...")
            client.delete_collection(name)
        else:
            info = client.get_collection(name)
            print(f"[restore] Collection '{name}' exists ({info.points_count or 0} pts). "
                  "Upsert-by-id (idempotent). Use --recreate to force full rebuild.")
            return

    print(f"[restore] Creating '{name}' (dim={VECTOR_DIM}, COSINE, on_disk=True) ...")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.COSINE,
            on_disk=True,           # vectors on disk — low-RAM safeguard
        ),
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
            on_disk=True,           # HNSW graph also on disk
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0,   # disable HNSW build during bulk upsert
        ),
        on_disk_payload=True,       # payload also on disk
    )
    print(f"[restore] Collection '{name}' created.")


def enable_indexing(client, name: str) -> None:
    """Re-enable HNSW indexing after bulk upsert completes."""
    from qdrant_client.models import OptimizersConfigDiff  # type: ignore
    client.update_collection(
        collection_name=name,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=20_000),
    )
    print(f"[restore] HNSW indexing re-enabled on '{name}'.")


def build_text_index(client, name: str) -> None:
    from qdrant_client.models import PayloadSchemaType  # type: ignore
    try:
        client.create_payload_index(
            collection_name=name,
            field_name="categories",
            field_schema=PayloadSchemaType.TEXT,
        )
        print("[restore] Text index on 'categories' created.")
    except Exception as e:
        if "already exists" in str(e).lower() or "conflict" in str(e).lower():
            print("[restore] Text index already exists.")
        else:
            print(f"[restore] Warning: text index failed: {e}")


# ── Parquet reading ─────────────────────────────────────────────────────────────

def iter_parquet_points(parquet_files: list[Path]) -> Iterator[tuple[str, list[float], dict]]:
    """Yield (id, vector, payload) from a list of local parquet files."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:
        sys.exit("[restore] pyarrow not installed: pip install pyarrow")

    for pf in parquet_files:
        print(f"[restore] Reading {pf.name} ...")
        table = pq.read_table(str(pf))
        ids = table["id"].to_pylist()
        vectors = table["vector"].to_pylist()
        names = table["name"].to_pylist()
        categories = table["categories"].to_pylist()
        characteristics = table["characteristics"].to_pylist()
        sources = table["source"].to_pylist()
        for i in range(len(ids)):
            payload = {
                "name": names[i] or "",
                "categories": categories[i] or "",
                "characteristics": characteristics[i] or "{}",
                "source": sources[i] or "",
            }
            yield ids[i], vectors[i], payload


# ── Upsert loop ────────────────────────────────────────────────────────────────

def upsert_parquet(
    client,
    collection: str,
    parquet_files: list[Path],
    batch_size: int,
) -> int:
    from qdrant_client.models import PointStruct  # type: ignore

    total = 0
    t0 = time.time()
    batch_ids: list[str] = []
    batch_vecs: list[list[float]] = []
    batch_payloads: list[dict] = []

    def flush() -> None:
        nonlocal total
        points = [
            PointStruct(id=pid, vector=vec, payload=pl)
            for pid, vec, pl in zip(batch_ids, batch_vecs, batch_payloads)
        ]
        client.upsert(collection_name=collection, points=points)
        total += len(points)
        batch_ids.clear()
        batch_vecs.clear()
        batch_payloads.clear()
        if total % 10_000 == 0:
            elapsed = time.time() - t0
            speed = total / elapsed if elapsed > 0 else 0
            print(f"[restore] {total:,} upserted | {speed:.0f} rows/s | {elapsed:.0f}s")

    for pid, vec, payload in iter_parquet_points(parquet_files):
        batch_ids.append(pid)
        batch_vecs.append(vec)
        batch_payloads.append(payload)
        if len(batch_ids) >= batch_size:
            flush()

    if batch_ids:
        flush()

    elapsed = time.time() - t0
    speed = total / elapsed if elapsed > 0 else 0
    print(f"[restore] Upsert complete: {total:,} points in {elapsed:.1f}s ({speed:.0f} rows/s)")
    return total


# ── Verification ───────────────────────────────────────────────────────────────

def verify(client, collection: str) -> None:
    info = client.get_collection(collection)
    print(f"\n[restore] === Verification ===")
    print(f"  collection   : {collection}")
    print(f"  points_count : {info.points_count or 0:,}")

    query_text = "шоколадная паста Нутелла"
    print(f"\n[restore] Sample query: '{query_text}'")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        vec = model.encode([query_text], normalize_embeddings=True)[0].tolist()
    except ImportError:
        print("[restore] sentence-transformers not installed — using zero vector for count check.")
        vec = [0.0] * VECTOR_DIM
    except Exception as e:
        print(f"[restore] WARNING: embed failed ({e}) — using zero vector.")
        vec = [0.0] * VECTOR_DIM

    result = client.query_points(
        collection_name=collection,
        query=vec,
        limit=5,
        with_payload=True,
    )
    print(f"[restore] Top-5 results:")
    for i, hit in enumerate(result.points):
        p = hit.payload or {}
        name = str(p.get("name", ""))[:70]
        source = p.get("source", "")
        cats = str(p.get("categories", ""))[:60]
        print(f"  [{i}] score={hit.score:.3f} | {source:8s} | {name}")
        print(f"       cats: {cats}")

    print(f"\n[restore] datasets_rag is live at {QDRANT_URL}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    qdrant_url = args.qdrant_url
    collection = args.collection

    client = get_client(qdrant_url)

    if args.verify_only:
        verify(client, collection)
        return

    # Resolve HF repo
    hf_repo_id = resolve_hf_repo(args)
    print(f"[restore] HF repo: {hf_repo_id}")

    # Work directory for parquet files
    if args.work_dir:
        work_dir = Path(args.work_dir)
    else:
        import tempfile
        work_dir = Path(tempfile.gettempdir()) / "datasets_rag_parquet"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"[restore] Work dir: {work_dir}")

    # List parquet files on HF
    print(f"[restore] Listing parquet files on HF ...")
    remote_files = list_hf_parquet_files(hf_repo_id)
    if not remote_files:
        sys.exit(
            f"[restore] ERROR: no parquet files found under parquet_all/ (or legacy parquet/) "
            f"in {hf_repo_id}.\n"
            "  The pod job may not have finished yet — check DONE_ALL (or DONE_DATASETS) "
            "marker on HF."
        )
    print(f"[restore] Found {len(remote_files)} parquet file(s):")
    for rf in remote_files:
        print(f"  {rf}")

    # Download (skip if already present with non-zero size)
    local_files: list[Path] = []
    for rf in remote_files:
        fname = Path(rf).name
        local_path = work_dir / fname
        if local_path.exists() and local_path.stat().st_size > 0:
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"[restore] Already downloaded: {fname} ({size_mb:.1f} MB) — skipping.")
        else:
            download_parquet_file(hf_repo_id, rf, local_path)
        local_files.append(local_path)

    # Create / verify collection
    ensure_collection(client, collection, args.recreate)

    # Upsert
    print(f"\n[restore] === Upserting {len(local_files)} file(s) → '{collection}' ===")
    upsert_parquet(client, collection, local_files, batch_size=args.batch)

    # Re-enable HNSW indexing + build text index
    enable_indexing(client, collection)
    build_text_index(client, collection)

    # Verify
    verify(client, collection)


if __name__ == "__main__":
    main()
