#!/usr/bin/env bash
# Pod run.sh for datasets_rag Phase-2 indexing (parquet path).
#
# Phase-2 datasets:
#   abo          Amazon Berkeley Objects  ~147k home/furniture  --limit 150000
#   rebrickable  LEGO sets               ~20k   --full
#   amazon       Amazon Reviews 2023 metadata (3 RU-relevant categories):
#                  Electronics           --limit 150000
#                  Home_and_Kitchen      --limit 150000
#                  Clothing_Shoes_and_Jewelry --limit 150000
#
# Flow: install deps → verify CUDA → embed Phase-2 datasets → write parquet shards
#       → upload parquet/ to HF → DONE_DATASETS_P2 / FAILED_DATASETS_P2 marker.
#
# Note on parquet/ folder: Phase-2 shards are uploaded alongside Phase-1 shards
# (same parquet/ prefix, new shard names file_NNNN.parquet starting from 0 per
# run).  On restore, restore_datasets_rag.py downloads ALL parquet/*.parquet and
# upserts them — points are keyed by stable UUID5 so Phase-1 points are untouched.
#
# Required env: HF_TOKEN, HF_REPO_ID
# SHA-pinned indexer fetch: HF_REV is injected by runpod_spawn_datasets_p2.py
# NO Qdrant is installed or started on the pod. Qdrant upsert happens locally.
set -uo pipefail
exec > >(tee -a /workspace/datasets_pod_p2.log) 2>&1

LOG=/workspace/datasets_pod_p2.log
echo "[pod-p2] start at $(date)"
echo "[pod-p2] HF_REPO_ID=${HF_REPO_ID}"

# ── Upload log on exit (success or failure) ────────────────────────────────────
upload_log() {
    python3 - <<'PYEOF' || true
import os
try:
    from huggingface_hub import HfApi
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj="/workspace/datasets_pod_p2.log",
        path_in_repo="datasets_pod_p2.log",
        repo_id=os.environ["HF_REPO_ID"],
        repo_type="dataset",
        commit_message="pod-p2 log (final)",
    )
    print("[pod-p2] log uploaded to HF")
except Exception as e:
    print(f"[pod-p2] log upload failed: {e}")
PYEOF
}
trap upload_log EXIT
trap 'echo "[pod-p2] ERROR at line $LINENO"' ERR

echo "[pod-p2] python: $(which python3) $(python3 --version)"

# ── Clean stale conda packages that conflict with sentence-transformers ─────────
echo "[pod-p2] === clean stale conda packages ==="
pip uninstall -y transformers tokenizers sentence-transformers 2>&1 | tail -5 || true
rm -rf /opt/conda/lib/python3.11/site-packages/transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/sentence_transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/tokenizers* 2>/dev/null || true

# ── Deps ───────────────────────────────────────────────────────────────────────
echo "[pod-p2] === pip install ==="
pip install --no-cache-dir --upgrade pip 2>&1 | tail -3
pip install --no-cache-dir \
    "transformers==4.46.*" \
    "tokenizers==0.20.*" \
    "sentence-transformers==3.*" \
    "datasets==2.*" \
    "huggingface_hub==0.25.*" \
    "python-dotenv" "pyarrow" "pandas" "boto3" "fsspec" "py7zr" 2>&1 | tail -20

# ── Verify CUDA is available ───────────────────────────────────────────────────
echo "[pod-p2] === verify torch CUDA ==="
python3 - <<'PYEOF'
import sys
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"[pod-p2] torch {torch.__version__}, cuda={cuda_ok}")
    if not cuda_ok:
        print("[pod-p2] WARNING: CUDA not available — embedding will be slow on CPU", file=sys.stderr)
    import sentence_transformers
    print(f"[pod-p2] sentence-transformers {sentence_transformers.__version__}")
    import datasets
    print(f"[pod-p2] datasets {datasets.__version__}")
    import huggingface_hub
    print(f"[pod-p2] huggingface_hub {huggingface_hub.__version__}")
    import pyarrow
    print(f"[pod-p2] pyarrow {pyarrow.__version__}")
    print("[pod-p2] all imports OK")
except Exception as e:
    print(f"[pod-p2] IMPORT FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ── Download indexer from HF (SHA-pinned via HF_REV) ──────────────────────────
echo "[pod-p2] === download indexer from HF (rev=${HF_REV:-main}) ==="
curl -fsSL \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/datasets/${HF_REPO_ID}/resolve/${HF_REV:-main}/ingest_dataset_to_qdrant.py" \
    -o /workspace/ingest_dataset_to_qdrant.py
ls -l /workspace/ingest_dataset_to_qdrant.py

# Confirm --out-parquet and --amazon-category args are present
python3 -c "
import subprocess, sys
r = subprocess.run(['python3', '/workspace/ingest_dataset_to_qdrant.py', '--help'],
                   capture_output=True, text=True)
missing = []
for arg in ('--out-parquet', '--amazon-category'):
    if arg not in r.stdout:
        missing.append(arg)
if missing:
    print(f'ERROR: missing args in indexer --help: {missing}', file=sys.stderr)
    sys.exit(1)
else:
    print('[pod-p2] --out-parquet + --amazon-category confirmed in indexer')
"

mkdir -p /workspace/out

ENCODE_BATCH=256   # GPU batch — T4/3090 can handle 256 at 384-d MiniLM
OUT=/workspace/out

# ── Phase-2 ingest: embed to parquet ──────────────────────────────────────────
# ABO   ~147k rows  --limit 150000
# Rebrickable  ~20k rows  --full (no meaningful limit needed)
# Amazon Electronics         --limit 150000
# Amazon Home_and_Kitchen    --limit 150000
# Amazon Clothing_Shoes_and_Jewelry --limit 150000
# Total target: ~570k rows.
#
# NOTE on ABO shard 0: documented as containing Dutch/NL items; the mapper
# _abo_multilang_value prefers en_US/en_GB/en tags first, so Dutch names are
# only used as fallback when no English value exists.  Translation happens at
# query time — Dutch items are safe to ingest as-is.

echo "[pod-p2] === Phase-2: ABO (furniture/home) cap=150000 → parquet ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset abo \
    --full \
    --limit 150000 \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet "${OUT}" \
    || echo "[pod-p2] WARN: abo ingest failed, continuing"

echo "[pod-p2] === Phase-2: Rebrickable (LEGO) full → parquet ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset rebrickable \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet "${OUT}" \
    || echo "[pod-p2] WARN: rebrickable ingest failed, continuing"

echo "[pod-p2] === Phase-2: Amazon Electronics cap=150000 → parquet ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset amazon \
    --amazon-category Electronics \
    --full \
    --limit 150000 \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet "${OUT}" \
    || echo "[pod-p2] WARN: amazon Electronics ingest failed, continuing"

echo "[pod-p2] === Phase-2: Amazon Home_and_Kitchen cap=150000 → parquet ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset amazon \
    --amazon-category Home_and_Kitchen \
    --full \
    --limit 150000 \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet "${OUT}" \
    || echo "[pod-p2] WARN: amazon Home_and_Kitchen ingest failed, continuing"

echo "[pod-p2] === Phase-2: Amazon Clothing_Shoes_and_Jewelry cap=150000 → parquet ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset amazon \
    --amazon-category Clothing_Shoes_and_Jewelry \
    --full \
    --limit 150000 \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet "${OUT}" \
    || echo "[pod-p2] WARN: amazon Clothing_Shoes_and_Jewelry ingest failed, continuing"

# ── Summary of produced parquet files ────────────────────────────────────────
echo "[pod-p2] === parquet output summary ==="
ls -lh "${OUT}"/file_*.parquet 2>/dev/null || echo "[pod-p2] WARNING: no parquet files found in ${OUT}"
TOTAL_ROWS=$(python3 -c "
import pyarrow.parquet as pq, pathlib, sys
total = 0
for f in sorted(pathlib.Path('${OUT}').glob('file_*.parquet')):
    t = pq.read_metadata(str(f))
    total += t.num_rows
print(total)
" 2>/dev/null || echo "unknown")
echo "[pod-p2] total rows across all parquet shards: ${TOTAL_ROWS}"

# ── Upload parquet/ shards + write DONE or FAILED marker ──────────────────────
echo "[pod-p2] === upload parquet shards to HF (or write FAILED marker) ==="
python3 - <<PYEOF
import os, pathlib, datetime, sys
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_REPO_ID"]
out_dir = pathlib.Path("/workspace/out")
files = sorted(out_dir.glob("file_*.parquet"))
ts = datetime.datetime.utcnow().isoformat()

if not files:
    print("[pod-p2] ERROR: no parquet files produced by any dataset — writing FAILED_DATASETS_P2 marker.")
    marker = f"FAILED at {ts}\nno parquet shards found in /workspace/out\n"
    api.upload_file(
        path_or_fileobj=marker.encode(),
        path_in_repo="FAILED_DATASETS_P2",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="FAILED marker — no parquet output (Phase-2)",
    )
    print(f"[pod-p2] FAILED_DATASETS_P2 uploaded at {ts}")
    sys.exit(0)   # not a fatal error for the outer script; log is still uploaded

print(f"[pod-p2] uploading {len(files)} parquet file(s) to {repo_id} ...")
for f in files:
    dest = f"parquet/{f.name}"
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}  ({size_mb:.1f} MB) -> {dest}")
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=dest,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"p2 parquet shard {f.name}",
    )
print(f"[pod-p2] all {len(files)} shards uploaded")

# Write DONE_DATASETS_P2 marker
total_rows = "${TOTAL_ROWS}"
marker = f"DONE at {ts}\ntotal_rows={total_rows}\nshards={len(files)}\nphase=2\n"
api.upload_file(
    path_or_fileobj=marker.encode(),
    path_in_repo="DONE_DATASETS_P2",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="DONE marker Phase-2",
)
print(f"[pod-p2] DONE_DATASETS_P2 uploaded at {ts}")
PYEOF

# ── Upload running log ─────────────────────────────────────────────────────────
echo "[pod-p2] === upload datasets_pod_p2.log to HF ==="
python3 - <<'PYEOF' || true
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/workspace/datasets_pod_p2.log",
    path_in_repo="datasets_pod_p2.log",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    commit_message="pod-p2 log (progress)",
)
print("[pod-p2] log uploaded")
PYEOF

echo "[pod-p2] === ALL DONE at $(date) ==="
echo "[pod-p2] Pod idle — manager controls terminate."
# Do NOT terminate — manager does it after verifying parquet files on HF.
sleep 7200
