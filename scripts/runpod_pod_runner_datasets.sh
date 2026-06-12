#!/usr/bin/env bash
# Pod run.sh for datasets_rag Phase-1 indexing (parquet path).
#
# Flow: install deps → verify CUDA → embed Phase-1 datasets → write parquet shards
#       → upload parquet/ to HF → DONE_DATASETS marker.
#
# Required env: HF_TOKEN, HF_REPO_ID
# NO Qdrant is installed or started on the pod. Qdrant upsert happens locally.
set -euo pipefail
exec > >(tee -a /workspace/datasets_pod.log) 2>&1

LOG=/workspace/datasets_pod.log
echo "[pod] start at $(date)"
echo "[pod] HF_REPO_ID=${HF_REPO_ID}"

# ── Upload log on exit (success or failure) ────────────────────────────────────
upload_log() {
    python3 - <<'PYEOF' || true
import os
try:
    from huggingface_hub import HfApi
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj="/workspace/datasets_pod.log",
        path_in_repo="datasets_pod.log",
        repo_id=os.environ["HF_REPO_ID"],
        repo_type="dataset",
        commit_message="pod log (final)",
    )
    print("[pod] log uploaded to HF")
except Exception as e:
    print(f"[pod] log upload failed: {e}")
PYEOF
}
trap upload_log EXIT
trap 'echo "[pod] ERROR at line $LINENO"' ERR

echo "[pod] python: $(which python3) $(python3 --version)"

# ── Clean stale conda packages that conflict with sentence-transformers ─────────
echo "[pod] === clean stale conda packages ==="
pip uninstall -y transformers tokenizers sentence-transformers 2>&1 | tail -5 || true
rm -rf /opt/conda/lib/python3.11/site-packages/transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/sentence_transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/tokenizers* 2>/dev/null || true

# ── Deps ───────────────────────────────────────────────────────────────────────
echo "[pod] === pip install ==="
pip install --no-cache-dir --upgrade pip 2>&1 | tail -3
pip install --no-cache-dir \
    "transformers==4.46.*" \
    "tokenizers==0.20.*" \
    "sentence-transformers==3.*" \
    "datasets==2.*" \
    "huggingface_hub==0.25.*" \
    "python-dotenv" "pyarrow" "pandas" "boto3" "fsspec" "py7zr" 2>&1 | tail -20

# ── Verify CUDA is available ───────────────────────────────────────────────────
echo "[pod] === verify torch CUDA ==="
python3 - <<'PYEOF'
import sys
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"[pod] torch {torch.__version__}, cuda={cuda_ok}")
    if not cuda_ok:
        print("[pod] WARNING: CUDA not available — embedding will be slow on CPU", file=sys.stderr)
    import sentence_transformers
    print(f"[pod] sentence-transformers {sentence_transformers.__version__}")
    import datasets
    print(f"[pod] datasets {datasets.__version__}")
    import huggingface_hub
    print(f"[pod] huggingface_hub {huggingface_hub.__version__}")
    import pyarrow
    print(f"[pod] pyarrow {pyarrow.__version__}")
    print("[pod] all imports OK")
except Exception as e:
    print(f"[pod] IMPORT FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ── Download indexer from HF ───────────────────────────────────────────────────
echo "[pod] === download indexer from HF ==="
curl -fsSL \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/datasets/${HF_REPO_ID}/resolve/${HF_REV:-main}/ingest_dataset_to_qdrant.py" \
    -o /workspace/ingest_dataset_to_qdrant.py
ls -l /workspace/ingest_dataset_to_qdrant.py

# Confirm --out-parquet arg is present
python3 -c "
import subprocess, sys
r = subprocess.run(['python3', '/workspace/ingest_dataset_to_qdrant.py', '--help'],
                   capture_output=True, text=True)
if '--out-parquet' not in r.stdout:
    print('ERROR: --out-parquet not in --help; wrong indexer version uploaded', file=sys.stderr)
    sys.exit(1)
else:
    print('[pod] --out-parquet arg confirmed in indexer')
"

mkdir -p /workspace/out

ENCODE_BATCH=256   # GPU batch — T4/3090 can handle 256 at 384-d MiniLM
OUT=/workspace/out

# ── Phase-1 ingest: embed to parquet ─────────────────────────────────────────
# OFF (food)  ~400k rows cap  |  OBF (beauty) full  |  OPFF (pet food) full  |  IKEA full
# Total target: ~<1M rows.

echo "[pod] === Phase-1: OFF (food) cap=400000 → parquet ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset off \
    --full \
    --limit 400000 \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet "${OUT}"

echo "[pod] === Phase-1: OBF (beauty) full → parquet ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset obf \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet "${OUT}"

echo "[pod] === Phase-1: OPFF (pet food) full → parquet ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset opff \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet "${OUT}"

echo "[pod] === Phase-1: IKEA full → parquet ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset ikea \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet "${OUT}"

# ── Summary of produced parquet files ────────────────────────────────────────
echo "[pod] === parquet output summary ==="
ls -lh "${OUT}"/file_*.parquet 2>/dev/null || echo "[pod] WARNING: no parquet files found in ${OUT}"
TOTAL_ROWS=$(python3 -c "
import pyarrow.parquet as pq, pathlib, sys
total = 0
for f in sorted(pathlib.Path('${OUT}').glob('file_*.parquet')):
    t = pq.read_metadata(str(f))
    total += t.num_rows
print(total)
" 2>/dev/null || echo "unknown")
echo "[pod] total rows across all parquet shards: ${TOTAL_ROWS}"

# ── Upload parquet/ shards to HF ──────────────────────────────────────────────
echo "[pod] === upload parquet shards to HF ==="
python3 - <<PYEOF
import os, pathlib
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_REPO_ID"]
out_dir = pathlib.Path("/workspace/out")
files = sorted(out_dir.glob("file_*.parquet"))

if not files:
    import sys
    print("[pod] ERROR: no parquet files to upload", file=sys.stderr)
    sys.exit(1)

print(f"[pod] uploading {len(files)} parquet file(s) to {repo_id} ...")
for f in files:
    dest = f"parquet/{f.name}"
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"  {f.name}  ({size_mb:.1f} MB) -> {dest}")
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=dest,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"parquet shard {f.name}",
    )
print(f"[pod] all {len(files)} shards uploaded")
PYEOF

# ── Upload running log ─────────────────────────────────────────────────────────
echo "[pod] === upload datasets_pod.log to HF ==="
python3 - <<'PYEOF' || true
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/workspace/datasets_pod.log",
    path_in_repo="datasets_pod.log",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    commit_message="pod log (progress)",
)
print("[pod] log uploaded")
PYEOF

# ── DONE marker ───────────────────────────────────────────────────────────────
echo "[pod] === upload DONE_DATASETS marker ==="
python3 - <<PYEOF
import os, datetime
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
ts = datetime.datetime.utcnow().isoformat()
marker = f"DONE at {ts}\ntotal_rows={TOTAL_ROWS}\n"
api.upload_file(
    path_or_fileobj=marker.encode(),
    path_in_repo="DONE_DATASETS",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    commit_message="DONE marker",
)
print(f"[pod] DONE_DATASETS uploaded at {ts}")
PYEOF

echo "[pod] === ALL DONE at $(date) ==="
echo "[pod] Pod idle — manager controls terminate."
# Do NOT terminate — manager does it after verifying parquet files on HF.
sleep 7200
