#!/usr/bin/env bash
# Pod run.sh for datasets_rag ALL datasets (single comprehensive run).
#
# FIX for overwrite bug: each dataset gets its OWN subdir under /workspace/out/
# so shard names file_0000.parquet never collide between datasets.
#
# Datasets:
#   off          (food)               --limit 400000  → /workspace/out/off
#   obf          (beauty)             --full           → /workspace/out/obf
#   opff         (pet food)           --full           → /workspace/out/opff
#   ikea         (furniture)          --full           → /workspace/out/ikea
#   abo          (Amazon Berkeley)    --limit 150000   → /workspace/out/abo
#   rebrickable  (LEGO)               --full           → /workspace/out/rebrickable
#   amazon       Electronics          --limit 150000   → /workspace/out/amazon_electronics
#   amazon       Home_and_Kitchen     --limit 150000   → /workspace/out/amazon_home
#   amazon       Clothing_Shoes_and_Jewelry --limit 150000 → /workspace/out/amazon_clothing
#
# Upload: all /workspace/out/*/file_*.parquet → HF parquet_all/<subdir>_<filename>
#         (e.g. off_file_0000.parquet, abo_file_0000.parquet) — globally unique.
#
# Required env: HF_TOKEN, HF_REPO_ID
# SHA-pinned indexer fetch: HF_REV injected by runpod_spawn_datasets_all.py
# NO Qdrant on the pod. Upsert happens locally via restore_datasets_rag.py.
set -uo pipefail
exec > >(tee -a /workspace/datasets_pod_all.log) 2>&1

LOG=/workspace/datasets_pod_all.log
echo "[pod-all] start at $(date)"
echo "[pod-all] HF_REPO_ID=${HF_REPO_ID}"

# ── Upload log on exit (success or failure) ────────────────────────────────────
upload_log() {
    python3 - <<'PYEOF' || true
import os
try:
    from huggingface_hub import HfApi
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj="/workspace/datasets_pod_all.log",
        path_in_repo="datasets_pod_all.log",
        repo_id=os.environ["HF_REPO_ID"],
        repo_type="dataset",
        commit_message="pod-all log (final)",
    )
    print("[pod-all] log uploaded to HF")
except Exception as e:
    print(f"[pod-all] log upload failed: {e}")
PYEOF
}
trap upload_log EXIT
trap 'echo "[pod-all] ERROR at line $LINENO"' ERR

echo "[pod-all] python: $(which python3) $(python3 --version)"

# ── Clean stale conda packages that conflict with sentence-transformers ─────────
echo "[pod-all] === clean stale conda packages ==="
pip uninstall -y transformers tokenizers sentence-transformers 2>&1 | tail -5 || true
rm -rf /opt/conda/lib/python3.11/site-packages/transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/sentence_transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/tokenizers* 2>/dev/null || true

# ── Deps ───────────────────────────────────────────────────────────────────────
echo "[pod-all] === pip install ==="
pip install --no-cache-dir --upgrade pip 2>&1 | tail -3
pip install --no-cache-dir \
    "transformers==4.46.*" \
    "tokenizers==0.20.*" \
    "sentence-transformers==3.*" \
    "datasets==2.*" \
    "huggingface_hub==0.25.*" \
    "python-dotenv" "pyarrow" "pandas" "boto3" "fsspec" "py7zr" 2>&1 | tail -20

# ── Verify CUDA is available ───────────────────────────────────────────────────
echo "[pod-all] === verify torch CUDA ==="
python3 - <<'PYEOF'
import sys
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"[pod-all] torch {torch.__version__}, cuda={cuda_ok}")
    if not cuda_ok:
        print("[pod-all] WARNING: CUDA not available — embedding will be slow on CPU", file=sys.stderr)
    import sentence_transformers
    print(f"[pod-all] sentence-transformers {sentence_transformers.__version__}")
    import datasets
    print(f"[pod-all] datasets {datasets.__version__}")
    import huggingface_hub
    print(f"[pod-all] huggingface_hub {huggingface_hub.__version__}")
    import pyarrow
    print(f"[pod-all] pyarrow {pyarrow.__version__}")
    print("[pod-all] all imports OK")
except Exception as e:
    print(f"[pod-all] IMPORT FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ── Download indexer from HF (SHA-pinned via HF_REV) ──────────────────────────
echo "[pod-all] === download indexer from HF (rev=${HF_REV:-main}) ==="
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
    print('[pod-all] --out-parquet + --amazon-category confirmed in indexer')
"

ENCODE_BATCH=256   # GPU batch — T4/3090 can handle 256 at 384-d MiniLM

# ── Per-dataset ingest: each gets its OWN subdir (overwrite bug fix) ───────────

# OFF (food) cap=400000 → /workspace/out/off
mkdir -p /workspace/out/off
echo "[pod-all] === OFF (food) cap=400000 → /workspace/out/off ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset off \
    --full \
    --limit 400000 \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet /workspace/out/off \
    || echo "[pod-all] WARN: off ingest failed, continuing"

# Upload intermediate progress log
python3 - <<'PYEOF' || true
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/workspace/datasets_pod_all.log",
    path_in_repo="datasets_pod_all.log",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    commit_message="pod-all log (after off)",
)
PYEOF

# OBF (beauty) full → /workspace/out/obf
mkdir -p /workspace/out/obf
echo "[pod-all] === OBF (beauty) full → /workspace/out/obf ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset obf \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet /workspace/out/obf \
    || echo "[pod-all] WARN: obf ingest failed, continuing"

# OPFF (pet food) full → /workspace/out/opff
mkdir -p /workspace/out/opff
echo "[pod-all] === OPFF (pet food) full → /workspace/out/opff ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset opff \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet /workspace/out/opff \
    || echo "[pod-all] WARN: opff ingest failed, continuing"

# IKEA full → /workspace/out/ikea
mkdir -p /workspace/out/ikea
echo "[pod-all] === IKEA full → /workspace/out/ikea ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset ikea \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet /workspace/out/ikea \
    || echo "[pod-all] WARN: ikea ingest failed, continuing"

# Upload progress after Phase-1 group
python3 - <<'PYEOF' || true
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/workspace/datasets_pod_all.log",
    path_in_repo="datasets_pod_all.log",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    commit_message="pod-all log (after phase-1 group: off+obf+opff+ikea)",
)
PYEOF

# ABO (Amazon Berkeley Objects) cap=150000 → /workspace/out/abo
mkdir -p /workspace/out/abo
echo "[pod-all] === ABO (furniture/home) cap=150000 → /workspace/out/abo ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset abo \
    --full \
    --limit 150000 \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet /workspace/out/abo \
    || echo "[pod-all] WARN: abo ingest failed, continuing"

# Rebrickable (LEGO) full → /workspace/out/rebrickable
mkdir -p /workspace/out/rebrickable
echo "[pod-all] === Rebrickable (LEGO) full → /workspace/out/rebrickable ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset rebrickable \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet /workspace/out/rebrickable \
    || echo "[pod-all] WARN: rebrickable ingest failed, continuing"

# Amazon Electronics cap=150000 → /workspace/out/amazon_electronics
mkdir -p /workspace/out/amazon_electronics
echo "[pod-all] === Amazon Electronics cap=150000 → /workspace/out/amazon_electronics ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset amazon \
    --amazon-category Electronics \
    --full \
    --limit 150000 \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet /workspace/out/amazon_electronics \
    || echo "[pod-all] WARN: amazon Electronics ingest failed, continuing"

# Amazon Home_and_Kitchen cap=150000 → /workspace/out/amazon_home
mkdir -p /workspace/out/amazon_home
echo "[pod-all] === Amazon Home_and_Kitchen cap=150000 → /workspace/out/amazon_home ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset amazon \
    --amazon-category Home_and_Kitchen \
    --full \
    --limit 150000 \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet /workspace/out/amazon_home \
    || echo "[pod-all] WARN: amazon Home_and_Kitchen ingest failed, continuing"

# Amazon Clothing_Shoes_and_Jewelry cap=150000 → /workspace/out/amazon_clothing
mkdir -p /workspace/out/amazon_clothing
echo "[pod-all] === Amazon Clothing_Shoes_and_Jewelry cap=150000 → /workspace/out/amazon_clothing ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset amazon \
    --amazon-category Clothing_Shoes_and_Jewelry \
    --full \
    --limit 150000 \
    --encode-batch "${ENCODE_BATCH}" \
    --out-parquet /workspace/out/amazon_clothing \
    || echo "[pod-all] WARN: amazon Clothing_Shoes_and_Jewelry ingest failed, continuing"

# ── Summary of all produced parquet files ────────────────────────────────────
echo "[pod-all] === parquet output summary ==="
find /workspace/out -name "file_*.parquet" -ls 2>/dev/null || echo "[pod-all] WARNING: no parquet files found"
TOTAL_ROWS=$(python3 -c "
import pyarrow.parquet as pq, pathlib, sys
total = 0
for f in sorted(pathlib.Path('/workspace/out').rglob('file_*.parquet')):
    try:
        t = pq.read_metadata(str(f))
        total += t.num_rows
        print(f'  {f.relative_to(\"/workspace/out\")}  {t.num_rows:,} rows')
    except Exception as e:
        print(f'  WARNING: {f.name} unreadable: {e}')
print(f'TOTAL: {total}')
" 2>/dev/null || echo "unknown")
echo "[pod-all] total rows: ${TOTAL_ROWS}"

# ── Upload all shards to HF parquet_all/ with unique names ────────────────────
# Name format: <subdir>_<filename>  e.g. off_file_0000.parquet, abo_file_0000.parquet
echo "[pod-all] === upload all shards to HF parquet_all/ ==="
python3 - <<'PYEOF'
import os, pathlib, datetime, sys
import pyarrow.parquet as pq
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_REPO_ID"]
out_root = pathlib.Path("/workspace/out")
ts = datetime.datetime.utcnow().isoformat()

# Collect all shards: /workspace/out/<subdir>/file_NNNN.parquet
all_shards = []
for subdir in sorted(out_root.iterdir()):
    if not subdir.is_dir():
        continue
    for shard in sorted(subdir.glob("file_*.parquet")):
        # unique HF name: <subdir>_<filename>  e.g. off_file_0000.parquet
        unique_name = f"{subdir.name}_{shard.name}"
        all_shards.append((shard, unique_name))

if not all_shards:
    print("[pod-all] ERROR: no parquet files produced by any dataset — writing FAILED_ALL marker.")
    marker = f"FAILED at {ts}\nno parquet shards found in /workspace/out\n"
    api.upload_file(
        path_or_fileobj=marker.encode(),
        path_in_repo="FAILED_ALL",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="FAILED_ALL marker — no parquet output",
    )
    print(f"[pod-all] FAILED_ALL uploaded at {ts}")
    sys.exit(0)

print(f"[pod-all] uploading {len(all_shards)} parquet shard(s) to {repo_id}/parquet_all/ ...")
total_rows = 0
for shard_path, unique_name in all_shards:
    dest = f"parquet_all/{unique_name}"
    size_mb = shard_path.stat().st_size / (1024 * 1024)
    try:
        meta = pq.read_metadata(str(shard_path))
        rows = meta.num_rows
        total_rows += rows
    except Exception:
        rows = "?"
    print(f"  {shard_path.relative_to(out_root)}  ({size_mb:.1f} MB, {rows} rows) -> {dest}")
    api.upload_file(
        path_or_fileobj=str(shard_path),
        path_in_repo=dest,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"parquet_all shard {unique_name}",
    )
print(f"[pod-all] all {len(all_shards)} shards uploaded to parquet_all/")

# Write DONE_ALL marker — Python computes summary, no shell interpolation
subdirs_done = sorted(set(s[1].split("_file_")[0] for s in all_shards))
marker = (
    f"DONE at {ts}\n"
    f"total_rows={total_rows}\n"
    f"shards={len(all_shards)}\n"
    f"datasets={','.join(subdirs_done)}\n"
)
api.upload_file(
    path_or_fileobj=marker.encode(),
    path_in_repo="DONE_ALL",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="DONE_ALL marker",
)
print(f"[pod-all] DONE_ALL uploaded at {ts}")
PYEOF

# ── Upload final log ───────────────────────────────────────────────────────────
echo "[pod-all] === upload datasets_pod_all.log to HF ==="
python3 - <<'PYEOF' || true
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj="/workspace/datasets_pod_all.log",
    path_in_repo="datasets_pod_all.log",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    commit_message="pod-all log (final)",
)
print("[pod-all] log uploaded")
PYEOF

echo "[pod-all] === ALL DONE at $(date) ==="
echo "[pod-all] Pod idle — manager controls terminate."
# Do NOT terminate — manager does it after verifying parquet_all/ on HF.
sleep 7200
