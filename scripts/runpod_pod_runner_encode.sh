#!/usr/bin/env bash
# Encode-only pod runner.  Encodes 2.25M rows to vectors/payloads shards,
# uploads to HF in ONE folder commit (~5 commits total → well under rate limit).
set -euo pipefail
exec > >(tee -a /workspace/pod.log) 2>&1

LOG=/workspace/pod.log
echo "[pod] start at $(date)"
echo "[pod] HF_REPO_ID=${HF_REPO_ID}"

upload_log() {
    cp "$LOG" /workspace/pod-current.log 2>/dev/null || true
    python3 - <<EOF || true
try:
    from huggingface_hub import HfApi
    HfApi(token="${HF_TOKEN}").upload_file(
        path_or_fileobj="/workspace/pod-current.log",
        path_in_repo="pod_encode.log",
        repo_id="${HF_REPO_ID}",
        repo_type="dataset",
    )
    print("[pod] log uploaded as pod_encode.log")
except Exception as e:
    print(f"[pod] log upload failed: {e}")
EOF
}

trap upload_log EXIT
trap 'echo "[pod] ERROR at line $LINENO"' ERR

echo "[pod] python: $(which python3) $(python3 --version)"

echo "[pod] === force-remove conda's stale transformers ==="
pip uninstall -y transformers tokenizers sentence-transformers 2>&1 | tail -5
rm -rf /opt/conda/lib/python3.11/site-packages/transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/sentence_transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/tokenizers* 2>/dev/null || true

echo "[pod] === pip install fresh ==="
pip install --no-cache-dir --upgrade pip 2>&1 | tail -3
pip install --no-cache-dir \
    "transformers==4.46.*" \
    "tokenizers==0.20.*" \
    "sentence-transformers==3.*" \
    "datasets==2.*" \
    "huggingface_hub==0.25.*" \
    "pyarrow" "pandas" 2>&1 | tail -15

echo "[pod] === verify imports ==="
python3 -c "
import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')
import sentence_transformers; print(f'sentence-transformers {sentence_transformers.__version__}')
import datasets; print(f'datasets {datasets.__version__}')
import pyarrow, pandas
print('all imports OK')
"

cd /workspace

echo "[pod] === download build_encode_only.py from HF ==="
curl -fsSL \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/datasets/${HF_REPO_ID}/resolve/main/build_encode_only.py?ts=$(date +%s)" \
  -o encode.py
ls -l encode.py

echo "[pod] === run encoder (GPU only, no Qdrant) ==="
HF_TOKEN="${HF_TOKEN}" python3 encode.py

echo "[pod] === list shards ==="
ls -lh /workspace/vectors_*.npy /workspace/payloads_*.parquet 2>/dev/null | head -30
TOTAL_SIZE=$(du -sh /workspace/vectors_*.npy /workspace/payloads_*.parquet 2>/dev/null | tail -1)
echo "Total shards size: ${TOTAL_SIZE}"

echo "[pod] === upload shards to HF in ONE folder commit ==="
python3 - <<EOF
from huggingface_hub import HfApi
from pathlib import Path
api = HfApi(token="${HF_TOKEN}")
files_dir = "/workspace"
api.upload_folder(
    folder_path=files_dir,
    repo_id="${HF_REPO_ID}",
    repo_type="dataset",
    allow_patterns=["vectors_*.npy", "payloads_*.parquet"],
    commit_message="encoded vectors + payloads (variant X)",
)
print("[pod] all shards uploaded in ONE commit")
EOF

echo "[pod] DONE_ENCODING at $(date)"
sleep 3600
