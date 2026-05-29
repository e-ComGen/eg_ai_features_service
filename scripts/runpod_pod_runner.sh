#!/usr/bin/env bash
# Runs on the RunPod pod.  Minimal-commit policy: ONE commit on success
# (qdrant.tar.gz), ONE commit on failure (pod.log via EXIT trap).
# Required env: HF_TOKEN, HF_REPO_ID
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
        path_in_repo="pod.log",
        repo_id="${HF_REPO_ID}",
        repo_type="dataset",
    )
    print("[pod] log uploaded to HF")
except Exception as e:
    print(f"[pod] log upload failed: {e}")
EOF
}

# Upload log ONCE on exit (success or failure).
trap upload_log EXIT
trap 'echo "[pod] ERROR at line $LINENO"' ERR

echo "[pod] python: $(which python3) $(python3 --version)"
echo "[pod] pip: $(which pip) $(pip --version 2>&1 | head -1)"

echo "[pod] === force-remove conda's stale transformers ==="
pip uninstall -y transformers tokenizers sentence-transformers 2>&1 | tail -5
# Also clear any leftover egg-info / metadata
rm -rf /opt/conda/lib/python3.11/site-packages/transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/sentence_transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/tokenizers* 2>/dev/null || true

echo "[pod] === pip install fresh ==="
pip install --no-cache-dir --upgrade pip 2>&1 | tail -3

echo "[pod] === actual install with == constraint ==="
# IMPORTANT: do NOT pipe to head with set -o pipefail — head closes stdin early,
# pip gets SIGPIPE, pipefail catches the failure and the whole script dies.
pip install --no-cache-dir \
    "transformers==4.46.*" \
    "tokenizers==0.20.*" \
    "sentence-transformers==3.*" \
    "qdrant-client==1.*" \
    "datasets==2.*" \
    "huggingface_hub==0.25.*" \
    "python-dotenv" "pyarrow" "pandas" 2>&1 | tail -20

echo "[pod] === verify what's actually installed ==="
pip show transformers | grep -E "^(Name|Version)"
pip show sentence-transformers | grep -E "^(Name|Version)"
pip show tokenizers | grep -E "^(Name|Version)"
pip show huggingface_hub | grep -E "^(Name|Version)"

echo "[pod] === verify imports ==="
python3 -c "
import sys
try:
    import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')
    import sentence_transformers; print(f'sentence-transformers {sentence_transformers.__version__}')
    import qdrant_client; print(f'qdrant-client OK (no __version__)')
    import datasets; print(f'datasets {datasets.__version__}')
    import huggingface_hub; print(f'huggingface_hub {huggingface_hub.__version__}')
    import pyarrow, pandas
    print('all imports OK')
except Exception as e:
    print(f'IMPORT FAILED: {type(e).__name__}: {e}', file=sys.stderr)
    sys.exit(1)
"

cd /workspace

echo "[pod] === download build_index.py from HF ==="
curl -fsSL \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  "https://huggingface.co/datasets/${HF_REPO_ID}/resolve/main/build_index.py" \
  -o build_index.py
ls -l build_index.py

echo "[pod] === run indexer ==="
HF_TOKEN="${HF_TOKEN}" python3 build_index.py --index-path /workspace/ozon_rag.qdrant 2>&1

echo "[pod] === tarball ==="
cd /workspace
tar czf qdrant.tar.gz ozon_rag.qdrant
ls -lh qdrant.tar.gz

echo "[pod] === upload qdrant.tar.gz to HF (ONE commit) ==="
python3 - <<EOF
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
api.upload_file(
    path_or_fileobj="/workspace/qdrant.tar.gz",
    path_in_repo="qdrant.tar.gz",
    repo_id="${HF_REPO_ID}",
    repo_type="dataset",
    commit_message="qdrant index result",
)
print("[pod] qdrant.tar.gz uploaded")
EOF

echo "[pod] DONE_INDEXING at $(date)"
sleep 7200  # keep-alive 2h
