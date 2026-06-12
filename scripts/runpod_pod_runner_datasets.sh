#!/usr/bin/env bash
# Pod run.sh for datasets_rag indexing job.
# Required env: HF_TOKEN, HF_REPO_ID
# Flow: install deps → start qdrant v1.18.0 → create datasets_rag collection (on_disk=true)
#       → run ingest for Phase-1 datasets with caps → snapshot → upload to HF → DONE marker.
set -euo pipefail
exec > >(tee -a /workspace/pod.log) 2>&1

LOG=/workspace/pod.log
echo "[pod] start at $(date)"
echo "[pod] HF_REPO_ID=${HF_REPO_ID}"

# ── Upload log on exit (success or failure) ────────────────────────────────────
upload_log() {
    cp "$LOG" /workspace/pod-current.log 2>/dev/null || true
    python3 - <<'PYEOF' || true
import os
try:
    from huggingface_hub import HfApi
    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
        path_or_fileobj="/workspace/pod-current.log",
        path_in_repo="datasets_pod.log",
        repo_id=os.environ["HF_REPO_ID"],
        repo_type="dataset",
    )
    print("[pod] log uploaded to HF")
except Exception as e:
    print(f"[pod] log upload failed: {e}")
PYEOF
}
trap upload_log EXIT
trap 'echo "[pod] ERROR at line $LINENO"' ERR

echo "[pod] python: $(which python3) $(python3 --version)"

# ── Deps ───────────────────────────────────────────────────────────────────────
echo "[pod] === clean stale conda packages ==="
pip uninstall -y transformers tokenizers sentence-transformers 2>&1 | tail -5 || true
rm -rf /opt/conda/lib/python3.11/site-packages/transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/sentence_transformers* 2>/dev/null || true
rm -rf /opt/conda/lib/python3.11/site-packages/tokenizers* 2>/dev/null || true

echo "[pod] === pip install ==="
pip install --no-cache-dir --upgrade pip 2>&1 | tail -3
pip install --no-cache-dir \
    "transformers==4.46.*" \
    "tokenizers==0.20.*" \
    "sentence-transformers==3.*" \
    "qdrant-client==1.*" \
    "datasets==2.*" \
    "huggingface_hub==0.25.*" \
    "python-dotenv" "pyarrow" "pandas" "boto3" "py7zr" "fsspec" 2>&1 | tail -20

echo "[pod] === verify imports ==="
python3 - <<'PYEOF'
import sys
try:
    import torch; print(f"torch {torch.__version__}, cuda={torch.cuda.is_available()}")
    import sentence_transformers; print(f"sentence-transformers {sentence_transformers.__version__}")
    import qdrant_client; print("qdrant-client OK")
    import datasets; print(f"datasets {datasets.__version__}")
    import huggingface_hub; print(f"huggingface_hub {huggingface_hub.__version__}")
    print("all imports OK")
except Exception as e:
    print(f"IMPORT FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ── Qdrant v1.18.0 install ────────────────────────────────────────────────────
echo "[pod] === install qdrant v1.18.0 ==="
mkdir -p /workspace/qdrant_server
cd /workspace/qdrant_server
VER="v1.18.0"
BASE="https://github.com/qdrant/qdrant/releases/download/${VER}"
ok=0
for variant in qdrant-x86_64-unknown-linux-gnu.tar.gz qdrant-x86_64-unknown-linux-musl.tar.gz; do
    echo "  Trying ${variant} ..."
    if curl -fsSL -o qdrant.tar.gz "${BASE}/${variant}"; then
        echo "  Downloaded ${variant}"
        ok=1
        break
    fi
done
if [ "$ok" != "1" ]; then
    echo "FAILED to download qdrant binary" >&2; exit 1
fi
tar xzf qdrant.tar.gz
chmod +x qdrant
echo "[pod] qdrant binary: $(./qdrant --version 2>&1 | head -1)"

# ── Qdrant config (low-RAM: on_disk_payload + no HNSW during build) ───────────
mkdir -p /workspace/qdrant_storage /workspace/qdrant_snapshots
cat > /workspace/qdrant_server/config.yaml <<'YAMLEOF'
log_level: INFO

storage:
  storage_path: /workspace/qdrant_storage
  snapshots_path: /workspace/qdrant_snapshots
  on_disk_payload: true
  optimizers:
    indexing_threshold_kb: 0
    default_segment_number: 4

service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334

cluster:
  enabled: false
YAMLEOF

# ── Start Qdrant ──────────────────────────────────────────────────────────────
echo "[pod] === start qdrant server ==="
cd /workspace/qdrant_server
nohup ./qdrant --config-path /workspace/qdrant_server/config.yaml \
    >/workspace/qdrant.log 2>&1 &
QDRANT_PID=$!
echo "[pod] qdrant pid=${QDRANT_PID}"

# Wait up to 60s for health
for i in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:6333/healthz >/dev/null 2>&1; then
        echo "[pod] qdrant HEALTHY after ${i}s"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo "[pod] qdrant did not start" >&2
        tail -30 /workspace/qdrant.log >&2
        exit 1
    fi
    sleep 1
done

# ── Create datasets_rag collection with on_disk vectors (low-RAM safeguard) ───
echo "[pod] === create datasets_rag collection ==="
python3 - <<'PYEOF'
import sys, time
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, VectorsConfig,
        HnswConfigDiff, OptimizersConfigDiff,
        OnDiskConfig,
    )
except ImportError as e:
    sys.exit(f"[pod] qdrant-client import error: {e}")

client = QdrantClient(url="http://127.0.0.1:6333", timeout=60)

COLLECTION = "datasets_rag"
existing = [c.name for c in client.get_collections().collections]
if COLLECTION in existing:
    print(f"[pod] collection '{COLLECTION}' already exists — skipping create")
else:
    print(f"[pod] creating '{COLLECTION}' with on_disk vectors + on_disk payload ...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
            on_disk=True,           # vectors live on disk, not RAM — key low-RAM safeguard
        ),
        # Keep HNSW small during build so we don't blow VRAM/RAM
        hnsw_config=HnswConfigDiff(
            m=16,
            ef_construct=100,
            on_disk=True,           # HNSW graph also on disk
        ),
        optimizers_config=OptimizersConfigDiff(
            indexing_threshold=0,   # disable indexing during bulk build
        ),
        on_disk_payload=True,       # payload also on disk
    )
    info = client.get_collection(COLLECTION)
    print(f"[pod] created: {info.config.params.vectors}")
print("[pod] collection ready")
PYEOF

# ── Download ingest_dataset_to_qdrant.py from HF ─────────────────────────────
echo "[pod] === download indexer from HF ==="
cd /workspace
curl -fsSL \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    "https://huggingface.co/datasets/${HF_REPO_ID}/resolve/main/ingest_dataset_to_qdrant.py" \
    -o ingest_dataset_to_qdrant.py
ls -l ingest_dataset_to_qdrant.py

# Verify the --limit arg exists in the downloaded script (we added it)
python3 -c "
import subprocess, sys
r = subprocess.run(['python3', '/workspace/ingest_dataset_to_qdrant.py', '--help'],
                   capture_output=True, text=True)
if '--limit' not in r.stdout:
    print('WARNING: --limit not in --help output; check indexer version', file=sys.stderr)
else:
    print('[pod] --limit arg confirmed in indexer')
"

QDRANT_URL="http://127.0.0.1:6333"
COLLECTION="datasets_rag"
ENCODE_BATCH=256   # GPU batch — tune up if VRAM allows
UPSERT_BATCH=512

# ── Phase-1 ingest: capped proof run ─────────────────────────────────────────
# OFF (food) ~400k rows | OBF (beauty) all ~64k | OPFF (pet food) all ~300k | IKEA all ~25k
# Total target: ~<1M rows.  Use --limit to cap OFF; others pass --full (they fit).

echo "[pod] === Phase-1 ingest: OFF (food) cap=400000 ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset off \
    --collection "${COLLECTION}" \
    --qdrant-url "${QDRANT_URL}" \
    --full \
    --limit 400000 \
    --encode-batch "${ENCODE_BATCH}" \
    --batch "${UPSERT_BATCH}" \
    --no-text-index \
    --recreate 2>&1

echo "[pod] === Phase-1 ingest: OBF (beauty) full ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset obf \
    --collection "${COLLECTION}" \
    --qdrant-url "${QDRANT_URL}" \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --batch "${UPSERT_BATCH}" \
    --no-text-index 2>&1

echo "[pod] === Phase-1 ingest: OPFF (pet food) full ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset opff \
    --collection "${COLLECTION}" \
    --qdrant-url "${QDRANT_URL}" \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --batch "${UPSERT_BATCH}" \
    --no-text-index 2>&1

echo "[pod] === Phase-1 ingest: IKEA full ==="
HF_TOKEN="${HF_TOKEN}" python3 /workspace/ingest_dataset_to_qdrant.py \
    --dataset ikea \
    --collection "${COLLECTION}" \
    --qdrant-url "${QDRANT_URL}" \
    --full \
    --encode-batch "${ENCODE_BATCH}" \
    --batch "${UPSERT_BATCH}" \
    --no-text-index 2>&1

# ── Build text index after all ingest complete ────────────────────────────────
echo "[pod] === build categories text index ==="
python3 - <<'PYEOF'
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType
client = QdrantClient(url="http://127.0.0.1:6333", timeout=120)
try:
    client.create_payload_index(
        collection_name="datasets_rag",
        field_name="categories",
        field_schema=PayloadSchemaType.TEXT,
    )
    print("[pod] text index on categories created")
except Exception as e:
    if "already exists" in str(e).lower() or "conflict" in str(e).lower():
        print("[pod] text index already exists")
    else:
        print(f"[pod] WARNING: text index failed: {e}")
PYEOF

# ── Collection info before snapshot ───────────────────────────────────────────
echo "[pod] === collection info ==="
curl -s http://127.0.0.1:6333/collections/datasets_rag | python3 -c "
import sys, json
d = json.load(sys.stdin)
pts = d.get('result', {}).get('points_count', 'unknown')
print(f'[pod] datasets_rag: {pts} points')
"

# ── Snapshot ──────────────────────────────────────────────────────────────────
echo "[pod] === snapshot datasets_rag ==="
SNAPSHOT_RESPONSE=$(curl -sS -X POST \
    "http://127.0.0.1:6333/collections/datasets_rag/snapshots?wait=true" \
    -H "Content-Type: application/json")
echo "[pod] snapshot response: ${SNAPSHOT_RESPONSE}"

SNAPSHOT_NAME=$(python3 -c "
import sys, json
d = json.loads('''${SNAPSHOT_RESPONSE}''')
name = d.get('result', {}).get('name', '')
if not name:
    print('ERROR: no snapshot name in response', file=sys.stderr)
    sys.exit(1)
print(name)
")
echo "[pod] snapshot name: ${SNAPSHOT_NAME}"

SNAPSHOT_FILE="/workspace/qdrant_snapshots/${SNAPSHOT_NAME}"
echo "[pod] snapshot file: ${SNAPSHOT_FILE}"
ls -lh "${SNAPSHOT_FILE}"

# ── Tar the snapshot ──────────────────────────────────────────────────────────
echo "[pod] === creating qdrant_datasets.tar.gz ==="
cd /workspace/qdrant_snapshots
tar czf /workspace/qdrant_datasets.tar.gz "${SNAPSHOT_NAME}"
ls -lh /workspace/qdrant_datasets.tar.gz

# ── Upload to HF ──────────────────────────────────────────────────────────────
echo "[pod] === upload qdrant_datasets.tar.gz to HF ==="
python3 - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj="/workspace/qdrant_datasets.tar.gz",
    path_in_repo="qdrant_datasets.tar.gz",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    commit_message="datasets_rag snapshot Phase-1",
)
print("[pod] qdrant_datasets.tar.gz uploaded")
PYEOF

# ── DONE marker ───────────────────────────────────────────────────────────────
echo "[pod] === upload DONE marker ==="
python3 - <<'PYEOF'
import os, datetime
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
ts = datetime.datetime.utcnow().isoformat()
api.upload_file(
    path_or_fileobj=ts.encode(),
    path_in_repo="DONE_DATASETS",
    repo_id=os.environ["HF_REPO_ID"],
    repo_type="dataset",
    commit_message="DONE marker",
)
print(f"[pod] DONE marker uploaded at {ts}")
PYEOF

echo "[pod] === ALL DONE at $(date) ==="
echo "[pod] Pod will stay alive — manager controls terminate."
# Do NOT terminate — manager does it after verified download.
sleep 7200
