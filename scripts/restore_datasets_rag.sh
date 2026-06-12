#!/usr/bin/env bash
# Restore datasets_rag from HF snapshot to local Qdrant server (WSL, localhost:6333).
#
# Usage (run from WSL):
#   bash /mnt/c/Users/Venya/PycharmProjects/CpAiFeatures-web-fetch/scripts/restore_datasets_rag.sh
#
# Requires:
#   - HF_TOKEN env var  (or set in .env below)
#   - Local Qdrant v1.18.0 running at localhost:6333 (run qdrant_start.sh first)
#   - huggingface_hub installed:  pip install huggingface_hub
#
# Mirrors the qdrant_restore2.sh pattern:
#   download tar.gz → extract snapshot → copy to ~/qdrant/snapshots/ → PUT /recover
set -euo pipefail

# ── Config ─────────────────────────────────────────────────────────────────────
QDRANT_URL="http://127.0.0.1:6333"
COLLECTION="datasets_rag"
SNAPSHOT_DIR="$HOME/qdrant/snapshots"
WORK_DIR="/tmp/datasets_rag_restore"
TAR_NAME="qdrant_datasets.tar.gz"

# State file lives on the Windows filesystem — read hf_repo_id from it.
STATE_FILE="/mnt/c/Users/Venya/PycharmProjects/CpAiFeatures-web-fetch/scripts/.runpod_datasets_state.json"
ENV_FILE="/mnt/c/Users/Venya/PycharmProjects/CpAiFeatures-web-fetch/.env"

# ── Load HF_TOKEN ──────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ] && [ -f "$ENV_FILE" ]; then
    # Parse HF_TOKEN=... from .env (handle optional quotes)
    HF_TOKEN=$(grep -E '^HF_TOKEN=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'" | tr -d '[:space:]') || true
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[restore] ERROR: HF_TOKEN not set and could not read from .env" >&2
    exit 1
fi
echo "[restore] HF_TOKEN: ${HF_TOKEN:0:8}..."

# ── Read hf_repo_id from state file ───────────────────────────────────────────
if [ -f "$STATE_FILE" ]; then
    HF_REPO_ID=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d['hf_repo_id'])")
    echo "[restore] hf_repo_id from state: ${HF_REPO_ID}"
else
    echo "[restore] ERROR: state file not found: $STATE_FILE" >&2
    echo "[restore] Pass HF_REPO_ID as first argument or run runpod_spawn_datasets.py first." >&2
    if [ -n "${1:-}" ]; then
        HF_REPO_ID="$1"
        echo "[restore] Using arg: ${HF_REPO_ID}"
    else
        exit 1
    fi
fi

# Allow override via argument
if [ -n "${1:-}" ]; then
    HF_REPO_ID="$1"
    echo "[restore] HF_REPO_ID overridden by arg: ${HF_REPO_ID}"
fi

# ── Check local Qdrant is healthy ──────────────────────────────────────────────
echo "[restore] Checking local Qdrant at ${QDRANT_URL} ..."
if ! curl -fsS "${QDRANT_URL}/healthz" >/dev/null 2>&1; then
    echo "[restore] ERROR: Qdrant not reachable at ${QDRANT_URL}" >&2
    echo "[restore] Start it first: bash ~/path/to/qdrant_start.sh" >&2
    exit 1
fi
echo "[restore] Qdrant is healthy."

# ── Download qdrant_datasets.tar.gz from HF ───────────────────────────────────
mkdir -p "$WORK_DIR"
TAR_PATH="$WORK_DIR/$TAR_NAME"

echo "[restore] Downloading ${TAR_NAME} from https://huggingface.co/datasets/${HF_REPO_ID} ..."
python3 - <<PYEOF
import os, sys, urllib.request

url = f"https://huggingface.co/datasets/${HF_REPO_ID}/resolve/main/${TAR_NAME}"
token = "${HF_TOKEN}"
dest = "${TAR_PATH}"

req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
print(f"[restore] GET {url}")
with urllib.request.urlopen(req, timeout=300) as r:
    total = 0
    with open(dest, "wb") as f:
        while True:
            chunk = r.read(4 * 1024 * 1024)  # 4 MB chunks
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"  {total // (1024*1024)} MB", end="\r", flush=True)
size_mb = os.path.getsize(dest) / (1024*1024)
print(f"\n[restore] Saved: {dest} ({size_mb:.1f} MB)")
PYEOF

# ── Extract snapshot from tar ─────────────────────────────────────────────────
echo "[restore] Extracting snapshot from tar ..."
cd "$WORK_DIR"
tar xzf "$TAR_NAME"

# Find the .snapshot file(s)
SNAPSHOT_FILE=$(find "$WORK_DIR" -name "*.snapshot" | head -1)
if [ -z "$SNAPSHOT_FILE" ]; then
    echo "[restore] ERROR: no .snapshot file found in tar" >&2
    ls -la "$WORK_DIR"
    exit 1
fi
SNAPSHOT_BASENAME=$(basename "$SNAPSHOT_FILE")
echo "[restore] Snapshot file: ${SNAPSHOT_FILE} ($(du -sh "$SNAPSHOT_FILE" | cut -f1))"

# ── Copy snapshot into ~/qdrant/snapshots/ (path-restriction fix) ─────────────
# Qdrant only allows recover from its own snapshots directory.
mkdir -p "$SNAPSHOT_DIR"
DST="$SNAPSHOT_DIR/$SNAPSHOT_BASENAME"

if [ -f "$DST" ]; then
    echo "[restore] Snapshot already at dst: ${DST} — skipping copy"
else
    echo "[restore] Copying snapshot to ${DST} ..."
    date
    cp "$SNAPSHOT_FILE" "$DST"
    date
fi
echo "[restore] dst: $(ls -lh "$DST")"

# ── Recover into Qdrant ───────────────────────────────────────────────────────
echo "[restore] === PUT /collections/${COLLECTION}/snapshots/recover ==="
date
curl -sS -X PUT \
    "${QDRANT_URL}/collections/${COLLECTION}/snapshots/recover?wait=true" \
    -H "Content-Type: application/json" \
    --data-binary "{\"location\":\"file://${DST}\"}"
echo
date

# ── Verify: point count + sample search ───────────────────────────────────────
echo "[restore] === Verification ==="

echo "[restore] Collection info:"
curl -s "${QDRANT_URL}/collections/${COLLECTION}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
r = d.get('result', {})
pts = r.get('points_count', 'unknown')
status = r.get('status', 'unknown')
print(f'  status       : {status}')
print(f'  points_count : {pts}')
cfg = r.get('config', {}).get('params', {}).get('vectors', {})
print(f'  vectors_cfg  : {cfg}')
"

echo
echo "[restore] Sample search (query: 'шоколадная паста Нутелла') ..."
python3 - <<'PYEOF'
import sys
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vec = model.encode(["шоколадная паста Нутелла"], normalize_embeddings=True)[0].tolist()
except Exception as e:
    print(f"[restore] WARNING: embed failed ({e}) — using zero vector for count check")
    vec = [0.0] * 384

try:
    from qdrant_client import QdrantClient
    client = QdrantClient(url="http://127.0.0.1:6333", timeout=30)
    result = client.query_points(
        collection_name="datasets_rag",
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
except Exception as e:
    print(f"[restore] ERROR during search: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

echo
echo "[restore] === DONE ==="
echo "[restore] datasets_rag is live at ${QDRANT_URL}"
