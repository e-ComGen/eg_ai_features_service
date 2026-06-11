#!/usr/bin/env bash
# Start qdrant server (detached, survives the launching wsl session).
set -e
cd "$HOME/qdrant"
mkdir -p storage snapshots

# Already running?
if curl -fsS http://127.0.0.1:6333/healthz >/dev/null 2>&1; then
  echo "ALREADY RUNNING"
  curl -s http://127.0.0.1:6333/ | head -c 300
  exit 0
fi

export QDRANT__SERVICE__HOST=0.0.0.0
export QDRANT__SERVICE__HTTP_PORT=6333
export QDRANT__SERVICE__GRPC_PORT=6334
export QDRANT__STORAGE__STORAGE_PATH="$HOME/qdrant/storage"
export QDRANT__STORAGE__SNAPSHOTS_PATH="$HOME/qdrant/snapshots"

setsid nohup ./qdrant >"$HOME/qdrant/qdrant.log" 2>&1 < /dev/null &
echo "launched pid $!"

# Wait up to 40s for health.
for i in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:6333/healthz >/dev/null 2>&1; then
    echo "HEALTHY after ${i}s"
    curl -s http://127.0.0.1:6333/collections
    echo
    exit 0
  fi
  sleep 1
done
echo "DID NOT BECOME HEALTHY — last log lines:"
tail -30 "$HOME/qdrant/qdrant.log"
exit 1
