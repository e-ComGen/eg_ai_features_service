#!/usr/bin/env bash
# Copy snapshot into qdrant's snapshots dir, then recover (path-restriction fix).
set -e
SRC="/mnt/c/Users/Venya/PycharmProjects/CpAiFeatures-web-fetch/app/services/enrichment/strategies/dictionaries/data/ozon_rag.snapshot"
DST="$HOME/qdrant/snapshots/ozon_rag.snapshot"
mkdir -p "$HOME/qdrant/snapshots"

if [ ! -f "$DST" ]; then
  echo "=== copying 6.2G snapshot to ext4 (this is the slow part) ==="; date
  cp "$SRC" "$DST"
  date
fi
echo "copied:"; ls -lh "$DST"

echo "=== recover into collection ozon_products ==="; date
curl -sS -X PUT 'http://127.0.0.1:6333/collections/ozon_products/snapshots/recover?wait=true' \
  -H 'Content-Type: application/json' \
  --data-binary "{\"location\":\"file://${DST}\"}"
echo; date
echo "=== collection info ==="
curl -s 'http://127.0.0.1:6333/collections/ozon_products'
echo
