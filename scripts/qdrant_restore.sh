#!/usr/bin/env bash
# Recover the ozon_rag collection snapshot into the running qdrant server.
set -e
SNAP="/mnt/c/Users/Venya/PycharmProjects/CpAiFeatures-web-fetch/app/services/enrichment/strategies/dictionaries/data/ozon_rag.snapshot"
echo "snapshot size:"; ls -lh "$SNAP"
echo "=== starting recover into collection ozon_products (wait=true, may take minutes) ==="
date
curl -sS -X PUT 'http://127.0.0.1:6333/collections/ozon_products/snapshots/recover?wait=true' \
  -H 'Content-Type: application/json' \
  --data-binary "{\"location\":\"file://${SNAP}\"}"
echo
date
echo "=== collection info after recover ==="
curl -s 'http://127.0.0.1:6333/collections/ozon_products'
echo
