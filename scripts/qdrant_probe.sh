#!/usr/bin/env bash
# Confirm the category-filtered vector search works on the server (no 4xx).
set -e
vec=$(python3 -c "print('['+','.join(['0.05']*384)+']')")
echo "=== filtered query: MatchText categories ~ 'Детские товары' ==="
curl -sS -X POST 'http://127.0.0.1:6333/collections/ozon_products/points/query' \
  -H 'Content-Type: application/json' \
  --data-binary "{\"query\":${vec},\"limit\":3,\"with_payload\":[\"name\"],\"filter\":{\"must\":[{\"key\":\"categories\",\"match\":{\"text\":\"Детские товары\"}}]}}" | head -c 800
echo; echo "=== unfiltered query (sanity) ==="
curl -sS -X POST 'http://127.0.0.1:6333/collections/ozon_products/points/query' \
  -H 'Content-Type: application/json' \
  --data-binary "{\"query\":${vec},\"limit\":2,\"with_payload\":[\"name\"]}" | head -c 500
echo
