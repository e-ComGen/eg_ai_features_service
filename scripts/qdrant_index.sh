#!/usr/bin/env bash
set -e
echo "=== sample point payload keys ==="
curl -s -X POST 'http://127.0.0.1:6333/collections/ozon_products/points/scroll' \
  -H 'Content-Type: application/json' \
  --data-binary '{"limit":1,"with_payload":true,"with_vector":false}' | head -c 1200
echo; echo "=== create TEXT index on categories ==="
curl -sS -X PUT 'http://127.0.0.1:6333/collections/ozon_products/index?wait=true' \
  -H 'Content-Type: application/json' \
  --data-binary '{"field_name":"categories","field_schema":"text"}'
echo; echo "=== payload_schema now ==="
curl -s 'http://127.0.0.1:6333/collections/ozon_products' | grep -o '"payload_schema":{[^}]*}'
echo
