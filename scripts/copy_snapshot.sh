#!/usr/bin/env bash
SNAPSHOT="/tmp/qdrant_snapshots/ozon_products/ozon_products-6554676618774636-2026-05-28-11-09-34.snapshot"
DEST="/mnt/c/Users/Venya/PycharmProjects/CpAiFeatures-web-fetch/app/services/enrichment/strategies/dictionaries/data/ozon_rag.snapshot"
echo "src=$SNAPSHOT"
echo "dst=$DEST"
START=$(date +%s)
cp "$SNAPSHOT" "$DEST"
END=$(date +%s)
echo "Copy: $((END-START))s"
ls -lh "$DEST"
