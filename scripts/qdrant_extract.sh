#!/usr/bin/env bash
set -e
DEST="$HOME/qdrant"
mkdir -p "$DEST"
cd "$DEST"
cp "/mnt/c/Users/Venya/AppData/Local/Temp/qdrant-v1.18.0-linux-gnu.tar.gz" ./qdrant.tar.gz
tar xzf qdrant.tar.gz
echo "=== contents ==="
ls -la
echo "=== version ==="
./qdrant --version
