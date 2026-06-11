#!/usr/bin/env bash
# Download qdrant server v1.18.0 (matches qdrant-client 1.18.0) into ~/qdrant.
set -e
VER=v1.18.0
DEST="$HOME/qdrant"
mkdir -p "$DEST"
cd "$DEST"

base="https://github.com/qdrant/qdrant/releases/download/${VER}"
ok=0
for variant in qdrant-x86_64-unknown-linux-gnu.tar.gz qdrant-x86_64-unknown-linux-musl.tar.gz; do
  echo "Trying ${variant} ..."
  if curl -fsSL -o qdrant.tar.gz "${base}/${variant}"; then
    echo "Downloaded ${variant}"
    ok=1
    break
  fi
done

if [ "$ok" != "1" ]; then
  echo "FAILED to download a binary archive for ${VER}. Listing available assets:"
  curl -s "https://api.github.com/repos/qdrant/qdrant/releases/tags/${VER}" | grep -oE 'name[^,]*tar.gz' | head -40
  exit 1
fi

tar xzf qdrant.tar.gz
ls -la
echo "=== version ==="
./qdrant --version || echo "binary did not run (maybe needs config dir)"
