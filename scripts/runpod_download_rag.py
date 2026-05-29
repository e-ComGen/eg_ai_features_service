"""Download qdrant.tar.gz from HF dataset and extract into data/.

Usage:
    python scripts/runpod_download_rag.py [hf_repo_id]
"""
from __future__ import annotations
import json
import os
import sys
import tarfile
from pathlib import Path
import urllib.request
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN = os.environ["HF_TOKEN"]

state_path = PROJECT_ROOT / "scripts" / ".runpod_rag_state.json"
if len(sys.argv) > 1:
    hf_repo_id = sys.argv[1]
else:
    if not state_path.exists():
        sys.exit("No hf_repo_id given and no state file found")
    hf_repo_id = json.loads(state_path.read_text())["hf_repo_id"]

DEST_DIR = PROJECT_ROOT / "app" / "services" / "enrichment" / "strategies" / "dictionaries" / "data"
DEST_DIR.mkdir(parents=True, exist_ok=True)
TAR_PATH = DEST_DIR / "qdrant.tar.gz"

url = f"https://huggingface.co/datasets/{hf_repo_id}/resolve/main/qdrant.tar.gz"
print(f"Downloading {url} ...")

req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}"})
with urllib.request.urlopen(req, timeout=120) as r:
    with open(TAR_PATH, "wb") as f:
        total = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"  {total // (1024*1024)} MB", end="\r")
print(f"\nSaved: {TAR_PATH} ({TAR_PATH.stat().st_size / 1024 / 1024:.1f} MB)")

print(f"Extracting to {DEST_DIR} ...")
with tarfile.open(TAR_PATH, "r:gz") as tar:
    tar.extractall(DEST_DIR)
print("Done.")
print(f"Index: {DEST_DIR / 'ozon_rag.qdrant'}")
TAR_PATH.unlink()
print(f"Removed tar: {TAR_PATH.name}")
