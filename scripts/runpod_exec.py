"""Run a shell command inside a pod via RunPod's exec mutation.

Usage:
    python scripts/runpod_exec.py "ls /workspace"
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from urllib import request, error
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]
UA = "Mozilla/5.0 (compatible; eg-ai-features/1.0)"

state_path = PROJECT_ROOT / "scripts" / ".runpod_rag_state.json"
pod_id = json.loads(state_path.read_text())["pod_id"]
cmd = sys.argv[1] if len(sys.argv) > 1 else "ls -la /workspace"

# Try REST exec endpoint first
url = f"https://rest.runpod.io/v1/pods/{pod_id}/exec"
req = request.Request(
    url,
    data=json.dumps({"cmd": cmd}).encode(),
    headers={
        "Authorization": f"Bearer {RUNPOD_KEY}",
        "Content-Type": "application/json",
        "User-Agent": UA,
    },
    method="POST",
)
try:
    with request.urlopen(req, timeout=120) as r:
        print(r.read().decode("utf-8", errors="replace"))
except error.HTTPError as e:
    print(f"REST HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}")
