"""Terminate a RunPod pod.

Usage:
    python scripts/runpod_terminate.py [pod_id]
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
if len(sys.argv) > 1:
    pod_id = sys.argv[1]
else:
    if not state_path.exists():
        sys.exit("No pod_id given and no state file found")
    pod_id = json.loads(state_path.read_text())["pod_id"]

MUTATION = """
mutation t($id: String!) {
  podTerminate(input: {podId: $id})
}
"""

req = request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({"query": MUTATION, "variables": {"id": pod_id}}).encode(),
    headers={
        "Authorization": f"Bearer {RUNPOD_KEY}",
        "Content-Type": "application/json",
        "User-Agent": UA,
    },
    method="POST",
)
try:
    with request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read())
except error.HTTPError as e:
    sys.exit(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}")

if "errors" in body:
    sys.exit(f"GraphQL errors: {body['errors']}")

print(f"Terminated pod: {pod_id}")
