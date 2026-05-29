"""Poll RunPod pod status.

Usage:
    python scripts/runpod_status_rag.py [pod_id]
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

QUERY = """
query pod($id: String!) {
  pod(input: {podId: $id}) {
    id
    name
    desiredStatus
    runtime {
      uptimeInSeconds
      ports { ip publicPort privatePort isIpPublic type }
    }
    lastStatusChange
    machine { gpuDisplayName }
    costPerHr
  }
}
"""

req = request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({"query": QUERY, "variables": {"id": pod_id}}).encode(),
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

pod = body["data"]["pod"]
if not pod:
    sys.exit(f"Pod {pod_id} not found")

print(f"Pod ID:        {pod['id']}")
print(f"Name:          {pod['name']}")
print(f"Status:        {pod['desiredStatus']}")
print(f"GPU:           {(pod.get('machine') or {}).get('gpuDisplayName')}")
cost = pod.get("costPerHr")
if cost is not None:
    print(f"Cost:          ${cost:.3f}/hr")
runtime = pod.get("runtime")
if runtime:
    secs = runtime.get("uptimeInSeconds", 0)
    print(f"Uptime:        {secs // 60}m {secs % 60}s")
    ports = runtime.get("ports") or []
    for p in ports:
        if p.get("isIpPublic"):
            print(f"SSH:           ssh root@{p['ip']} -p {p['publicPort']}")
else:
    print("Uptime:        (not yet running)")
print(f"Last change:   {pod.get('lastStatusChange')}")
