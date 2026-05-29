"""Fetch container logs for a pod via REST API."""
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
    pod_id = json.loads(state_path.read_text())["pod_id"]

# REST API endpoint for logs
url = f"https://rest.runpod.io/v1/pods/{pod_id}/logs"
req = request.Request(
    url,
    headers={"Authorization": f"Bearer {RUNPOD_KEY}", "User-Agent": UA},
)
try:
    with request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8", errors="replace")
except error.HTTPError as e:
    body_err = e.read().decode("utf-8", errors="replace")
    print(f"HTTP {e.code}: {body_err[:500]}")
    # Try GraphQL fallback
    print("\nTrying GraphQL fallback...")
    query = '{ pod(input: {podId: "%s"}) { containerLogs { line timestamp } } }' % pod_id
    req2 = request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query}).encode(),
        headers={
            "Authorization": f"Bearer {RUNPOD_KEY}",
            "Content-Type": "application/json",
            "User-Agent": UA,
        },
        method="POST",
    )
    try:
        with request.urlopen(req2, timeout=30) as r:
            b2 = json.loads(r.read())
            if "errors" in b2:
                print(json.dumps(b2["errors"], indent=2))
            else:
                logs = (b2["data"]["pod"] or {}).get("containerLogs") or []
                for l in logs[-100:]:
                    print(l.get("line", ""))
    except error.HTTPError as e2:
        print(f"GraphQL HTTP {e2.code}: {e2.read().decode('utf-8', errors='replace')[:500]}")
    sys.exit(1)

print(body[-5000:])
