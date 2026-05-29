"""Get full Pod field info incl runtime metrics."""
import json, os
from pathlib import Path
from urllib import request
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
KEY = os.environ["RUNPOD_API_KEY"]

state = json.loads((Path(__file__).resolve().parent / ".runpod_rag_state.json").read_text())
pod_id = state["pod_id"]

q = """
query pod($id: String!) {
  pod(input: {podId: $id}) {
    id
    name
    desiredStatus
    runtime {
      uptimeInSeconds
      gpus { id gpuUtilPercent memoryUtilPercent }
      container { cpuPercent memoryPercent }
    }
    machine { gpuDisplayName }
    costPerHr
  }
}
"""
req = request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({"query": q, "variables": {"id": pod_id}}).encode(),
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
             "User-Agent": "Mozilla/5.0 (compatible; eg-ai/1.0)"},
    method="POST",
)
with request.urlopen(req, timeout=30) as r:
    body = json.loads(r.read())
print(json.dumps(body, indent=2))
