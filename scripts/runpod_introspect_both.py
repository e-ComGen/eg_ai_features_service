"""Status for BOTH pods (slow A + fast B)."""
import json, os
from pathlib import Path
from urllib import request
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
KEY = os.environ["RUNPOD_API_KEY"]

A = json.loads((Path(__file__).resolve().parent / ".runpod_rag_state.json").read_text())
B = json.loads((Path(__file__).resolve().parent / ".runpod_rag_fast_state.json").read_text())

q = """
query pod($id: String!) {
  pod(input: {podId: $id}) {
    id
    desiredStatus
    runtime {
      uptimeInSeconds
      gpus { gpuUtilPercent memoryUtilPercent }
      container { cpuPercent memoryPercent }
    }
    costPerHr
  }
}
"""
for label, state in [("A slow", A), ("B fast", B)]:
    r = request.urlopen(request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": q, "variables": {"id": state["pod_id"]}}).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 (compatible; eg-ai/1.0)"},
        method="POST",
    ), timeout=30)
    body = json.loads(r.read())
    pod = body["data"]["pod"]
    if not pod:
        print(f"{label}: TERMINATED")
        continue
    rt = pod.get("runtime") or {}
    gpus = rt.get("gpus") or [{}]
    cont = rt.get("container") or {}
    print(f"{label}: {pod['desiredStatus']}  uptime={rt.get('uptimeInSeconds')}s  "
          f"GPU={gpus[0].get('gpuUtilPercent')}%  GPUmem={gpus[0].get('memoryUtilPercent')}%  "
          f"CPU={cont.get('cpuPercent')}%  RAM={cont.get('memoryPercent')}%  "
          f"${pod.get('costPerHr')}/h")
