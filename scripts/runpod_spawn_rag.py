"""Spawn RunPod GPU pod to build the Ozon RAG index, upload result to HuggingFace.

Flow:
  1. Read RUNPOD_API_KEY + HF_TOKEN from .env.
  2. Resolve HF username via /api/whoami-v2.
  3. Create private HF dataset {user}/ozon-rag-index-tmp (idempotent).
  4. Build pod startup CMD with build_ozon_rag_index.py embedded base64.
  5. POST /graphql podFindAndDeployOnDemand → spawn community-cloud RTX A2000 pod.
  6. Print pod ID + HF target.  Use scripts/runpod_status_rag.py to poll.

Cost: ~$0.12/h × 1h max = ~$0.12.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from urllib import request, error

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RUNPOD_KEY = os.getenv("RUNPOD_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
if not RUNPOD_KEY:
    sys.exit("RUNPOD_API_KEY not set in .env")
if not HF_TOKEN:
    sys.exit("HF_TOKEN not set in .env")

GPU_TYPE_IDS = [
    "NVIDIA RTX A2000",                     # 6GB,  $0.12/h
    "NVIDIA GeForce RTX 3070",              # 8GB,  $0.13/h
    "NVIDIA RTX A4000",                     # 16GB, $0.17/h
    "NVIDIA GeForce RTX 3080",              # 10GB, $0.17/h
    "NVIDIA RTX A5000",                     # 24GB, $0.16/h
    "NVIDIA GeForce RTX 3090",              # 24GB, $0.22/h
]
DOCKER_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK_GB = 50
HF_DATASET = "ozon-rag-index-tmp"

UA = "Mozilla/5.0 (compatible; eg-ai-features/1.0)"


def http_json(url: str, headers: dict, payload: dict | None = None, method: str = "GET") -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = request.Request(url, data=data, headers={**headers, "User-Agent": UA}, method=method)
    try:
        with request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        sys.exit(f"HTTP {e.code} for {method} {url}\n{body[:1000]}")


def hf_whoami() -> str:
    body = http_json(
        "https://huggingface.co/api/whoami-v2",
        {"Authorization": f"Bearer {HF_TOKEN}"},
    )
    name = body.get("name") or (body.get("auth", {}).get("accessToken") or {}).get("displayName")
    if not name:
        sys.exit(f"Could not resolve HF user from token. body={body!r}")
    return name


def hf_create_dataset(repo_id: str) -> None:
    """Idempotent: 200 if exists, 201 if created, raises on real error."""
    req = request.Request(
        "https://huggingface.co/api/repos/create",
        data=json.dumps({"name": HF_DATASET, "type": "dataset", "private": True}).encode(),
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
            "User-Agent": UA,
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as r:
            print(f"[HF] Created dataset: {repo_id} (HTTP {r.status})")
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        if "already exists" in body.lower() or e.code == 409:
            print(f"[HF] Dataset already exists: {repo_id}")
        else:
            sys.exit(f"[HF] Create failed HTTP {e.code}: {body[:500]}")


def build_startup_cmd(indexer_b64: str, hf_repo_id: str) -> str:
    """Bash startup CMD for the pod (single line, runs indexer + uploads tar)."""
    return (
        "bash -lc '"
        "set -e; "
        "echo [pod] start at $(date); "
        "pip install -q qdrant-client datasets sentence-transformers python-dotenv pyarrow pandas huggingface_hub; "
        "cd /workspace; "
        f"echo {indexer_b64} | base64 -d > build_index.py; "
        f"export HF_TOKEN={HF_TOKEN}; "
        "python build_index.py --index-path /workspace/ozon_rag.qdrant; "
        "echo [pod] indexer done, tarring...; "
        "cd /workspace && tar czf qdrant.tar.gz ozon_rag.qdrant; "
        "ls -lh qdrant.tar.gz; "
        f"python -c \"from huggingface_hub import HfApi; HfApi().upload_file(path_or_fileobj='/workspace/qdrant.tar.gz', path_in_repo='qdrant.tar.gz', repo_id='{hf_repo_id}', repo_type='dataset', token='{HF_TOKEN}')\"; "
        "echo [pod] DONE_INDEXING; "
        "sleep 3600"  # keep alive 1h for inspection if needed
        "'"
    )


_RETRYABLE_CODES = {"SUPPLY_CONSTRAINT", "RUNPOD"}


def runpod_graphql(query: str, variables: dict | None = None, allow_retry: bool = False) -> dict | None:
    body = http_json(
        "https://api.runpod.io/graphql",
        {"Authorization": f"Bearer {RUNPOD_KEY}", "Content-Type": "application/json"},
        {"query": query, "variables": variables or {}},
        method="POST",
    )
    if "errors" in body:
        codes = {(e.get("extensions") or {}).get("code") for e in body["errors"]}
        if allow_retry and codes & _RETRYABLE_CODES:
            msg = body["errors"][0].get("message", "")
            print(f"[Pod]   skip: {msg[:90]}")
            return None
        sys.exit(f"RunPod GraphQL errors: {json.dumps(body['errors'], indent=2)}")
    return body["data"]


def spawn_pod(startup_cmd: str) -> tuple[str, str]:
    mutation = """
    mutation podDeployment($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        machineId
        imageName
        machine { podHostId }
      }
    }
    """
    for gpu_id in GPU_TYPE_IDS:
        variables = {
            "input": {
                "cloudType": "COMMUNITY",
                "gpuCount": 1,
                "volumeInGb": 0,
                "containerDiskInGb": CONTAINER_DISK_GB,
                "minVcpuCount": 2,
                "minMemoryInGb": 8,
                "gpuTypeId": gpu_id,
                "name": "eg-ai-rag-indexer",
                "imageName": DOCKER_IMAGE,
                "dockerArgs": startup_cmd,
                "ports": "22/tcp",
                "volumeMountPath": "/workspace",
                "env": [{"key": "HF_TOKEN", "value": HF_TOKEN}],
            }
        }
        print(f"[Pod] Trying {gpu_id} ...")
        data = runpod_graphql(mutation, variables, allow_retry=True)
        if data is None:
            print(f"[Pod]   no supply, trying next")
            continue
        pod = data["podFindAndDeployOnDemand"]
        if pod:
            return pod["id"], gpu_id
    sys.exit("No GPU available in community cloud across all candidates")


def main() -> None:
    indexer_path = PROJECT_ROOT / "scripts" / "build_ozon_rag_index.py"
    indexer_b64 = base64.b64encode(indexer_path.read_bytes()).decode()

    hf_user = hf_whoami()
    hf_repo_id = f"{hf_user}/{HF_DATASET}"
    print(f"[HF] User: {hf_user}  ->  dataset: {hf_repo_id}")
    hf_create_dataset(hf_repo_id)

    startup = build_startup_cmd(indexer_b64, hf_repo_id)
    print(f"[Pod] Startup CMD size: {len(startup)} bytes")
    pod_id, gpu_used = spawn_pod(startup)

    print(f"[Pod] Created pod ID: {pod_id}")
    print(f"[Pod] GPU: {gpu_used}")
    print(f"[Pod] Image: {DOCKER_IMAGE}")
    print(f"[HF]  Result will be at: https://huggingface.co/datasets/{hf_repo_id}")
    print()
    print("Next steps:")
    print(f"  1. Poll status: python scripts/runpod_status_rag.py {pod_id}")
    print(f"  2. After ~45 min, download: python scripts/runpod_download_rag.py {hf_repo_id}")
    print(f"  3. Terminate when done: python scripts/runpod_terminate.py {pod_id}")

    # Save state for later scripts
    state_path = PROJECT_ROOT / "scripts" / ".runpod_rag_state.json"
    state_path.write_text(json.dumps({
        "pod_id": pod_id,
        "hf_repo_id": hf_repo_id,
        "spawned_at": int(time.time()),
        "gpu": gpu_used,
    }))
    print(f"[State] Saved: {state_path}")


if __name__ == "__main__":
    main()
