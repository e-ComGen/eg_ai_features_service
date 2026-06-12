"""Spawn RunPod pod for datasets_rag Phase-1 indexing.

Flow:
  1. Upload runpod_pod_runner_datasets.sh (as run.sh) +
     ingest_dataset_to_qdrant.py to HF dataset `datasets-rag-index` (private).
  2. Spawn a pod with dockerArgs that curl+exec run.sh from HF.
  3. Save state to scripts/.runpod_datasets_state.json.

Monitor with:  python scripts/runpod_status_rag.py --state scripts/.runpod_datasets_state.json
Download with: python scripts/restore_datasets_rag.py   (after DONE_DATASETS marker appears)
Terminate with: python scripts/runpod_terminate.py --state scripts/.runpod_datasets_state.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib import request, error

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RUNPOD_KEY = os.environ["RUNPOD_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
UA = "Mozilla/5.0 (compatible; eg-ai-features/1.0)"

HF_DATASET = "datasets-rag-index"
STATE_FILE = PROJECT_ROOT / "scripts" / ".runpod_datasets_state.json"
DOCKER_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
CONTAINER_DISK_GB = 80   # qdrant storage + model cache + tar

# GPU candidates: prefer 8+ GB VRAM, cheap community cloud.
# RTX 3070 (8 GB) is fine for 384-d MiniLM; 3090/A4000 gives faster embed throughput.
GPU_TYPE_IDS = [
    "NVIDIA GeForce RTX 3070",
    "NVIDIA RTX A4000",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX A2000",          # 12 GB variant; slower but available
]


# ── HF helpers ─────────────────────────────────────────────────────────────────

def hf_whoami() -> str:
    req = request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {HF_TOKEN}", "User-Agent": UA},
    )
    with request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["name"]


def hf_create_dataset(repo_id: str) -> None:
    name = repo_id.split("/", 1)[1]
    req = request.Request(
        "https://huggingface.co/api/repos/create",
        data=json.dumps({"name": name, "type": "dataset", "private": True}).encode(),
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
        if "already" in body.lower() or e.code == 409:
            print(f"[HF] Dataset already exists: {repo_id}")
        else:
            sys.exit(f"[HF] Create failed HTTP {e.code}: {body[:500]}")


def hf_upload_folder(repo_id: str, files: list[tuple[Path, str]]) -> str:
    """Upload multiple files in a single commit using HfApi.upload_folder.

    Returns the immutable commit SHA so callers can build cache-proof URLs.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import HfApi
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for src, dst in files:
            shutil.copy(src, td_path / dst)
        api = HfApi(token=HF_TOKEN)
        commit = api.upload_folder(
            folder_path=str(td_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="datasets-rag bootstrap scripts",
        )
    sha: str = commit.oid
    for _, dst in files:
        print(f"[HF] uploaded {dst}")
    print(f"[HF] commit SHA: {sha}")
    return sha


# ── RunPod helpers ─────────────────────────────────────────────────────────────

def runpod_graphql(query: str, variables: dict, allow_retry: bool = False) -> dict | None:
    req = request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
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
        sys.exit(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:600]}")
    if "errors" in body:
        codes = {(e.get("extensions") or {}).get("code") for e in body["errors"]}
        if allow_retry and codes & {"SUPPLY_CONSTRAINT", "RUNPOD"}:
            print(f"[Pod]   skip: {body['errors'][0].get('message', '')[:90]}")
            return None
        sys.exit(f"RunPod GraphQL errors: {json.dumps(body['errors'], indent=2)}")
    return body["data"]


def spawn_pod(hf_repo_id: str, run_sh_url: str, sha: str) -> tuple[str, str]:
    """Try each GPU candidate in order; return (pod_id, gpu_used)."""
    cache_bust = int(time.time())
    busted_url = f"{run_sh_url}?ts={cache_bust}"
    docker_cmd = (
        "/bin/bash -c "
        "\"apt-get update -qq && apt-get install -y -qq curl ca-certificates && "
        f"curl -fsSL -H 'Authorization: Bearer {HF_TOKEN}' "
        f"'{busted_url}' -o /tmp/run.sh && "
        "chmod +x /tmp/run.sh && "
        "exec bash /tmp/run.sh\""
    )
    print(f"[Pod] dockerArgs ({len(docker_cmd)} bytes): {docker_cmd[:160]}...")

    mutation = """
    mutation pd($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id machineId imageName
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
                "minMemoryInGb": 12,
                "gpuTypeId": gpu_id,
                "name": "eg-datasets-rag-indexer",
                "imageName": DOCKER_IMAGE,
                "dockerArgs": docker_cmd,
                "ports": "22/tcp",
                "volumeMountPath": "/workspace",
                "env": [
                    {"key": "HF_TOKEN", "value": HF_TOKEN},
                    {"key": "HF_REPO_ID", "value": hf_repo_id},
                    {"key": "HF_REV", "value": sha},
                ],
            }
        }
        print(f"[Pod] Trying {gpu_id} ...")
        data = runpod_graphql(mutation, variables, allow_retry=True)
        if data is None:
            continue
        pod = data["podFindAndDeployOnDemand"]
        if pod:
            return pod["id"], gpu_id

    sys.exit("No GPU available across all candidates — try again later.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    user = hf_whoami()
    hf_repo_id = f"{user}/{HF_DATASET}"
    print(f"[HF] User: {user}  ->  {hf_repo_id}")

    hf_create_dataset(hf_repo_id)

    runner = PROJECT_ROOT / "scripts" / "runpod_pod_runner_datasets.sh"
    indexer = PROJECT_ROOT / "scripts" / "ingest_dataset_to_qdrant.py"

    if not runner.exists():
        sys.exit(f"[Error] Missing: {runner}")
    if not indexer.exists():
        sys.exit(f"[Error] Missing: {indexer}")

    sha = hf_upload_folder(
        hf_repo_id,
        [
            (runner, "run.sh"),
            (indexer, "ingest_dataset_to_qdrant.py"),
        ],
    )

    run_sh_url = f"https://huggingface.co/datasets/{hf_repo_id}/resolve/{sha}/run.sh"
    pod_id, gpu_used = spawn_pod(hf_repo_id, run_sh_url, sha)
    print(f"[Pod] Created: {pod_id}  GPU: {gpu_used}")

    state = {
        "pod_id": pod_id,
        "hf_repo_id": hf_repo_id,
        "spawned_at": int(time.time()),
        "gpu": gpu_used,
        "job": "datasets_rag_phase1",
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))
    print(f"[State] saved to {STATE_FILE}")
    print()
    print(f"Monitor : check HF https://huggingface.co/datasets/{hf_repo_id}")
    print(f"          look for DONE_DATASETS file + datasets_pod.log")
    print(f"Restore : python scripts/restore_datasets_rag.py")
    print(f"Terminate: python scripts/runpod_terminate.py")
    print(f"  (state file: {STATE_FILE})")


if __name__ == "__main__":
    main()
