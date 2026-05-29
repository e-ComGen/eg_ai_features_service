"""V2: Spawn RunPod pod, host bootstrap scripts in HF dataset.

Flow:
  1. Upload build_ozon_rag_index.py + runpod_pod_runner.sh to HF dataset.
  2. Spawn pod with minimal dockerArgs that bootstraps from HF.
  3. Save state to .runpod_rag_state.json.

Bootstrap pattern:  pod's `dockerArgs` -> curl run.sh -> bash run.sh.
Avoids base64-in-dockerArgs limits and shell-quoting hell.
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

GPU_TYPE_IDS = [
    "NVIDIA RTX A2000",
    "NVIDIA GeForce RTX 3070",
    "NVIDIA RTX A4000",
    "NVIDIA GeForce RTX 3080",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX 4000 Ada Generation",
]
DOCKER_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
CONTAINER_DISK_GB = 50
HF_DATASET = "ozon-rag-index-2"


def hf_whoami() -> str:
    req = request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {HF_TOKEN}", "User-Agent": UA},
    )
    with request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["name"]


def hf_upload_file(repo_id: str, local_path: Path, repo_path: str) -> None:
    """Upload single file to HF dataset using the LFS-free path (small files)."""
    import base64
    # HF supports a simple commit API: POST /api/datasets/{repo}/commit/main
    # with files in multipart. Easier: use huggingface_hub.
    try:
        from huggingface_hub import HfApi
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"[HF] uploaded {repo_path}")


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


def spawn_pod(hf_repo_id: str, run_sh_url: str) -> tuple[str, str]:
    # dockerArgs: short bootstrap that curls and runs run.sh from HF.
    # Note: HF requires Authorization for private datasets; pass token in URL header.
    # Cache-bust HF URL — append timestamp so CDN can't serve stale.
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
    print(f"[Pod] dockerArgs ({len(docker_cmd)} bytes):\n  {docker_cmd[:200]}...")

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
                "minMemoryInGb": 8,
                "gpuTypeId": gpu_id,
                "name": "eg-ai-rag-indexer-v2",
                "imageName": DOCKER_IMAGE,
                "dockerArgs": docker_cmd,
                "ports": "22/tcp",
                "volumeMountPath": "/workspace",
                "env": [
                    {"key": "HF_TOKEN", "value": HF_TOKEN},
                    {"key": "HF_REPO_ID", "value": hf_repo_id},
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
    sys.exit("No GPU available across all candidates")


def hf_create_dataset(repo_id: str) -> None:
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
        if "already" in body.lower() or e.code == 409:
            print(f"[HF] Dataset already exists: {repo_id}")
        else:
            sys.exit(f"[HF] Create failed HTTP {e.code}: {body[:500]}")


def hf_upload_folder(repo_id: str, files: list[tuple[Path, str]]) -> None:
    """Upload multiple files in a single commit using HfApi.upload_folder."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import HfApi
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        for src, dst in files:
            shutil.copy(src, td_path / dst)
        api = HfApi(token=HF_TOKEN)
        api.upload_folder(
            folder_path=str(td_path),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="bootstrap scripts",
        )
    for _, dst in files:
        print(f"[HF] uploaded {dst} (folder commit)")


def main() -> None:
    user = hf_whoami()
    hf_repo_id = f"{user}/{HF_DATASET}"
    print(f"[HF] User: {user}  ->  {hf_repo_id}")
    hf_create_dataset(hf_repo_id)

    # Upload bootstrap scripts in ONE commit.
    indexer = PROJECT_ROOT / "scripts" / "build_ozon_rag_index.py"
    runner = PROJECT_ROOT / "scripts" / "runpod_pod_runner.sh"
    hf_upload_folder(hf_repo_id, [(indexer, "build_index.py"), (runner, "run.sh")])

    run_sh_url = f"https://huggingface.co/datasets/{hf_repo_id}/resolve/main/run.sh"

    pod_id, gpu_used = spawn_pod(hf_repo_id, run_sh_url)
    print(f"[Pod] Created: {pod_id}  GPU: {gpu_used}")

    state = {
        "pod_id": pod_id,
        "hf_repo_id": hf_repo_id,
        "spawned_at": int(time.time()),
        "gpu": gpu_used,
    }
    (PROJECT_ROOT / "scripts" / ".runpod_rag_state.json").write_text(json.dumps(state))
    print(f"[State] saved. Result will appear at: https://huggingface.co/datasets/{hf_repo_id}/blob/main/qdrant.tar.gz")


if __name__ == "__main__":
    main()
