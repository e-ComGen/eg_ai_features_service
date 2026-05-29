"""Spawn RunPod GPU pod for ENCODE-ONLY workflow (variant X).

Output: shards uploaded to HF, downloaded locally, then `import_shards_to_qdrant.py`.
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
    "NVIDIA GeForce RTX 3080 Ti",
    "NVIDIA RTX 4000 SFF Ada Generation",
    "NVIDIA RTX A4500",
    "NVIDIA GeForce RTX 4070 Ti",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA GeForce RTX 3090 Ti",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA RTX A6000",
    "NVIDIA GeForce RTX 4090",
]
DOCKER_IMAGE = "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime"
CONTAINER_DISK_GB = 50  # 10-15GB for HF dataset cache + 4GB shards + overhead
HF_DATASET = "ozon-rag-index-2"


def hf_whoami() -> str:
    req = request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {HF_TOKEN}", "User-Agent": UA},
    )
    with request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["name"]


def hf_upload_folder(repo_id: str, files: list[tuple[Path, str]]) -> None:
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
            commit_message="encode bootstrap",
        )
    for _, dst in files:
        print(f"[HF] uploaded {dst}")


def runpod_graphql(query, variables, allow_retry=False):
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


def spawn_pod(hf_repo_id, run_sh_url):
    docker_cmd = (
        "/bin/bash -c "
        "\"apt-get update -qq && apt-get install -y -qq curl ca-certificates && "
        f"curl -fsSL -H 'Authorization: Bearer {HF_TOKEN}' "
        f"'{run_sh_url}' -o /tmp/run.sh && "
        "chmod +x /tmp/run.sh && "
        "exec bash /tmp/run.sh\""
    )
    print(f"[Pod] dockerArgs ({len(docker_cmd)} bytes)")

    mutation = """
    mutation pd($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) { id machineId imageName }
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
                "name": "eg-ai-encode-only",
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
        print(f"[Pod] Trying {gpu_id}")
        data = runpod_graphql(mutation, variables, allow_retry=True)
        if data is None:
            continue
        pod = data["podFindAndDeployOnDemand"]
        if pod:
            return pod["id"], gpu_id
    sys.exit("No GPU available")


def main():
    user = hf_whoami()
    hf_repo_id = f"{user}/{HF_DATASET}"
    print(f"[HF] User: {user}  ->  {hf_repo_id}")

    encode_script = PROJECT_ROOT / "scripts" / "build_encode_only.py"
    run_sh = PROJECT_ROOT / "scripts" / "runpod_pod_runner_encode.sh"
    hf_upload_folder(hf_repo_id, [(encode_script, "build_encode_only.py"), (run_sh, "run_encode.sh")])

    run_sh_url = f"https://huggingface.co/datasets/{hf_repo_id}/resolve/main/run_encode.sh"
    pod_id, gpu_used = spawn_pod(hf_repo_id, run_sh_url)
    print(f"[Pod] Created: {pod_id}  GPU: {gpu_used}")

    state = {
        "pod_id": pod_id,
        "hf_repo_id": hf_repo_id,
        "spawned_at": int(time.time()),
        "gpu": gpu_used,
        "type": "encode_only",
    }
    (PROJECT_ROOT / "scripts" / ".runpod_encode_state.json").write_text(json.dumps(state))
    print(f"[State] saved. Output will appear as vectors_*.npy + payloads_*.parquet at:")
    print(f"  https://huggingface.co/datasets/{hf_repo_id}/tree/main")


if __name__ == "__main__":
    main()
