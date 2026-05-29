"""Quick: list RunPod community GPUs with pricing, sorted cheapest first.

Reads RUNPOD_API_KEY from .env. Prints only displayName / memory / community price.
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
import urllib.request

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

key = os.getenv("RUNPOD_API_KEY")
if not key:
    sys.exit("RUNPOD_API_KEY not set")

query = """
{
  gpuTypes {
    id
    displayName
    memoryInGb
    communityPrice
    securePrice
    communitySpotPrice
    lowestPrice(input:{gpuCount:1}) {
      minimumBidPrice
      uninterruptablePrice
    }
  }
}
"""
req = urllib.request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({"query": query}).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; eg-ai-features/1.0)",
    },
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        body = json.loads(r.read())
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:500]}")
    sys.exit(1)
if "errors" in body:
    print("GraphQL errors:", body["errors"])
    sys.exit(1)
data = body["data"]["gpuTypes"]

rows = []
for g in data:
    lp = g.get("lowestPrice") or {}
    price = lp.get("uninterruptablePrice") or g.get("communityPrice") or g.get("securePrice")
    if price is None:
        continue
    rows.append((price, g.get("displayName", "?"), g.get("memoryInGb"), g.get("id")))

rows.sort()
print(f"{'price/h':>8s}  {'GPU':30s}  {'VRAM':>6s}  id")
for price, name, mem, gid in rows[:20]:
    print(f"${price:>6.3f}/h  {name:30s}  {mem!s:>5s}GB  {gid}")
