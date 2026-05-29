"""Introspect RunPod GraphQL Pod type to find log/exec fields."""
import json
import os
from pathlib import Path
from urllib import request
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
KEY = os.environ["RUNPOD_API_KEY"]

q = """
{
  __type(name: "Pod") {
    fields { name type { name kind ofType { name kind } } }
  }
}
"""
r = request.urlopen(request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps({"query": q}).encode(),
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json",
             "User-Agent": "Mozilla/5.0 (compatible; eg-ai/1.0)"},
    method="POST",
), timeout=30)
body = json.loads(r.read())
for f in body["data"]["__type"]["fields"]:
    t = f["type"]
    tn = t.get("name") or (t.get("ofType") or {}).get("name") or t.get("kind")
    print(f"  {f['name']:40s} {tn}")
