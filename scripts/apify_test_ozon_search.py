"""Test Apify Ozon Scraper Pro — search for 1 product, see what we get back."""
import os, sys, urllib.request, json, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
TOK = os.environ["APIFY_TOKEN"]

ACTOR = "zen-studio~ozon-scraper-pro"  # ~ is URL-safe for /
QUERY = "Cooler Master MWE Gold 750 V2"


def http(method, url, body=None):
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


# 1. Start a run with our query
print(f"=== Starting actor run for query: {QUERY!r} ===")
run_url = f"https://api.apify.com/v2/acts/{ACTOR}/runs?token={TOK}"
status, resp = http("POST", run_url, {
    "queries": [QUERY],
    "maxItems": 5,           # we want top 5 search results
    "skipDetails": False,    # we want full characteristics
})
if status >= 300:
    print(f"HTTP {status}: {resp}")
    sys.exit(1)

run_id = resp["data"]["id"]
status_url = f"https://api.apify.com/v2/actor-runs/{run_id}"
print(f"Run ID: {run_id}")
print(f"Status: {resp['data']['status']}")
print(f"Available fields: {list(resp['data'].keys())[:15]}")
print()

# 2. Poll until done
print("=== Polling for completion ===")
for i in range(60):  # 60 × 5 = 5 min max
    time.sleep(5)
    s, r = http("GET", f"{status_url}?token={TOK}")
    if s >= 300:
        print(f"Status check HTTP {s}: {r}")
        break
    state = r["data"]["status"]
    print(f"  [{i+1}] {state}")
    if state in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED_OUT"):
        break
final_status = r["data"]["status"]
print(f"\nFinal: {final_status}")
print(f"Stats: {r['data'].get('stats', {})}")

if final_status != "SUCCEEDED":
    print(f"Full response: {json.dumps(r['data'], indent=2, ensure_ascii=False)[:1000]}")
    sys.exit(1)

# 3. Fetch results
ds_id = r["data"]["defaultDatasetId"]
items_url = f"https://api.apify.com/v2/datasets/{ds_id}/items?token={TOK}&format=json"
s, items = http("GET", items_url)
print(f"\n=== Got {len(items)} items ===")
for i, item in enumerate(items[:5]):
    print(f"\n--- Item {i+1} ---")
    print(f"  title: {item.get('title', '?')[:80]}")
    print(f"  sku: {item.get('sku')}")
    print(f"  url: {item.get('url', '?')[:80]}")
    print(f"  price: {item.get('price')}")
    chars = item.get("characteristics") or item.get("shortCharacteristics") or []
    print(f"  characteristics count: {len(chars)}")
    if chars:
        print(f"  first 8 chars:")
        for c in chars[:8]:
            if isinstance(c, dict):
                key = c.get("key") or c.get("name") or "?"
                val = c.get("value") or "?"
                print(f"    [{key}] {val}")
