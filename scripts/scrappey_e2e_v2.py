"""HTML-only flow: search HTML → tile slug → /features/ HTML → extract chars."""
import os, sys, urllib.request, json, time, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
KEY = os.environ["SCRAPPEY_KEY"]
ENDPOINT = f"https://publisher.scrappey.com/api/v1?key={KEY}"


def post(payload, timeout=180):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(ENDPOINT, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()), time.time() - t0
    except Exception as e:
        return {"_err": f"{type(e).__name__}: {e}"}, time.time() - t0


def balance():
    try:
        with urllib.request.urlopen(urllib.request.Request(f"https://publisher.scrappey.com/api/v1/balance?key={KEY}"), timeout=20) as r:
            return json.loads(r.read()).get("balance")
    except Exception:
        return "?"


def find_tiles_in_search_html(html: str) -> list[dict]:
    """Extract /product/<slug>-<pid>/ links from search HTML.

    Also tries to grab title from surrounding `<span>title</span>` if possible.
    """
    seen = set()
    tiles = []
    # Find product anchors with text
    pat = re.compile(
        r'<a[^>]+href="(/product/([a-z0-9\-]+)-(\d+)/)[^"]*"[^>]*>(.*?)</a>',
        re.DOTALL | re.IGNORECASE,
    )
    for m in pat.finditer(html):
        href, slug, pid, body = m.group(1), m.group(2), m.group(3), m.group(4)
        if pid in seen:
            continue
        seen.add(pid)
        # Strip HTML tags from body to get title
        title = re.sub(r"<[^>]+>", " ", body)
        title = re.sub(r"\s+", " ", title).strip()
        tiles.append({"pid": pid, "slug": slug, "href": href, "title": title[:120]})
    return tiles


def extract_chars_from_features(html: str) -> list[dict]:
    """Extract characteristics from /features/ HTML data-state SSR payloads."""
    out = []
    seen = set()
    pat = re.compile(
        r'<div\s+id="state-webCharacteristics-[^"]+"\s+data-state=\'([^\']+)\'',
        re.DOTALL,
    )
    for raw in pat.findall(html):
        try:
            data = json.loads(raw)
        except Exception:
            decoded = raw.replace("&quot;", '"').replace("&#39;", "'").replace("&amp;", "&")
            try:
                data = json.loads(decoded)
            except Exception:
                continue
        for c in data.get("characteristics", []):
            # c can have keys: short, long, full — all lists of {key, name, values:[{text,id}]}
            for kind in ("short", "long", "full"):
                for item in c.get(kind, []) or []:
                    name = item.get("name")
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    out.append({
                        "key": item.get("key"),
                        "name": name,
                        "values": [{"text": v.get("text"), "id": v.get("id")} for v in (item.get("values") or [])],
                    })
    return out


# --- run ---
print(f"Start balance: {balance()}")
print()
print("=" * 70)
print("=== STEP 1: HTML search ===")
search_url = "https://www.ozon.ru/search/?text=Cooler+Master+MWE+Gold+750+V2"
b, e = post({"cmd": "request.get", "url": search_url})
if "_err" in b:
    print(f"  FAIL: {b}")
    sys.exit(1)
sol = b.get("solution") or {}
sc = sol.get("statusCode")
html = sol.get("response") or ""
print(f"  HTTP{sc} ({e:.1f}s) len={len(html)} | balance={balance()}")
if sc != 200:
    print(f"  bad status, first 400: {html[:400]}")
    sys.exit(1)

tiles = find_tiles_in_search_html(html)
print(f"  found {len(tiles)} unique product tiles (top 8):")
for t in tiles[:8]:
    print(f"    pid={t['pid']} title={t['title'][:60]}")

if not tiles:
    sys.exit(1)

# Pick best tile — prefer one with "750" in slug and "gold" and "v2"
def score(t):
    s = t["slug"].lower()
    sc = 0
    if "mwe" in s: sc += 1
    if "750" in s: sc += 2
    if "gold" in s: sc += 2
    if "v2" in s: sc += 3
    return sc
tiles_sorted = sorted(tiles, key=score, reverse=True)
chosen = tiles_sorted[0]
print(f"\n  ✅ chosen: {chosen['title'][:80]}")
print(f"     pid={chosen['pid']} href={chosen['href']}")

# --- STEP 2: features ---
features_url = f"https://www.ozon.ru/product/{chosen['slug']}-{chosen['pid']}/features/"
print()
print("=" * 70)
print(f"=== STEP 2: HTML /features/ → {features_url[:80]}... ===")
b2, e2 = post({"cmd": "request.get", "url": features_url})
if "_err" in b2:
    print(f"  FAIL: {b2}")
    sys.exit(1)
sol2 = b2.get("solution") or {}
sc2 = sol2.get("statusCode")
html2 = sol2.get("response") or ""
print(f"  HTTP{sc2} ({e2:.1f}s) len={len(html2)} | balance={balance()}")
if sc2 != 200:
    print(f"  bad status, first 400: {html2[:400]}")
    sys.exit(1)

Path("_dump_bp_features_v2.html").write_text(html2, encoding="utf-8")

# --- STEP 3: extract characteristics ---
print()
print("=" * 70)
print("=== STEP 3: extract characteristics ===")
chars = extract_chars_from_features(html2)
print(f"  ✅ extracted {len(chars)} unique characteristics:")
for c in chars:
    vals = ", ".join(v.get("text", "") for v in c.get("values", []) if v.get("text"))[:80]
    print(f"    [{(c.get('name') or '?')[:35]:35s}] = {vals}")

print()
print(f"Final balance: {balance()}")
print(f"\n🎯 SUCCESS: 2 credits / товар, {len(chars)} характеристик extracted")
