"""
Ozon DataDome bypass v3 — 3 techniques with different IP sources.

Run from the worktree root:
    venv/Scripts/python scripts/_test_ozon_bypass_v3.py

Safe to run automatically:   Approach 1 (archive.org)
Requires user action:        Approach 2 (Chrome profile — close Chrome first)
                             Approach 3 (residential proxy — register on provider first)
"""
import asyncio
import json
import os
import re
import sys
import urllib.parse

TARGET_URL = "https://www.ozon.ru/category/elektronika-15500/"
TARGET_OZON_HOST = "www.ozon.ru"

# ---------------------------------------------------------------------------
# Approach 1 — Wayback Machine (archive.org)
# ---------------------------------------------------------------------------
# archive.org snapshots Ozon from its own IPs; DataDome has no effect here.
# Category trees change slowly, so even a months-old snapshot is useful.
# ---------------------------------------------------------------------------

def _wayback_available_url(url: str) -> str | None:
    """Query Wayback CDX API to find the latest snapshot URL."""
    import urllib.request as _urlrequest
    import urllib.parse as _urlparse

    api = f"https://archive.org/wayback/available?url={_urlparse.quote(url, safe=':/')}"
    try:
        from curl_cffi import requests as cffi_requests
        r = cffi_requests.get(api, timeout=15)
        data = r.json()
    except ImportError:
        with _urlrequest.urlopen(api, timeout=15) as resp:
            data = json.loads(resp.read())

    snapshot = data.get("archived_snapshots", {}).get("closest", {})
    if snapshot.get("available"):
        return snapshot["url"]
    return None


def _wayback_cdx_latest(url: str, limit: int = 1) -> str | None:
    """Use CDX API to find latest snapshot (more reliable than /wayback/available)."""
    import urllib.parse as _urlparse
    try:
        from curl_cffi import requests as cffi_requests
        cdx = (
            "https://web.archive.org/cdx/search/cdx"
            f"?url={_urlparse.quote(url, safe=':/')}"
            "&output=json&limit=1&fl=timestamp,statuscode,original&filter=statuscode:200"
            "&from=20240101&fastLatest=true"
        )
        r = cffi_requests.get(cdx, timeout=20)
        rows = r.json()
        if len(rows) > 1:  # first row is header
            ts = rows[1][0]
            orig = rows[1][2]
            return f"https://web.archive.org/web/{ts}/{orig}"
        return None
    except Exception:
        return None


def _extract_ozon_data(html: str) -> dict:
    """Extract useful data from archived Ozon HTML."""
    result = {
        "has_next_data": False,
        "product_links": [],
        "title": "",
        "category_names": [],
    }

    # Page title
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if title_match:
        result["title"] = re.sub(r"<[^>]+>", "", title_match.group(1)).strip()

    # __NEXT_DATA__ JSON blob
    next_data_match = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html,
        re.DOTALL,
    )
    if next_data_match:
        result["has_next_data"] = True
        try:
            nd = json.loads(next_data_match.group(1))
            # Walk looking for category-like structures
            nd_str = json.dumps(nd, ensure_ascii=False)
            cats = re.findall(r'"title"\s*:\s*"([^"]{5,60})"', nd_str)
            result["category_names"] = list(dict.fromkeys(cats))[:20]
        except json.JSONDecodeError:
            pass

    # Product links (both live and wayback-wrapped)
    raw_links = re.findall(r'href=["\']([^"\']*?/product/[^"\']+)["\']', html)
    clean = set()
    for lnk in raw_links:
        # Strip Wayback Machine prefix if present
        m = re.search(r"/(https?://[^/]*/product/[^?\"']+)", lnk)
        if m:
            clean.add(m.group(1).split("?")[0])
        elif "/product/" in lnk and "ozon.ru" in lnk:
            clean.add(lnk.split("?")[0])
    result["product_links"] = list(clean)[:30]

    return result


def test_wayback() -> dict:
    """Approach 1: fetch Ozon category via archive.org Wayback Machine."""
    label = "1 archive.org"
    print(f"\n[{label}] Looking up snapshot for {TARGET_URL} ...")

    snapshot_url = _wayback_available_url(TARGET_URL)
    if not snapshot_url:
        print(f"[{label}] /wayback/available returned nothing — trying CDX ...")
        snapshot_url = _wayback_cdx_latest(TARGET_URL)

    if not snapshot_url:
        print(f"[{label}] No snapshot found in Wayback Machine.")
        return {
            "label": label,
            "snapshot_url": None,
            "status": None,
            "title": "no snapshot found",
            "product_links": 0,
            "category_names": [],
            "verdict": "no_snapshot",
        }

    print(f"[{label}] Snapshot URL: {snapshot_url}")

    try:
        from curl_cffi import requests as cffi_requests
        resp = cffi_requests.get(snapshot_url, timeout=30, impersonate="chrome120")
        status = resp.status_code
        html = resp.text
    except ImportError:
        import urllib.request
        req = urllib.request.Request(snapshot_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            status = r.status
            html = r.read().decode("utf-8", errors="replace")

    print(f"[{label}] HTTP {status}, page length: {len(html):,} chars")

    data = _extract_ozon_data(html)
    print(f"[{label}] title:          {data['title']!r}")
    print(f"[{label}] has __NEXT_DATA__: {data['has_next_data']}")
    print(f"[{label}] product_links:  {len(data['product_links'])}")
    if data["product_links"]:
        print(f"[{label}] sample links:")
        for lnk in data["product_links"][:5]:
            print(f"           {lnk}")
    if data["category_names"]:
        print(f"[{label}] category_names (up to 10): {data['category_names'][:10]}")

    verdict = "WORKS" if status == 200 and len(html) > 5000 else "blocked"
    return {
        "label": label,
        "snapshot_url": snapshot_url,
        "status": status,
        "title": data["title"],
        "product_links": len(data["product_links"]),
        "category_names": data["category_names"][:10],
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Approach 2 — Real Chrome user profile (launch_persistent_context)
# ---------------------------------------------------------------------------
# Uses the user's existing Chrome cookies + fingerprint.
# DataDome cannot distinguish this from a regular browser session.
#
# REQUIREMENT: Close Chrome before running this approach.
#   venv/Scripts/python scripts/_test_ozon_bypass_v3.py --profile
# ---------------------------------------------------------------------------

async def test_user_profile() -> dict:
    """Approach 2: Playwright with real Chrome user profile."""
    label = "2 chrome-profile"
    user_data = os.path.expanduser("~/AppData/Local/Google/Chrome/User Data")
    if not os.path.isdir(user_data):
        return {
            "label": label,
            "status": None,
            "title": "Chrome User Data dir not found",
            "anchors": 0,
            "verdict": "skip",
        }

    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            ctx = await p.chromium.launch_persistent_context(
                user_data_dir=user_data,
                headless=False,          # visible window — much less suspicious to DataDome
                args=[
                    "--profile-directory=Default",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-blink-features=AutomationControlled",
                ],
            )
            page = await ctx.new_page()
            resp = await page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30_000)
            status = resp.status if resp else None
            await page.wait_for_timeout(6_000)
            title = await page.title()
            body = (await page.evaluate("document.body.innerText"))[:300]
            anchors = await page.eval_on_selector_all('a[href*="/product/"]', "els => els.length")
            await ctx.close()

        print(f"\n[{label}] status={status} title={title!r}")
        print(f"  body[:300]: {body!r}")
        print(f"  product anchors: {anchors}")
        verdict = "WORKS" if status == 200 and anchors > 0 else "blocked"
        return {"label": label, "status": status, "title": title, "anchors": anchors, "verdict": verdict}

    except Exception as exc:
        msg = str(exc)
        print(f"\n[{label}] EXCEPTION: {msg}")
        if "user data directory is already in use" in msg.lower() or "singleton" in msg.lower():
            note = "Chrome is open — close all Chrome windows and retry"
            print(f"  → {note}")
            return {"label": label, "status": None, "title": note, "anchors": 0, "verdict": "blocked (Chrome open)"}
        return {"label": label, "status": None, "title": msg[:80], "anchors": 0, "verdict": "blocked"}


# ---------------------------------------------------------------------------
# Approach 3 — Residential proxy (Webshare free tier / Smartproxy / Bright Data)
# ---------------------------------------------------------------------------
# With a residential IP, DataDome typically allows the request through.
#
# HOW TO GET FREE RESIDENTIAL PROXIES:
#
#   Option A — Webshare.io (recommended, easiest)
#     1. Go to https://www.webshare.io/ → "Start for Free"
#     2. Register with email; no credit card needed for free tier
#     3. Free tier: 10 residential proxies, 1 GB/month
#     4. Go to Dashboard → Proxy List → Download list
#     5. Format:  <host>:<port>:<username>:<password>
#     6. Set WEBSHARE_PROXY below and run: python scripts/_test_ozon_bypass_v3.py --proxy
#
#   Option B — Smartproxy free trial
#     1. https://smartproxy.com/ → "Start Free Trial"
#     2. 100 MB free residential bandwidth
#     3. Endpoint: gate.smartproxy.com:10000, user: spuser-XXXX, pass from dashboard
#
#   Option C — Bright Data 3-day trial
#     1. https://brightdata.com/ → "Start Free Trial" (requires credit card verification)
#     2. Residential proxy endpoint in dashboard
#
# After registration, set env var or edit PROXY_URL below:
#
#   export WEBSHARE_PROXY="http://user:pass@p.webshare.io:80"
#
# ---------------------------------------------------------------------------

PROXY_URL = os.environ.get("WEBSHARE_PROXY", "")  # set this env var after registration


def test_residential_proxy(proxy_url: str = PROXY_URL) -> dict:
    """Approach 3: curl_cffi with residential proxy."""
    label = "3 residential-proxy"
    if not proxy_url:
        print(f"\n[{label}] No proxy configured. Set WEBSHARE_PROXY env var.")
        return {
            "label": label,
            "status": None,
            "title": "WEBSHARE_PROXY not set",
            "anchors": 0,
            "verdict": "skip (no proxy configured)",
        }

    try:
        from curl_cffi import requests as cffi_requests
        proxies = {"http": proxy_url, "https": proxy_url}
        r = cffi_requests.get(
            TARGET_URL,
            impersonate="chrome120",
            proxies=proxies,
            timeout=20,
        )
        status = r.status_code
        title_m = re.search(r"<title[^>]*>(.*?)</title>", r.text, re.IGNORECASE)
        title = title_m.group(1).strip() if title_m else ""
        anchors = len(re.findall(r'href="[^"]*?/product/', r.text))
        print(f"\n[{label}] status={status} title={title!r} product_anchors≈{anchors}")
        verdict = "WORKS" if status == 200 and anchors > 0 else "blocked"
        return {"label": label, "status": status, "title": title, "anchors": anchors, "verdict": verdict}

    except Exception as exc:
        print(f"\n[{label}] EXCEPTION: {exc}")
        return {"label": label, "status": None, "title": str(exc)[:80], "anchors": 0, "verdict": "blocked"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = sys.argv[1:]
    run_profile = "--profile" in args
    run_proxy = "--proxy" in args
    run_all = "--all" in args

    print("=" * 70)
    print("Ozon DataDome bypass v3")
    print(f"Target: {TARGET_URL}")
    print("=" * 70)

    results = []

    # --- Approach 1: always run (safe, no user action needed) ---
    r1 = test_wayback()
    results.append(r1)

    # --- Approach 2: only when --profile or --all (requires Chrome closed) ---
    if run_profile or run_all:
        r2 = asyncio.run(test_user_profile())
        results.append(r2)
    else:
        print(
            "\n[2 chrome-profile] SKIPPED."
            " Run with --profile flag after closing Chrome."
        )
        results.append({
            "label": "2 chrome-profile",
            "status": None,
            "title": "skipped (use --profile flag)",
            "product_links": 0,
            "verdict": "skip",
        })

    # --- Approach 3: only when --proxy or --all (requires WEBSHARE_PROXY set) ---
    if run_proxy or run_all:
        r3 = test_residential_proxy()
        results.append(r3)
    else:
        print(
            "\n[3 residential-proxy] SKIPPED."
            " Set WEBSHARE_PROXY env var and run with --proxy flag."
        )
        results.append({
            "label": "3 residential-proxy",
            "status": None,
            "title": "skipped (use --proxy flag + WEBSHARE_PROXY env var)",
            "product_links": 0,
            "verdict": "skip",
        })

    # Summary table
    print("\n")
    print("=" * 70)
    print(f"{'Approach':<25} {'Status':>6}  {'Verdict':<30}  Title")
    print("-" * 70)
    any_works = False
    for r in results:
        works = r.get("verdict") == "WORKS"
        if works:
            any_works = True
        print(
            f"{r['label']:<25} {str(r.get('status') or '-'):>6}  "
            f"{r.get('verdict','?'):<30}  {str(r.get('title',''))[:40]!r}"
        )
    print("=" * 70)

    if any_works:
        working = [r["label"] for r in results if r.get("verdict") == "WORKS"]
        print(f"\nWorking approaches: {working}")
        sys.exit(0)
    else:
        print("\nNo approach returned live data yet.")
        print("\nNext steps for user:")
        print("  Approach 2 — Browser profile:")
        print("    1. Close ALL Chrome windows.")
        print("    2. Run: venv/Scripts/python scripts/_test_ozon_bypass_v3.py --profile")
        print()
        print("  Approach 3 — Free residential proxy:")
        print("    Option A (Webshare, recommended):")
        print("      1. Register at https://www.webshare.io/  (free, no CC)")
        print("      2. Dashboard → Proxy List → copy one proxy")
        print("      3. set WEBSHARE_PROXY=http://user:pass@p.webshare.io:80")
        print("      4. Run: venv/Scripts/python scripts/_test_ozon_bypass_v3.py --proxy")
        print("    Option B (Smartproxy 100 MB trial):")
        print("      1. https://smartproxy.com/ → Start Free Trial")
        print("      2. set WEBSHARE_PROXY=http://spuser-XXX:PASS@gate.smartproxy.com:10000")
        print("    Option C (Bright Data 3-day trial):")
        print("      1. https://brightdata.com/ → Start Free Trial (CC required)")
        sys.exit(1)


if __name__ == "__main__":
    main()
