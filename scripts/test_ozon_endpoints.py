"""Test multiple Ozon scraping approaches from local IP."""
from __future__ import annotations
import sys, json, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Try curl_cffi first (it impersonates real Chrome TLS fingerprint)
try:
    from curl_cffi import requests as cc_requests
    HAS_CURL_CFFI = True
    print("[OK] curl_cffi available")
except ImportError:
    HAS_CURL_CFFI = False
    print("[!] curl_cffi NOT installed — pip install curl_cffi")

import httpx
import uuid

PRODUCT = "Блок питания Cooler Master MWE Gold 750 V2"

# Mobile-app headers
MOBILE_HEADERS = {
    "User-Agent": "ozonapp_android/17.48.0+2528",
    "x-o3-app-name": "ozonapp_android",
    "x-o3-app-version": "17.48.0",
    "x-o3-fp": "1.01ae145142fa31f9",
    "MOBILE-GAID": str(uuid.uuid4()).upper(),
    "Accept": "application/json",
    "Accept-Language": "ru-RU,ru;q=0.9",
}

# Chrome web headers (alternative — pretend desktop browser)
CHROME_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.ozon.ru/",
}

TESTS = [
    ("api.ozon.ru mobile",   "https://api.ozon.ru/composer-api.bx/page/json/v2",   MOBILE_HEADERS),
    ("api.ozon.ru chrome",   "https://api.ozon.ru/composer-api.bx/page/json/v2",   CHROME_HEADERS),
    ("www.ozon.ru mobile",   "https://www.ozon.ru/api/composer-api.bx/page/json/v2", MOBILE_HEADERS),
    ("www.ozon.ru chrome",   "https://www.ozon.ru/api/composer-api.bx/page/json/v2", CHROME_HEADERS),
    ("entrypoint mobile",    "https://www.ozon.ru/api/entrypoint-api.bx/page/json/v2", MOBILE_HEADERS),
]
params = {"url": f"/search/?text={PRODUCT}"}

print()
print("=" * 60)
print("Test 1: httpx with various endpoints/headers")
print("=" * 60)
for name, url, hdrs in TESTS:
    try:
        r = httpx.get(url, params=params, headers=hdrs, timeout=15, follow_redirects=True)
        print(f"  {name:30s} → HTTP {r.status_code}  body={len(r.content)}")
        if r.status_code == 200:
            body = r.text[:200]
            print(f"      body start: {body}")
    except Exception as e:
        print(f"  {name:30s} → ERR: {type(e).__name__}: {e}")
    time.sleep(1)

if HAS_CURL_CFFI:
    print()
    print("=" * 60)
    print("Test 2: curl_cffi (Chrome TLS impersonation)")
    print("=" * 60)
    for impersonate in ["chrome131", "chrome120", "safari17_0"]:
        for name, url, hdrs in TESTS[:3]:  # only test top 3
            try:
                r = cc_requests.get(url, params=params, headers=hdrs, timeout=15, impersonate=impersonate)
                print(f"  {impersonate:12s} {name:30s} → HTTP {r.status_code}  body={len(r.content)}")
                if r.status_code == 200:
                    print(f"      first 300 bytes of JSON:")
                    print(f"      {r.text[:300]}")
                    break
            except Exception as e:
                print(f"  {impersonate:12s} {name:30s} → ERR: {e}")
            time.sleep(1)
