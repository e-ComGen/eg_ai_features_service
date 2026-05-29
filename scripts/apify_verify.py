"""Verify Apify token works and check account balance."""
import os, sys, urllib.request, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
TOK = os.environ["APIFY_TOKEN"]

req = urllib.request.Request(f"https://api.apify.com/v2/users/me?token={TOK}")
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        d = json.loads(r.read())["data"]
    print(f"User: {d.get('username')}")
    print(f"Plan: {d.get('plan', {}).get('id', '?')}")
    print(f"Email: {d.get('email')}")
    print(f"Balance USD: {d.get('usageCycle', {})}")
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.read().decode()[:200]}")
