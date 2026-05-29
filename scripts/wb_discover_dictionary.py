"""Explore WB Content API — what endpoints return what for dictionary building.

Endpoints (per https://dev.wildberries.ru/openapi/api-information):
  - GET /content/v2/object/parent/all      — top-level categories (groups)
  - GET /content/v2/object/all?locale=ru   — flat list of subject categories
  - GET /content/v2/object/charcs/{subjID} — characteristics for a subject
  - GET /content/v2/directory/charcs/{ID}  — possible values for a charc

Goal: understand schema to write build_wb_dictionary.py.
"""
from __future__ import annotations
import json
import os
import sys
import urllib.request, urllib.parse
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

TOKEN = os.environ["WB_API_KEY"]
BASE = "https://content-api.wildberries.ru"


def get(path: str, **params) -> dict | list:
    q = urllib.parse.urlencode(params, encoding="utf-8")
    url = f"{BASE}{path}" + (f"?{q}" if q else "")
    req = urllib.request.Request(url, headers={"Authorization": TOKEN})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        return {"_error": f"HTTP {e.code}: {body}"}


print("=" * 70)
print("1. PARENT CATEGORIES (groups)")
print("=" * 70)
r = get("/content/v2/object/parent/all", locale="ru")
if isinstance(r, dict) and "data" in r:
    parents = r["data"]
    print(f"Total parent categories: {len(parents)}")
    for p in parents[:8]:
        print(f"  id={p.get('id')} name={p.get('name')[:60]}")

print()
print("=" * 70)
print("2. ALL SUBJECTS (subcategories) — first page")
print("=" * 70)
r = get("/content/v2/object/all", locale="ru", limit=50, offset=0)
if isinstance(r, dict) and "data" in r:
    subjects = r["data"]
    print(f"Subjects in first page: {len(subjects)}")
    # show ones related to PSU
    psu_keywords = ["блок", "питан"]
    for s in subjects[:5]:
        print(f"  id={s.get('subjectID')} parent={s.get('parentID')} name={s.get('subjectName')}")
    print(f"  ... showing first 5")

print()
print("=" * 70)
print("3. SEARCH: 'Блок питания' in subjects")
print("=" * 70)
r = get("/content/v2/object/all", locale="ru", limit=1000, offset=0, name="Блок питания")
if isinstance(r, dict) and "data" in r:
    psu_subjects = r["data"]
    print(f"Matching: {len(psu_subjects)}")
    for s in psu_subjects[:10]:
        print(f"  id={s.get('subjectID')} parent={s.get('parentID')} {s.get('subjectName')}")
    if psu_subjects:
        # Pick one and explore characteristics
        SUBJ_ID = psu_subjects[0]["subjectID"]
        print()
        print("=" * 70)
        print(f"4. CHARACTERISTICS for subject {SUBJ_ID}: {psu_subjects[0]['subjectName']}")
        print("=" * 70)
        r = get(f"/content/v2/object/charcs/{SUBJ_ID}", locale="ru")
        if isinstance(r, dict) and "data" in r:
            chars = r["data"]
            print(f"Total characteristics: {len(chars)}")
            for c in chars[:10]:
                req = "REQ" if c.get("required") else "opt"
                pop = "POP" if c.get("popular") else ""
                ctype = c.get("charcType")
                print(f"  id={c.get('id')} {req} {pop} type={ctype} name={c.get('name')[:50]}")
            # Pick first with chartType=1 (enum) and explore values
            for c in chars:
                if c.get("charcType") in (1, 4):  # dictionary types
                    cid = c.get("id")
                    print()
                    print("=" * 70)
                    print(f"5. VALUES for char {cid}: {c.get('name')}")
                    print("=" * 70)
                    r = get(f"/content/v2/directory/charcs/{cid}", locale="ru", limit=20)
                    if isinstance(r, dict) and "data" in r:
                        vals = r["data"]
                        print(f"Total values: {r.get('total','?')}, first 10:")
                        for v in vals[:10]:
                            print(f"  id={v.get('id')} value={v.get('value')}")
                    break
        elif "_error" in r:
            print(r["_error"])
