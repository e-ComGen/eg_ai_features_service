"""Build Wildberries category/characteristic dictionary via Content API (auth required).

Schema mirrors Ozon dictionary:
  {
    "schema_version": 1,
    "source": "wildberries_content_api",
    "generated_at": "YYYY-MM-DD",
    "categories": {
      "<subject_id>": {
        "subject_id": int, "subject_name": str, "parent_id": int,
        "characteristics": [
          {"id": int, "name": str, "required": bool, "popular": bool,
           "charcType": int, "unitName": str, "maxCount": int,
           "values": [{"id": int, "value": str}, ...]}  # only for charcType ∈ {1,4}
        ]
      }
    }
  }

ETA ~50-65 min:
  Phase 1: subjects (~8 paginated calls, 3 sec)
  Phase 2: chars per subject (7175 × 0.3s = 36 min)
  Phase 3: values for unique dict-backed charcs (~3-5k calls × 0.3s = 15-25 min)

Resume support via .partial files.
"""
from __future__ import annotations
import gzip
import json
import logging
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
sys.stdout.reconfigure(line_buffering=True)

TOKEN = os.environ.get("WB_API_KEY")
if not TOKEN:
    sys.exit("WB_API_KEY not set in .env")

BASE = "https://content-api.wildberries.ru"
DATA_DIR = PROJECT_ROOT / "app" / "services" / "enrichment" / "strategies" / "dictionaries" / "data"
OUT_PATH = DATA_DIR / "wb_dictionary.json.gz"
SUBJECTS_PARTIAL = DATA_DIR / "wb_subjects.partial.json"
CHARS_PARTIAL = DATA_DIR / "wb_chars.partial.json"
VALUES_PARTIAL = DATA_DIR / "wb_values.partial.json"

RATE_LIMIT_SLEEP = 0.3
VALUES_LIMIT = 10000
PARTIAL_SAVE_EVERY = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def http_get(path: str, **params):
    q = urllib.parse.urlencode(params, encoding="utf-8") if params else ""
    url = f"{BASE}{path}" + (f"?{q}" if q else "")
    req = urllib.request.Request(url, headers={"Authorization": TOKEN})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                log.warning("429 on %s, sleep %ds", path, wait)
                time.sleep(wait)
                continue
            if e.code in (400, 404):
                return None
            log.warning("HTTP %d %s: %s", e.code, path, e.read()[:200])
            return None
        except Exception as e:
            log.warning("net error %s: %s", path, e)
            time.sleep(2)
    return None


def fetch_all_subjects() -> list[dict]:
    if SUBJECTS_PARTIAL.exists():
        cached = json.loads(SUBJECTS_PARTIAL.read_text(encoding="utf-8"))
        log.info("Subjects cached: %d", len(cached))
        return cached
    log.info("Phase 1: fetching all subjects")
    out: list[dict] = []
    offset = 0
    while True:
        r = http_get("/content/v2/object/all", locale="ru", limit=1000, offset=offset)
        if not r or not r.get("data"):
            break
        chunk = r["data"]
        out.extend(chunk)
        log.info("  ... %d subjects", len(out))
        if len(chunk) < 1000:
            break
        offset += 1000
        time.sleep(RATE_LIMIT_SLEEP)
    SUBJECTS_PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    SUBJECTS_PARTIAL.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    log.info("Phase 1 done: %d subjects (cached)", len(out))
    return out


def fetch_chars(subjects: list[dict]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {}
    if CHARS_PARTIAL.exists():
        result = {int(k): v for k, v in json.loads(CHARS_PARTIAL.read_text(encoding="utf-8")).items()}
        log.info("Chars cached: %d subjects already processed", len(result))
    total = len(subjects)
    done = 0
    for i, s in enumerate(subjects):
        sid = s.get("subjectID")
        if sid is None or sid in result:
            continue
        r = http_get(f"/content/v2/object/charcs/{sid}", locale="ru")
        result[sid] = r.get("data", []) if r else []
        done += 1
        time.sleep(RATE_LIMIT_SLEEP)
        if done % PARTIAL_SAVE_EVERY == 0:
            CHARS_PARTIAL.write_text(json.dumps({str(k): v for k, v in result.items()}, ensure_ascii=False), encoding="utf-8")
            log.info("Chars: %d/%d (%.1f%%) — +%d this run", i + 1, total, (i + 1) / total * 100, done)
    CHARS_PARTIAL.write_text(json.dumps({str(k): v for k, v in result.items()}, ensure_ascii=False), encoding="utf-8")
    log.info("Phase 2 done: %d subjects covered", len(result))
    return result


def fetch_values(subj_chars: dict[int, list[dict]]) -> dict[int, list[dict]]:
    unique: dict[int, dict] = {}
    for chars in subj_chars.values():
        for c in chars:
            cid = c.get("charcID")
            if cid is None or cid in unique:
                continue
            if c.get("charcType") in (1, 4):
                unique[cid] = c
    log.info("Unique dict-backed charcIDs: %d", len(unique))

    result: dict[int, list[dict]] = {}
    if VALUES_PARTIAL.exists():
        result = {int(k): v for k, v in json.loads(VALUES_PARTIAL.read_text(encoding="utf-8")).items()}
        log.info("Values cached: %d charcIDs already processed", len(result))

    todo = [cid for cid in unique if cid not in result]
    log.info("Values to fetch: %d (skipping %d cached)", len(todo), len(result))
    done = 0
    for cid in todo:
        r = http_get(f"/content/v2/directory/charcs/{cid}", locale="ru", limit=VALUES_LIMIT)
        if r and "data" in r:
            result[cid] = [{"id": v.get("id"), "value": v.get("value")} for v in r["data"]]
        else:
            result[cid] = []
        done += 1
        time.sleep(RATE_LIMIT_SLEEP)
        if done % PARTIAL_SAVE_EVERY == 0:
            VALUES_PARTIAL.write_text(json.dumps({str(k): v for k, v in result.items()}, ensure_ascii=False), encoding="utf-8")
            log.info("Values: %d/%d done this run", done, len(todo))
    VALUES_PARTIAL.write_text(json.dumps({str(k): v for k, v in result.items()}, ensure_ascii=False), encoding="utf-8")
    log.info("Phase 3 done: %d charcIDs covered", len(result))
    return result


def build_final(subjects: list[dict], subj_chars: dict[int, list[dict]], values: dict[int, list[dict]]) -> dict:
    by_id = {s["subjectID"]: s for s in subjects}
    cats = {}
    for sid, chars in subj_chars.items():
        if not chars:
            continue
        s = by_id.get(sid, {})
        out_chars = []
        for c in chars:
            cid = c.get("charcID")
            entry = {
                "id": cid,
                "name": c.get("name"),
                "required": c.get("required", False),
                "popular": c.get("popular", False),
                "charcType": c.get("charcType"),
                "unitName": c.get("unitName") or None,
                "maxCount": c.get("maxCount", 1),
                "isVariable": c.get("isVariable", False),
            }
            if cid in values:
                entry["values"] = values[cid]
            out_chars.append(entry)
        cats[str(sid)] = {
            "subject_id": sid,
            "subject_name": s.get("subjectName"),
            "parent_id": s.get("parentID"),
            "characteristics": out_chars,
        }
    return {
        "schema_version": 1,
        "source": "wildberries_content_api",
        "generated_at": time.strftime("%Y-%m-%d"),
        "categories": cats,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subjects = fetch_all_subjects()
    subj_chars = fetch_chars(subjects)
    values = fetch_values(subj_chars)

    log.info("Assembling final dict")
    final = build_final(subjects, subj_chars, values)
    log.info("Categories with chars: %d", len(final["categories"]))

    log.info("Writing %s", OUT_PATH)
    with gzip.open(OUT_PATH, "wt", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False)
    log.info("DONE %s (%.1f MB)", OUT_PATH, OUT_PATH.stat().st_size / 1024 / 1024)


if __name__ == "__main__":
    main()
