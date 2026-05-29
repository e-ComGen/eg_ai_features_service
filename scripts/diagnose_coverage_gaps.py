"""Расследование: почему value_id resolution ~65% и почему Optional только 41%."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Берём v2 (текущий лучший с PDF+RAG fixed): 75.0% / 41.4%
RESULT = Path(__file__).resolve().parent / "eval_results" / "ps_by_name_20260528_163317.json"
data = json.loads(RESULT.read_text(encoding="utf-8"))

# Также загружаем Ozon dictionary чтобы знать какие чары REQ vs OPT и есть ли values
import gzip
DICT_PATH = Path(__file__).resolve().parent.parent / "app" / "services" / "enrichment" / "strategies" / "dictionaries" / "data" / "ozon_dictionary.json.gz"
with gzip.open(DICT_PATH, "rt", encoding="utf-8") as f:
    dict_data = json.load(f)

# Найдём блок для нашей категории (description_category_id=17028612, type_id=91910)
TARGET_CAT_KEY = "17028612:91910"
chars = None
for k, v in dict_data.get("categories", {}).items():
    if str(v.get("description_category_id")) == "17028612" and str(v.get("type_id")) == "91910":
        chars = v.get("characteristics", [])
        break
if not chars:
    sys.exit("Could not find PSU chars in dictionary")

char_by_id = {c["id"]: c for c in chars}
n_total = len(chars)
n_req = sum(1 for c in chars if c.get("is_required"))
n_opt = n_total - n_req
print(f"PSU category: {n_total} chars ({n_req} required, {n_opt} optional)")
print()

# Подсчёт по атрибутам через 20 продуктов
products = data["products"]
n = len(products)
print(f"Products: {n}")
print()

# ===== 1. Required анализ =====
print("=" * 70)
print("PART 1: REQUIRED attributes — какие пропускаем?")
print("=" * 70)
req_chars = [c for c in chars if c.get("is_required")]
for c in req_chars:
    aid = c["id"]
    nm = c["name"]
    filled = [p for p in products if any(f["attribute_id"] == aid for f in p["filled"])]
    fill_count = len(filled)
    has_values = bool(c.get("values"))
    print(f"  [{aid}] {nm[:50]:50s}  {fill_count}/{n}  has_dict_values={has_values}")
    if fill_count < n:
        missing = [p["name"][:50] for p in products if not any(f["attribute_id"] == aid for f in p["filled"])]
        for m in missing[:5]:
            print(f"      MISSED: {m}")

# ===== 2. Value_id resolution failures =====
print()
print("=" * 70)
print("PART 2: VALUE_ID FAILURES — атрибуты с values но без value_id в результате")
print("=" * 70)
vid_failures: Counter = Counter()
vid_total: Counter = Counter()
samples_per_attr: dict = defaultdict(list)
for p in products:
    for f in p["filled"]:
        aid = f["attribute_id"]
        c = char_by_id.get(aid)
        if not c or not c.get("values"):
            continue
        vid_total[aid] += 1
        if not f.get("value_id"):
            vid_failures[aid] += 1
            if len(samples_per_attr[aid]) < 3:
                samples_per_attr[aid].append((f.get("value"), f.get("source")))

print(f"\nTop attrs with value_id failure rate:")
for aid, fail_count in vid_failures.most_common(15):
    total = vid_total[aid]
    rate = fail_count / total * 100
    nm = char_by_id[aid]["name"][:40]
    n_allowed = len(char_by_id[aid].get("values", []))
    print(f"  [{aid}] {nm:40s} fail {fail_count}/{total} ({rate:.0f}%)  dict has {n_allowed} values")
    for value, source in samples_per_attr[aid][:2]:
        v = str(value)[:50]
        print(f"      EMITTED: '{v}' (src={source})")

# ===== 3. Optional never filled =====
print()
print("=" * 70)
print("PART 3: OPTIONAL — какие НИКОГДА не заполняются и почему")
print("=" * 70)
opt_chars = [c for c in chars if not c.get("is_required")]
attr_fill = Counter()
for p in products:
    for f in p["filled"]:
        attr_fill[f["attribute_id"]] += 1

never = [c for c in opt_chars if attr_fill[c["id"]] == 0]
rare = [c for c in opt_chars if 0 < attr_fill[c["id"]] <= 3]
common = [c for c in opt_chars if attr_fill[c["id"]] > 3]
print(f"\nOptional never filled: {len(never)} / {n_opt}")
for c in never:
    name = c["name"][:55]
    has_v = bool(c.get("values"))
    typ = c.get("type") or "?"
    print(f"  [{c['id']}] {name:55s} type={typ[:8]} has_values={has_v}")
print(f"\nOptional rarely filled (1-3 of 20): {len(rare)}")
for c in rare:
    name = c["name"][:55]
    print(f"  [{c['id']}] {name:55s} {attr_fill[c['id']]}/{n}")
print(f"\nOptional common (>3 of 20): {len(common)}")
