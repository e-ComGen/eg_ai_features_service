"""One-shot script: drop truncated values lists and gzip the Ozon dictionary."""
import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "app/services/enrichment/strategies/dictionaries/data"
SRC = DATA_DIR / "ozon_dictionary.json"
DST = DATA_DIR / "ozon_dictionary.json.gz"

print(f"Source: {SRC}")
print(f"Size before: {SRC.stat().st_size / 1024 / 1024:.1f} MB")

try:
    import ijson  # noqa: F401
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False

print(f"ijson available: {HAS_IJSON} (using json.load fallback)")

print("Loading JSON (this may take a while for 1.8 GB)...")
with open(SRC, "r", encoding="utf-8") as fh:
    data = json.load(fh)

cats = data.get("categories", data)

total_chars = 0
truncated_dropped = 0
vals_before = 0
vals_after = 0

for entry in cats.values():
    for char in entry.get("characteristics", []):
        total_chars += 1
        v = char.get("values")
        if v:
            vals_before += len(v)
        if char.get("values_truncated") is True:
            if "values" in char:
                truncated_dropped += 1
                vals_before_this = len(char["values"])
                del char["values"]
                vals_after += 0  # removed entirely
        else:
            if v:
                vals_after += len(v)

print(f"Total characteristics processed: {total_chars}")
print(f"Truncated entries with values dropped: {truncated_dropped}")
print(f"Values count before: {vals_before:,}")
print(f"Values count after:  {vals_after:,}")
print(f"Values removed:      {vals_before - vals_after:,}")

print(f"Writing gzipped output to {DST} ...")
with gzip.open(DST, "wt", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False)

gz_size = DST.stat().st_size
print(f"Size after (.gz): {gz_size / 1024 / 1024:.1f} MB")

if gz_size < 50 * 1024 * 1024:
    print(f"ERROR: .gz file is only {gz_size / 1024 / 1024:.1f} MB — below 50 MB safety threshold. NOT deleting original.")
    sys.exit(1)

print("Safety check passed. Deleting original 1.8 GB .json ...")
os.remove(SRC)
print("Done.")
