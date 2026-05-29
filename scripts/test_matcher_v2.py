"""Test new matcher only on attributes with VALID Ozon dict values (skip broken ones)."""
from __future__ import annotations
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Block sentence_transformers import to avoid segfault
import sys as _s
_s.modules['sentence_transformers'] = None  # type: ignore

from app.services.enrichment.strategies.dictionaries.ozon_loader import resolve_value_id

CAT_ID, TYPE_ID = 17028612, 91910

# Test cases on KNOWN-GOOD dict attrs (skip broken 8229 Тип, 23278 Назначение)
TESTS = [
    # (attr_id, name, input)
    (22473, "PFC", "Aктивный"),       # Latin A homoglyph → should resolve to 'Активный'
    (22473, "PFC", "Активный"),       # direct
    (5126,  "Voltage", "100-240В"),   # direct
    (5126,  "Voltage", "100 - 240 V"),  # spaces + Latin V
    (5126,  "Voltage", "110 - 240 V"),
    (5127,  "MoboConn", "ATX 20+4 пин"),  # direct
    (5127,  "MoboConn", "20+4 pin ATX"),  # word order + Latin pin
    (5127,  "MoboConn", "ATX 24+4 пин"),
    (22476, "Protect", "OCP"),        # alias → "защита от перегрузки по току" → match prefix
    (22476, "Protect", "OCP (защита от перегрузки по току)"),  # exact
    (22476, "Protect", "защита от перегрузки по току"),  # synonym
    (10096, "Color", "['чёрный']"),   # array unwrap
    (10096, "Color", "Чёрный"),       # ё→е
    (4389,  "Country", "Китай"),
    (4389,  "Country", "['Китай']"),  # array unwrap
    (4389,  "Country", "Китай (Тайвань)"),  # strip parens
    (5125,  "ATX Spec", "ATX 2.4"),
    (5125,  "ATX Spec", "ATX 2.52"),  # not in dict, should rapidfuzz to 2.5
    (5125,  "ATX Spec", "ATX12V"),    # different format
]

passed = 0
failed = 0
for aid, name, val in TESTS:
    vid = resolve_value_id(CAT_ID, TYPE_ID, aid, val)
    status = "✓" if vid else "✗"
    if vid: passed += 1
    else: failed += 1
    print(f" {status} attr={aid} ({name:9s}) input={val!r:40s} -> vid={vid}")

print()
print(f"Passed: {passed}/{len(TESTS)} ({passed/len(TESTS)*100:.0f}%)")
