"""Unit-test value matcher upgrade with our failing cases from the eval."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    resolve_value_id, get_ozon_characteristics_for_type,
)

CAT_ID = 17028612
TYPE_ID = 91910

# Failing cases from diagnose_coverage_gaps.py
TESTS = [
    # (attribute_id, attribute_name, input_value, should_resolve)
    (22473, "Корректор коэффициента мощности (PFC)", "Aктивный",       True),
    (22473, "Корректор коэффициента мощности (PFC)", "Активный",       True),
    (8229,  "Тип",                                   "Немодульный",    True),
    (8229,  "Тип",                                   "Fully-Modular",  True),
    (8229,  "Тип",                                   "Полностью модульный", True),
    (8229,  "Тип",                                   "Модульный",      True),
    (5126,  "Входное напряжение",                    "100 - 240 V",    True),
    (5126,  "Входное напряжение",                    "110 - 240 V",    True),
    (5127,  "Разъемы питания материнской платы",     "20+4 pin ATX",   True),
    (10096, "Цвет товара",                           "['бронза']",     True),
    (10096, "Цвет товара",                           "['черный']",     True),
    (4389,  "Страна-изготовитель",                   "['Китай']",      True),
    (4389,  "Страна-изготовитель",                   "['Тайвань']",    True),
    (4389,  "Страна-изготовитель",                   "Китай (Тайвань)", True),
    (5125,  "Спецификация БП",                       "ATX12V",         True),
    (23278, "Назначение",                            "ПК",             True),
]

# Get available dict values for context
chars = get_ozon_characteristics_for_type(CAT_ID, TYPE_ID)
char_by_id = {c["id"]: c for c in chars}

passed = 0
failed = 0
for aid, name, val, should_resolve in TESTS:
    vid = resolve_value_id(CAT_ID, TYPE_ID, aid, val)
    result = "PASS" if (vid is not None) == should_resolve else "FAIL"
    if result == "PASS":
        passed += 1
    else:
        failed += 1
    char = char_by_id.get(aid, {})
    dict_values = char.get("values", [])
    dict_preview = [v["value"] for v in dict_values[:8]]
    print(f"{result}: attr={aid} '{name[:30]:30s}' input='{val[:25]:25s}' -> vid={vid}")
    if result == "FAIL":
        print(f"      Dict has {len(dict_values)} options. First 8: {dict_preview}")

print()
print(f"Passed: {passed}/{len(TESTS)}")
print(f"Failed: {failed}/{len(TESTS)}")
