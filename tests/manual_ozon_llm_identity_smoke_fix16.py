"""Live smoke: real DeepSeek call for FIX-16 _verify_product_identity on the
manifest oracle pairs (docs/MANIFEST_ozon_llm_identity_verifier_fix16.md).

NOT a pytest test -- makes real network calls (small $ cost). Judge = the LLM
itself; this script's ONLY job is to print the REAL verdicts so a human/QA can
confirm the prompt actually discriminates siblings/accessories/generations
correctly, per the manifest's acceptance criterion #2.

Run: PYTHONUTF8=1 PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe tests/manual_ozon_llm_identity_smoke_fix16.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from app.services.enrichment.sources.ozon_card_source import OzonCardSource

# (label, query, title, expected_verdict) -- expected per manifest oracle table.
_PAIRS = [
    ("V3 sibling",    "POCO X6 5G", "Смартфон POCO M8 Pro 5G 8/256 ГБ", "different"),
    ("V5 gen-sibling", "iPhone 15", "Смартфон Apple iPhone 14 128 ГБ", "different"),
    ("V7 accessory",  "POCO X6 5G", "Чехол для POCO X6 5G силиконовый", "different"),
    ("V1 same",       "POCO X6 5G", "Смартфон POCO X6 5G 8/256 ГБ чёрный", "same"),
    ("V2 same-variant", "POCO X6 5G", "Смартфон POCO X6 5G 12/512 ГБ синий", "same"),
]


async def main() -> int:
    src = OzonCardSource()
    mismatches = []
    for label, query, title, expected in _PAIRS:
        verdict = await src._verify_product_identity(query, title)
        key = src._identity_cache.get((query, title))
        ok = "OK" if verdict == expected else "MISMATCH"
        if verdict != expected:
            mismatches.append((label, expected, verdict))
        print(
            f"[{ok}] {label}: query={query!r} title={title!r} "
            f"-> verdict={verdict!r} (expected={expected!r}) cached={key!r}",
            flush=True,
        )

    print("\n=== SUMMARY ===")
    if mismatches:
        print(f"{len(mismatches)}/{len(_PAIRS)} MISMATCH:")
        for label, expected, got in mismatches:
            print(f"  - {label}: expected={expected} got={got}")
        # Any sibling/accessory judged "same" is the dangerous failure mode
        # the manifest calls out explicitly -- fail loudly, never silent-pass.
        return 1
    print(f"All {len(_PAIRS)} pairs matched expectation.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
