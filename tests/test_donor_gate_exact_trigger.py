# -*- coding: utf-8 -*-
"""Tests for the exact-mode donor-gate trigger: WbCardSource._model_index_mismatch
and _should_run_donor_gate. These decide WHEN the LLM donor gate is invoked —
no LLM call here, pure deterministic logic.

Motivating case: a "Mi Band 7" card matched a "Mi Band 8" target at fuzzy score
78.2 (>= exact threshold) and bypassed the gate (which previously fired only on
brand_line, 60-78), letting Mi Band 7 specs leak into Mi Band 8."""
import pytest

from app.services.enrichment.sources.wb_card_source import WbCardSource


# (target, donor, expect_mismatch)
MISMATCH_CASES = [
    ("Умные часы Xiaomi Mi Band 8", "Фитнес браслет Xiaomi Mi Band 7", True),
    ("Apple Watch Series 9 45mm", "Apple Watch Series 8 45mm", True),
    ("70mai A500S", "70mai A50", True),
    ("Osprey Farpoint 40", "Osprey Daylite 13", True),
    # same product — donor a richer superset OR identical model index → trust
    ("Xiaomi Mi Band 8", "Xiaomi Smart Band 8 NFC", False),
    ("Очки Ray-Ban Wayfarer RB2140", "Ray-Ban Wayfarer RB2140 черные", False),
    ("Газонокосилка Bosch ARM 37", "Bosch ARM 37 газонокосилка", False),
    ("Nokian Nordman 8 205/55 R16", "Nokian Nordman 8 205/55 R16 зимняя", False),
    ("Xiaomi Mi Band 8", "Xiaomi Mi Band 8 Pro", False),
]


@pytest.mark.parametrize("target,donor,expected", MISMATCH_CASES)
def test_model_index_mismatch(target, donor, expected):
    assert WbCardSource._model_index_mismatch(target, donor) is expected


class _GateProbe(WbCardSource):
    """Avoid running WbCardSource.__init__ (network clients etc.) — we only
    exercise the pure-logic gate decision."""
    def __init__(self):  # noqa: D401 - intentional no-op
        pass


@pytest.mark.parametrize("mode,target,donor,expected", [
    # brand_line always gates
    ("brand_line", "Whatever A", "Whatever B", True),
    # exact + model mismatch → gate
    ("exact", "Mi Band 8", "Mi Band 7", True),
    # exact + matching model index → trust, no gate
    ("exact", "Mi Band 8", "Xiaomi Smart Band 8 NFC", False),
    # skip never gates
    ("skip", "Mi Band 8", "Mi Band 7", False),
])
def test_should_run_donor_gate(mode, target, donor, expected):
    probe = _GateProbe()
    assert probe._should_run_donor_gate(mode, target, donor) is expected
