"""Оракул #2: donor-gate re-pick в _select_gated_card (WB).

Прежде donor-gate на LLM-вердикте «не тот товар» дропал ВЕСЬ донор (`return []`).
Теперь: трёхзначный вердикт (SAME/DIFFERENT/UNKNOWN, UNKNOWN=ретрай×1), на
DIFFERENT/двойной-UNKNOWN — re-pick следующей-лучшей карты из пула; кап LLM =
_GateBudget. Sentinel: Mi Band 8 против пула [Mi Band 7, Mi Band 6] (обе DIFFERENT)
ОБЯЗАН вернуть None (галлюн-донор не воскрешаем). Решение: fable. Не удалять без
пере-обоснования. Тест изолирует re-pick логику (monkeypatch _rank_cards/скоринга).
"""
import asyncio

from app.services.enrichment.sources.donor_gate import DonorVerdict as V
from app.services.enrichment.sources.wb_card_source import WbCardSource, _GateBudget


class FakeGate:
    """DI-подмена DonorMatchGate: title → очередь вердиктов, журнал вызовов."""
    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    async def verdict(self, target, donor):
        self.calls.append(donor)
        q = self.script.get(donor)
        v = q.pop(0) if q else V.UNKNOWN
        return (v, True)  # в тесте каждый verdict = реальный LLM-вызов


def _wb(ranked, gate, clean=(), skip_below=60):
    wb = WbCardSource.__new__(WbCardSource)  # без тяжёлого __init__
    wb._rank_cards = lambda *a, **k: ranked
    wb._classify_match = lambda s: "skip" if s < skip_below else "brand_line"
    wb._should_run_donor_gate = lambda mode, fn, title: title not in clean
    wb._donor_gate = gate
    return wb


def _run(wb, budget=None):
    return asyncio.run(wb._select_gated_card(
        "Mi Band 8", None, None, [], "Mi Band 8", budget or _GateBudget()))


def test_first_same_accepted():
    g = FakeGate({"Mi Band 8 черный": [V.SAME]})
    r = _run(_wb([(7, {}, "Mi Band 8 черный", 80.0), (6, {}, "Mi Band 6", 76.0)], g))
    assert r[0] == 7 and r[4] == "brand_line" and len(g.calls) == 1


def test_different_then_repick_same():
    g = FakeGate({"Mi Band 7": [V.DIFFERENT], "Mi Band 8 черный": [V.SAME]})
    r = _run(_wb([(7, {}, "Mi Band 7", 78.0), (8, {}, "Mi Band 8 черный", 80.0)], g))
    assert r[0] == 8 and g.calls == ["Mi Band 7", "Mi Band 8 черный"]


def test_sentinel_all_different_abstains():
    g = FakeGate({"Mi Band 7": [V.DIFFERENT], "Mi Band 6": [V.DIFFERENT]})
    r = _run(_wb([(7, {}, "Mi Band 7", 78.2), (6, {}, "Mi Band 6", 76.0)], g))
    assert r is None and g.calls == ["Mi Band 7", "Mi Band 6"]


def test_unknown_then_retry_same():
    g = FakeGate({"Mi Band X": [V.UNKNOWN, V.SAME]})
    b = _GateBudget()
    r = _run(_wb([(7, {}, "Mi Band X", 80.0)], g), b)
    assert r[0] == 7 and len(g.calls) == 2 and b.used == 2


def test_double_unknown_drops_then_next():
    g = FakeGate({"A": [V.UNKNOWN, V.UNKNOWN], "B": [V.SAME]})
    r = _run(_wb([(7, {}, "A", 80.0), (8, {}, "B", 79.0)], g))
    assert r[0] == 8 and g.calls == ["A", "A", "B"]


def test_budget_cap_stops():
    g = FakeGate({f"C{i}": [V.DIFFERENT] for i in range(5)})
    b = _GateBudget(3)
    r = _run(_wb([(i, {}, f"C{i}", 80.0) for i in range(5)], g), b)
    assert r is None and b.used == 3 and len(g.calls) == 3


def test_clean_accepted_no_llm():
    g = FakeGate({})
    r = _run(_wb([(7, {}, "CleanExact", 90.0)], g, clean={"CleanExact"}))
    assert r[0] == 7 and g.calls == []


def test_skip_low_score_candidate():
    g = FakeGate({"Good": [V.SAME]})
    r = _run(_wb([(7, {}, "Low", 50.0), (8, {}, "Good", 80.0)], g))
    assert r[0] == 8 and g.calls == ["Good"]
