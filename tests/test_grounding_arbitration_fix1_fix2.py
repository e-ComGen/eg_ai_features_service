"""Arch/property tests for grounding-arbitration Round 1 (FIX-1 + FIX-2).

Spec: docs/MANIFEST_grounding_arbitration.md
Covers INV-1 (authority-over-confidence), INV-5 (regression / single-source
correctness preserved), INV-6 (collections still union). INV-2 (last-resort
abstain) and the O1-O3/R1-R4 oracle live in
tests/test_grounding_arbitration_oracle.py (functional, authored by a
different model family per the no-self-bias rule).

These are PROPERTY tests: they assert the invariant holds across many
source/confidence combinations, not just one worked example.
"""
from itertools import combinations

import pytest

from app.services.enrichment.base import AttributeValue, Source, SOURCE_PRIORITY
from app.services.enrichment.pipeline import PipelineOrchestrator, _merge_winner


def _av(attribute_id: int, value, confidence: float, source: Source, evidence=None,
        is_collection: bool = False) -> AttributeValue:
    return AttributeValue(
        attribute_id=attribute_id,
        value=value,
        confidence=confidence,
        source=source,
        evidence=evidence,
        is_collection=is_collection,
    )


# ---------------------------------------------------------------------------
# INV-1 — authority-over-confidence: higher SOURCE_PRIORITY ALWAYS wins,
# regardless of confidence. Confidence changes NO merge outcome.
# ---------------------------------------------------------------------------

ALL_SOURCES = list(Source)


@pytest.mark.parametrize("hi_src,lo_src", [
    (hi, lo)
    for hi, lo in combinations(ALL_SOURCES, 2)
    if SOURCE_PRIORITY[hi] != SOURCE_PRIORITY[lo]
])
def test_inv1_higher_priority_always_wins_regardless_of_confidence(hi_src, lo_src):
    """For every pair of sources with different SOURCE_PRIORITY, the
    higher-priority one wins the merge no matter which side has higher
    self-reported confidence."""
    if SOURCE_PRIORITY[hi_src] < SOURCE_PRIORITY[lo_src]:
        hi_src, lo_src = lo_src, hi_src  # normalize: hi_src really is higher

    # Case A: priority-winner has LOW confidence, priority-loser has HIGH confidence.
    winner_should_be = _av(1, "winner_value", confidence=0.10, source=hi_src, evidence="x")
    loser = _av(1, "loser_value", confidence=0.99, source=lo_src, evidence="a much longer evidence string here")
    assert _merge_winner(winner_should_be, loser).source == hi_src
    assert _merge_winner(loser, winner_should_be).source == hi_src  # order-independent

    # Case B: flip confidences the other way — outcome must NOT change.
    winner_should_be2 = _av(1, "winner_value", confidence=0.99, source=hi_src, evidence="x")
    loser2 = _av(1, "loser_value", confidence=0.10, source=lo_src, evidence="a much longer evidence string here")
    assert _merge_winner(winner_should_be2, loser2).source == hi_src
    assert _merge_winner(loser2, winner_should_be2).source == hi_src


def test_inv1_confidence_never_changes_outcome_within_same_priority_class():
    """Two values from the SAME priority class: swapping confidence values
    around must not change which one wins — only evidence length (the
    declared tiebreak) may."""
    a = _av(1, "a", confidence=0.20, source=Source.WEB_SEARCH, evidence="short")
    b = _av(1, "b", confidence=0.95, source=Source.COMPETITOR_RAG, evidence="short")
    assert SOURCE_PRIORITY[a.source] == SOURCE_PRIORITY[b.source]

    winner_low_conf_a = _merge_winner(a, b)
    a_hi = a.model_copy(update={"confidence": 0.95})
    b_lo = b.model_copy(update={"confidence": 0.20})
    winner_swapped = _merge_winner(a_hi, b_lo)

    # Same evidence lengths on both sides in both runs -> same tiebreak result
    # (incumbent b) regardless of which side now holds the high confidence.
    assert winner_low_conf_a.source == winner_swapped.source == b.source


def test_inv1_tiebreak_within_same_priority_is_evidence_length_not_confidence():
    """Same priority class, different evidence lengths, LOWER-evidence side
    has higher confidence -> the LONGER-evidence side still wins."""
    short_ev_high_conf = _av(1, "x", confidence=0.99, source=Source.VISION, evidence="hi")
    long_ev_low_conf = _av(1, "y", confidence=0.05, source=Source.ICECAT,
                            evidence="a substantially longer piece of evidence text")
    assert SOURCE_PRIORITY[short_ev_high_conf.source] == SOURCE_PRIORITY[long_ev_low_conf.source]
    result = _merge_winner(short_ev_high_conf, long_ev_low_conf)
    assert result.value == "y"


def test_inv1_equal_priority_equal_evidence_keeps_incumbent_stable():
    challenger = _av(1, "c", confidence=0.99, source=Source.DESCRIPTION, evidence="")
    incumbent = _av(1, "i", confidence=0.01, source=Source.OZON_CARD, evidence="")
    assert SOURCE_PRIORITY[challenger.source] == SOURCE_PRIORITY[incumbent.source]
    result = _merge_winner(challenger, incumbent)
    assert result.value == "i"  # incumbent kept, confidence (0.01 vs 0.99) irrelevant


# ---------------------------------------------------------------------------
# INV-5 — regression: an attribute filled by exactly one EVIDENCED source is
# unchanged (existing correct single-select fills stay byte-identical).
# ---------------------------------------------------------------------------

def test_inv5_single_evidenced_nonguess_source_passthrough_unchanged():
    orchestrator = PipelineOrchestrator()
    only_value = _av(
        42, "44800", confidence=0.6, source=Source.WEB_SEARCH,
        evidence="official spec sheet lists 44800 impacts/min",
    )
    result = orchestrator._merge([only_value])
    assert len(result) == 1
    assert result[0].value == "44800"
    assert result[0].source == Source.WEB_SEARCH
    assert result[0].confidence == 0.6  # untouched, not rewritten


def test_inv5_single_scalar_llm_sole_fill_passthrough_unchanged():
    """A lone SCALAR LLM_KNOWLEDGE value is never an abstain candidate under the
    NARROW FIX-2 rule (abstain only fires for a sole-proposer is_collection LLM
    value). Scalar sole-fills are kept for coverage, evidence-content irrelevant."""
    orchestrator = PipelineOrchestrator()
    only_value = _av(
        7, "44800", confidence=0.9, source=Source.LLM_KNOWLEDGE,
        evidence="manufacturer datasheet quote: model XYZ123 rated at 44800 bpm",
    )
    result = orchestrator._merge([only_value])
    assert len(result) == 1
    assert result[0].value == "44800"
    assert result[0].source == Source.LLM_KNOWLEDGE


# ---------------------------------------------------------------------------
# INV-6 — collections still union across proposing sources (FIX-2 only stops
# a GUESS-source from being the sole restrictive proposer; it must not break
# the union behavior for non-guess or corroborated/evidenced proposers).
# ---------------------------------------------------------------------------

def test_inv6_collection_union_across_two_nonguess_sources():
    orchestrator = PipelineOrchestrator()
    a = _av(9, ["Сверление"], confidence=0.7, source=Source.DESCRIPTION,
            evidence="как указано в описании", is_collection=True)
    b = _av(9, ["Сверление с ударом"], confidence=0.7, source=Source.WEB_SEARCH,
            evidence="бетон Ø=16 confirms hammer mode", is_collection=True)
    result = orchestrator._merge([a, b])
    assert len(result) == 1
    values = {str(v).strip().lower() for v in result[0].value}
    assert values == {"сверление", "сверление с ударом"}


def test_inv6_collection_union_survives_when_llm_is_not_sole_proposer():
    """A card + a LLM_KNOWLEDGE collection value still union: the LLM value is
    NOT the sole proposer of the attribute (the card co-proposes it), so the
    narrow abstain rule does not fire — evidence content is irrelevant."""
    orchestrator = PipelineOrchestrator()
    card = _av(11, ["Красный"], confidence=0.9, source=Source.OZON_CARD,
               evidence="карточка Ozon", is_collection=True)
    guess = _av(11, ["Синий"], confidence=0.9, source=Source.LLM_KNOWLEDGE,
                evidence="каталожная спецификация указывает цвет для модели ABC456 — синий",
                is_collection=True)
    result = orchestrator._merge([card, guess])
    assert len(result) == 1
    values = {str(v).strip().lower() for v in result[0].value}
    assert values == {"красный", "синий"}


# ---------------------------------------------------------------------------
# FIX-2 NARROW-RULE regression (coordinator-mandated): the abstain must NOT
# over-reach beyond a sole-proposer is_collection LLM_KNOWLEDGE value.
# Both tests FAIL against the OLD broad rule (which abstained SAFE_ENUM_FILL and
# evidence-less scalar LLM guesses) and PASS under the narrow rule.
# ---------------------------------------------------------------------------

def test_fix2_safe_enum_fill_sole_collection_kept():
    """SAFE_ENUM_FILL is EXCLUDED from abstain: a lone, evidence-less
    SAFE_ENUM_FILL value for an is_collection attribute is KEPT (its own
    Gate A/B handle grounding — it is the legit type-default mechanism, e.g.
    Сезон/Стиль). The OLD broad rule wrongly dropped it."""
    orchestrator = PipelineOrchestrator()
    only_value = _av(
        21, ["Демисезон"], confidence=0.82, source=Source.SAFE_ENUM_FILL,
        evidence=None, is_collection=True,
    )
    result = orchestrator._merge([only_value])
    assert len(result) == 1
    assert result[0].source == Source.SAFE_ENUM_FILL
    assert [str(x).strip().lower() for x in result[0].value] == ["демисезон"]


def test_fix2_scalar_llm_evidenceless_sole_kept():
    """A lone SCALAR LLM_KNOWLEDGE value with NO evidence is KEPT (coverage):
    the narrow abstain fires only for is_collection sole-proposer LLM values.
    The OLD broad rule wrongly dropped this (evidence-less scalar guess)."""
    orchestrator = PipelineOrchestrator()
    only_value = _av(
        22, "Пластик", confidence=0.9, source=Source.LLM_KNOWLEDGE,
        evidence=None, is_collection=False,
    )
    result = orchestrator._merge([only_value])
    assert len(result) == 1
    assert result[0].value == "Пластик"
    assert result[0].source == Source.LLM_KNOWLEDGE
