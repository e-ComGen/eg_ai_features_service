"""B-I1..B-I7: unit tests for _track_a_corroboration_filter (Finishing re-gate).

Spec: docs/MANIFEST_finishing_gate_option_b.md
All tests are deterministic, no LLM calls, no network, no DB.
"""
from app.services.enrichment.pipeline import PipelineOrchestrator, _is_objective_spec_attr
from app.services.enrichment.base import AttributeValue, TargetAttribute, Source


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric_target(attr_id: int = 5379) -> TargetAttribute:
    """Numeric target — _is_objective_spec_attr returns True (type='numeric')."""
    return TargetAttribute(id=attr_id, name="Срок годности", type="numeric")


def _text_target(attr_id: int = 9999) -> TargetAttribute:
    """Free-text target — _is_objective_spec_attr returns False."""
    return TargetAttribute(id=attr_id, name="Описание", type="text")


def _av(
    attr_id: int,
    value: object,
    source: Source,
    evidence: str | None = None,
    confidence: float = 0.9,
) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=confidence,
        source=source,
        evidence=evidence,
    )


def _run_filter(
    all_values: list[AttributeValue],
    targets: list[TargetAttribute],
) -> list[AttributeValue]:
    """Call the method without triggering PipelineOrchestrator.__init__."""
    pipeline = object.__new__(PipelineOrchestrator)
    return pipeline._track_a_corroboration_filter(all_values, targets)


# ---------------------------------------------------------------------------
# Precondition: verify _is_objective_spec_attr behaves as expected for targets used below
# ---------------------------------------------------------------------------

def test_precondition_numeric_target_is_objective_spec():
    """Precondition: TargetAttribute(type='numeric') must be classified as objective-spec."""
    t = _numeric_target()
    assert _is_objective_spec_attr(t) is True, (
        "Precondition FAIL: numeric target should be objective-spec"
    )


def test_precondition_text_target_not_objective_spec():
    """Precondition: TargetAttribute(type='text') must NOT be objective-spec."""
    t = _text_target()
    assert _is_objective_spec_attr(t) is False, (
        "Precondition FAIL: text target should not be objective-spec"
    )


# ---------------------------------------------------------------------------
# B-I1: uncorroborated objective-spec from guess sources → DROPPED
# ---------------------------------------------------------------------------

def test_b_i1_web_search_uncorroborated_objective_spec_dropped():
    """B-I1: web_search fill for numeric attr without authoritative corroboration is dropped."""
    target = _numeric_target(5379)
    v = _av(5379, 1095, Source.WEB_SEARCH)
    result = _run_filter([v], [target])
    assert not any(r.attribute_id == 5379 for r in result), (
        "B-I1 FAIL: uncorroborated web_search objective-spec fill should be absent"
    )


def test_b_i1_llm_knowledge_uncorroborated_objective_spec_dropped():
    """B-I1: llm_knowledge fill for numeric attr without authoritative corroboration is dropped."""
    target = _numeric_target(5379)
    v = _av(5379, 1095, Source.LLM_KNOWLEDGE)
    result = _run_filter([v], [target])
    assert not any(r.attribute_id == 5379 for r in result), (
        "B-I1 FAIL: uncorroborated llm_knowledge objective-spec fill should be absent"
    )


def test_b_i1_competitor_rag_uncorroborated_objective_spec_dropped():
    """B-I1: competitor_rag fill for numeric attr without authoritative corroboration is dropped."""
    target = _numeric_target(5379)
    v = _av(5379, 1095, Source.COMPETITOR_RAG)
    result = _run_filter([v], [target])
    assert not any(r.attribute_id == 5379 for r in result), (
        "B-I1 FAIL: uncorroborated competitor_rag objective-spec fill should be absent"
    )


# ---------------------------------------------------------------------------
# B-I2: corroborated by authoritative source → KEPT
# ---------------------------------------------------------------------------

def test_b_i2_corroborated_by_icecat_kept():
    """B-I2: web_search fill corroborated by icecat (same attr+value) is preserved."""
    target = _numeric_target(5379)
    guess = _av(5379, 1095, Source.WEB_SEARCH)
    authoritative = _av(5379, 1095, Source.ICECAT)
    result = _run_filter([guess, authoritative], [target])
    attr_ids = [r.attribute_id for r in result]
    assert attr_ids.count(5379) == 2, (
        "B-I2 FAIL: both the guess fill and authoritative fill should be preserved"
    )


def test_b_i2_corroborated_by_wb_card_kept():
    """B-I2: llm_knowledge fill corroborated by wb_card is preserved."""
    target = _numeric_target(5379)
    guess = _av(5379, 548, Source.LLM_KNOWLEDGE)
    authoritative = _av(5379, 548, Source.WB_CARD)
    result = _run_filter([guess, authoritative], [target])
    assert len([r for r in result if r.attribute_id == 5379]) == 2, (
        "B-I2 FAIL: wb_card-corroborated fill should be kept"
    )


# ---------------------------------------------------------------------------
# B-I3: non-spec (Track B) fills are NOT dropped
# ---------------------------------------------------------------------------

def test_b_i3_non_spec_llm_knowledge_not_dropped():
    """B-I3: llm_knowledge fill for text (non-spec) attr is never dropped by re-gate."""
    target = _text_target(9999)
    v = _av(9999, "синий", Source.LLM_KNOWLEDGE)
    result = _run_filter([v], [target])
    assert any(r.attribute_id == 9999 for r in result), (
        "B-I3 FAIL: non-spec fill must not be dropped by Track-A re-gate"
    )


def test_b_i3_non_spec_web_search_not_dropped():
    """B-I3: web_search fill for text (non-spec) attr is never dropped."""
    target = _text_target(9999)
    v = _av(9999, "Зелёный", Source.WEB_SEARCH)
    result = _run_filter([v], [target])
    assert any(r.attribute_id == 9999 for r in result), (
        "B-I3 FAIL: non-spec web_search fill must not be dropped"
    )


# ---------------------------------------------------------------------------
# B-I4: authoritative fills and verbatim-anchored fills pass through untouched
# ---------------------------------------------------------------------------

def test_b_i4_icecat_authoritative_not_dropped():
    """B-I4: icecat fill passes through regardless of corroboration status."""
    target = _numeric_target(5379)
    v = _av(5379, 548, Source.ICECAT)
    result = _run_filter([v], [target])
    assert any(r.attribute_id == 5379 for r in result), (
        "B-I4 FAIL: authoritative icecat fill must not be dropped"
    )


def test_b_i4_description_authoritative_not_dropped():
    """B-I4: description fill (authoritative) passes through."""
    target = _numeric_target(5379)
    v = _av(5379, 548, Source.DESCRIPTION)
    result = _run_filter([v], [target])
    assert any(r.attribute_id == 5379 for r in result), (
        "B-I4 FAIL: description fill must not be dropped"
    )


def test_b_i4_verbatim_anchored_llm_not_dropped():
    """B-I4: llm_knowledge fill with safe_enum:verbatim_gate evidence is kept."""
    target = _numeric_target(5379)
    v = _av(5379, 1095, Source.LLM_KNOWLEDGE, evidence="safe_enum:verbatim_gate|matched")
    result = _run_filter([v], [target])
    assert any(r.attribute_id == 5379 for r in result), (
        "B-I4 FAIL: verbatim-anchored fill must not be dropped"
    )


def test_b_i4_verbatim_anchored_web_search_not_dropped():
    """B-I4: web_search fill with safe_enum:verbatim_gate evidence is kept."""
    target = _numeric_target(5379)
    v = _av(5379, 1095, Source.WEB_SEARCH, evidence="safe_enum:verbatim_gate|text from page")
    result = _run_filter([v], [target])
    assert any(r.attribute_id == 5379 for r in result), (
        "B-I4 FAIL: verbatim-anchored web_search fill must not be dropped"
    )


# ---------------------------------------------------------------------------
# B-I7: idempotency
# ---------------------------------------------------------------------------

def test_b_i7_idempotent_on_filtered_output():
    """B-I7: running the filter twice produces the same result as running it once."""
    target = _numeric_target(5379)
    auth = _av(5379, 548, Source.ICECAT)
    good_guess = _av(5379, 548, Source.WEB_SEARCH)   # corroborated — kept
    bad_guess = _av(5379, 1095, Source.WEB_SEARCH)   # uncorroborated — dropped
    non_spec = _av(9999, "синий", Source.LLM_KNOWLEDGE)

    targets = [target, _text_target(9999)]
    first_pass = _run_filter([auth, good_guess, bad_guess, non_spec], targets)
    second_pass = _run_filter(first_pass, targets)

    assert len(first_pass) == len(second_pass), (
        "B-I7 FAIL: second filter pass changed the result set (not idempotent)"
    )
    first_ids = sorted(r.attribute_id for r in first_pass)
    second_ids = sorted(r.attribute_id for r in second_pass)
    assert first_ids == second_ids, "B-I7 FAIL: attr_id sets differ between passes"


# ---------------------------------------------------------------------------
# MUTATION self-check sentinel
# Used by the mutation script to verify the drop logic is load-bearing.
# When objective-spec drop is removed (patched to always append), this MUST fail.
# ---------------------------------------------------------------------------

def test_mutation_sentinel_b_i1():
    """Sentinel for mutation self-check.

    If the drop condition in _track_a_corroboration_filter is removed
    (mutation: always append regardless of corroboration), this test FAILS —
    proving the test catches the defect.
    """
    target = _numeric_target(5379)
    v = _av(5379, 1095, Source.WEB_SEARCH)
    result = _run_filter([v], [target])
    assert not any(r.attribute_id == 5379 for r in result), (
        "MUTATION DETECTED: drop logic is missing — 1095 should not survive re-gate"
    )
