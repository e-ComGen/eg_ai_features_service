"""Unit tests for AttributeMerger."""

import pytest
from app.services.enrichment.attribute_merger import AttributeMerger, AttributeValue, Source


def _av(attr_id: str, value, confidence: float, source: Source) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=confidence,
        source=source,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_merge_picks_highest_confidence():
    """When two branches disagree on a value, the higher confidence wins."""
    merger = AttributeMerger()
    branch_a = [_av("colour", "red",  confidence=0.9, source=Source.DESCRIPTION)]
    branch_b = [_av("colour", "blue", confidence=0.6, source=Source.VISION)]

    result = merger.merge([branch_a, branch_b])

    assert len(result) == 1
    assert result[0].value == "red"
    assert result[0].confidence == 0.9


def test_merge_equal_confidence_uses_source_priority():
    """On equal confidence, DESCRIPTION beats WEB_SEARCH beats VISION beats LLM_KNOWLEDGE."""
    merger = AttributeMerger()

    # VISION vs WEB_SEARCH at same confidence → WEB_SEARCH wins.
    b1 = [_av("weight", "500g", confidence=0.7, source=Source.VISION)]
    b2 = [_av("weight", "510g", confidence=0.7, source=Source.WEB_SEARCH)]
    result = merger.merge([b1, b2])
    assert result[0].source == Source.WEB_SEARCH

    # WEB_SEARCH vs DESCRIPTION at same confidence → DESCRIPTION wins.
    b3 = [_av("size", "L",    confidence=0.8, source=Source.WEB_SEARCH)]
    b4 = [_av("size", "Large", confidence=0.8, source=Source.DESCRIPTION)]
    result2 = merger.merge([b3, b4])
    assert result2[0].source == Source.DESCRIPTION

    # LLM_KNOWLEDGE vs everything else at same confidence → others win.
    b5 = [_av("brand", "ACME",  confidence=0.5, source=Source.LLM_KNOWLEDGE)]
    b6 = [_av("brand", "Brand", confidence=0.5, source=Source.VISION)]
    result3 = merger.merge([b5, b6])
    assert result3[0].source == Source.VISION


def test_merge_empty_branches_returns_empty():
    """All-empty input yields an empty list."""
    merger = AttributeMerger()
    assert merger.merge([[], [], []]) == []


def test_merge_single_branch():
    """Single branch → all its attrs returned as-is."""
    merger = AttributeMerger()
    branch = [
        _av("colour", "green", confidence=0.8, source=Source.DESCRIPTION),
        _av("weight", "1kg",   confidence=0.7, source=Source.DESCRIPTION),
    ]
    result = merger.merge([branch])
    attr_ids = {a.attribute_id for a in result}
    assert attr_ids == {"colour", "weight"}


def test_merge_no_branches_returns_empty():
    """Calling merge with no branches returns empty list."""
    merger = AttributeMerger()
    assert merger.merge([]) == []


def test_merge_different_attributes_from_different_branches():
    """Non-overlapping attributes from all branches are all included."""
    merger = AttributeMerger()
    b1 = [_av("colour", "red",  confidence=0.9, source=Source.DESCRIPTION)]
    b2 = [_av("weight", "200g", confidence=0.7, source=Source.VISION)]
    b3 = [_av("brand",  "Foo",  confidence=0.8, source=Source.WEB_SEARCH)]

    result = merger.merge([b1, b2, b3])
    attr_ids = {a.attribute_id for a in result}
    assert attr_ids == {"colour", "weight", "brand"}


def test_merge_source_priority_full_order():
    """Full priority chain: DESCRIPTION > WEB_SEARCH > VISION > LLM_KNOWLEDGE."""
    merger = AttributeMerger()
    conf = 0.5  # same for all
    branches = [
        [_av("x", "llm",  conf, Source.LLM_KNOWLEDGE)],
        [_av("x", "vis",  conf, Source.VISION)],
        [_av("x", "web",  conf, Source.WEB_SEARCH)],
        [_av("x", "desc", conf, Source.DESCRIPTION)],
    ]
    result = merger.merge(branches)
    assert result[0].value == "desc"
    assert result[0].source == Source.DESCRIPTION


def test_merge_one_branch_all_zeros():
    """Branch with confidence 0 still contributes if no other branch covers the attr."""
    merger = AttributeMerger()
    branch = [_av("material", "plastic", confidence=0.0, source=Source.VISION)]
    result = merger.merge([branch])
    assert len(result) == 1
    assert result[0].value == "plastic"
