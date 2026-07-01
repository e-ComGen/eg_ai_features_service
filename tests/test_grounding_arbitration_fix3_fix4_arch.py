"""Arch/property tests for grounding-arbitration Round 2 (FIX-3 + FIX-4).

Spec: docs/MANIFEST_grounding_arbitration.md
Covers INV-3 (class-grounding: COUNTRY survives only with cross-host
corroboration or model-adjacency) and INV-4 (no surviving contradiction:
concrete-Ø vs Режимы работы; Комплектация-named chuck vs Тип патрона).

These are PROPERTY tests exercising the unit-level functions directly
(_drop_ungrounded_hard_facts, _reconcile_cross_field_contradictions) with
synthetic AttributeValue/TargetAttribute inputs -- no live enrichment.
The O2/O2b/O3/O1b worked-example oracle lives in
tests/test_grounding_arbitration_fix3_fix4_oracle.py (functional, authored
by a different model family per the no-self-bias rule).
"""
import pytest

from app.services.enrichment.base import AttributeValue, Source, TargetAttribute
from app.services.enrichment.pipeline import (
    _drop_ungrounded_hard_facts,
    _reconcile_cross_field_contradictions,
)


def _av(attribute_id, value, confidence, source, evidence=None, is_collection=False):
    return AttributeValue(
        attribute_id=attribute_id,
        value=value,
        confidence=confidence,
        source=source,
        evidence=evidence,
        is_collection=is_collection,
    )


def _target(id, name, type="text", is_collection=False):
    return TargetAttribute(id=id, name=name, type=type, is_collection=is_collection)


COUNTRY_ID = 900
COUNTRY_TARGETS = [_target(COUNTRY_ID, "Страна-изготовитель")]


# ---------------------------------------------------------------------------
# INV-3 -- COUNTRY-class web_search survives ONLY with cross-host
# corroboration OR model-adjacency; a lone generic snippet -> abstain (empty).
# ---------------------------------------------------------------------------

def test_inv3_lone_generic_websearch_snippet_abstains():
    """A single web_search COUNTRY fill with a generic brand-only snippet
    (no second host, no model mention) must be dropped."""
    values = [
        _av(COUNTRY_ID, "Япония", 0.7, Source.WEB_SEARCH,
            evidence="[https://ru.wikipedia.org/wiki/Makita] Makita — японская компания, "
                      "производитель электроинструмента."),
    ]
    result = _drop_ungrounded_hard_facts(values, "", COUNTRY_TARGETS, product_model_text="")
    assert not any(v.attribute_id == COUNTRY_ID for v in result)


@pytest.mark.parametrize("n_hosts", [0, 1])
def test_inv3_fewer_than_two_hosts_abstains(n_hosts):
    """Property: with < 2 distinct real hosts and no model-adjacency, the
    fill never survives, regardless of how many same-host duplicates exist."""
    values = []
    for i in range(3):  # duplicate the SAME host 3x -- still only 1 distinct host
        host = "https://site-a.example.com/page" if n_hosts >= 1 else None
        evidence = f"[{host}] Makita — японская компания." if host else "Makita — японская компания."
        values.append(_av(COUNTRY_ID, "Япония", 0.7, Source.WEB_SEARCH, evidence=evidence))
    result = _drop_ungrounded_hard_facts(values, "", COUNTRY_TARGETS, product_model_text="")
    assert not any(v.attribute_id == COUNTRY_ID for v in result)


def test_inv3_two_distinct_hosts_corroborated_survives():
    """Property: >= 2 DISTINCT real hosts asserting the SAME normalized
    country value -> the fill is kept (INV-3 doesn't over-abstain)."""
    values = [
        _av(COUNTRY_ID, "Китай", 0.7, Source.WEB_SEARCH,
            evidence="[https://site-a.example.com/p1] Страна производства: Китай."),
        _av(COUNTRY_ID, "Китай", 0.6, Source.WEB_SEARCH,
            evidence="[https://site-b.example.ru/p2] Произведено в Китае, завод в Шэньчжэне."),
    ]
    result = _drop_ungrounded_hard_facts(values, "", COUNTRY_TARGETS, product_model_text="")
    kept = [v for v in result if v.attribute_id == COUNTRY_ID]
    assert len(kept) == 2  # both corroborating fills survive


def test_inv3_model_adjacency_survives_without_corroboration():
    """Property: a SINGLE web_search fill survives if the product's own
    model designator (NOT the brand) co-occurs with the country claim in
    the same evidence snippet -- even with only one host."""
    values = [
        _av(COUNTRY_ID, "Япония", 0.7, Source.WEB_SEARCH,
            evidence="[https://shop.example.com/hp1630] Makita HP1630 официально "
                      "произведена в Японии на заводе Anjo."),
    ]
    result = _drop_ungrounded_hard_facts(
        values, "", COUNTRY_TARGETS, product_model_text="HP1630 Дрель-шуруповёрт",
    )
    kept = [v for v in result if v.attribute_id == COUNTRY_ID]
    assert len(kept) == 1
    assert kept[0].value == "Япония"


def test_inv3_brand_only_mention_is_not_model_adjacency():
    """Property: the BRAND appearing in the evidence (trivially true for any
    'X is a Japanese company' snippet) must NOT count as model-adjacency --
    only the product's own MODEL token counts. This is the exact bug FIX-3
    closes (brand-home-country leak)."""
    values = [
        _av(COUNTRY_ID, "Япония", 0.7, Source.WEB_SEARCH,
            evidence="[https://ru.wikipedia.org/wiki/Makita] Makita — японская компания."),
    ]
    # product_model_text is the model designator (brand already stripped by the caller,
    # see _strip_leading_brand) -- "Makita" itself would never appear here.
    result = _drop_ungrounded_hard_facts(
        values, "", COUNTRY_TARGETS, product_model_text="HP1630 Дрель-шуруповёрт",
    )
    assert not any(v.attribute_id == COUNTRY_ID for v in result)


def test_inv3_other_trusted_retrieval_sources_still_exempt():
    """Regression: FIX-3 narrows ONLY the web_search COUNTRY exemption.
    Card/IceCat/vision COUNTRY fills stay fully exempt (real marketplace
    data), unaffected by cross-host/model-adjacency requirements."""
    values = [
        _av(COUNTRY_ID, "Вьетнам", 0.9, Source.OZON_CARD, evidence=None),
    ]
    result = _drop_ungrounded_hard_facts(values, "", COUNTRY_TARGETS, product_model_text="")
    kept = [v for v in result if v.attribute_id == COUNTRY_ID]
    assert len(kept) == 1
    assert kept[0].value == "Вьетнам"


def test_inv3_llm_knowledge_country_still_always_dropped():
    """Regression: pre-existing behavior unchanged -- LLM_KNOWLEDGE COUNTRY
    fills are always dropped (FIX-3 only touches the web_search branch)."""
    values = [
        _av(COUNTRY_ID, "Германия", 0.8, Source.LLM_KNOWLEDGE, evidence="likely German engineering"),
    ]
    result = _drop_ungrounded_hard_facts(values, "", COUNTRY_TARGETS, product_model_text="")
    assert not any(v.attribute_id == COUNTRY_ID for v in result)


# ---------------------------------------------------------------------------
# INV-4 -- no surviving contradiction after finalize.
# ---------------------------------------------------------------------------

BORE_ID, MODES_ID, KOMP_ID, CHUCK_ID = 910, 911, 912, 913
DRILL_TARGETS = [
    _target(BORE_ID, "Макс. диаметр отверстия в бетоне, мм", type="numeric"),
    _target(MODES_ID, "Режимы работы", is_collection=True),
]
CHUCK_TARGETS = [
    _target(KOMP_ID, "Комплектация"),
    _target(CHUCK_ID, "Тип патрона"),
]


def test_inv4_concrete_bore_real_source_adds_impact_mode():
    merged = [
        _av(BORE_ID, "16", 0.8, Source.WEB_SEARCH, evidence="concrete bore up to 16mm"),
        _av(MODES_ID, ["Сверление"], 0.7, Source.LLM_KNOWLEDGE, evidence="drilling"),
    ]
    result = _reconcile_cross_field_contradictions(merged, DRILL_TARGETS)
    modes = next(v for v in result if v.attribute_id == MODES_ID)
    lower_vals = [x.lower() for x in modes.value]
    assert any("удар" in x for x in lower_vals)
    assert "сверление" in lower_vals  # original mode preserved


def test_inv4_concrete_bore_llm_guess_drops_inconsistent_modes():
    """When the concrete-Ø itself is a pure LLM_KNOWLEDGE guess (no real
    evidence backing it), it is too weak to justify correcting Режимы
    работы -- the inconsistent partial is dropped instead."""
    merged = [
        _av(BORE_ID, "16", 0.5, Source.LLM_KNOWLEDGE, evidence="probably drills concrete"),
        _av(MODES_ID, ["Сверление"], 0.7, Source.WEB_SEARCH, evidence="drilling only"),
    ]
    result = _reconcile_cross_field_contradictions(merged, DRILL_TARGETS)
    assert not any(v.attribute_id == MODES_ID for v in result)


def test_inv4_no_bore_present_leaves_modes_unchanged():
    merged = [
        _av(MODES_ID, ["Сверление"], 0.7, Source.WEB_SEARCH, evidence="drilling only"),
    ]
    result = _reconcile_cross_field_contradictions(merged, DRILL_TARGETS)
    modes = next(v for v in result if v.attribute_id == MODES_ID)
    assert modes.value == ["Сверление"]


def test_inv4_modes_already_has_impact_mode_is_idempotent():
    merged = [
        _av(BORE_ID, "16", 0.8, Source.WEB_SEARCH, evidence="concrete bore up to 16mm"),
        _av(MODES_ID, ["Сверление", "Сверление с ударом"], 0.7, Source.WEB_SEARCH, evidence="both modes"),
    ]
    result = _reconcile_cross_field_contradictions(merged, DRILL_TARGETS)
    modes = next(v for v in result if v.attribute_id == MODES_ID)
    assert modes.value == ["Сверление", "Сверление с ударом"]  # untouched


def test_inv4_komplektatsiya_corrects_conflicting_chuck_type():
    merged = [
        _av(KOMP_ID, "Аккумулятор, зарядное устройство, ключевой патрон, кейс",
            0.9, Source.DESCRIPTION, evidence="verbatim from description"),
        _av(CHUCK_ID, "Быстрозажимной", 0.6, Source.SAFE_ENUM_FILL, evidence="enum default"),
    ]
    result = _reconcile_cross_field_contradictions(merged, CHUCK_TARGETS)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert "ключев" in chuck.value.lower()


def test_inv4_komplektatsiya_matching_chuck_type_is_idempotent():
    merged = [
        _av(KOMP_ID, "ключевой патрон, кейс", 0.9, Source.DESCRIPTION, evidence="x"),
        _av(CHUCK_ID, "Ключевой", 0.6, Source.SAFE_ENUM_FILL, evidence="x"),
    ]
    result = _reconcile_cross_field_contradictions(merged, CHUCK_TARGETS)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert chuck.value == "Ключевой"  # untouched, no spurious re-write


def test_inv4_komplektatsiya_does_not_override_higher_priority_chuck_type():
    """Gate-2 fix: a LOW-priority Комплектация claim (LLM_KNOWLEDGE, prio 1)
    must NOT override a HIGH-priority Тип патрона (OZON_CARD, prio 4) --
    the stronger-grounded field wins (manifest architecture rule), backwards
    from the previous unconditional-override bug caught by cross-family review."""
    merged = [
        _av(KOMP_ID, "ключевой патрон, кейс", 0.5, Source.LLM_KNOWLEDGE, evidence="guess"),
        _av(CHUCK_ID, "Быстрозажимной", 0.9, Source.OZON_CARD, evidence="real card data"),
    ]
    result = _reconcile_cross_field_contradictions(merged, CHUCK_TARGETS)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert chuck.value == "Быстрозажимной"  # untouched -- card outranks the LLM guess


def test_inv4_komplektatsiya_ambiguous_mention_skips_correction():
    """Both chuck types named in free text -> ambiguous, never guess."""
    merged = [
        _av(KOMP_ID, "быстрозажимной патрон и запасной ключевой патрон", 0.9,
            Source.DESCRIPTION, evidence="x"),
        _av(CHUCK_ID, "Быстрозажимной", 0.6, Source.SAFE_ENUM_FILL, evidence="x"),
    ]
    result = _reconcile_cross_field_contradictions(merged, CHUCK_TARGETS)
    chuck = next(v for v in result if v.attribute_id == CHUCK_ID)
    assert chuck.value == "Быстрозажимной"  # left as-is, ambiguous case not touched


@pytest.mark.parametrize("bore_source", [Source.WEB_SEARCH, Source.OZON_CARD, Source.ICECAT])
def test_inv4_property_never_leaves_bore_present_without_impact_mode(bore_source):
    """Property (INV-4 core): for ANY non-guess bore source, after
    reconciliation, Режимы работы is NEVER left present-but-missing an
    удар mode alongside a present concrete-Ø."""
    merged = [
        _av(BORE_ID, "13", 0.7, bore_source, evidence="handles concrete drilling"),
        _av(MODES_ID, ["Сверление"], 0.6, Source.LLM_KNOWLEDGE, evidence="guess"),
    ]
    result = _reconcile_cross_field_contradictions(merged, DRILL_TARGETS)
    modes = next((v for v in result if v.attribute_id == MODES_ID), None)
    if modes is not None:
        lower_vals = [x.lower() for x in modes.value] if isinstance(modes.value, list) else [str(modes.value).lower()]
        assert any("удар" in x for x in lower_vals)
    # else: modes dropped entirely -- also satisfies "no contradiction survives"
