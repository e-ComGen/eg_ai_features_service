"""
Tests for docs/MANIFEST_grounding_arbitration.md FIX-3/FIX-4 functional oracle (O2, O2b, O3, O1b).
"""

from app.services.enrichment.pipeline import (
    _drop_ungrounded_hard_facts,
    _reconcile_cross_field_contradictions,
)
from app.services.enrichment.base import AttributeValue, Source, TargetAttribute


def _av(attribute_id, value, confidence, source, evidence=None, is_collection=False):
    return AttributeValue(
        attribute_id=attribute_id, value=value, confidence=confidence,
        source=source, evidence=evidence, is_collection=is_collection,
    )


def _target(id, name, type="text", is_collection=False):
    return TargetAttribute(id=id, name=name, type=type, is_collection=is_collection)


def test_o2_drop_ungrounded_hard_fact_no_model_mention():
    """O2: Generic brand-fact with no product-model mention should be dropped."""
    target = _target(901, "Страна-изготовитель")
    value = _av(
        attribute_id=901,
        value="Япония",
        confidence=0.7,
        source=Source.WEB_SEARCH,
        evidence="[https://ru.wikipedia.org/wiki/Makita] Makita — крупный японский производитель электроинструмента, основанный в 1915 году.",
    )
    
    result = _drop_ungrounded_hard_facts([value], "", [target], product_model_text="HP1630")
    
    assert not any(v.attribute_id == 901 for v in result)


def test_o2b_cross_host_corroboration():
    """O2b(a): Cross-host corroboration should preserve both values."""
    target = _target(901, "Страна-изготовитель")
    v1 = _av(
        attribute_id=901,
        value="Китай",
        confidence=0.7,
        source=Source.WEB_SEARCH,
        evidence="[https://host-one.example.com/p] Страна происхождения: Китай.",
    )
    v2 = _av(
        attribute_id=901,
        value="Китай",
        confidence=0.7,
        source=Source.WEB_SEARCH,
        evidence="[https://host-two.example.ru/p] Товар произведён в Китае.",
    )
    
    result = _drop_ungrounded_hard_facts([v1, v2], "", [target], product_model_text="")
    
    assert len([v for v in result if v.attribute_id == 901]) == 2


def test_o2b_model_adjacency():
    """O2b(b): Model-adjacency in evidence should preserve the value."""
    target = _target(901, "Страна-изготовитель")
    value = _av(
        attribute_id=901,
        value="Япония",
        confidence=0.7,
        source=Source.WEB_SEARCH,
        evidence="[https://shop.example.com/hp1630] Makita HP1630 производится в Японии, завод Anjo.",
    )
    
    result = _drop_ungrounded_hard_facts([value], "", [target], product_model_text="HP1630 Дрель-шуруповёрт")
    
    assert len([v for v in result if v.attribute_id == 901]) == 1
    result_901 = next(v for v in result if v.attribute_id == 901)
    assert result_901.value == "Япония"


def test_o3_cross_field_contradiction_reconciliation():
    """O3: Cross-field contradiction between Комплектация and Тип патрона."""
    targets = [
        _target(912, "Комплектация"),
        _target(913, "Тип патрона"),
    ]
    merged = [
        _av(913, "Быстрозажимной", 0.6, Source.SAFE_ENUM_FILL, evidence="enum default"),
        _av(912, "Дрель, ключевой патрон, кейс для переноски", 0.9, Source.DESCRIPTION, evidence="verbatim product description"),
    ]
    
    result = _reconcile_cross_field_contradictions(merged, targets)
    
    result_913 = next(v for v in result if v.attribute_id == 913)
    assert "ключев" in result_913.value.lower()
    assert "быстрозажимн" not in result_913.value.lower()


def test_o1b_drilling_modes_contradiction():
    """O1b: Concrete drilling diameter implies impact mode should be present."""
    targets = [
        _target(910, "Макс. диаметр отверстия в бетоне, мм", type="numeric"),
        _target(911, "Режимы работы", is_collection=True),
    ]
    merged = [
        _av(910, "16", 0.8, Source.WEB_SEARCH, evidence="handles concrete drilling up to 16mm"),
        _av(911, ["Сверление"], 0.7, Source.LLM_KNOWLEDGE, evidence="drilling mode only", is_collection=True),
    ]
    
    result = _reconcile_cross_field_contradictions(merged, targets)
    
    result_911 = next((v for v in result if v.attribute_id == 911), None)
    if result_911 is not None:
        # corrected: must now include an удар/impact mode -- never left as bare "Сверление" alone
        values_lower = [x.lower() for x in result_911.value] if isinstance(result_911.value, list) else [str(result_911.value).lower()]
        assert any("удар" in x for x in values_lower)
    else:
        # OR: the inconsistent partial was dropped entirely (also an acceptable resolution)
        pass
