# -*- coding: utf-8 -*-
"""ТН ВЭД (customs code) must survive the required-enum drop-guard.

Bug: the Ozon dict ships a generic sample `values` list (~66 unrelated codes —
candles, plastic dishware) for the ТН ВЭД attribute on EVERY category, so a
valid TnvedSource code resolves to value_id=None and was silently dropped by
_drop_unresolved_required_enums as an "unresolved required enum" — emptying
ТН ВЭД on ~all products. The customs-code field is effectively free-text; it
must be exempt from the enum drop-guards."""
from app.services.enrichment.pipeline import (
    _drop_unresolved_required_enums,
    _drop_unresolved_optional_enums,
)
from app.services.enrichment.base import (
    AttributeValue, TargetAttribute, Source,
)


def _tnved_target():
    return TargetAttribute(
        id=22232,
        name="ТН ВЭД коды ЕАЭС",
        type="text",
        # generic sample codes Ozon attaches to every category (garbage)
        allowed_values=["3406000000 - Свечи", "3924100000 - Посуда из пластмасс"],
        is_collection=False,
        is_required=True,
    )


def _tnved_value():
    return AttributeValue(
        attribute_id=22232,
        value="9504500000",            # correct code for a game console
        confidence=0.90,
        source=Source.LLM_KNOWLEDGE,
        value_id=None,                  # never resolves against the garbage list
        evidence="tnved_resolver: ТН ВЭД ЕАЭС, резолв по категории",
    )


def test_tnved_value_survives_required_drop_guard():
    targets = [_tnved_target()]
    out = _drop_unresolved_required_enums([_tnved_value()], targets)
    assert len(out) == 1
    assert out[0].value == "9504500000"


def test_tnved_value_survives_optional_drop_guard():
    # even if somehow marked optional, the customs code must not be dropped
    t = _tnved_target()
    t.is_required = False
    out = _drop_unresolved_optional_enums([_tnved_value()], [t])
    assert len(out) == 1


def test_real_required_enum_without_value_id_still_dropped():
    """Control: a genuine required enum (Тип) with value_id=None IS dropped —
    the guard must still work for real enums, the exemption is ТН ВЭД-only."""
    typ = TargetAttribute(
        id=8229, name="Тип", type="text",
        allowed_values=["Секатор", "Дальномер"],
        is_collection=False, is_required=True,
    )
    bad = AttributeValue(
        attribute_id=8229, value="без маятникового хода",
        confidence=0.8, source=Source.WB_CARD, value_id=None,
    )
    out = _drop_unresolved_required_enums([bad], [typ])
    assert out == []
