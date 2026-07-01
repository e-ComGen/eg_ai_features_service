"""Functional/oracle test for docs/MANIFEST_grounding_arbitration.md FIX-1/FIX-2 Round 1."""
from app.services.enrichment.pipeline import PipelineOrchestrator, _merge_winner
from app.services.enrichment.base import AttributeValue, Source, SOURCE_PRIORITY


def _av(attribute_id, value, confidence, source, evidence=None, is_collection=False):
    return AttributeValue(
        attribute_id=attribute_id,
        value=value,
        confidence=confidence,
        source=source,
        evidence=evidence,
        is_collection=is_collection,
    )


def test_r1_authority_over_confidence_flip():
    card = _av(1, "card_val", 0.5, Source.OZON_CARD, evidence="")
    web = _av(1, "web_val", 0.95, Source.WEB_SEARCH, evidence="a fairly long web search snippet about this exact product")

    winner_fw = _merge_winner(card, web)
    winner_bw = _merge_winner(web, card)
    assert winner_fw.source == Source.OZON_CARD
    assert winner_bw.source == Source.OZON_CARD

    # Old logic: strictly higher confidence wins, tie-break by SOURCE_PRIORITY
    if web.confidence > card.confidence:
        old_winner = web
    elif card.confidence > web.confidence:
        old_winner = card
    else:
        old_winner = card if SOURCE_PRIORITY[card.source] > SOURCE_PRIORITY[web.source] else web

    assert old_winner.source == Source.WEB_SEARCH
    assert old_winner.source != winner_fw.source


def test_r2_last_resort_abstain():
    orch = PipelineOrchestrator()
    result = orch._merge([
        _av(100, "Сверление", 0.95, Source.LLM_KNOWLEDGE, evidence="this is a drill, only drilling mode", is_collection=True)
    ])
    assert not any(v.attribute_id == 100 for v in result)


def test_r3_regression_passthrough():
    orch = PipelineOrchestrator()
    result = orch._merge([
        _av(200, "44800", 0.6, Source.WEB_SEARCH, evidence="official spec sheet lists 44800 impacts/min")
    ])
    assert len(result) == 1
    v = result[0]
    assert v.attribute_id == 200
    assert v.value == "44800"
    assert v.source == Source.WEB_SEARCH
    assert v.confidence == 0.6


def test_r4_collections_union_preserved():
    orch = PipelineOrchestrator()
    vals = [
        _av(300, ["Сверление"], 0.8, Source.LLM_KNOWLEDGE, evidence="каталожная спецификация указывает функции: сверление и ударное сверление для модели XYZ123", is_collection=True),
        _av(300, ["Сверление с ударом"], 0.8, Source.LLM_KNOWLEDGE, evidence="официальный даташит модели XYZ123 подтверждает режим удара, параграф 4.2", is_collection=True),
    ]
    result = orch._merge(vals)
    merged_300 = [v for v in result if v.attribute_id == 300]
    assert len(merged_300) == 1
    merged_val = merged_300[0]
    assert isinstance(merged_val.value, list)
    lower_vals = [x.lower() for x in merged_val.value]
    assert "сверление" in lower_vals
    assert "сверление с ударом" in lower_vals
