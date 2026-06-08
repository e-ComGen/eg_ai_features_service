"""Unit tests for image_card_search GATING layer (no network, no real LLM).

Покрываем слоистый гейт false-positive mitigation:
  - matching card (правильный бренд+тип, высокий visual-score) → ПРОХОДИТ;
  - wrong-brand card → REJECT (G1);
  - wrong-type card → REJECT (G2);
  - low visual-score → REJECT (G3);
  - top-K agreement (2 движка) понижает порог visual-score;
  - LLM-verify reject (G4) режет, даже если G1-G3 прошли;
  - enum-gate (G5): значение без value_id не филлит enum;
  - confidence cap (G5).
Reverse-image вызов — STUB (возвращает []), реальный API не дёргаем.
"""
import asyncio

import pytest

from app.services.enrichment.sources import image_card_search as ics
from app.services.enrichment.sources.image_card_search import (
    ImageCandidate, gate_candidate, find_matching_card,
    brand_matches, type_matches, enum_gate_value, capped_confidence,
    IMAGE_CARD_CONF_CAP, VISUAL_SCORE_MIN_SINGLE,
)


def _run(coro):
    return asyncio.run(coro)


# --- мок LLM-verifier (G4) ---

class _StubVerifier:
    def __init__(self, same: bool, conf: float):
        self._same = same
        self._conf = conf
        self.calls = 0

    async def verify_same_product(self, product_name, candidate_title, candidate_attrs):
        self.calls += 1
        return self._same, self._conf


class _RaisingVerifier:
    async def verify_same_product(self, product_name, candidate_title, candidate_attrs):
        raise RuntimeError("llm down")


# ---------------------------------------------------------------------------
# G1 brand cross-check
# ---------------------------------------------------------------------------

def test_brand_match_from_title():
    c = ImageCandidate(url="u", marketplace="ozon", title="Nike Куртка мужская")
    assert brand_matches("Nike", c) is True


def test_brand_mismatch_rejects():
    c = ImageCandidate(url="u", marketplace="ozon", title="Adidas Куртка", brand="Adidas")
    assert brand_matches("Nike", c) is False


def test_brand_unknown_returns_none():
    c = ImageCandidate(url="u", marketplace="ozon", title="Куртка зимняя")
    assert brand_matches(None, c) is None
    assert brand_matches("Nike", c) is None  # бренда нет ни в title ни в .brand


# ---------------------------------------------------------------------------
# G2 type cross-check
# ---------------------------------------------------------------------------

def test_type_match_same_lemma():
    c = ImageCandidate(url="u", marketplace="wb", title="Nike Куртки мужские")
    assert type_matches("куртка", c) is True


def test_type_mismatch_rejects():
    c = ImageCandidate(url="u", marketplace="wb", title="Nike Брюки спортивные")
    assert type_matches("куртка", c) is False


# ---------------------------------------------------------------------------
# Полный гейт: matching card проходит
# ---------------------------------------------------------------------------

def test_matching_card_passes():
    c = ImageCandidate(
        url="https://ozon.ru/p/1", marketplace="ozon", title="Nike Куртка мужская",
        brand="Nike", card_id="1", visual_score=0.9,
    )
    res = _run(gate_candidate(
        c, our_brand="Nike", target_type="куртка", product_name="Nike Куртка мужская",
    ))
    assert res.accepted is True
    assert res.match_confidence == pytest.approx(0.9)


def test_wrong_brand_rejected():
    c = ImageCandidate(
        url="u", marketplace="ozon", title="Adidas Куртка", brand="Adidas",
        card_id="2", visual_score=0.95,
    )
    res = _run(gate_candidate(
        c, our_brand="Nike", target_type="куртка", product_name="Nike Куртка",
    ))
    assert res.accepted is False
    assert "G1" in res.reason


def test_wrong_type_rejected():
    c = ImageCandidate(
        url="u", marketplace="wb", title="Nike Брюки", brand="Nike",
        card_id="3", visual_score=0.95,
    )
    res = _run(gate_candidate(
        c, our_brand="Nike", target_type="куртка", product_name="Nike Куртка",
    ))
    assert res.accepted is False
    assert "G2" in res.reason


def test_low_visual_score_rejected():
    c = ImageCandidate(
        url="u", marketplace="ozon", title="Nike Куртка", brand="Nike",
        card_id="4", visual_score=0.5,  # < VISUAL_SCORE_MIN_SINGLE
    )
    res = _run(gate_candidate(
        c, our_brand="Nike", target_type="куртка", product_name="Nike Куртка",
    ))
    assert res.accepted is False
    assert "G3" in res.reason


def test_multi_engine_lowers_threshold():
    # 0.70 ниже single-порога (0.82) но выше multi-порога (0.65).
    assert 0.70 < VISUAL_SCORE_MIN_SINGLE
    c = ImageCandidate(
        url="u", marketplace="ozon", title="Nike Куртка", brand="Nike",
        card_id="5", visual_score=0.70, engines={"yandex_images", "google_lens"},
    )
    res = _run(gate_candidate(
        c, our_brand="Nike", target_type="куртка", product_name="Nike Куртка",
    ))
    assert res.accepted is True


# ---------------------------------------------------------------------------
# G4 LLM-verify
# ---------------------------------------------------------------------------

def test_llm_verify_accepts():
    c = ImageCandidate(
        url="u", marketplace="ozon", title="Nike Куртка Windrunner", brand="Nike",
        card_id="6", visual_score=0.9,
    )
    v = _StubVerifier(same=True, conf=0.9)
    res = _run(gate_candidate(
        c, our_brand="Nike", target_type="куртка",
        product_name="Nike Куртка Windrunner", llm_verifier=v,
    ))
    assert res.accepted is True
    assert v.calls == 1
    assert res.match_confidence == pytest.approx(0.9)


def test_llm_verify_reject_no_match():
    c = ImageCandidate(
        url="u", marketplace="ozon", title="Nike Куртка Other", brand="Nike",
        card_id="7", visual_score=0.95,  # G1-G3 проходят
    )
    v = _StubVerifier(same=False, conf=0.9)
    res = _run(gate_candidate(
        c, our_brand="Nike", target_type="куртка",
        product_name="Nike Куртка Windrunner", llm_verifier=v,
    ))
    assert res.accepted is False
    assert "G4" in res.reason


def test_llm_verify_reject_low_conf():
    c = ImageCandidate(
        url="u", marketplace="ozon", title="Nike Куртка", brand="Nike",
        card_id="8", visual_score=0.95,
    )
    v = _StubVerifier(same=True, conf=0.5)  # < LLM_VERIFY_CONF_MIN
    res = _run(gate_candidate(
        c, our_brand="Nike", target_type="куртка",
        product_name="Nike Куртка", llm_verifier=v,
    ))
    assert res.accepted is False
    assert "G4" in res.reason


def test_llm_error_rejects_fail_closed():
    c = ImageCandidate(
        url="u", marketplace="ozon", title="Nike Куртка", brand="Nike",
        card_id="9", visual_score=0.95,
    )
    res = _run(gate_candidate(
        c, our_brand="Nike", target_type="куртка",
        product_name="Nike Куртка", llm_verifier=_RaisingVerifier(),
    ))
    assert res.accepted is False
    assert "G4" in res.reason


# ---------------------------------------------------------------------------
# find_matching_card: STUB вернёт [] → None (безопасно)
# ---------------------------------------------------------------------------

def test_find_matching_card_stub_returns_none():
    res = _run(find_matching_card(
        "https://img/photo.jpg", "Nike Куртка", our_brand="Nike", target_type="куртка",
    ))
    assert res is None


def test_find_matching_card_picks_best_when_candidates(monkeypatch):
    async def fake_search(image_url, marketplaces=()):
        return [
            ImageCandidate(url="a", marketplace="ozon", title="Nike Куртка",
                           brand="Nike", card_id="A", visual_score=0.85),
            ImageCandidate(url="b", marketplace="ozon", title="Nike Куртка",
                           brand="Nike", card_id="B", visual_score=0.95),
            ImageCandidate(url="c", marketplace="ozon", title="Adidas Куртка",
                           brand="Adidas", card_id="C", visual_score=0.99),  # G1 reject
        ]
    monkeypatch.setattr(ics, "_reverse_image_search", fake_search)
    res = _run(find_matching_card(
        "img", "Nike Куртка", our_brand="Nike", target_type="куртка",
    ))
    assert res is not None
    assert res.candidate.card_id == "B"  # высший visual_score среди прошедших


def test_merge_by_card_id_accumulates_engines(monkeypatch):
    async def fake_search(image_url, marketplaces=()):
        return [
            ImageCandidate(url="a", marketplace="ozon", title="Nike Куртка",
                           brand="Nike", card_id="X", visual_score=0.60,
                           engines={"yandex_images"}),
            ImageCandidate(url="a", marketplace="ozon", title="Nike Куртка",
                           brand="Nike", card_id="X", visual_score=0.68,
                           engines={"google_lens"}),
        ]
    monkeypatch.setattr(ics, "_reverse_image_search", fake_search)
    # По отдельности 0.68 < single-порог → reject; слитые 2 движка → multi-порог → pass.
    res = _run(find_matching_card(
        "img", "Nike Куртка", our_brand="Nike", target_type="куртка",
    ))
    assert res is not None
    assert res.candidate.card_id == "X"


# ---------------------------------------------------------------------------
# G5 enum gate + confidence cap
# ---------------------------------------------------------------------------

def test_enum_gate_no_dict_returns_none():
    # Без cat/type → нельзя резолвить value_id → None (значение не пишем).
    assert enum_gate_value(None, None, 123, "Чёрный") is None
    assert enum_gate_value(0, 0, 123, "Чёрный") is None


def test_enum_gate_resolves_via_loader(monkeypatch):
    import app.services.enrichment.strategies.dictionaries.ozon_loader as loader
    monkeypatch.setattr(loader, "resolve_value_id", lambda c, t, a, v: 777)
    assert enum_gate_value(1, 2, 3, "Чёрный") == 777


def test_confidence_cap():
    assert capped_confidence(0.99) == pytest.approx(IMAGE_CARD_CONF_CAP)
    assert capped_confidence(0.40) == pytest.approx(0.40)
