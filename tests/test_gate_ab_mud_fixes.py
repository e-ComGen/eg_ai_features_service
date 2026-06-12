"""Tests for Gate A (web_search self-consistency) and Fix B (extended audio spec attrs).

These cover the two mud leaks from the 25-product eval:
  1. "Бязь" source=web_search, evidence="Состав ткани: 100% хлопок" → must be DROPPED
     (value token "бязь" absent from evidence → Gate A fires).
  2. "2.0" source=llm_knowledge, attr=Звуковая схема, no authoritative corroboration → DROPPED
     (Track A; extended by Fix B for new audio keyword variants).

KEEP (must NOT be dropped):
  - "Хлопок" source=web_search, evidence="состав: 100% хлопок" → KEPT (token present).
  - "2.0" source=llm_knowledge, Звуковая схема, WITH ozon_card "2.0" → KEPT (corroborated).
  - vision fill → always passes through untouched.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    _is_objective_spec_attr,
    _web_search_grounded_in_evidence,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ctx(product_name: str = "Товар", brand: str = "Бренд") -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name=product_name,
        category_id=100,
        category_path=["Электроника"],
        brand=brand,
    )


def _target(
    attr_id: int,
    name: str,
    type_: str = "enum",
    allowed_values: list[str] | None = None,
) -> TargetAttribute:
    return TargetAttribute(
        id=attr_id,
        name=name,
        type=type_,
        allowed_values=allowed_values,
    )


def _av(
    attr_id: int,
    value: str,
    source: Source = Source.LLM_KNOWLEDGE,
    confidence: float = 0.9,
    evidence: str | None = None,
) -> AttributeValue:
    return AttributeValue(
        attribute_id=attr_id,
        value=value,
        confidence=confidence,
        source=source,
        evidence=evidence,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fix A: _web_search_grounded_in_evidence unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestWebSearchGroundedInEvidence:
    """Unit tests for the self-consistency gate helper."""

    def test_byaz_not_in_khlopok_evidence(self):
        """'Бязь' is absent from evidence '100% хлопок' → returns False (drop)."""
        assert _web_search_grounded_in_evidence("Бязь", "Состав ткани: 100% хлопок") is False

    def test_khlopok_in_khlopok_evidence(self):
        """'Хлопок' is present in evidence '100% хлопок' → returns True (keep)."""
        assert _web_search_grounded_in_evidence("Хлопок", "состав: 100% хлопок") is True

    def test_100_khlopok_in_evidence(self):
        """Composite '100% Хлопок': token 'хлопок' present in evidence → True."""
        assert _web_search_grounded_in_evidence("100% Хлопок", "ткань: 100% хлопок, гипоаллергенно") is True

    def test_empty_evidence_returns_true(self):
        """No evidence → conservative: do not drop."""
        assert _web_search_grounded_in_evidence("Бязь", None) is True
        assert _web_search_grounded_in_evidence("Бязь", "") is True

    def test_boolean_да_skips_check(self):
        """Boolean fill 'Да' — no meaningful token to check → True (keep)."""
        assert _web_search_grounded_in_evidence("Да", "product has noise cancelling") is True

    def test_boolean_нет_skips_check(self):
        assert _web_search_grounded_in_evidence("Нет", "no wireless") is True

    def test_ё_normalisation(self):
        """ё→е normalisation must allow matching across ё/е variants."""
        assert _web_search_grounded_in_evidence("Ёлочка", "ткань: елочка, хлопок") is True

    def test_case_insensitive(self):
        """Matching must be case-insensitive."""
        assert _web_search_grounded_in_evidence("хлопок", "СОСТАВ: 100% ХЛОПОК") is True

    def test_very_short_value_token_skips_check(self):
        """Value whose all tokens are < 3 chars → conservative, keep."""
        assert _web_search_grounded_in_evidence("AB", "something else entirely") is True

    def test_polyester_in_evidence(self):
        """'Полиэстер' → token 'полиэстер' found in evidence."""
        assert _web_search_grounded_in_evidence(
            "Полиэстер", "состав: полиэстер 100%, износостойкий"
        ) is True

    def test_polyester_not_in_khlopok_evidence(self):
        """'Полиэстер' absent from 'хлопок' evidence → drop."""
        assert _web_search_grounded_in_evidence("Полиэстер", "100% хлопок, натуральный") is False


# ──────────────────────────────────────────────────────────────────────────────
# Fix B: extended audio/channel spec fragments in _is_objective_spec_attr
# ──────────────────────────────────────────────────────────────────────────────


class TestFixBAudioSpecFragments:
    """New audio/channel keyword variants must be classified as objective-spec."""

    def test_zvukovaya_skhema_is_spec(self):
        """'Звуковая схема' was already present — ensure it still matches."""
        t = _target(5089, "Звуковая схема")
        assert _is_objective_spec_attr(t) is True

    def test_zvukovaya_sistema_is_spec(self):
        """'Звуковая система' — newly added fragment."""
        t = _target(5090, "Звуковая система")
        assert _is_objective_spec_attr(t) is True

    def test_audiokanal_is_spec(self):
        """'Аудиоканал' — newly added fragment."""
        t = _target(5091, "Аудиоканал")
        assert _is_objective_spec_attr(t) is True

    def test_konfiguratsiya_kanalov_is_spec(self):
        """'Конфигурация каналов' — newly added fragment."""
        t = _target(5092, "Конфигурация каналов")
        assert _is_objective_spec_attr(t) is True

    def test_kolichestvo_kanalov_is_spec(self):
        """'Количество каналов' was already present — still matches."""
        t = _target(5093, "Количество каналов")
        assert _is_objective_spec_attr(t) is True

    def test_stil_is_not_spec(self):
        """'Стиль' is not a spec attr — must remain False."""
        t = _target(400, "Стиль")
        assert _is_objective_spec_attr(t) is False


# ──────────────────────────────────────────────────────────────────────────────
# Integration: full adversarial pass — 4 mandated cases
# ──────────────────────────────────────────────────────────────────────────────


class TestGateABIntegration:
    """Mandated integration cases: drop/keep exactly as specified."""

    # ── Case 1: "Бязь" web_search evidence="100% хлопок" → DROPPED ──────────

    @pytest.mark.asyncio
    async def test_byaz_web_search_dropped_evidence_mismatch(self):
        """Gate A: 'Бязь' value absent from evidence '100% хлопок' → DROP."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        mud = _av(
            attr_id=4496,
            value="Бязь",
            source=Source.WEB_SEARCH,
            confidence=0.9,
            evidence="Состав ткани: 100% хлопок",
        )
        target = _target(4496, "Материал")
        ctx = _ctx("Постельное бельё хлопок", "HomeTextile")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [mud], [target], ctx, [],
        )
        assert not any(v.attribute_id == 4496 for v in result), (
            "Бязь with evidence='100% хлопок' must be DROPPED by Gate A"
        )

    # ── Case 2: "Хлопок" web_search evidence contains token → KEPT ───────────

    @pytest.mark.asyncio
    async def test_khlopok_web_search_kept_when_evidence_contains_token(self):
        """Gate A: 'Хлопок' token present in evidence → fill is NOT dropped by Gate A.

        'Материал' is objective-spec → goes to Track A (corroboration required).
        We add a description corroborator so it survives Track A too.
        """
        from app.services.enrichment.pipeline import PipelineOrchestrator

        good = _av(
            attr_id=4496,
            value="Хлопок",
            source=Source.WEB_SEARCH,
            confidence=0.88,
            evidence="состав: 100% хлопок, натуральная ткань",
        )
        # Description corroborator so Track A passes
        corr = _av(
            attr_id=4496,
            value="Хлопок",
            source=Source.DESCRIPTION,
            confidence=0.95,
        )
        target = _target(4496, "Материал")
        ctx = _ctx("Футболка хлопковая", "Brand")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [good, corr], [target], ctx, [],
        )
        assert any(v.attribute_id == 4496 and v.source == Source.WEB_SEARCH for v in result), (
            "Хлопок with matching evidence must NOT be dropped by Gate A"
        )

    # ── Case 3: "2.0" llm_knowledge, Звуковая схема, no authoritative → DROPPED

    @pytest.mark.asyncio
    async def test_zvukovaya_2_0_llm_knowledge_dropped_without_corroboration(self):
        """Track A: '2.0' Звуковая схема from llm_knowledge, no authoritative source → DROP.

        Яндекс Станция Мини 2 is mono (1.0). LLM evidence is self-generated.
        """
        from app.services.enrichment.pipeline import PipelineOrchestrator

        mud = _av(
            attr_id=5089,
            value="2.0",
            source=Source.LLM_KNOWLEDGE,
            confidence=0.95,
            evidence="The device is known to have a 2.0 sound scheme (stereo speakers).",
        )
        target = _target(5089, "Звуковая схема")
        ctx = _ctx("Яндекс Станция Мини 2", "Яндекс")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [mud], [target], ctx, [],
        )
        assert not any(v.attribute_id == 5089 for v in result), (
            "Звуковая схема=2.0 without authoritative corroboration must be DROPPED"
        )

    # ── Case 4: "2.0" llm_knowledge, WITH ozon_card "2.0" → KEPT ─────────────

    @pytest.mark.asyncio
    async def test_zvukovaya_2_0_llm_knowledge_kept_with_ozon_card(self):
        """Track A: '2.0' corroborated by ozon_card → KEPT."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        llm_fill = _av(
            attr_id=5089,
            value="2.0",
            source=Source.LLM_KNOWLEDGE,
            confidence=0.95,
            evidence="LLM self-generated evidence",
        )
        corr = _av(
            attr_id=5089,
            value="2.0",
            source=Source.OZON_CARD,
            confidence=0.99,
        )
        target = _target(5089, "Звуковая схема")
        ctx = _ctx("Колонка стерео", "Brand")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [llm_fill, corr], [target], ctx, [],
        )
        assert any(v.attribute_id == 5089 and v.source == Source.LLM_KNOWLEDGE for v in result), (
            "Звуковая схема=2.0 corroborated by ozon_card must SURVIVE"
        )

    # ── Case 5: vision fill is untouched ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_vision_fill_passes_through_untouched(self):
        """Vision source is NOT in _ADVERSARIAL_SOURCES → always passes through."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        vision_fill = _av(
            attr_id=11859,
            value="False",
            source=Source.VISION,
            confidence=0.92,
            evidence="image shows over-ear headphones",
        )
        target = _target(11859, "True Wireless", type_="bool")
        ctx = _ctx("Sony WH-1000XM5", "Sony")

        verify_calls: list = []

        async def _track_verify(ctx, proposals, resolved_attrs=None, llm_manager=None):
            verify_calls.extend(proposals)
            return set()

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_track_verify,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [vision_fill], [target], ctx, [],
            )

        assert any(v.attribute_id == 11859 and v.source == Source.VISION for v in result), (
            "Vision fill must pass through untouched (not in adversarial sources)"
        )
        assert not verify_calls, "Gate B must NOT be called for vision fills"
