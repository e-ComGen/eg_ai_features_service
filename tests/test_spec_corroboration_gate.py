"""Tests for the deterministic source-corroboration gate (Stage 4.9, Track A).

FIX CORR-A: Objective-spec attrs (material, bool features, numeric, connectivity)
from LLM_KNOWLEDGE / WEB_SEARCH are DROPPED unless an authoritative source
(wb_card, ozon_card, icecat, pdf_datasheet, description) independently produced
the SAME (attribute_id, normalised value).

FIX CORR-B: Non-spec attrs still go through the existing LLM Gate B path.

All tests use mocked LLM — no real API calls, fully deterministic.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.pipeline import (
    _is_objective_spec_attr,
    _normalize_for_corroboration,
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
    semantic_type: str | None = None,
) -> TargetAttribute:
    return TargetAttribute(
        id=attr_id,
        name=name,
        type=type_,
        allowed_values=allowed_values,
        semantic_type=semantic_type,
    )


def _av(
    attr_id: int,
    value: str,
    source: Source = Source.LLM_KNOWLEDGE,
    confidence: float = 0.85,
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
# _is_objective_spec_attr predicate
# ──────────────────────────────────────────────────────────────────────────────


class TestIsObjectiveSpecAttr:
    """Unit tests for the spec-class predicate."""

    # ── semantic_type signal ──────────────────────────────────────────────────

    def test_semantic_type_material_is_spec(self):
        t = _target(1, "Какой-то атрибут", semantic_type="material")
        assert _is_objective_spec_attr(t) is True

    def test_semantic_type_connectivity_is_spec(self):
        t = _target(2, "Что-то", semantic_type="connectivity")
        assert _is_objective_spec_attr(t) is True

    def test_semantic_type_composition_is_spec(self):
        t = _target(3, "Состав чего-то", semantic_type="composition")
        assert _is_objective_spec_attr(t) is True

    def test_semantic_type_color_is_not_spec(self):
        """Color is explicitly NOT spec-class (Gate B LLM is appropriate)."""
        t = _target(4, "Цвет", semantic_type="color")
        assert _is_objective_spec_attr(t) is False

    # ── structural type signal ────────────────────────────────────────────────

    def test_bool_type_is_spec(self):
        """True Wireless (id=11859) is type=bool — must be spec."""
        t = _target(11859, "True Wireless", type_="bool")
        assert _is_objective_spec_attr(t) is True

    def test_numeric_type_is_spec(self):
        """Any numeric measurement attr is spec."""
        t = _target(5, "Мощность, Вт", type_="numeric")
        assert _is_objective_spec_attr(t) is True

    def test_text_type_no_match_is_not_spec(self):
        """Free-text attr with no name fragment is not spec."""
        t = _target(6, "Аннотация", type_="text")
        assert _is_objective_spec_attr(t) is False

    def test_enum_type_no_match_is_not_spec(self):
        """A generic lifestyle enum is not spec."""
        t = _target(7, "Стиль", type_="enum", allowed_values=["Спортивный", "Повседневный"])
        assert _is_objective_spec_attr(t) is False

    # ── name-fragment signal ──────────────────────────────────────────────────

    def test_name_материал_is_spec(self):
        """'Материал' substring triggers spec."""
        t = _target(4496, "Материал", type_="enum")
        assert _is_objective_spec_attr(t) is True

    def test_name_материал_верха_is_spec(self):
        t = _target(99, "Материал верха", type_="enum")
        assert _is_objective_spec_attr(t) is True

    def test_name_состав_is_spec(self):
        """'Состав' substring triggers spec."""
        t = _target(4604, "Состав материала", type_="text")
        assert _is_objective_spec_attr(t) is True

    def test_name_подкладка_is_spec(self):
        t = _target(88, "Материал подкладки", type_="enum")
        assert _is_objective_spec_attr(t) is True

    def test_name_звуковая_схема_is_spec(self):
        """'Звуковая схема' triggers spec (audio config mud case)."""
        t = _target(5089, "Звуковая схема", type_="enum")
        assert _is_objective_spec_attr(t) is True

    def test_name_true_wireless_is_spec(self):
        """'True Wireless' substring triggers spec (mud case: over-ear=Да)."""
        t = _target(11859, "True Wireless", type_="enum")
        assert _is_objective_spec_attr(t) is True

    def test_name_тип_подключения_is_spec(self):
        t = _target(50, "Тип подключения", type_="enum")
        assert _is_objective_spec_attr(t) is True

    def test_name_интерфейс_is_spec(self):
        t = _target(4526, "Интерфейс", type_="enum")
        assert _is_objective_spec_attr(t) is True

    def test_name_шумоподавлен_is_spec(self):
        t = _target(200, "Активное шумоподавление", type_="enum")
        assert _is_objective_spec_attr(t) is True

    def test_name_цвет_is_not_spec(self):
        t = _target(300, "Цвет", type_="enum")
        assert _is_objective_spec_attr(t) is False

    def test_name_бренд_is_not_spec(self):
        t = _target(31, "Бренд", type_="enum")
        assert _is_objective_spec_attr(t) is False

    def test_name_стиль_is_not_spec(self):
        t = _target(400, "Стиль", type_="enum")
        assert _is_objective_spec_attr(t) is False

    def test_name_сезон_is_not_spec(self):
        """Season is subjective/categorical — not spec."""
        t = _target(500, "Сезон", type_="enum")
        assert _is_objective_spec_attr(t) is False

    # ── case insensitivity ────────────────────────────────────────────────────

    def test_name_case_insensitive(self):
        """Predicate must be case-insensitive."""
        t = _target(600, "МАТЕРИАЛ", type_="enum")
        assert _is_objective_spec_attr(t) is True

    def test_semantic_type_case_insensitive(self):
        t = _target(700, "Что-то", semantic_type="MATERIAL")
        assert _is_objective_spec_attr(t) is True


# ──────────────────────────────────────────────────────────────────────────────
# _normalize_for_corroboration
# ──────────────────────────────────────────────────────────────────────────────


class TestNormalizeForCorroboration:

    def test_true_to_да(self):
        assert _normalize_for_corroboration(True) == "да"
        assert _normalize_for_corroboration("True") == "да"
        assert _normalize_for_corroboration("true") == "да"
        assert _normalize_for_corroboration("Да") == "да"

    def test_false_to_нет(self):
        assert _normalize_for_corroboration(False) == "нет"
        assert _normalize_for_corroboration("False") == "нет"
        assert _normalize_for_corroboration("нет") == "нет"
        assert _normalize_for_corroboration("Нет") == "нет"

    def test_ё_to_е(self):
        assert _normalize_for_corroboration("Хлёб") == "хлеб"

    def test_strips_whitespace(self):
        assert _normalize_for_corroboration("  2.0  ") == "2.0"

    def test_collapses_internal_whitespace(self):
        assert _normalize_for_corroboration("2.1  канала") == "2.1 канала"

    def test_lowercases(self):
        assert _normalize_for_corroboration("Bluetooth") == "bluetooth"


# ──────────────────────────────────────────────────────────────────────────────
# Corroboration gate integration (via _run_llm_knowledge_adversarial_pass)
# ──────────────────────────────────────────────────────────────────────────────


class TestSpecCorroborationGate:
    """Mud cases are DROPPED; corroborated fills SURVIVE; non-spec goes to Gate B."""

    # ── MUD CASE 1: True Wireless=true on over-ear headphones ────────────────

    @pytest.mark.asyncio
    async def test_true_wireless_mud_dropped_without_corroboration(self):
        """LLM says True Wireless=Да for Sony WH-1000XM5 (over-ear).
        No authoritative source confirms it → DROPPED."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        # bool type → is_objective_spec_attr == True
        mud = _av(attr_id=11859, value="Да", source=Source.LLM_KNOWLEDGE, confidence=0.85)
        target = _target(11859, "True Wireless", type_="bool")
        ctx = _ctx("Sony WH-1000XM5 наушники", "Sony")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [mud], [target], ctx, [],
        )
        assert not any(v.attribute_id == 11859 for v in result), (
            "True Wireless=Да without authoritative corroboration must be DROPPED"
        )

    # ── MUD CASE 2: Звуковая схема=2.0 on mono Яндекс Станция Мини 2 ────────

    @pytest.mark.asyncio
    async def test_zvukovaya_skhema_mud_dropped_without_corroboration(self):
        """LLM says Звуковая схема=2.0 for Яндекс Станция Мини 2 (mono device).
        No authoritative source confirms it → DROPPED."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        mud = _av(attr_id=5089, value="2.0", source=Source.LLM_KNOWLEDGE, confidence=0.88)
        target = _target(5089, "Звуковая схема", type_="enum")
        ctx = _ctx("Яндекс Станция Мини 2", "Яндекс")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [mud], [target], ctx, [],
        )
        assert not any(v.attribute_id == 5089 for v in result), (
            "Звуковая схема=2.0 without authoritative corroboration must be DROPPED"
        )

    # ── MUD CASE 3: Бязь on Nike tee (web_search) ────────────────────────────

    @pytest.mark.asyncio
    async def test_byaz_web_search_dropped_without_corroboration(self):
        """WEB_SEARCH says Материал=Бязь for Nike tee.
        No authoritative source confirms it → DROPPED."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        mud = _av(
            attr_id=4496, value="Бязь", source=Source.WEB_SEARCH, confidence=0.85,
            evidence="web snippet about a different product",
        )
        target = _target(4496, "Материал", type_="enum")
        ctx = _ctx("Футболка Nike Dri-FIT мужская", "Nike")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [mud], [target], ctx, [],
        )
        assert not any(v.attribute_id == 4496 for v in result), (
            "Бязь on Nike tee without authoritative corroboration must be DROPPED"
        )

    # ── CORROBORATION SURVIVAL: ozon_card confirms same value ────────────────

    @pytest.mark.asyncio
    async def test_spec_fill_survives_when_ozon_card_corroborates(self):
        """LLM says Интерфейс=Bluetooth for a headphone.
        ozon_card also has Интерфейс=Bluetooth → fill SURVIVES."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        llm_fill = _av(attr_id=4526, value="Bluetooth", source=Source.LLM_KNOWLEDGE)
        ozon_corr = _av(attr_id=4526, value="Bluetooth", source=Source.OZON_CARD, confidence=0.92)
        target = _target(4526, "Интерфейс", type_="enum")
        ctx = _ctx("Наушники JBL Live 660NC", "JBL")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [llm_fill, ozon_corr], [target], ctx, [],
        )
        # ozon_card is keep_as_is (not an inference source), llm_fill is corroborated
        attr_ids = [v.attribute_id for v in result]
        assert attr_ids.count(4526) == 2, (
            "Both the ozon_card fill and the corroborated llm_knowledge fill must survive"
        )

    @pytest.mark.asyncio
    async def test_spec_fill_survives_when_wb_card_corroborates(self):
        """Состав=100% Хлопок from LLM_KNOWLEDGE, corroborated by wb_card."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        llm_fill = _av(attr_id=4604, value="100% Хлопок", source=Source.LLM_KNOWLEDGE)
        wb_corr = _av(attr_id=4604, value="100% хлопок", source=Source.WB_CARD, confidence=0.90)
        target = _target(4604, "Состав материала", type_="text")
        ctx = _ctx("Футболка Levi's 501", "Levi's")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [llm_fill, wb_corr], [target], ctx, [],
        )
        # llm fill must survive (corroborated by wb_card via normalised match)
        assert any(v.attribute_id == 4604 and v.source == Source.LLM_KNOWLEDGE for v in result), (
            "LLM fill corroborated by wb_card must SURVIVE"
        )

    @pytest.mark.asyncio
    async def test_normalisation_ensures_case_ё_match(self):
        """Normalisation: LLM says «Ёлочка», icecat says «елочка» → same after normalise."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        llm_fill = _av(attr_id=777, value="Ёлочка", source=Source.LLM_KNOWLEDGE)
        icecat_corr = _av(attr_id=777, value="елочка", source=Source.ICECAT, confidence=0.95)
        target = _target(777, "Материал", type_="enum")
        ctx = _ctx("Ёлка декоративная", "Декор")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [llm_fill, icecat_corr], [target], ctx, [],
        )
        assert any(v.attribute_id == 777 and v.source == Source.LLM_KNOWLEDGE for v in result), (
            "ё→е normalisation must allow «Ёлочка» to match «елочка» from icecat"
        )

    # ── Two guess-prone sources agreeing is NOT corroboration ────────────────

    @pytest.mark.asyncio
    async def test_two_inference_sources_agreeing_is_not_corroboration(self):
        """LLM_KNOWLEDGE=Бязь AND WEB_SEARCH=Бязь for Nike tee.
        They agree but NEITHER is authoritative → BOTH dropped."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        llm_fill = _av(attr_id=4496, value="Бязь", source=Source.LLM_KNOWLEDGE)
        web_fill = _av(attr_id=4496, value="Бязь", source=Source.WEB_SEARCH)
        target = _target(4496, "Материал", type_="enum")
        ctx = _ctx("Футболка Nike", "Nike")

        orchestrator = PipelineOrchestrator()
        result = await orchestrator._run_llm_knowledge_adversarial_pass(
            [llm_fill, web_fill], [target], ctx, [],
        )
        assert not any(v.attribute_id == 4496 for v in result), (
            "LLM+WEB agreement on spec-class attr must NOT be treated as corroboration; "
            "both fills must be dropped"
        )

    # ── Non-spec fill goes to Gate B, not corroboration ──────────────────────

    @pytest.mark.asyncio
    async def test_non_spec_fill_routed_to_gate_b_not_corroboration(self):
        """A non-spec attr (style, season) must go through Gate B LLM verify,
        NOT the deterministic corroboration path."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        non_spec_fill = _av(
            attr_id=8888, value="Спортивный", source=Source.LLM_KNOWLEDGE, confidence=0.85,
        )
        target = _target(8888, "Стиль", type_="enum", allowed_values=["Спортивный", "Повседневный"])
        ctx = _ctx()

        gate_b_called_with: list = []

        async def _mock_gate_b(ctx, proposals, resolved_attrs=None, llm_manager=None):
            gate_b_called_with.extend(proposals)
            return {p[0] for p in proposals}  # confirm all

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_mock_gate_b,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [non_spec_fill], [target], ctx, [],
            )

        # Gate B must have been called with the non-spec fill
        assert any(p[0] == 8888 for p in gate_b_called_with), (
            "Non-spec fill must be routed to Gate B adversarial verify"
        )
        # And it survives when Gate B confirms
        assert any(v.attribute_id == 8888 for v in result)

    @pytest.mark.asyncio
    async def test_non_spec_fill_dropped_when_gate_b_retracts(self):
        """A non-spec attr retracted by Gate B is dropped (existing behaviour unchanged)."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        fill = _av(attr_id=9999, value="Повседневный", source=Source.LLM_KNOWLEDGE)
        target = _target(9999, "Стиль", type_="enum")
        ctx = _ctx()

        async def _retract_all(ctx, proposals, resolved_attrs=None, llm_manager=None):
            return set()

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_retract_all,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [fill], [target], ctx, [],
            )

        assert not any(v.attribute_id == 9999 for v in result)

    # ── Verbatim-anchored fills skip both tracks ──────────────────────────────

    @pytest.mark.asyncio
    async def test_verbatim_anchored_spec_fill_is_kept_as_is(self):
        """A spec-class fill with verbatim_gate evidence skips corroboration entirely
        (already gated by Gate A — no extra corroboration needed)."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        verbatim = _av(
            attr_id=4526, value="USB-C",
            source=Source.WEB_SEARCH,
            evidence="safe_enum:verbatim_gate: found «USB-C» in page text",
        )
        target = _target(4526, "Интерфейс", type_="enum")
        ctx = _ctx()

        gate_b_calls: list = []

        async def _track_gate_b(ctx, proposals, resolved_attrs=None, llm_manager=None):
            gate_b_calls.extend(proposals)
            return set()

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_track_gate_b,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [verbatim], [target], ctx, [],
            )

        # Verbatim fill survives (not routed to any gate)
        assert any(v.attribute_id == 4526 for v in result), (
            "Verbatim-anchored fill must be kept as-is, not sent to corroboration or Gate B"
        )
        assert not gate_b_calls, "Gate B must NOT be called for verbatim-anchored fills"

    # ── Authoritative sources passthrough unchanged ───────────────────────────

    @pytest.mark.asyncio
    async def test_authoritative_source_always_passes_through(self):
        """ozon_card, wb_card, icecat, description, pdf_datasheet fills are
        never touched by the adversarial pass."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        authoritative_fills = [
            _av(1, "Хлопок", source=Source.OZON_CARD),
            _av(2, "Нет", source=Source.WB_CARD),
            _av(3, "2.0", source=Source.ICECAT),
            _av(4, "100% Polyester", source=Source.DESCRIPTION),
            _av(5, "USB-C", source=Source.PDF_DATASHEET),
        ]
        targets = [_target(i, f"Attr{i}") for i in range(1, 6)]
        ctx = _ctx()

        gate_b_calls: list = []

        async def _track(ctx, proposals, resolved_attrs=None, llm_manager=None):
            gate_b_calls.extend(proposals)
            return set()

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_track,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                authoritative_fills, targets, ctx, [],
            )

        assert len(result) == 5, "All authoritative fills must pass through unchanged"
        assert not gate_b_calls, "Gate B must NOT be called for authoritative sources"
