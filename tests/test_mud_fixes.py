"""Tests for the three MUD-plug fixes.

FIX #2(a): conf<=0.0 global hard-drop in _finalize_async.
FIX #2(b): Vision blocked from Материал/Состав by both id and name.
FIX #3:    Gate B adversarial verifier receives resolved attrs → retracts contradictions.
FIX #1:    llm_knowledge adversarial pass: mocked Gate B retracts mud, confirms good fills.

All tests use mocked LLM — no real API calls, deterministic.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.sources.safe_enum_fill_source import (
    SafeEnumFillSource,
    _ProposalResponse,
    _ProposedFill,
    _AdversarialResponse,
    _VerifiedItem,
    run_adversarial_verify,
)
from app.services.enrichment.sources.vision_source import (
    VisionSource,
    _VISION_BLOCKED_ATTR_IDS,
    _VISION_BLOCKED_NAME_FRAGMENTS,
)
from app.services.providers.structured_adapter import StructuredLlmManager


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ctx(
    product_name: str = "Футболка Nike",
    brand: str = "Nike",
    product_id: int = 1,
    category_path: list[str] | None = None,
    image_urls: list[str] | None = None,
) -> ExtractionContext:
    return ExtractionContext(
        product_id=product_id,
        product_name=product_name,
        category_id=100,
        category_path=category_path or ["Одежда", "Футболки"],
        brand=brand,
        image_urls=image_urls or [],
    )


def _target(
    attr_id: int,
    name: str,
    allowed_values: list[str] | None = None,
    is_required: bool = False,
    semantic_type: str | None = None,
) -> TargetAttribute:
    return TargetAttribute(
        id=attr_id,
        name=name,
        type="enum",
        allowed_values=allowed_values,
        is_required=is_required,
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


def _mock_llm(responses: list) -> StructuredLlmManager:
    mock = AsyncMock(spec=StructuredLlmManager)
    mock.structured_request.side_effect = [(r, 10) for r in responses]
    return mock


def _adversarial_response(items: list[dict]) -> _AdversarialResponse:
    return _AdversarialResponse(verifications=[_VerifiedItem(**i) for i in items])


# ──────────────────────────────────────────────────────────────────────────────
# FIX #2(a): conf <= 0.0 global hard-drop
# ──────────────────────────────────────────────────────────────────────────────


class TestConfZeroHardDrop:
    """conf<=0.0 values must be dropped BEFORE they reach the final filled set."""

    @pytest.mark.asyncio
    async def test_nike_abs_plastik_conf_zero_dropped(self):
        """Nike tee Материал=ABS пластик conf=0.0 evidence=«No material listed» → DROPPED."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        # Simulate what _finalize_async receives: a conf=0.0 Vision fill
        bad_av = _av(
            attr_id=4496,
            value="ABS пластик",
            source=Source.VISION,
            confidence=0.0,
            evidence="No material listed",
        )

        targets = [_target(4496, "Материал")]
        context = _ctx(product_name="Футболка Nike Dri-FIT")

        # Patch _finalize and llm_resolve_tail to be no-ops so we can inspect
        # what _finalize_async does with the conf=0.0 value.
        with (
            patch.object(orchestrator, "_finalize", return_value=[]),
            patch.object(
                orchestrator._strategy, "llm_resolve_tail",
                new=AsyncMock(return_value=[]),
            ),
            patch("app.services.enrichment.pipeline._apply_brand_from_title_llm",
                  new=AsyncMock(side_effect=lambda v, *a, **kw: v)),
        ):
            result = await orchestrator._finalize_async([bad_av], targets, context)

        # The conf=0.0 value must have been dropped before reaching _finalize
        assert result == [], f"Expected empty list but got: {result}"

    @pytest.mark.asyncio
    async def test_positive_conf_value_not_dropped(self):
        """A conf=0.85 fill must NOT be dropped by the conf<=0.0 guard."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()

        good_av = _av(attr_id=10, value="Синий", source=Source.LLM_KNOWLEDGE, confidence=0.85)
        targets = [_target(10, "Цвет")]
        context = _ctx()

        captured = []

        def _fake_finalize(vals, *args, **kwargs):
            captured.extend(vals)
            return []

        with (
            patch.object(orchestrator, "_finalize", side_effect=_fake_finalize),
            patch.object(
                orchestrator._strategy, "llm_resolve_tail",
                new=AsyncMock(return_value=[]),
            ),
            patch("app.services.enrichment.pipeline._apply_brand_from_title_llm",
                  new=AsyncMock(side_effect=lambda v, *a, **kw: v)),
        ):
            await orchestrator._finalize_async([good_av], targets, context)

        assert any(v.attribute_id == 10 for v in captured), \
            "conf=0.85 fill must reach _finalize"

    @pytest.mark.asyncio
    async def test_conf_exactly_zero_dropped(self):
        """Confidence exactly 0.0 is dropped."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        orchestrator = PipelineOrchestrator()
        zero_av = _av(attr_id=999, value="X", confidence=0.0)
        targets = [_target(999, "Test")]
        context = _ctx()

        captured = []

        def _fake_finalize(vals, *args, **kwargs):
            captured.extend(vals)
            return []

        with (
            patch.object(orchestrator, "_finalize", side_effect=_fake_finalize),
            patch.object(
                orchestrator._strategy, "llm_resolve_tail",
                new=AsyncMock(return_value=[]),
            ),
            patch("app.services.enrichment.pipeline._apply_brand_from_title_llm",
                  new=AsyncMock(side_effect=lambda v, *a, **kw: v)),
        ):
            await orchestrator._finalize_async([zero_av], targets, context)

        assert not any(v.attribute_id == 999 for v in captured), \
            "conf=0.0 must be dropped before _finalize"


# ──────────────────────────────────────────────────────────────────────────────
# FIX #2(b): Vision blocked from Материал/Состав by id AND name
# ──────────────────────────────────────────────────────────────────────────────


class TestVisionMaterialBlockedById:
    """Vision must not fill attr 4496 (Материал) or 4604 (Состав) by id."""

    def test_blocked_ids_present_in_constant(self):
        assert 4496 in _VISION_BLOCKED_ATTR_IDS
        assert 4604 in _VISION_BLOCKED_ATTR_IDS

    def test_is_applicable_false_for_attr_4496(self):
        """is_applicable returns False for Материал (id=4496) regardless of semantic_type."""
        from app.services.enrichment.sources.vision_source import VisionSource
        from app.services.enrichment.vision_producer import VisionProducer
        source = VisionSource(
            vision_producer=AsyncMock(spec=VisionProducer),
            extraction_manager=AsyncMock(spec=StructuredLlmManager),
        )
        ctx = _ctx(image_urls=["https://img.example.com/photo.jpg"])
        # semantic_type=material_visual is a VISUAL type — the old bug: it would pass
        # the semantic_type check but must now be blocked by id check.
        target = _target(4496, "Материал", semantic_type="material_visual")
        assert source.is_applicable(ctx, target) is False

    def test_is_applicable_false_for_attr_4604(self):
        from app.services.enrichment.sources.vision_source import VisionSource
        from app.services.enrichment.vision_producer import VisionProducer
        source = VisionSource(
            vision_producer=AsyncMock(spec=VisionProducer),
            extraction_manager=AsyncMock(spec=StructuredLlmManager),
        )
        ctx = _ctx(image_urls=["https://img.example.com/photo.jpg"])
        target = _target(4604, "Состав", semantic_type=None)
        assert source.is_applicable(ctx, target) is False

    def test_is_applicable_false_for_material_name(self):
        """Any attr with 'материал' in its name is blocked regardless of id."""
        from app.services.enrichment.sources.vision_source import VisionSource
        from app.services.enrichment.vision_producer import VisionProducer
        source = VisionSource(
            vision_producer=AsyncMock(spec=VisionProducer),
            extraction_manager=AsyncMock(spec=StructuredLlmManager),
        )
        ctx = _ctx(image_urls=["https://img.example.com/photo.jpg"])
        target = _target(9999, "Материал верха", semantic_type=None)
        assert source.is_applicable(ctx, target) is False

    def test_is_applicable_false_for_sostav_name(self):
        """Any attr with 'состав' in its name is blocked."""
        from app.services.enrichment.sources.vision_source import VisionSource
        from app.services.enrichment.vision_producer import VisionProducer
        source = VisionSource(
            vision_producer=AsyncMock(spec=VisionProducer),
            extraction_manager=AsyncMock(spec=StructuredLlmManager),
        )
        ctx = _ctx(image_urls=["https://img.example.com/photo.jpg"])
        target = _target(8888, "Состав материала", semantic_type=None)
        assert source.is_applicable(ctx, target) is False

    def test_is_applicable_false_for_podkladka_name(self):
        """Any attr with 'подкладк' in its name is blocked."""
        from app.services.enrichment.sources.vision_source import VisionSource
        from app.services.enrichment.vision_producer import VisionProducer
        source = VisionSource(
            vision_producer=AsyncMock(spec=VisionProducer),
            extraction_manager=AsyncMock(spec=StructuredLlmManager),
        )
        ctx = _ctx(image_urls=["https://img.example.com/photo.jpg"])
        target = _target(7777, "Материал подкладки", semantic_type=None)
        assert source.is_applicable(ctx, target) is False

    def test_is_applicable_true_for_color(self):
        """Color target is still applicable — guard should not over-block."""
        from app.services.enrichment.sources.vision_source import VisionSource
        from app.services.enrichment.vision_producer import VisionProducer
        source = VisionSource(
            vision_producer=AsyncMock(spec=VisionProducer),
            extraction_manager=AsyncMock(spec=StructuredLlmManager),
        )
        ctx = _ctx(image_urls=["https://img.example.com/photo.jpg"])
        target = _target(100, "Цвет", semantic_type="color")
        assert source.is_applicable(ctx, target) is True

    @pytest.mark.asyncio
    async def test_extract_drops_4496_from_output(self):
        """Even if the LLM emits attr_id=4496, the output chokepoint must drop it."""
        from app.services.enrichment.sources.vision_source import (
            VisionSource,
            _VisionExtractionResponse,
            _VisionExtractedAttr,
        )
        from app.services.enrichment.vision_producer import VisionProducer

        extracted_attr = _VisionExtractedAttr(
            attribute_id=4496,
            value="Хлопок",
            confidence=0.8,
            evidence="looks like cotton",
        )
        resp = _VisionExtractionResponse(extracted=[extracted_attr])

        mock_vision = AsyncMock(spec=VisionProducer)
        mock_vision.produce_description = AsyncMock(return_value="Cotton t-shirt.")

        mock_extractor = AsyncMock(spec=StructuredLlmManager)
        mock_extractor.structured_request = AsyncMock(return_value=(resp, 100))

        source = VisionSource(vision_producer=mock_vision, extraction_manager=mock_extractor)
        ctx = _ctx(image_urls=["https://img.example.com/photo.jpg"])
        # Include 4496 in the target list to test the output-filter path
        target = _target(4496, "Материал", semantic_type="material_visual")

        result = await source.extract(ctx, [target])
        assert result == [], f"Expected empty list but got: {result}"


# ──────────────────────────────────────────────────────────────────────────────
# FIX #3: Gate B receives resolved attrs → retracts contradictions
# ──────────────────────────────────────────────────────────────────────────────


class TestGateBContradictionRetract:
    """Gate B adversarial verifier must retract proposals contradicting resolved attrs."""

    @pytest.mark.asyncio
    async def test_windows_version_retracted_when_os_is_none(self):
        """Windows version proposed for a «Без ОС» laptop → Gate B retracts."""
        # Simulate: resolved attrs have OS=«Без ОС»
        resolved_os = _av(attr_id=4000, value="Без ОС", source=Source.OZON_CARD, confidence=0.95)

        # Windows version proposal
        proposals = [
            _ProposedFill(attribute_id=4001, value="Windows 11", reasoning="common OS"),
        ]
        target_by_id = {4001: _target(4001, "Версия Windows", ["Windows 11", "Windows 10"])}

        # Mock the adversarial LLM to retract (NOT_CONFIRMED).
        # _adversarial_verify makes exactly ONE LLM call (the verify call only).
        mock_resp = _adversarial_response([
            {"attribute_id": 4001, "verdict": "NOT_CONFIRMED"},
        ])
        mock_llm = _mock_llm([mock_resp])

        source = SafeEnumFillSource(llm_manager=mock_llm)
        confirmed = await source._adversarial_verify(
            _ctx(product_name="Ноутбук HP без ОС", brand="HP"),
            proposals,
            target_by_id,
            resolved_attrs=[resolved_os],
        )
        assert 4001 not in confirmed, "Windows 11 should be retracted when OS=Без ОС"

    @pytest.mark.asyncio
    async def test_hdd_count_retracted_when_hdd_zero(self):
        """HDD count=1 proposed when HDD capacity=0 → Gate B retracts."""
        resolved_hdd = _av(attr_id=4010, value="0", source=Source.WB_CARD, confidence=0.92)

        proposals = [
            _ProposedFill(attribute_id=4011, value="1", reasoning="default laptop"),
        ]
        target_by_id = {4011: _target(4011, "Кол-во HDD", ["0", "1", "2"])}

        mock_resp = _adversarial_response([
            {"attribute_id": 4011, "verdict": "NOT_CONFIRMED"},
        ])
        mock_llm = _mock_llm([mock_resp])

        source = SafeEnumFillSource(llm_manager=mock_llm)
        confirmed = await source._adversarial_verify(
            _ctx(product_name="Ноутбук Lenovo", brand="Lenovo"),
            proposals,
            target_by_id,
            resolved_attrs=[resolved_hdd],
        )
        assert 4011 not in confirmed

    @pytest.mark.asyncio
    async def test_resolved_context_passed_to_verifier(self):
        """Verify resolved_attrs are included in the verifier user_text block."""
        resolved = _av(attr_id=888, value="Без ОС", source=Source.OZON_CARD, confidence=0.90)
        proposals = [_ProposedFill(attribute_id=4001, value="Windows 11", reasoning="")]
        target_by_id = {4001: _target(4001, "Версия Windows", ["Windows 11"])}

        mock_resp = _adversarial_response([{"attribute_id": 4001, "verdict": "NOT_CONFIRMED"}])
        mock_llm = AsyncMock(spec=StructuredLlmManager)
        mock_llm.structured_request = AsyncMock(return_value=(mock_resp, 10))

        source = SafeEnumFillSource(llm_manager=mock_llm)
        await source._adversarial_verify(
            _ctx(product_name="Ноутбук", brand="HP"),
            proposals,
            target_by_id,
            resolved_attrs=[resolved],
        )

        # Inspect the user_text passed to LLM — it should contain the resolved attr
        call_kwargs = mock_llm.structured_request.call_args
        user_text = call_kwargs[1].get("user_text") or call_kwargs[0][1]
        assert "888" in user_text or "Без ОС" in user_text, \
            "Resolved attr must appear in adversarial verifier user_text"


# ──────────────────────────────────────────────────────────────────────────────
# FIX #1: LLM_KNOWLEDGE adversarial pass
# ──────────────────────────────────────────────────────────────────────────────


class TestLlmKnowledgeAdversarialPass:
    """Mud cases are dropped; known-good fills survive."""

    @pytest.mark.asyncio
    async def test_byaz_retracted_for_levis(self):
        """Levi's 501 Материал=Бязь conf=0.93 → adversarial RETRACTS → dropped."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        mud_fill = _av(attr_id=4496, value="Бязь", source=Source.LLM_KNOWLEDGE, confidence=0.93)
        targets = [_target(4496, "Материал")]
        context = _ctx(product_name="Джинсы Levi's 501", brand="Levi's")

        # Mock run_adversarial_verify to retract attr 4496.
        # The function is imported inside _run_llm_knowledge_adversarial_pass via local import;
        # patch at the source module path so the local import picks up the mock.
        async def _fake_verify(ctx, proposals, resolved_attrs=None, llm_manager=None):
            return set()  # retract all

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_fake_verify,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [mud_fill], targets, context, [],
            )

        assert not any(v.attribute_id == 4496 for v in result), \
            "Бязь fill must be retracted (dropped)"

    @pytest.mark.asyncio
    async def test_series3_model_retracted_for_series9_watch(self):
        """Apple Watch S9 Модель=Apple Watch Series 3 42mm → RETRACTS → dropped."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        mud_fill = _av(
            attr_id=7001,
            value="Apple Watch Series 3 42mm",
            source=Source.LLM_KNOWLEDGE,
            confidence=0.88,
        )
        targets = [_target(7001, "Модель")]
        context = _ctx(product_name="Apple Watch Series 9 41mm", brand="Apple")

        async def _retract_all(ctx, proposals, resolved_attrs=None, llm_manager=None):
            return set()

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_retract_all,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [mud_fill], targets, context, [],
            )

        assert not any(v.attribute_id == 7001 for v in result)

    @pytest.mark.asyncio
    async def test_sony_true_wireless_retracted(self):
        """Sony WH-1000XM5 True Wireless=Да → RETRACTS (over-ear, not TWS) → dropped."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        mud_fill = _av(attr_id=5555, value="Да", source=Source.LLM_KNOWLEDGE, confidence=0.85)
        targets = [_target(5555, "True Wireless")]
        context = _ctx(product_name="Sony WH-1000XM5 наушники", brand="Sony")

        async def _retract_all(ctx, proposals, resolved_attrs=None, llm_manager=None):
            return set()

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_retract_all,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [mud_fill], targets, context, [],
            )

        assert not any(v.attribute_id == 5555 for v in result)

    @pytest.mark.asyncio
    async def test_android_os_confirmed_for_samsung(self):
        """Samsung Galaxy OS=Android → adversarial CONFIRMS → fill SURVIVES."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        good_fill = _av(attr_id=3000, value="Android", source=Source.LLM_KNOWLEDGE, confidence=0.95)
        targets = [_target(3000, "Операционная система")]
        context = _ctx(product_name="Samsung Galaxy S23 смартфон", brand="Samsung")

        async def _confirm_all(ctx, proposals, resolved_attrs=None, llm_manager=None):
            return {p[0] for p in proposals}  # confirm all

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_confirm_all,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [good_fill], targets, context, [],
            )

        assert any(v.attribute_id == 3000 and v.value == "Android" for v in result), \
            "Android OS for Samsung Galaxy must SURVIVE the adversarial pass"

    @pytest.mark.asyncio
    async def test_color_from_card_confirmed(self):
        """Color fill from product card info → CONFIRMED → survives."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        good_fill = _av(attr_id=2000, value="Чёрный", source=Source.LLM_KNOWLEDGE, confidence=0.90)
        targets = [_target(2000, "Цвет")]
        context = _ctx(product_name="Смартфон Samsung Galaxy S23 чёрный", brand="Samsung")

        async def _confirm_all(ctx, proposals, resolved_attrs=None, llm_manager=None):
            return {p[0] for p in proposals}

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_confirm_all,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [good_fill], targets, context, [],
            )

        assert any(v.attribute_id == 2000 for v in result), \
            "Color fill must SURVIVE when adversarial confirms"

    @pytest.mark.asyncio
    async def test_non_llm_knowledge_fills_not_routed(self):
        """Only LLM_KNOWLEDGE fills go through the pass; others are kept as-is."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        card_fill = _av(attr_id=1000, value="Красный", source=Source.OZON_CARD, confidence=0.92)
        llm_fill = _av(attr_id=2000, value="Синий", source=Source.LLM_KNOWLEDGE, confidence=0.80)
        targets = [_target(1000, "Цвет1"), _target(2000, "Цвет2")]
        context = _ctx()

        verify_calls = []

        async def _track_verify(ctx, proposals, resolved_attrs=None, llm_manager=None):
            verify_calls.extend(proposals)
            return {p[0] for p in proposals}  # confirm all proposed

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_track_verify,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [card_fill, llm_fill], targets, context, [],
            )

        # card_fill must be in result (not touched by adversarial)
        assert any(v.attribute_id == 1000 for v in result)
        # Only the llm_fill was routed through verify
        assert all(p[0] == 2000 for p in verify_calls), \
            "Only LLM_KNOWLEDGE fills should be sent to adversarial verify"

    @pytest.mark.asyncio
    async def test_verbatim_anchored_fills_skipped(self):
        """Fills with safe_enum:verbatim_gate evidence skip the adversarial pass."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        verbatim_fill = _av(
            attr_id=3333,
            value="Повседневный",
            source=Source.LLM_KNOWLEDGE,
            confidence=0.82,
            evidence="safe_enum:verbatim_gate: found in card text",
        )
        targets = [_target(3333, "Стиль")]
        context = _ctx()

        verify_calls = []

        async def _track_verify(ctx, proposals, resolved_attrs=None, llm_manager=None):
            verify_calls.extend(proposals)
            return set()

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_track_verify,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [verbatim_fill], targets, context, [],
            )

        # Verbatim-anchored fill should NOT be sent to verify
        assert not verify_calls, "Verbatim-anchored fills must not go through adversarial verify"
        # And should STILL be in result (kept as-is)
        assert any(v.attribute_id == 3333 for v in result)


# ──────────────────────────────────────────────────────────────────────────────
# run_adversarial_verify (standalone, public function)
# ──────────────────────────────────────────────────────────────────────────────


class TestRunAdversarialVerifyStandalone:
    """Test the public run_adversarial_verify helper."""

    @pytest.mark.asyncio
    async def test_confirms_correct_fill(self):
        mock_resp = _adversarial_response([{"attribute_id": 1, "verdict": "CONFIRMED"}])
        mock_llm = _mock_llm([mock_resp])
        ctx = _ctx()

        confirmed = await run_adversarial_verify(
            ctx,
            [(1, "Цвет", "Чёрный")],
            llm_manager=mock_llm,
        )
        assert 1 in confirmed

    @pytest.mark.asyncio
    async def test_retracts_wrong_fill(self):
        mock_resp = _adversarial_response([{"attribute_id": 2, "verdict": "NOT_CONFIRMED"}])
        mock_llm = _mock_llm([mock_resp])
        ctx = _ctx()

        confirmed = await run_adversarial_verify(
            ctx,
            [(2, "Материал", "Бязь")],
            llm_manager=mock_llm,
        )
        assert 2 not in confirmed

    @pytest.mark.asyncio
    async def test_fail_closed_on_llm_error(self):
        mock_llm = AsyncMock(spec=StructuredLlmManager)
        mock_llm.structured_request.side_effect = RuntimeError("LLM down")
        ctx = _ctx()

        confirmed = await run_adversarial_verify(
            ctx,
            [(3, "Модель", "Series 3")],
            llm_manager=mock_llm,
        )
        assert confirmed == set(), "Must retract all when LLM fails (fail-closed)"

    @pytest.mark.asyncio
    async def test_empty_proposals_returns_empty(self):
        ctx = _ctx()
        confirmed = await run_adversarial_verify(ctx, [])
        assert confirmed == set()

    @pytest.mark.asyncio
    async def test_gate_b_prompt_does_not_contain_proposer_evidence(self):
        """FIX A: Gate B verify prompt must NOT include the proposer's own evidence string.

        Root cause of Яндекс Станция Мини 2 / «Звуковая схема=2.0» mud leak:
        the llm_knowledge fill carried evidence='stereo 2.0 according to official
        specifications' and Gate B received it → self-confirmation loop. The fix:
        proposals are (attr_id, attr_name, value) tuples — no evidence field.
        This test asserts the self-reported evidence text is absent from the prompt.
        """
        evidence_text = "stereo 2.0 according to official specifications"

        captured_user_text = []

        async def _mock_structured_request(system_prompt, user_text, response_model):
            captured_user_text.append(user_text)
            return (_adversarial_response([{"attribute_id": 777, "verdict": "CONFIRMED"}]), 10)

        mock_llm = AsyncMock(spec=StructuredLlmManager)
        mock_llm.structured_request.side_effect = _mock_structured_request

        ctx = _ctx(product_name="Яндекс Станция Мини 2", brand="Яндекс")
        await run_adversarial_verify(
            ctx,
            [(777, "Звуковая схема", "2.0")],
            llm_manager=mock_llm,
        )

        assert captured_user_text, "LLM must have been called"
        prompt = captured_user_text[0]
        assert evidence_text not in prompt, (
            f"Proposer evidence MUST NOT appear in Gate B verify prompt, "
            f"but found: {evidence_text!r} in prompt"
        )


# ──────────────────────────────────────────────────────────────────────────────
# FIX B: web_search fills covered by the adversarial pass
# ──────────────────────────────────────────────────────────────────────────────


class TestWebSearchAdversarialPass:
    """web_search fills must be routed through the adversarial Gate B pass.

    Root cause of «Бязь» on Nike tee (source=web_search): Gate A verbatim-check
    may have fired (word appears in some web snippet about a different product),
    but Gate B's «correct for THIS exact product» check would catch the wrong
    attribution. Extending the pass to WEB_SEARCH closes this leak.
    """

    @pytest.mark.asyncio
    async def test_web_search_mud_fill_is_dropped_when_retracted(self):
        """WEB_SEARCH Бязь on Nike tee → Gate B retracts → dropped."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        mud_fill = _av(
            attr_id=4496,
            value="Бязь",
            source=Source.WEB_SEARCH,
            confidence=0.85,
            evidence="web_search found бязь in snippet",
        )
        targets = [_target(4496, "Материал")]
        context = _ctx(product_name="Футболка Nike Dri-FIT мужская", brand="Nike")

        async def _retract_all(ctx, proposals, resolved_attrs=None, llm_manager=None):
            return set()  # retract everything

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_retract_all,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [mud_fill], targets, context, [],
            )

        assert not any(v.attribute_id == 4496 for v in result), (
            "WEB_SEARCH Бязь fill must be DROPPED when Gate B retracts"
        )

    @pytest.mark.asyncio
    async def test_web_search_good_fill_survives_when_confirmed(self):
        """WEB_SEARCH confirmed fill (e.g. correct color from web snippet) survives."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        good_fill = _av(
            attr_id=2000,
            value="Чёрный",
            source=Source.WEB_SEARCH,
            confidence=0.88,
            evidence="цвет чёрный, найдено на странице товара Nike Air Max 270",
        )
        targets = [_target(2000, "Цвет")]
        context = _ctx(product_name="Кроссовки Nike Air Max 270 чёрные", brand="Nike")

        async def _confirm_all(ctx, proposals, resolved_attrs=None, llm_manager=None):
            return {p[0] for p in proposals}

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_confirm_all,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [good_fill], targets, context, [],
            )

        assert any(v.attribute_id == 2000 and v.value == "Чёрный" for v in result), (
            "Good WEB_SEARCH fill must SURVIVE when Gate B confirms"
        )

    @pytest.mark.asyncio
    async def test_web_search_routed_alongside_llm_knowledge(self):
        """Both WEB_SEARCH and LLM_KNOWLEDGE non-spec fills are sent to Gate B together.

        NOTE: Uses non-spec attr names («Стиль», «Сезон») — spec-class attrs (material,
        bool, numeric, connectivity) now go through deterministic corroboration (Track A)
        instead of Gate B. This test validates Track B (Gate B) routing for non-spec attrs.
        """
        from app.services.enrichment.pipeline import PipelineOrchestrator

        llm_fill = _av(attr_id=1001, value="Спортивный", source=Source.LLM_KNOWLEDGE, confidence=0.85)
        web_fill = _av(attr_id=1002, value="Повседневный", source=Source.WEB_SEARCH, confidence=0.80)
        card_fill = _av(attr_id=1003, value="Синий", source=Source.OZON_CARD, confidence=0.95)

        targets = [
            _target(1001, "Стиль"),    # non-spec: no fragment match, type=enum → Gate B
            _target(1002, "Сезон"),    # non-spec: no fragment match, type=enum → Gate B
            _target(1003, "Цвет"),     # authoritative source → passthrough
        ]
        context = _ctx()

        verified_attr_ids: list[int] = []

        async def _track_and_confirm(ctx, proposals, resolved_attrs=None, llm_manager=None):
            verified_attr_ids.extend(p[0] for p in proposals)
            return {p[0] for p in proposals}  # confirm all

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_track_and_confirm,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [llm_fill, web_fill, card_fill], targets, context, [],
            )

        # Both non-spec inference fills must be routed to Gate B; card fill must NOT be routed
        assert 1001 in verified_attr_ids, "LLM_KNOWLEDGE non-spec fill must be sent to Gate B"
        assert 1002 in verified_attr_ids, "WEB_SEARCH non-spec fill must be sent to Gate B"
        assert 1003 not in verified_attr_ids, "OZON_CARD fill must NOT be sent to Gate B"

        # All three fills survive (card passthrough + both confirmed by mock)
        all_attr_ids = {v.attribute_id for v in result}
        assert all_attr_ids == {1001, 1002, 1003}

    @pytest.mark.asyncio
    async def test_web_search_verbatim_anchored_skipped(self):
        """WEB_SEARCH fills with verbatim_gate evidence skip adversarial (already gated)."""
        from app.services.enrichment.pipeline import PipelineOrchestrator

        verbatim_anchored = _av(
            attr_id=5000,
            value="Повседневный",
            source=Source.WEB_SEARCH,
            confidence=0.82,
            evidence="safe_enum:verbatim_gate: found in web text",
        )
        targets = [_target(5000, "Стиль")]
        context = _ctx()

        verify_calls: list = []

        async def _track_verify(ctx, proposals, resolved_attrs=None, llm_manager=None):
            verify_calls.extend(proposals)
            return set()  # retract all — but this fill should be skipped

        orchestrator = PipelineOrchestrator()
        with patch(
            "app.services.enrichment.sources.safe_enum_fill_source.run_adversarial_verify",
            new=_track_verify,
        ):
            result = await orchestrator._run_llm_knowledge_adversarial_pass(
                [verbatim_anchored], targets, context, [],
            )

        assert not verify_calls, "Verbatim-anchored WEB_SEARCH fill must skip Gate B"
        assert any(v.attribute_id == 5000 for v in result), (
            "Verbatim-anchored fill must be kept as-is"
        )
