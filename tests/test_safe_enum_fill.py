"""Tests for SafeEnumFillSource — gated LLM fill for short optional enum attrs.

ALL tests use mocked LLM calls (no network). The four critical test cases:
  (a) correct in-domain value with verbatim evidence → FILLED via Gate A.
  (b) Mud case: Levi's jeans Материал="Бязь" / Назначение="для дома" proposed
      at conf=0.95 (self-reported) → adversarial verifier RETRACTS → EMPTY.
  (c) Uncertain/generic guess, no evidence → EMPTY.
  (d) Correct famous-product spec the verifier confirms → FILLED via Gate B.
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
from app.services.enrichment.sources.safe_enum_fill_source import (
    SafeEnumFillSource,
    _ProposalResponse,
    _ProposedFill,
    _AdversarialResponse,
    _VerifiedItem,
    _verbatim_check,
    _is_short_enum,
    SAFE_ENUM_MAX_OPTIONS,
    _EVIDENCE_PREFIX_VERBATIM,
    _EVIDENCE_PREFIX_ADVERSARIAL,
)
from app.services.providers.structured_adapter import StructuredLlmManager


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────


def _ctx(
    product_name: str = "Product",
    brand: str = "Brand",
    product_id: int = 1,
) -> ExtractionContext:
    return ExtractionContext(
        product_id=product_id,
        product_name=product_name,
        category_id=100,
        category_path=["Одежда", "Джинсы"],
        brand=brand,
    )


def _target(
    attr_id: int,
    name: str,
    allowed_values: list[str],
    is_required: bool = False,
) -> TargetAttribute:
    return TargetAttribute(
        id=attr_id,
        name=name,
        type="enum",
        allowed_values=allowed_values,
        is_required=is_required,
    )


def _proposal_response(fills: list[dict]) -> _ProposalResponse:
    return _ProposalResponse(fills=[_ProposedFill(**f) for f in fills])


def _adversarial_response(items: list[dict]) -> _AdversarialResponse:
    return _AdversarialResponse(verifications=[_VerifiedItem(**i) for i in items])


def _mock_llm(responses: list) -> StructuredLlmManager:
    """Build a mock LLM that returns sequential (response, tokens) tuples."""
    mock = AsyncMock(spec=StructuredLlmManager)
    mock.structured_request.side_effect = [(r, 10) for r in responses]
    return mock


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests for helpers
# ──────────────────────────────────────────────────────────────────────────────


class TestVerbatimCheck:
    def test_phrase_match_whole_word(self):
        assert _verbatim_check("Повседневный", "Стиль: Повседневный")

    def test_phrase_match_cyrillic_multiword(self):
        assert _verbatim_check("для ежедневного", "Назначение: для ежедневного ношения")

    def test_false_when_not_present(self):
        assert not _verbatim_check("Бязь", "Состав: 100% хлопок, джинсовая ткань")

    def test_false_for_household_term_not_in_jeans_text(self):
        # "для дома" is a household term that would NOT appear in a jeans description
        source = "Джинсы прямого кроя, состав 99% хлопок 1% эластан, синий цвет"
        assert not _verbatim_check("для дома", source)

    def test_false_empty_source(self):
        assert not _verbatim_check("Повседневный", "")

    def test_false_empty_value(self):
        assert not _verbatim_check("", "some text with повседневный style")

    def test_case_insensitive(self):
        assert _verbatim_check("ПОВСЕДНЕВНЫЙ", "стиль ПОВСЕДНЕВНЫЙ")

    def test_yo_normalization(self):
        assert _verbatim_check("лёгкий", "лёгкий вес")


class TestIsShortEnum:
    def test_qualifies(self):
        t = _target(1001, "Стиль", ["Повседневный", "Спортивный", "Деловой"])
        assert _is_short_enum(t)

    def test_required_excluded(self):
        t = _target(1001, "Стиль", ["Повседневный", "Спортивный"], is_required=True)
        assert not _is_short_enum(t)

    def test_no_allowed_values_excluded(self):
        t = TargetAttribute(id=1, name="Описание", type="text")
        assert not _is_short_enum(t)

    def test_too_many_options_excluded(self):
        t = _target(1, "Бренд", [f"brand_{i}" for i in range(SAFE_ENUM_MAX_OPTIONS + 1)])
        assert not _is_short_enum(t)

    def test_brand_target_excluded(self):
        t = _target(31, "Бренд", ["Nike", "Adidas"])
        assert not _is_short_enum(t)

    def test_brand_by_name_excluded(self):
        t = _target(9999, "Торговая марка", ["Nike", "Adidas"])
        assert not _is_short_enum(t)


# ──────────────────────────────────────────────────────────────────────────────
# Integration-style tests: mocked LLM calls
# ──────────────────────────────────────────────────────────────────────────────


class TestSafeEnumFillSource:

    @pytest.mark.asyncio
    async def test_a_verbatim_gate_fills_value(self):
        """(a) Correct in-domain value with verbatim evidence → FILLED via Gate A."""
        style_target = _target(2001, "Стиль", ["Повседневный", "Спортивный", "Деловой"])
        ctx = _ctx("Джинсы Levi's 501 прямого кроя мужские", brand="Levi's")

        # LLM proposes "Повседневный" for Стиль
        proposal = _proposal_response([
            {"attribute_id": 2001, "value": "Повседневный", "reasoning": "classic jeans style"}
        ])
        # Adversarial should NOT be called (verbatim gate fires first)
        llm_mock = _mock_llm([proposal])

        source = SafeEnumFillSource(llm_manager=llm_mock)
        # Source text CONTAINS "Повседневный" — Gate A fires
        source_text = "Джинсы прямого кроя. Стиль: Повседневный. Состав: 99% хлопок."

        results = await source.extract(ctx, [style_target], source_text=source_text)

        assert len(results) == 1
        av = results[0]
        assert av.attribute_id == 2001
        assert av.value == "Повседневный"
        assert _EVIDENCE_PREFIX_VERBATIM in (av.evidence or "")
        # Only 1 LLM call (proposal); adversarial NOT called
        assert llm_mock.structured_request.call_count == 1

    @pytest.mark.asyncio
    async def test_b_mud_case_retracted_by_adversarial(self):
        """(b) Mud case: Бязь/для дома proposed at high conf → adversarial RETRACTS → EMPTY."""
        material_target = _target(4496, "Материал", ["Бязь", "Деним", "Вельвет", "Хлопок"])
        purpose_target = _target(5001, "Назначение", [
            "для дома", "для улицы", "для спорта", "для работы"
        ])
        ctx = _ctx("Джинсы мужские Levi's 501 Original прямые", brand="Levi's")

        # LLM proposes MUD: Бязь for Материал, для дома for Назначение
        proposal = _proposal_response([
            {"attribute_id": 4496, "value": "Бязь", "reasoning": "fabric"},
            {"attribute_id": 5001, "value": "для дома", "reasoning": "casual use"},
        ])
        # Adversarial: RETRACTS both — they are clearly wrong for Levi's 501 denim jeans
        adversarial = _adversarial_response([
            {"attribute_id": 4496, "verdict": "NOT_CONFIRMED"},
            {"attribute_id": 5001, "verdict": "NOT_CONFIRMED"},
        ])
        llm_mock = _mock_llm([proposal, adversarial])

        source = SafeEnumFillSource(llm_manager=llm_mock)
        # No source text → Gate A cannot fire → all go through Gate B
        results = await source.extract(
            ctx, [material_target, purpose_target], source_text=None
        )

        # Both MUD fills retracted → EMPTY
        assert results == [], f"Expected empty but got: {results}"
        # 2 LLM calls: proposal + adversarial (batched)
        assert llm_mock.structured_request.call_count == 2

    @pytest.mark.asyncio
    async def test_c_uncertain_guess_no_evidence_stays_empty(self):
        """(c) Uncertain/generic guess, no evidence → EMPTY."""
        season_target = _target(3001, "Сезон", ["Весна", "Лето", "Осень", "Зима"])
        ctx = _ctx("Носки хлопковые", brand=None)

        # LLM proposes "Лето" (generic guess for cotton socks)
        proposal = _proposal_response([
            {"attribute_id": 3001, "value": "Лето", "reasoning": "cotton is for summer"}
        ])
        # Adversarial retracts: socks are typically multi-season, not specifically summer
        adversarial = _adversarial_response([
            {"attribute_id": 3001, "verdict": "NOT_CONFIRMED"},
        ])
        llm_mock = _mock_llm([proposal, adversarial])

        source = SafeEnumFillSource(llm_manager=llm_mock)
        results = await source.extract(ctx, [season_target], source_text=None)

        assert results == []

    @pytest.mark.asyncio
    async def test_d_famous_product_spec_confirmed_by_adversarial(self):
        """(d) Correct famous-product spec the verifier confirms → FILLED via Gate B."""
        audience_target = _target(6001, "Целевая аудитория", [
            "Мужчины", "Женщины", "Дети", "Унисекс"
        ])
        ctx = _ctx("Джинсы мужские Levi's 501 Original прямые синие", brand="Levi's")

        # LLM proposes "Мужчины" — correct for men's jeans
        proposal = _proposal_response([
            {"attribute_id": 6001, "value": "Мужчины", "reasoning": "explicitly men's jeans"}
        ])
        # Adversarial CONFIRMS — "Мужчины" is unambiguous for men's product
        adversarial = _adversarial_response([
            {"attribute_id": 6001, "verdict": "CONFIRMED"},
        ])
        llm_mock = _mock_llm([proposal, adversarial])

        source = SafeEnumFillSource(llm_manager=llm_mock)
        # No source text → must go through Gate B
        results = await source.extract(ctx, [audience_target], source_text=None)

        assert len(results) == 1
        av = results[0]
        assert av.attribute_id == 6001
        assert av.value == "Мужчины"
        assert _EVIDENCE_PREFIX_ADVERSARIAL in (av.evidence or "")
        assert av.source == Source.SAFE_ENUM_FILL
        assert llm_mock.structured_request.call_count == 2  # proposal + adversarial

    @pytest.mark.asyncio
    async def test_proposal_value_not_in_allowed_values_dropped(self):
        """Safety: LLM proposes a value not in allowed_values → silently dropped."""
        target = _target(7001, "Стиль", ["Повседневный", "Спортивный"])
        ctx = _ctx("Кроссовки Nike Air Max", brand="Nike")

        # LLM hallucinates a value outside the enum
        proposal = _proposal_response([
            {"attribute_id": 7001, "value": "Высокотехнологичный", "reasoning": "Nike tech"}
        ])
        llm_mock = _mock_llm([proposal])

        source = SafeEnumFillSource(llm_manager=llm_mock)
        results = await source.extract(ctx, [target], source_text=None)

        # No valid proposals → adversarial never called, result empty
        assert results == []
        assert llm_mock.structured_request.call_count == 1

    @pytest.mark.asyncio
    async def test_required_targets_excluded(self):
        """Required targets must be ignored by SafeEnumFillSource."""
        required_target = _target(8001, "Тип", ["Джинсы", "Шорты"], is_required=True)
        ctx = _ctx("Джинсы Levi's 501", brand="Levi's")

        llm_mock = _mock_llm([])  # should never be called
        source = SafeEnumFillSource(llm_manager=llm_mock)

        results = await source.extract(ctx, [required_target], source_text=None)

        assert results == []
        assert llm_mock.structured_request.call_count == 0

    @pytest.mark.asyncio
    async def test_brand_target_excluded(self):
        """Brand targets must be excluded (handled by brand-from-name logic)."""
        brand_target = _target(31, "Бренд", ["Levi's", "Wrangler", "Lee"])
        ctx = _ctx("Джинсы мужские Levi's 501", brand="Levi's")

        llm_mock = _mock_llm([])
        source = SafeEnumFillSource(llm_manager=llm_mock)

        results = await source.extract(ctx, [brand_target], source_text=None)

        assert results == []
        assert llm_mock.structured_request.call_count == 0

    @pytest.mark.asyncio
    async def test_mixed_gate_outcomes(self):
        """Gate A fires for one attr (verbatim), Gate B fires for another (adversarial confirm)."""
        style_target = _target(2001, "Стиль", ["Повседневный", "Спортивный"])
        audience_target = _target(6001, "Целевая аудитория", ["Мужчины", "Женщины", "Унисекс"])
        ctx = _ctx("Джинсы мужские Levi's 501", brand="Levi's")

        proposal = _proposal_response([
            {"attribute_id": 2001, "value": "Повседневный", "reasoning": "classic style"},
            {"attribute_id": 6001, "value": "Мужчины", "reasoning": "men's product"},
        ])
        # Only audience_target needs adversarial (style was verbatim)
        adversarial = _adversarial_response([
            {"attribute_id": 6001, "verdict": "CONFIRMED"},
        ])
        llm_mock = _mock_llm([proposal, adversarial])

        source = SafeEnumFillSource(llm_manager=llm_mock)
        # Source text contains "Повседневный" but NOT "Мужчины" literally
        source_text = "Джинсы прямого кроя Повседневный стиль. Состав хлопок."

        results = await source.extract(
            ctx, [style_target, audience_target], source_text=source_text
        )

        assert len(results) == 2
        attr_ids = {r.attribute_id for r in results}
        assert attr_ids == {2001, 6001}

        style_av = next(r for r in results if r.attribute_id == 2001)
        audience_av = next(r for r in results if r.attribute_id == 6001)

        assert _EVIDENCE_PREFIX_VERBATIM in (style_av.evidence or "")
        assert _EVIDENCE_PREFIX_ADVERSARIAL in (audience_av.evidence or "")
        # 2 calls: proposal + adversarial (Gate A spared adversarial for style only)
        assert llm_mock.structured_request.call_count == 2

    @pytest.mark.asyncio
    async def test_llm_proposal_failure_returns_empty(self):
        """LLM proposal call fails → returns [] without crashing."""
        target = _target(2001, "Стиль", ["Повседневный", "Спортивный"])
        ctx = _ctx("Джинсы Levi's", brand="Levi's")

        llm_mock = AsyncMock(spec=StructuredLlmManager)
        llm_mock.structured_request.side_effect = RuntimeError("LLM connection error")

        source = SafeEnumFillSource(llm_manager=llm_mock)
        results = await source.extract(ctx, [target], source_text=None)

        assert results == []

    @pytest.mark.asyncio
    async def test_adversarial_failure_returns_empty(self):
        """Adversarial call fails → fails closed (returns [])."""
        target = _target(2001, "Стиль", ["Повседневный", "Спортивный"])
        ctx = _ctx("Джинсы Levi's", brand="Levi's")

        proposal = _proposal_response([
            {"attribute_id": 2001, "value": "Повседневный", "reasoning": "classic"}
        ])
        llm_mock = AsyncMock(spec=StructuredLlmManager)
        # First call (proposal) succeeds; second (adversarial) raises
        llm_mock.structured_request.side_effect = [
            (proposal, 10),
            RuntimeError("adversarial timeout"),
        ]

        source = SafeEnumFillSource(llm_manager=llm_mock)
        # No source_text → Gate A misses → Gate B (adversarial) runs
        results = await source.extract(ctx, [target], source_text=None)

        # Adversarial failed → nothing confirmed → empty (fail-closed)
        assert results == []
