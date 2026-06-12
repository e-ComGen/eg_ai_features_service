"""Unit tests for the extra_fields wiring in WebSearchSource.

Tests that sneakerhead extra_fields emitted by harvest_composition are correctly
wired through _emit_extra_fields_avs into AttributeValues, with:
  - enum gate: matched allowed option → emitted; no match → dropped
  - free-text target: verbatim value emitted
  - already-filled target → not overwritten
  - no matching target → skipped silently
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.sources.web_search_source import (
    WebSearchSource,
    _find_target_by_label,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_context(**kwargs) -> ExtractionContext:
    defaults = dict(
        product_id=1,
        product_name="Nike Air Max 90",
        category_id=100,
        brand="Nike",
        ozon_type_id=None,
    )
    defaults.update(kwargs)
    return ExtractionContext(**defaults)


def _make_target(
    tid: int,
    name: str,
    ttype: str = "enum",
    allowed_values: list[str] | None = None,
    semantic_type: str | None = None,
) -> TargetAttribute:
    return TargetAttribute(
        id=tid,
        name=name,
        type=ttype,
        allowed_values=allowed_values,
        semantic_type=semantic_type,
    )


# ---------------------------------------------------------------------------
# _find_target_by_label — unit tests for the label-matcher
# ---------------------------------------------------------------------------

class TestFindTargetByLabel:
    def test_exact_match(self):
        targets = [_make_target(1, "Пол"), _make_target(2, "Страна")]
        assert _find_target_by_label("пол", targets).id == 1

    def test_label_in_target_name(self):
        # "страна" ⊆ "Страна производства"
        targets = [_make_target(1, "Страна производства")]
        assert _find_target_by_label("страна", targets).id == 1

    def test_target_name_in_label(self):
        # target name "Пол" ⊆ label "пол покупателя"
        targets = [_make_target(1, "Пол")]
        assert _find_target_by_label("пол покупателя", targets).id == 1

    def test_no_match_returns_none(self):
        targets = [_make_target(1, "Цвет")]
        assert _find_target_by_label("размер", targets) is None

    def test_exact_wins_over_substring(self):
        # "цвет" exact should win over "основной цвет"
        targets = [_make_target(1, "Основной цвет"), _make_target(2, "Цвет")]
        result = _find_target_by_label("цвет", targets)
        assert result.id == 2


# ---------------------------------------------------------------------------
# _emit_extra_fields_avs — unit tests for the AV emitter
# ---------------------------------------------------------------------------

def _make_source() -> WebSearchSource:
    """Build a WebSearchSource with mocked dependencies (no real LLM/search)."""
    source = WebSearchSource.__new__(WebSearchSource)
    source._search = MagicMock()
    source._extractor = MagicMock()
    source._judge = MagicMock()
    source._strategy = MagicMock()
    source._summary_cache = {}
    source._browser_fetcher = None
    source._browser_fetcher_owned = False
    return source


class TestEmitExtraFieldsAvs:
    def test_free_text_target_emitted_verbatim(self):
        """Free-text (non-enum) targets accept verbatim value without gating."""
        source = _make_source()
        ctx = _make_context()
        targets = [_make_target(201, "Артикул", ttype="text")]
        avs = source._emit_extra_fields_avs(
            extra_fields={"Артикул": "DH2987-102"},
            effective_targets=targets,
            filled_ids=set(),
            context=ctx,
            site="sneakerhead.ru",
        )
        assert len(avs) == 1
        av = avs[0]
        assert av.attribute_id == 201
        assert av.value == "DH2987-102"
        assert av.source == Source.WEB_SEARCH
        assert "sneakerhead" in (av.evidence or "")
        assert av.value_id is None  # free-text → no value_id

    def test_enum_with_matching_allowed_value_emitted(self):
        """Enum target: value matching allowed_values is emitted (no type_id needed)."""
        source = _make_source()
        ctx = _make_context(ozon_type_id=None)  # no type_id — allowed_values path
        targets = [_make_target(
            101, "Пол", ttype="enum",
            allowed_values=["Мужской", "Женский", "Унисекс"],
        )]
        avs = source._emit_extra_fields_avs(
            extra_fields={"Пол": "Унисекс"},
            effective_targets=targets,
            filled_ids=set(),
            context=ctx,
            site="sneakerhead.ru",
        )
        assert len(avs) == 1
        assert avs[0].attribute_id == 101
        assert avs[0].value == "Унисекс"

    def test_enum_with_no_matching_allowed_value_dropped(self):
        """Enum target: value not in allowed_values (no fuzzy match) → dropped."""
        source = _make_source()
        ctx = _make_context(ozon_type_id=None)
        targets = [_make_target(
            101, "Пол", ttype="enum",
            allowed_values=["Мужской", "Женский"],
        )]
        # "Дракон" is not in allowed_values and won't fuzzy-match anything
        avs = source._emit_extra_fields_avs(
            extra_fields={"Пол": "Дракон"},
            effective_targets=targets,
            filled_ids=set(),
            context=ctx,
            site="sneakerhead.ru",
        )
        assert avs == []

    def test_already_filled_target_not_overwritten(self):
        """If target.id is in filled_ids the extra_field is silently skipped."""
        source = _make_source()
        ctx = _make_context(ozon_type_id=None)
        targets = [_make_target(
            101, "Пол", ttype="enum",
            allowed_values=["Мужской", "Женский", "Унисекс"],
        )]
        avs = source._emit_extra_fields_avs(
            extra_fields={"Пол": "Мужской"},
            effective_targets=targets,
            filled_ids={101},   # already filled
            context=ctx,
            site="sneakerhead.ru",
        )
        assert avs == []

    def test_no_matching_target_skipped(self):
        """Label with no matching target → empty result, no exception."""
        source = _make_source()
        ctx = _make_context()
        targets = [_make_target(300, "Размер", ttype="text")]
        avs = source._emit_extra_fields_avs(
            extra_fields={"Цвет": "Белый"},
            effective_targets=targets,
            filled_ids=set(),
            context=ctx,
            site="sneakerhead.ru",
        )
        assert avs == []

    def test_enum_with_type_id_calls_resolve_value_id(self):
        """Enum target + ozon_type_id → resolve_value_id called; value_id on AV."""
        source = _make_source()
        ctx = _make_context(ozon_type_id=999, category_id=100)
        targets = [_make_target(102, "Страна производства", ttype="enum")]

        with patch(
            "app.services.enrichment.sources.web_search_source.WebSearchSource"
            "._emit_extra_fields_avs",
            wraps=source._emit_extra_fields_avs,
        ):
            # Patch resolve_value_id inside the module namespace
            with patch(
                "app.services.enrichment.strategies.dictionaries.ozon_loader.resolve_value_id",
                return_value=42,
            ) as mock_resolve:
                avs = source._emit_extra_fields_avs(
                    extra_fields={"Страна": "Китай"},
                    effective_targets=targets,
                    filled_ids=set(),
                    context=ctx,
                    site="sneakerhead.ru",
                )

        # resolve_value_id should have been called
        mock_resolve.assert_called_once_with(100, 999, 102, "Китай")
        assert len(avs) == 1
        assert avs[0].value_id == 42
        assert avs[0].value == "Китай"

    def test_enum_with_type_id_no_resolve_dropped(self):
        """Enum target + ozon_type_id + resolve_value_id returns None → dropped."""
        source = _make_source()
        ctx = _make_context(ozon_type_id=999, category_id=100)
        targets = [_make_target(102, "Страна", ttype="enum")]

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.resolve_value_id",
            return_value=None,
        ):
            avs = source._emit_extra_fields_avs(
                extra_fields={"Страна": "НеизвестнаяСтрана"},
                effective_targets=targets,
                filled_ids=set(),
                context=ctx,
                site="sneakerhead.ru",
            )

        assert avs == []

    def test_empty_value_skipped(self):
        """Empty string values in extra_fields are ignored."""
        source = _make_source()
        ctx = _make_context()
        targets = [_make_target(201, "Артикул", ttype="text")]
        avs = source._emit_extra_fields_avs(
            extra_fields={"Артикул": ""},
            effective_targets=targets,
            filled_ids=set(),
            context=ctx,
            site="sneakerhead.ru",
        )
        assert avs == []

    def test_confidence_and_source_attribution(self):
        """Emitted AVs use WEB_SEARCH source and evidence includes site label."""
        source = _make_source()
        ctx = _make_context(ozon_type_id=None)
        targets = [_make_target(201, "Сезон", ttype="text")]
        avs = source._emit_extra_fields_avs(
            extra_fields={"Сезон": "Демисезон"},
            effective_targets=targets,
            filled_ids=set(),
            context=ctx,
            site="https://sneakerhead.ru/product/12345",
        )
        assert len(avs) == 1
        av = avs[0]
        assert av.source == Source.WEB_SEARCH
        assert av.confidence == pytest.approx(0.72)
        assert "Сезон" in (av.evidence or "")
        assert "verbatim" in (av.evidence or "")


# ---------------------------------------------------------------------------
# Integration smoke: _mine_composition_if_needed passes extra_fields through
# ---------------------------------------------------------------------------

class TestMineCompositionExtraFieldsIntegration:
    """Smoke test that _mine_composition_if_needed calls _emit_extra_fields_avs
    when harvest_composition returns extra_fields."""

    def test_extra_fields_propagated_from_harvest_result(self):
        source = _make_source()

        ctx = _make_context(ozon_type_id=None)
        # Targets include composition attrs (to trigger harvest) + Пол
        targets = [
            _make_target(4604, "Состав материала", ttype="text"),
            _make_target(301, "Пол", ttype="enum",
                         allowed_values=["Мужской", "Женский", "Унисекс"]),
        ]

        harvest_result = {
            "composition": "100% хлопок",
            "material": "хлопок",
            "source_url": "https://sneakerhead.ru/prod/1",
            "site": "sneakerhead.ru",
            "evidence": "100% хлопок",
            "route": "open",
            "extra_fields": {"Пол": "Унисекс"},
        }

        async def _run():
            # Patch where harvest_composition is defined; the local `from ... import`
            # inside _mine_composition_if_needed re-imports from this module each call.
            with patch(
                "app.services.enrichment.sources.multisite_composition.harvest_composition",
                new=AsyncMock(return_value=harvest_result),
            ):
                return await source._mine_composition_if_needed(
                    context=ctx,
                    effective_targets=targets,
                    already_filled=[],
                )

        avs = asyncio.run(_run())

        attr_ids = {av.attribute_id for av in avs}
        # Composition AV
        assert 4604 in attr_ids
        # Extra-fields AV for Пол
        assert 301 in attr_ids
        pol_av = next(av for av in avs if av.attribute_id == 301)
        assert pol_av.value == "Унисекс"
        assert pol_av.source == Source.WEB_SEARCH
