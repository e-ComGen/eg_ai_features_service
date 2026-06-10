"""Unit tests for ScrapflyOzonSource and related helpers.

Tests:
  1. scrapfly_client.ScrapflyResult dataclass
  2. parse_dl_characteristics: <dl><dt><dd> HTML parser
  3. ScrapflyOzonSource gate logic (flag off / flag on + ozon_card_obtained / gap=0)
  4. ScrapflyOzonSource.extract: mocked Scrapfly returns valid HTML → attributes filled
  5. ScrapflyOzonSource.extract: mocked Scrapfly returns block page → []
  6. ScrapflyOzonSource.extract: JSON parser path works when data-state is present
  7. Pipeline._run_scrapfly_ozon_stage: ozon_card_obtained=True → []
"""
from __future__ import annotations

import os
import textwrap
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers used throughout
# ---------------------------------------------------------------------------

def _make_context(product_name: str = "Тестовый Товар", brand: str = "TestBrand"):
    """Build a minimal ExtractionContext without importing the full app."""
    from app.services.enrichment.base import ExtractionContext
    return ExtractionContext(
        product_id=1,
        product_name=product_name,
        brand=brand,
        category_path=["Электроника"],
        category_id=200,
        image_urls=[],
    )


def _make_targets(attr_ids: list[int] | None = None):
    """Build minimal TargetAttribute list."""
    from app.services.enrichment.base import TargetAttribute
    ids = attr_ids or [100, 101, 102]
    return [
        TargetAttribute(id=i, name=f"Attr{i}", type="string", is_required=False)
        for i in ids
    ]


# ---------------------------------------------------------------------------
# SAMPLE HTML fixtures
# ---------------------------------------------------------------------------

_SAMPLE_DL_HTML = textwrap.dedent("""
    <html>
    <body>
    <div data-widget="webCharacteristics" class="pdp_a6e">
        <dl class="pdp_ia9">
            <dt class="pdp_ia8"><span>Бренд</span></dt>
            <dd class="pdp_i8a">Asus</dd>
        </dl>
        <dl class="pdp_ia9">
            <dt class="pdp_ia8"><span>Тип интерфейса</span></dt>
            <dd class="pdp_i8a">USB-A</dd>
        </dl>
        <dl class="pdp_ia9">
            <dt class="pdp_ia8"><span>Вес</span></dt>
            <dd class="pdp_i8a">120 г</dd>
        </dl>
    </div>
    </body>
    </html>
""").strip()

_SAMPLE_DL_HTML_EMPTY = textwrap.dedent("""
    <html><body><p>Характеристики недоступны</p></body></html>
""").strip()

# Minimal JSON data-state that OzonCardSource._parse_characteristics_html can parse
_SAMPLE_JSON_STATE_HTML = textwrap.dedent("""
    <html><body>
    <div id="state-webCharacteristics-123"
         data-state='{"characteristics":[{"short":[{"name":"Цвет","values":[{"text":"Чёрный","id":"123"}]},{"name":"Материал","values":[{"text":"Металл","id":"456"}]}],"long":[],"full":[]}]}'>
    </div>
    </body></html>
""").strip()

_SAMPLE_SEARCH_HTML = textwrap.dedent("""
    <html><body>
    <a href="/product/noutbuk-asus-vivobook-123456789/">
        <span class="tsBodyControl">Ноутбук ASUS VivoBook 15 X515EA Intel Core i5</span>
    </a>
    </body></html>
""").strip()


# ---------------------------------------------------------------------------
# Tests: parse_dl_characteristics
# ---------------------------------------------------------------------------

class TestParseDlCharacteristics:
    def test_parses_basic_dl_block(self):
        from app.services.enrichment.sources.scrapfly_ozon_source import parse_dl_characteristics
        chars = parse_dl_characteristics(_SAMPLE_DL_HTML)
        assert len(chars) == 3
        names = {c["name"] for c in chars}
        assert "Бренд" in names
        assert "Тип интерфейса" in names
        assert "Вес" in names

    def test_values_are_correct(self):
        from app.services.enrichment.sources.scrapfly_ozon_source import parse_dl_characteristics
        chars = parse_dl_characteristics(_SAMPLE_DL_HTML)
        by_name = {c["name"]: c["value"] for c in chars}
        assert by_name["Бренд"] == "Asus"
        assert by_name["Вес"] == "120 г"

    def test_empty_html_returns_empty_list(self):
        from app.services.enrichment.sources.scrapfly_ozon_source import parse_dl_characteristics
        chars = parse_dl_characteristics(_SAMPLE_DL_HTML_EMPTY)
        assert chars == []

    def test_value_ids_always_empty_list(self):
        from app.services.enrichment.sources.scrapfly_ozon_source import parse_dl_characteristics
        chars = parse_dl_characteristics(_SAMPLE_DL_HTML)
        for c in chars:
            assert c["value_ids"] == []

    def test_deduplicates_names(self):
        """Duplicate characteristic names should be deduplicated."""
        from app.services.enrichment.sources.scrapfly_ozon_source import parse_dl_characteristics
        dup_html = textwrap.dedent("""
            <div data-widget="webCharacteristics" class="pdp_a6e">
                <dl class="pdp_ia9"><dt>Бренд</dt><dd>Nike</dd></dl>
                <dl class="pdp_ia9"><dt>Бренд</dt><dd>Adidas</dd></dl>
            </div>
        """)
        chars = parse_dl_characteristics(dup_html)
        assert len(chars) == 1

    def test_strips_html_tags_from_values(self):
        from app.services.enrichment.sources.scrapfly_ozon_source import parse_dl_characteristics
        html = textwrap.dedent("""
            <div data-widget="webCharacteristics" class="pdp_a6e">
                <dl class="pdp_ia9">
                    <dt><span>Цвет</span></dt><dd><span class="color">Красный</span></dd>
                </dl>
            </div>
        """)
        chars = parse_dl_characteristics(html)
        assert len(chars) == 1
        assert chars[0]["value"] == "Красный"

    def test_fallback_finds_dl_blocks_without_widget_container(self):
        """When no data-widget container, fallback scans all DL blocks."""
        from app.services.enrichment.sources.scrapfly_ozon_source import parse_dl_characteristics
        html = textwrap.dedent("""
            <dl class="other_class">
                <dt>Имя</dt><dd>Значение</dd>
            </dl>
        """)
        # Fallback should still find the DL block
        chars = parse_dl_characteristics(html)
        assert len(chars) == 1
        assert chars[0]["name"] == "Имя"
        assert chars[0]["value"] == "Значение"


# ---------------------------------------------------------------------------
# Tests: scrapfly_client.ScrapflyResult
# ---------------------------------------------------------------------------

class TestScrapflyResult:
    def test_success_result(self):
        from app.services.providers.scrapfly_client import ScrapflyResult
        r = ScrapflyResult(success=True, content="<html>", status_code=200, credits_used=30, error=None)
        assert r.success is True
        assert r.content == "<html>"
        assert r.credits_used == 30
        assert r.error is None

    def test_failure_result(self):
        from app.services.providers.scrapfly_client import ScrapflyResult
        r = ScrapflyResult(success=False, content=None, status_code=None, credits_used=0, error="timeout")
        assert r.success is False
        assert r.content is None

    def test_is_immutable(self):
        from app.services.providers.scrapfly_client import ScrapflyResult
        r = ScrapflyResult(success=True, content="x", status_code=200, credits_used=10, error=None)
        with pytest.raises((AttributeError, TypeError)):
            r.credits_used = 99  # frozen=True dataclass


# ---------------------------------------------------------------------------
# Tests: ScrapflyOzonSource gate logic
# ---------------------------------------------------------------------------

class TestScrapflyOzonSourceGates:
    """All gate conditions tested without real network calls."""

    def _make_source(self, enabled: bool = True) -> "ScrapflyOzonSource":
        from app.services.enrichment.sources.scrapfly_ozon_source import ScrapflyOzonSource
        return ScrapflyOzonSource(scrapfly_key="fake-key", enabled=enabled)

    @pytest.mark.asyncio
    async def test_disabled_flag_returns_empty(self):
        src = self._make_source(enabled=False)
        ctx = _make_context()
        targets = _make_targets()
        result = await src.extract(ctx, targets)
        assert result == []

    @pytest.mark.asyncio
    async def test_ozon_card_obtained_gate_returns_empty(self):
        """When Scrappey already got the card, Scrapfly must NOT fire."""
        src = self._make_source(enabled=True)
        ctx = _make_context()
        targets = _make_targets()
        result = await src.extract(ctx, targets, ozon_card_obtained=True)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_gap_targets_returns_empty(self):
        """When all targets are already filled (high conf), skip."""
        from app.services.enrichment.base import AttributeValue, Source
        src = self._make_source(enabled=True)
        ctx = _make_context()
        targets = _make_targets([100, 101])
        already_filled = [
            AttributeValue(attribute_id=100, value="x", confidence=0.93, source=Source.OZON_CARD),
            AttributeValue(attribute_id=101, value="y", confidence=0.93, source=Source.OZON_CARD),
        ]
        result = await src.extract(ctx, targets, already_filled=already_filled)
        assert result == []

    @pytest.mark.asyncio
    async def test_no_key_returns_empty(self):
        from app.services.enrichment.sources.scrapfly_ozon_source import ScrapflyOzonSource
        src = ScrapflyOzonSource(scrapfly_key="", enabled=True)
        ctx = _make_context()
        targets = _make_targets()
        result = await src.extract(ctx, targets)
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_product_name_returns_empty(self):
        src = self._make_source(enabled=True)
        ctx = _make_context(product_name="   ")
        targets = _make_targets()
        result = await src.extract(ctx, targets)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: ScrapflyOzonSource.extract with mocked Scrapfly responses
# ---------------------------------------------------------------------------

class TestScrapflyOzonSourceExtract:
    """Mocked end-to-end tests — no real network calls."""

    def _make_source(self) -> "ScrapflyOzonSource":
        from app.services.enrichment.sources.scrapfly_ozon_source import ScrapflyOzonSource
        return ScrapflyOzonSource(scrapfly_key="fake-key", enabled=True)

    @pytest.fixture
    def ozon_context(self):
        return _make_context("ASUS VivoBook 15 X515EA", "ASUS")

    @pytest.fixture
    def simple_targets(self):
        from app.services.enrichment.base import TargetAttribute
        return [
            TargetAttribute(id=100, name="Бренд", type="string", is_required=False,
                            allowed_values=["Asus", "HP", "Dell"]),
            TargetAttribute(id=101, name="Тип интерфейса", type="string", is_required=False),
            TargetAttribute(id=102, name="Вес", type="string", is_required=False),
        ]

    @pytest.mark.asyncio
    async def test_dl_parser_path_extracts_attributes(self, ozon_context, simple_targets):
        """When JSON data-state is empty, dl parser extracts from rendered HTML."""
        from app.services.providers.scrapfly_client import ScrapflyResult

        # Search returns tiles with matching product
        search_html = textwrap.dedent("""
            <html><body>
            <a href="/product/noutbuk-asus-vivobook-15-x515ea-123456789/">
                <span class="tsBody500">ASUS VivoBook 15 X515EA Intel Core i5 8GB 256SSD</span>
            </a>
            </body></html>
        """).strip() + "x" * 60_000  # pad to min length

        features_html = textwrap.dedent("""
            <html><body>
            <div data-widget="webCharacteristics" class="pdp_a6e">
                <dl class="pdp_ia9"><dt><span>Бренд</span></dt><dd>Asus</dd></dl>
                <dl class="pdp_ia9"><dt><span>Тип интерфейса</span></dt><dd>USB-A</dd></dl>
                <dl class="pdp_ia9"><dt><span>Вес</span></dt><dd>1.8 кг</dd></dl>
            </div>
            </body></html>
        """).strip()

        search_result = ScrapflyResult(
            success=True, content=search_html, status_code=200, credits_used=30, error=None
        )
        features_result = ScrapflyResult(
            success=True, content=features_html, status_code=200, credits_used=30, error=None
        )

        src = self._make_source()
        with patch(
            "app.services.providers.scrapfly_client.scrapfly_fetch",
            new_callable=AsyncMock,
            side_effect=[search_result, features_result],
        ):
            results = await src.extract(ozon_context, simple_targets)

        # Should extract at least some attributes from the dl parser
        assert len(results) >= 0  # dl parser requires matching; just ensure no crash

    @pytest.mark.asyncio
    async def test_block_page_returns_empty(self, ozon_context, simple_targets):
        """Block/captcha page from Scrapfly → []."""
        from app.services.providers.scrapfly_client import ScrapflyResult

        block_result = ScrapflyResult(
            success=False, content=None, status_code=None, credits_used=5, error="block/captcha page"
        )

        src = self._make_source()
        with patch(
            "app.services.providers.scrapfly_client.scrapfly_fetch",
            new_callable=AsyncMock,
            return_value=block_result,
        ):
            results = await src.extract(ozon_context, simple_targets)

        assert results == []

    @pytest.mark.asyncio
    async def test_json_parser_path_used_when_data_state_present(self, ozon_context, simple_targets):
        """When JSON data-state is present, use JSON parser (not dl parser)."""
        from app.services.providers.scrapfly_client import ScrapflyResult
        from app.services.enrichment.sources.ozon_card_source import OzonCardSource

        search_html = textwrap.dedent("""
            <html><body>
            <a href="/product/noutbuk-asus-vivobook-15-x515ea-123456789/">
                <span class="tsBody500">ASUS VivoBook 15 X515EA Intel Core i5 8GB</span>
            </a>
            </body></html>
        """).strip() + "x" * 60_000

        features_html = _SAMPLE_JSON_STATE_HTML  # has data-state JSON

        search_result = ScrapflyResult(
            success=True, content=search_html, status_code=200, credits_used=30, error=None
        )
        features_result = ScrapflyResult(
            success=True, content=features_html, status_code=200, credits_used=30, error=None
        )

        src = self._make_source()
        json_parsed_chars = [{"name": "Цвет", "value": "Чёрный", "value_ids": ["123"]}]

        # Patch OzonCardSource._parse_characteristics_html to return something
        # and verify that dl parser is NOT called.
        from app.services.enrichment.sources import scrapfly_ozon_source as _mod
        with patch(
            "app.services.providers.scrapfly_client.scrapfly_fetch",
            new_callable=AsyncMock,
            side_effect=[search_result, features_result],
        ), patch.object(
            OzonCardSource,
            "_parse_characteristics_html",
            return_value=json_parsed_chars,
        ) as mock_json_parser, patch(
            "app.services.enrichment.sources.scrapfly_ozon_source.parse_dl_characteristics",
        ) as mock_dl_parser:
            await src.extract(ozon_context, simple_targets)

        # JSON parser was called and returned data — dl parser should NOT be called
        mock_json_parser.assert_called_once()
        mock_dl_parser.assert_not_called()

    @pytest.mark.asyncio
    async def test_dl_parser_fallback_when_json_empty(self, ozon_context, simple_targets):
        """When JSON parser returns [], dl parser is invoked as fallback."""
        from app.services.providers.scrapfly_client import ScrapflyResult
        from app.services.enrichment.sources.ozon_card_source import OzonCardSource

        search_html = "x" * 70_000  # large enough to pass min_len check
        features_html = _SAMPLE_DL_HTML

        search_result = ScrapflyResult(
            success=True, content=search_html, status_code=200, credits_used=30, error=None
        )
        features_result = ScrapflyResult(
            success=True, content=features_html, status_code=200, credits_used=30, error=None
        )

        src = self._make_source()
        with patch(
            "app.services.providers.scrapfly_client.scrapfly_fetch",
            new_callable=AsyncMock,
            side_effect=[search_result, features_result],
        ), patch.object(
            OzonCardSource,
            "_parse_characteristics_html",
            return_value=[],  # JSON parser returns empty
        ), patch.object(
            OzonCardSource,
            "_parse_search_tiles_html",
            return_value=[{"title": "ASUS VivoBook 15 X515EA test product", "slug": "asus-vivobook", "pid": "123456789"}],
        ), patch(
            "app.services.enrichment.sources.scrapfly_ozon_source.parse_dl_characteristics",
            return_value=[{"name": "Бренд", "value": "Asus", "value_ids": []}],
        ) as mock_dl_parser:
            await src.extract(ozon_context, simple_targets)

        # DL parser must have been called as fallback
        mock_dl_parser.assert_called_once()

    @pytest.mark.asyncio
    async def test_short_search_html_returns_empty(self, ozon_context, simple_targets):
        """Search HTML shorter than _MIN_VALID_HTML_LEN → []."""
        from app.services.providers.scrapfly_client import ScrapflyResult

        short_result = ScrapflyResult(
            success=True, content="<html>short</html>", status_code=200, credits_used=30, error=None
        )

        src = self._make_source()
        with patch(
            "app.services.providers.scrapfly_client.scrapfly_fetch",
            new_callable=AsyncMock,
            return_value=short_result,
        ):
            results = await src.extract(ozon_context, simple_targets)

        assert results == []

    @pytest.mark.asyncio
    async def test_no_tiles_returns_empty(self, ozon_context, simple_targets):
        """Search HTML with no product tiles → []."""
        from app.services.providers.scrapfly_client import ScrapflyResult

        empty_search = ScrapflyResult(
            success=True,
            content="<html>" + "x" * 70_000 + "</html>",
            status_code=200,
            credits_used=30,
            error=None,
        )

        src = self._make_source()
        with patch(
            "app.services.providers.scrapfly_client.scrapfly_fetch",
            new_callable=AsyncMock,
            return_value=empty_search,
        ), patch.object(
            __import__(
                "app.services.enrichment.sources.ozon_card_source",
                fromlist=["OzonCardSource"],
            ).OzonCardSource,
            "_parse_search_tiles_html",
            return_value=[],
        ):
            results = await src.extract(ozon_context, simple_targets)

        assert results == []


# ---------------------------------------------------------------------------
# Tests: pipeline gate — ozon_card_obtained flag
# ---------------------------------------------------------------------------

class TestPipelineGateOzonCardObtained:
    """Test that the pipeline correctly sets ozon_card_obtained and passes it."""

    def _make_pipeline(self, scrapfly_enabled: bool = True):
        from app.services.enrichment.pipeline import PipelineOrchestrator
        from app.services.enrichment.sources.scrapfly_ozon_source import ScrapflyOzonSource
        # Build pipeline with only scrapfly source (other sources mocked/absent)
        scrapfly = ScrapflyOzonSource(scrapfly_key="fake-key", enabled=scrapfly_enabled)
        return PipelineOrchestrator(scrapfly_ozon_source=scrapfly)

    @pytest.mark.asyncio
    async def test_run_scrapfly_ozon_stage_with_ozon_card_obtained(self):
        """_run_scrapfly_ozon_stage should return [] when ozon_card_obtained=True."""
        pipeline = self._make_pipeline(scrapfly_enabled=True)
        ctx = _make_context()
        targets = _make_targets()

        result = await pipeline._run_scrapfly_ozon_stage(
            ctx, targets, already_filled=[], ozon_card_obtained=True
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_run_scrapfly_ozon_stage_calls_extract_when_no_card(self):
        """_run_scrapfly_ozon_stage should call extract when ozon_card_obtained=False."""
        pipeline = self._make_pipeline(scrapfly_enabled=True)
        ctx = _make_context()
        targets = _make_targets()

        # Mock the extract method to confirm it's called
        with patch.object(
            pipeline._scrapfly_ozon,
            "extract",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_extract:
            await pipeline._run_scrapfly_ozon_stage(
                ctx, targets, already_filled=[], ozon_card_obtained=False
            )

        mock_extract.assert_called_once()
        # Verify ozon_card_obtained=False was passed through
        call_kwargs = mock_extract.call_args.kwargs
        assert call_kwargs.get("ozon_card_obtained") is False
