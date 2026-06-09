"""Unit tests for composition_extractor.

Tests:
  POSITIVE — real snippets must yield non-empty result containing material words.
  NEGATIVE — noise inputs must return [] (fail-closed).
  Brand verification — page_matches_brand logic.
  Wiring smoke — _mine_composition_if_needed emits / skips correctly.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.services.enrichment.composition_extractor import (
    extract_composition,
    normalize_material_en_ru,
    page_matches_brand,
    primary_material,
)


# ---------------------------------------------------------------------------
# Positive cases — must find composition
# ---------------------------------------------------------------------------

class TestExtractCompositionPositive:
    def test_ru_label_colon(self):
        html = "<p>Состав: 79% хлопок 21% полиэстер</p>"
        result = extract_composition(html)
        assert result, f"Expected composition, got: {result}"
        assert any("хлопок" in r for r in result)

    def test_ru_material_colon(self):
        html = "<div>Материал: 90% хлопок, 10% полиэстер</div>"
        result = extract_composition(html)
        assert result, f"Expected composition, got: {result}"
        assert any("хлопок" in r for r in result)

    def test_en_percentage_material(self):
        html = "<span>82% cotton, 18% polyester</span>"
        result = extract_composition(html)
        assert result, f"Expected composition, got: {result}"
        joined = " ".join(result).lower()
        assert "cotton" in joined or "хлопок" in joined

    def test_three_component_ru(self):
        html = """
        <table>
            <tr><td>Состав изделия</td><td>Хлопок 68%, Лиоцелл 31%, Эластан 1%</td></tr>
        </table>
        """
        result = extract_composition(html)
        assert result, f"Expected composition, got: {result}"
        joined = " ".join(result).lower()
        assert "хлопок" in joined or "cotton" in joined

    def test_fabric_label_en(self):
        html = "<p>Fabric: 95% cotton, 5% elastane</p>"
        result = extract_composition(html)
        assert result, f"Expected composition, got: {result}"
        joined = " ".join(result).lower()
        assert "cotton" in joined or "хлопок" in joined

    def test_composition_label_en(self):
        html = "<span>Composition: 100% polyester</span>"
        result = extract_composition(html)
        assert result, f"Expected composition, got: {result}"

    def test_material_with_dashes_separator(self):
        html = "<p>Состав - 80% хлопок / 20% полиэстер</p>"
        result = extract_composition(html)
        assert result, f"Expected composition, got: {result}"

    def test_multiline_table(self):
        """Typical kixbox.ru / brandshop.ru table format."""
        html = """
        <table class="product-properties">
          <tr><th>Состав</th><td>79% хлопок, 21% полиэстер</td></tr>
          <tr><th>Сезон</th><td>Демисезон</td></tr>
        </table>
        """
        result = extract_composition(html)
        assert result, f"Expected composition, got: {result}"

    def test_plaintext_inline(self):
        """Plain text without HTML tags."""
        text = "Состав: 95% хлопок, 5% эластан"
        result = extract_composition(text)
        assert result

    def test_ru_no_label_pct_plus_material(self):
        """Free-form pct+material without a label — should still find it."""
        html = "<p>80% хлопок, 20% полиэстер — отличный выбор</p>"
        result = extract_composition(html)
        assert result


# ---------------------------------------------------------------------------
# Negative cases — must return []
# ---------------------------------------------------------------------------

class TestExtractCompositionNegative:
    def test_discount_50_percent(self):
        html = "<p>скидка 50% на все товары!</p>"
        assert extract_composition(html) == []

    def test_minus_50_percent(self):
        html = "<div>-50% на вторую вещь</div>"
        assert extract_composition(html) == []

    def test_original_100_percent(self):
        html = "<p>100% оригинал. Гарантия подлинности.</p>"
        assert extract_composition(html) == []

    def test_installment_0_percent(self):
        html = "<span>рассрочка 0% на 12 месяцев</span>"
        assert extract_composition(html) == []

    def test_cashback_5_percent(self):
        html = "<p>кэшбэк 5% при оплате картой</p>"
        assert extract_composition(html) == []

    def test_size_50(self):
        html = "<p>размер 50, стандартная посадка</p>"
        assert extract_composition(html) == []

    def test_material_keyword_only_no_percent_no_label(self):
        """Word 'хлопок' with no % and no label must not be extracted."""
        html = "<p>Модный хлопок для летнего сезона.</p>"
        assert extract_composition(html) == []

    def test_sale_off_en(self):
        html = "<div>50% off on selected items</div>"
        assert extract_composition(html) == []

    def test_empty_string(self):
        assert extract_composition("") == []

    def test_no_material_content(self):
        html = "<p>Купите сейчас по лучшей цене! Только сегодня.</p>"
        assert extract_composition(html) == []

    def test_discount_in_same_string_as_material(self):
        """Noise word adjacent to % wins — reject."""
        html = "<p>скидка 30% хлопок</p>"
        assert extract_composition(html) == []

    def test_percentage_only_no_material(self):
        html = "<p>99.9% satisfaction guaranteed</p>"
        assert extract_composition(html) == []


# ---------------------------------------------------------------------------
# normalize_material_en_ru
# ---------------------------------------------------------------------------

class TestNormalizeMaterialEnRu:
    def test_cotton(self):
        assert normalize_material_en_ru("cotton") == "хлопок"

    def test_polyester(self):
        assert normalize_material_en_ru("polyester") == "полиэстер"

    def test_elastane(self):
        assert normalize_material_en_ru("elastane") == "эластан"

    def test_spandex(self):
        assert normalize_material_en_ru("spandex") == "эластан"

    def test_viscose(self):
        assert normalize_material_en_ru("viscose") == "вискоза"

    def test_nylon(self):
        assert normalize_material_en_ru("nylon") == "полиамид"

    def test_lyocell(self):
        assert normalize_material_en_ru("lyocell") == "лиоцелл"

    def test_tencel(self):
        assert normalize_material_en_ru("tencel") == "лиоцелл"

    def test_ru_passthrough(self):
        """Already-RU words should pass through unchanged."""
        assert normalize_material_en_ru("хлопок") == "хлопок"

    def test_unknown_passthrough(self):
        assert normalize_material_en_ru("unknownfiber") == "unknownfiber"

    def test_case_insensitive(self):
        assert normalize_material_en_ru("Cotton") == "хлопок"
        assert normalize_material_en_ru("POLYESTER") == "полиэстер"


# ---------------------------------------------------------------------------
# primary_material
# ---------------------------------------------------------------------------

class TestPrimaryMaterial:
    def test_picks_highest_pct(self):
        comps = ["79% хлопок, 21% полиэстер"]
        assert primary_material(comps) == "хлопок"

    def test_en_composition(self):
        comps = ["82% cotton, 18% polyester"]
        mat = primary_material(comps)
        # Should resolve to RU canonical
        assert mat in ("хлопок", "cotton")  # cotton maps to хлопок via vocab

    def test_empty(self):
        assert primary_material([]) is None

    def test_no_pct(self):
        assert primary_material(["хлопок"]) is None


# ---------------------------------------------------------------------------
# page_matches_brand
# ---------------------------------------------------------------------------

class TestPageMatchesBrand:
    def test_brand_in_url(self):
        assert page_matches_brand("<html></html>", "https://kixbox.ru/champion-hoodie", "Champion")

    def test_brand_in_title(self):
        html = "<html><head><title>Champion Reverse Weave Hoodie | Kixbox</title></head></html>"
        assert page_matches_brand(html, "https://kixbox.ru/product", "Champion")

    def test_brand_in_h1(self):
        html = "<html><body><h1>Champion Толстовка худи</h1></body></html>"
        assert page_matches_brand(html, "https://example.com/product", "Champion")

    def test_brand_in_body_text(self):
        html = "<html><body>" + ("x " * 500) + "<p>Champion hoodie состав хлопок</p></body></html>"
        assert page_matches_brand(html, "https://example.com/product", "Champion")

    def test_wrong_brand_rejected(self):
        html = "<html><head><title>Nike Air Max</title></head><body><p>Nike hoodie</p></body></html>"
        assert not page_matches_brand(html, "https://nike.com/product", "Champion")

    def test_no_brand_always_true(self):
        """No brand info → pass (caller uses low confidence)."""
        assert page_matches_brand("<html></html>", "https://example.com", None)


# ---------------------------------------------------------------------------
# Wiring smoke test: _mine_composition_if_needed
# ---------------------------------------------------------------------------

class TestMineCompositionWiring:
    """Smoke tests for the WebSearchSource._mine_composition_if_needed integration."""

    def _make_target(self, attr_id: int) -> MagicMock:
        t = MagicMock()
        t.id = attr_id
        return t

    def _make_context(self, brand: str = "Champion") -> MagicMock:
        ctx = MagicMock()
        ctx.product_name = "Толстовка худи Champion Reverse Weave"
        ctx.brand = brand
        ctx.category_id = 100
        ctx.ozon_type_id = 200
        ctx.product_id = 1
        ctx.llm_calls_so_far = 0
        ctx.languages = None
        return ctx

    def test_emits_4604_on_composition_found(self):
        """When mine_composition returns a hit, 4604 AV is emitted."""
        from app.services.enrichment.sources.web_search_source import WebSearchSource

        mock_producer = MagicMock()
        mock_producer.mine_composition = AsyncMock(return_value=["79% хлопок, 21% полиэстер"])
        mock_producer._use_serper = True

        source = WebSearchSource.__new__(WebSearchSource)
        source._search = mock_producer
        source._extractor = MagicMock()
        source._judge = MagicMock()
        source._strategy = MagicMock()
        source._summary_cache = {}

        ctx = self._make_context()
        target_4604 = self._make_target(4604)
        targets = [target_4604]

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.resolve_value_id",
            return_value=None,
        ):
            avs = asyncio.get_event_loop().run_until_complete(
                source._mine_composition_if_needed(ctx, targets, [])
            )

        assert any(av.attribute_id == 4604 for av in avs), f"Expected 4604 in {avs}"
        av_4604 = next(av for av in avs if av.attribute_id == 4604)
        assert "хлопок" in str(av_4604.value)

    def test_wrong_brand_page_emits_nothing(self):
        """When mine_composition finds nothing (brand filter rejected), return []."""
        from app.services.enrichment.sources.web_search_source import WebSearchSource

        mock_producer = MagicMock()
        mock_producer.mine_composition = AsyncMock(return_value=[])  # empty = brand-rejected
        mock_producer._use_serper = True

        source = WebSearchSource.__new__(WebSearchSource)
        source._search = mock_producer
        source._extractor = MagicMock()
        source._judge = MagicMock()
        source._strategy = MagicMock()
        source._summary_cache = {}

        ctx = self._make_context(brand="Champion")
        targets = [self._make_target(4604)]

        avs = asyncio.get_event_loop().run_until_complete(
            source._mine_composition_if_needed(ctx, targets, [])
        )
        assert avs == []

    def test_no_composition_targets_skips_mining(self):
        """When 4604/4496 not in targets, mine_composition is never called."""
        from app.services.enrichment.sources.web_search_source import WebSearchSource

        mock_producer = MagicMock()
        mock_producer.mine_composition = AsyncMock(return_value=["79% хлопок, 21% полиэстер"])

        source = WebSearchSource.__new__(WebSearchSource)
        source._search = mock_producer
        source._extractor = MagicMock()
        source._judge = MagicMock()
        source._strategy = MagicMock()
        source._summary_cache = {}

        ctx = self._make_context()
        # Target is some other attribute, not 4604/4496
        targets = [self._make_target(9999)]

        avs = asyncio.get_event_loop().run_until_complete(
            source._mine_composition_if_needed(ctx, targets, [])
        )
        assert avs == []
        mock_producer.mine_composition.assert_not_called()
