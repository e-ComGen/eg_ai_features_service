"""Unit tests for icecat_numeric_normalizer.

Tests cover all observed real-world IceCat cases plus conservatism guards.
No LLM, no network — pure offline.
"""
from __future__ import annotations

import pytest

from app.services.enrichment.sources.icecat_numeric_normalizer import normalize_icecat_numeric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm(attr_name: str, raw: str) -> str:
    return normalize_icecat_numeric(attr_name, raw)


# ---------------------------------------------------------------------------
# Observed real cases from eval run
# ---------------------------------------------------------------------------

class TestTrailingUnitStrip:
    """Trailing bare SI unit stripped when attr name encodes the same unit."""

    def test_cord_length_m(self):
        """'1,2 m' for attr 'Длина шнура, м' → '1.2'."""
        result = norm("Длина шнура, м", "1,2 m")
        assert result == "1.2"

    def test_charge_time_min(self):
        """'50 min' for attr 'Время зарядки до 100%, мин' → '50'."""
        result = norm("Время зарядки до 100%, мин", "50 min")
        assert result == "50"

    def test_already_bare_number_unchanged(self):
        """'785' for attr with known unit → stays '785' (no unit to strip)."""
        result = norm("Частота обновления, Гц", "785")
        assert result == "785"

    def test_dimension_pair_mm_unchanged(self):
        """'0,2331 x 0,2331 mm' for мм attr → unchanged (multi-value guard)."""
        result = norm("Пиксельный шаг, мм", "0,2331 x 0,2331 mm")
        assert result == "0,2331 x 0,2331 mm"

    def test_incompatible_unit_value_passthrough(self):
        """'1.2 kg' for attr 'Длина шнура, м' → unchanged (no m←kg conversion)."""
        result = norm("Длина шнура, м", "1.2 kg")
        assert result == "1.2 kg"


class TestObservedRealCases:
    """The exact specimens from the task description."""

    def test_diagonal_cm_with_explicit_inch_notation(self):
        """'68,6 cm (27\")' for attr '..., дюймы' → '27'."""
        result = norm('Диагональ экрана, дюймы', '68,6 cm (27")')
        assert result == "27"

    def test_weight_kg_strip_unit(self):
        """'6,3 kg' for attr '..., кг' → '6.3'."""
        result = norm("Вес, кг", "6,3 kg")
        assert result == "6.3"

    def test_brightness_cd_m2_strip_unit(self):
        """'320 cd/m²' for attr '..., кд/м2' → '320'."""
        result = norm("Яркость, кд/м2", "320 cd/m²")
        assert result == "320"

    def test_refresh_rate_hz_strip_unit(self):
        """'165 Hz' for attr '..., Гц' → '165'."""
        result = norm("Макс. частота обновления, Гц", "165 Hz")
        assert result == "165"

    def test_response_time_ms_strip_unit(self):
        """'1 ms' for attr '..., мс' → '1'."""
        result = norm("Время отклика, мс", "1 ms")
        assert result == "1"

    def test_pixel_size_ambiguous_dimension_pair_unchanged(self):
        """'0,2331 x 0,2331 mm' → unchanged (multi-value guard)."""
        result = norm("Размер пикселя, мм", "0,2331 x 0,2331 mm")
        assert result == "0,2331 x 0,2331 mm"


# ---------------------------------------------------------------------------
# Additional conversion cases
# ---------------------------------------------------------------------------

class TestConversions:
    def test_diagonal_cm_no_inch_notation_converts(self):
        """'68.6 cm' for inch attr → convert cm to inch (≈ 27.0)."""
        result = norm("Диагональ экрана, дюймы", "68.6 cm")
        val = float(result)
        assert abs(val - 68.6 / 2.54) < 0.01

    def test_diagonal_mm_converts_to_inch(self):
        """'685.8 mm' for inch attr → convert mm to inch."""
        result = norm("Диагональ экрана, дюймы", "685.8 mm")
        val = float(result)
        assert abs(val - 685.8 / 25.4) < 0.1

    def test_weight_g_no_conversion_for_kg_attr(self):
        """'6300 g' for кг attr → convert g to kg → '6.3'."""
        result = norm("Вес, кг", "6300 g")
        val = float(result)
        assert abs(val - 6.3) < 0.01

    def test_comma_decimal_fix_no_unit(self):
        """'6,3' (bare number with comma) for кг attr → '6.3'."""
        result = norm("Вес, кг", "6,3")
        assert result == "6.3"


# ---------------------------------------------------------------------------
# Conservatism guards — should pass through unchanged
# ---------------------------------------------------------------------------

class TestConservatismGuards:
    def test_unknown_attr_unit_passthrough(self):
        """Attr name with no known unit → unchanged."""
        result = norm("Тип разъёма", "HDMI 2.1")
        assert result == "HDMI 2.1"

    def test_no_number_in_value_passthrough(self):
        """Value with no parseable number → unchanged."""
        result = norm("Частота обновления, Гц", "HDR")
        assert result == "HDR"

    def test_multiple_numbers_without_x_passthrough(self):
        """Value with two numbers not in dimension pair → unchanged."""
        result = norm("Вес, кг", "from 1.5 to 2.5 kg")
        # Two numbers found → conservatism → unchanged
        assert result == "from 1.5 to 2.5 kg"

    def test_unknown_source_unit_no_conversion_passthrough(self):
        """Value unit cm for attr кг (no cm→kg conversion) → unchanged."""
        result = norm("Вес, кг", "15 cm")
        assert result == "15 cm"

    def test_ambiguous_dimension_with_x_unchanged(self):
        """'10 x 20 mm' → unchanged regardless of attr."""
        result = norm("Размер, мм", "10 x 20 mm")
        assert result == "10 x 20 mm"

    def test_empty_value_passthrough(self):
        """Empty string → empty string."""
        result = norm("Вес, кг", "")
        assert result == ""

    def test_plain_text_passthrough(self):
        """Completely non-numeric text → unchanged."""
        result = norm("Вес, кг", "нет данных")
        assert result == "нет данных"


# ---------------------------------------------------------------------------
# Integration: normalization applies inside IceCatSource._map_features_to_targets
# ---------------------------------------------------------------------------

class TestIceCatSourceIntegration:
    """Verify normalize_icecat_numeric is called inside IceCatSource."""

    @pytest.mark.asyncio
    async def test_icecat_source_normalizes_weight(self):
        from unittest.mock import AsyncMock
        from app.services.enrichment.base import ExtractionContext, TargetAttribute, Source
        from app.services.enrichment.sources.icecat_source import IceCatSource

        source = IceCatSource(email="test@example.com", token="test-token")
        features = [("Вес, кг", "6,3 kg")]
        source._search_and_fetch = AsyncMock(return_value=features)

        context = ExtractionContext(
            product_id=1,
            product_name="Monitor ASUS VP279QGL",
            category_id=42,
            brand="ASUS",
        )
        targets = [TargetAttribute(id=10, name="Вес, кг", type="numeric")]
        results = await source.extract(context, targets)

        assert len(results) == 1
        assert results[0].value == "6.3"
        # Evidence preserves original raw value
        assert "6,3 kg" in results[0].evidence

    @pytest.mark.asyncio
    async def test_icecat_source_normalizes_diagonal(self):
        from unittest.mock import AsyncMock
        from app.services.enrichment.base import ExtractionContext, TargetAttribute
        from app.services.enrichment.sources.icecat_source import IceCatSource

        source = IceCatSource(email="test@example.com", token="test-token")
        features = [("Диагональ дисплея", '68,6 cm (27")')]
        source._search_and_fetch = AsyncMock(return_value=features)

        context = ExtractionContext(
            product_id=2,
            product_name="Monitor ASUS VP279QGL",
            category_id=42,
            brand="ASUS",
        )
        targets = [TargetAttribute(id=11, name="Диагональ экрана, дюймы", type="numeric")]
        results = await source.extract(context, targets)

        assert len(results) == 1
        assert results[0].value == "27"

    @pytest.mark.asyncio
    async def test_icecat_source_passthrough_ambiguous_pixel_size(self):
        from unittest.mock import AsyncMock
        from app.services.enrichment.base import ExtractionContext, TargetAttribute
        from app.services.enrichment.sources.icecat_source import IceCatSource

        source = IceCatSource(email="test@example.com", token="test-token")
        raw = "0,2331 x 0,2331 mm"
        features = [("Размер пикселя", raw)]
        source._search_and_fetch = AsyncMock(return_value=features)

        context = ExtractionContext(
            product_id=3,
            product_name="Monitor ASUS VP279QGL",
            category_id=42,
            brand="ASUS",
        )
        targets = [TargetAttribute(id=12, name="Размер пикселя, мм", type="text")]
        results = await source.extract(context, targets)

        assert len(results) == 1
        assert results[0].value == raw  # unchanged
