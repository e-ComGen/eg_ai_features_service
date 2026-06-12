"""Tests for EN→RU translation recovery in the RAG value path.

Covers:
  - _match_enum_value: "Black" translated to "чёрный" matches RU enum option.
  - _match_enum_value: "Yes" → "да" matches boolean enum.
  - _match_enum_value: gender variants → Ozon canon.
  - _normalize_free_text_rag: "Cotton" → translated to "хлопок" (not dropped).
  - _normalize_free_text_rag: "100% Polyester" no whole-token match → still dropped.
  - _normalize_free_text_rag: "ftwwht/cblack" raw SKU code → still dropped.
  - _normalize_free_text_rag: existing RU values unchanged.
  - _normalize_free_text_rag: measurement-like values unchanged.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from app.services.enrichment.sources.competitor_rag_source import (
    _normalize_free_text_rag,
    _match_enum_value,
    _translate_for_enum,
)


# ---------------------------------------------------------------------------
# _normalize_free_text_rag
# ---------------------------------------------------------------------------

class TestNormalizeFreeTextRag:
    def test_cotton_translates_to_ru(self):
        """'Cotton' is a whole-token EN→RU key → translated to 'хлопок', not dropped."""
        result = _normalize_free_text_rag("Cotton")
        assert result == "хлопок", f"Expected 'хлопок', got {result!r}"

    def test_cotton_lowercase(self):
        """Case insensitive: 'cotton' also translates."""
        result = _normalize_free_text_rag("cotton")
        assert result == "хлопок"

    def test_polyester_single_token_translates(self):
        """'Polyester' → 'полиэстер' (single-token match in table)."""
        result = _normalize_free_text_rag("Polyester")
        assert result == "полиэстер"

    def test_100_percent_polyester_no_whole_token_match_dropped(self):
        """'100% Polyester' is NOT a whole-token key → no match → still dropped."""
        result = _normalize_free_text_rag("100% Polyester")
        # Not in _EN_TO_RU_VALUES as a whole key → None
        assert result is None

    def test_raw_colorway_code_still_dropped(self):
        """'ftwwht/cblack/solred' must still be dropped (raw SKU code, step 2 fires first)."""
        assert _normalize_free_text_rag("ftwwht/cblack/solred") is None

    def test_raw_code_with_dash_dropped(self):
        """'ABCD-123/XY' raw code → dropped."""
        assert _normalize_free_text_rag("ABCD-123/XY") is None

    def test_ru_value_unchanged(self):
        """Existing RU value 'красный' passes through unchanged."""
        result = _normalize_free_text_rag("красный")
        assert result == "красный"

    def test_ru_value_with_spaces_cleaned(self):
        """RU value with extra spaces is collapsed but not dropped."""
        result = _normalize_free_text_rag("  хлопок  ")
        assert result == "хлопок"

    def test_measurement_like_kept(self):
        """'750W' is measurement-like → kept, not dropped."""
        result = _normalize_free_text_rag("750W")
        assert result == "750W"

    def test_measurement_2_4ghz_kept(self):
        """'2.4GHz' is measurement-like → kept."""
        result = _normalize_free_text_rag("2.4GHz")
        assert result == "2.4GHz"

    def test_plain_english_non_dict_dropped(self):
        """'UK 6' is non-Cyrillic, non-measurement, not in table → dropped."""
        assert _normalize_free_text_rag("UK 6") is None

    def test_aluminium_translates(self):
        """'Aluminium' is in table → 'алюминий'."""
        result = _normalize_free_text_rag("Aluminium")
        assert result == "алюминий"

    def test_empty_string_returns_none(self):
        assert _normalize_free_text_rag("") is None

    def test_whitespace_only_returns_none(self):
        assert _normalize_free_text_rag("   ") is None


# ---------------------------------------------------------------------------
# _translate_for_enum
# ---------------------------------------------------------------------------

class TestTranslateForEnum:
    def test_black_to_cherny(self):
        result = _translate_for_enum("Black")
        # _normalize_token lowercases and translates; ё→е applied → "черный"
        assert "черн" in result.lower() or "чёрн" in result.lower()

    def test_yes_to_da(self):
        result = _translate_for_enum("Yes")
        assert "да" in result.lower()

    def test_gender_mens_to_muzhskoy(self):
        """'men's' maps to Ozon gender canon 'Мужской'."""
        result = _translate_for_enum("men's")
        assert result == "Мужской"

    def test_gender_womens_to_zhenskiy(self):
        result = _translate_for_enum("women")
        assert result == "Женский"

    def test_gender_boys(self):
        result = _translate_for_enum("boys")
        assert result == "Мальчики"

    def test_ru_value_passes_through(self):
        """RU value 'красный' is not corrupted by translation."""
        result = _translate_for_enum("красный")
        # After _normalize_token: lower, ё→е, strip punct.  "красный" → "красный".
        assert "красн" in result.lower()

    def test_aluminium_translates(self):
        result = _translate_for_enum("Aluminium")
        assert "алюмин" in result.lower()


# ---------------------------------------------------------------------------
# _match_enum_value  — integration with actual find_best_match mocked out
# ---------------------------------------------------------------------------

class TestMatchEnumValueTranslation:
    """These tests mock find_best_match to assert the TRANSLATED value is passed."""

    def test_black_translated_before_matching(self):
        """'Black' must be translated to a RU form before find_best_match is called."""
        ru_options = ["Чёрный", "Белый", "Красный"]
        calls = []

        def fake_find_best_match(target, options):
            calls.append(target)
            # Simulate: if 'черн' in target → return 'Чёрный'
            if "черн" in target.lower() or "чёрн" in target.lower():
                return "Чёрный"
            return None

        fake_matcher = MagicMock()
        fake_matcher.find_best_match = fake_find_best_match

        with patch(
            "app.services.enrichment.sources.competitor_rag_source"
            "._translate_for_enum",
            wraps=_translate_for_enum,
        ), patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.get_matcher",
            return_value=fake_matcher,
        ):
            result = _match_enum_value("Black", ru_options)

        assert result == "Чёрный", f"Expected 'Чёрный', got {result!r}"
        # Translated value was used (not raw "Black")
        assert any("черн" in c.lower() or "чёрн" in c.lower() for c in calls), (
            f"Expected translated (RU) value in calls, got: {calls}"
        )

    def test_yes_translated_before_matching(self):
        """'Yes' → 'да' before matching boolean enum."""
        bool_options = ["Да", "Нет"]
        calls = []

        def fake_find(target, options):
            calls.append(target)
            if target.lower() == "да":
                return "Да"
            return None

        fake_matcher = MagicMock()
        fake_matcher.find_best_match = fake_find

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.get_matcher",
            return_value=fake_matcher,
        ):
            result = _match_enum_value("Yes", bool_options)

        assert result == "Да"
        assert "да" in calls[0].lower()

    def test_raw_ru_value_still_matches(self):
        """Existing RU values like 'Красный' are not broken by translation."""
        options = ["Красный", "Синий"]

        def fake_find(target, options):
            # After _normalize_token: "красный" (ё→е)
            if "красн" in target.lower():
                return "Красный"
            return None

        fake_matcher = MagicMock()
        fake_matcher.find_best_match = fake_find

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.get_matcher",
            return_value=fake_matcher,
        ):
            result = _match_enum_value("Красный", options)

        assert result == "Красный"

    def test_untranslatable_falls_back_to_raw(self):
        """Value not in EN→RU table: translated == normalized raw → still tried once."""
        options = ["XL", "L", "M"]
        calls = []

        def fake_find(target, options):
            calls.append(target)
            return "XL" if "xl" in target.lower() else None

        fake_matcher = MagicMock()
        fake_matcher.find_best_match = fake_find

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.get_matcher",
            return_value=fake_matcher,
        ):
            result = _match_enum_value("XL", options)

        # Result may be "XL" (raw matched) — not None
        assert result == "XL"

    def test_no_matcher_translated_exact_match(self):
        """When matcher is None, translated form is tried for exact match first."""
        options = ["Да", "Нет"]
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.get_matcher",
            return_value=None,
        ):
            result = _match_enum_value("Yes", options)
        # "Yes" → translated "да" → case-insensitive exact match against "Да"
        assert result == "Да"

    def test_no_matcher_ru_exact_match_unchanged(self):
        """When matcher is None and value is RU, exact match still works."""
        options = ["Да", "Нет"]
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.get_matcher",
            return_value=None,
        ):
            result = _match_enum_value("Да", options)
        assert result == "Да"
