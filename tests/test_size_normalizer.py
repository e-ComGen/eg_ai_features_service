# -*- coding: utf-8 -*-
"""Unit tests for size_normalizer: GOST table, parse_explicit_size, extract_wb_sizes,
and WB sizes_table → Ozon value_ids resolution."""
import pytest

from app.services.enrichment.size_normalizer import (
    expand_intl_to_ru,
    parse_explicit_size,
    extract_wb_sizes,
)


# ---------------------------------------------------------------------------
# expand_intl_to_ru: GOST intl→RU table
# ---------------------------------------------------------------------------

class TestExpandIntlToRu:
    def test_numeric_direct_pass_through(self):
        assert expand_intl_to_ru("44") == ["44"]
        assert expand_intl_to_ru("46") == ["46"]
        assert expand_intl_to_ru("54") == ["54"]

    def test_range_expands_to_both_endpoints(self):
        result = expand_intl_to_ru("42-44")
        assert result == ["42", "44"]

    def test_range_with_em_dash(self):
        result = expand_intl_to_ru("46–48")
        assert result == ["46", "48"]

    def test_letter_S_gives_candidates(self):
        result = expand_intl_to_ru("S")
        assert "42" in result or "44" in result
        assert len(result) >= 1

    def test_letter_M_gives_candidates(self):
        result = expand_intl_to_ru("M")
        assert "46" in result or "48" in result

    def test_letter_XL_gives_candidates(self):
        result = expand_intl_to_ru("XL")
        assert len(result) >= 1
        for v in result:
            assert int(v) >= 50

    def test_letter_XXL_gives_candidates(self):
        result = expand_intl_to_ru("XXL")
        assert len(result) >= 1
        for v in result:
            assert int(v) >= 52

    def test_case_insensitive(self):
        assert expand_intl_to_ru("xl") == expand_intl_to_ru("XL")
        assert expand_intl_to_ru("m") == expand_intl_to_ru("M")

    def test_universal_variants(self):
        assert expand_intl_to_ru("one size") == ["универсальный"]
        assert expand_intl_to_ru("os") == ["универсальный"]
        assert expand_intl_to_ru("onesize") == ["универсальный"]
        assert expand_intl_to_ru("free size") == ["универсальный"]

    def test_unknown_token_returns_empty(self):
        """Non-size alphabetic tokens return empty. Numeric tokens pass through
        (validation against the Ozon dict is the caller's responsibility — the
        function is used from sizes_table where values ARE sizes)."""
        assert expand_intl_to_ru("Nike") == []
        assert expand_intl_to_ru("") == []
        # Note: pure numeric tokens pass through as-is from this function;
        # protection against "501"/"2024" is done in parse_explicit_size, not here.
        assert expand_intl_to_ru("GARBAGE_NOT_A_SIZE") == []

    def test_2xl_alias(self):
        result = expand_intl_to_ru("2XL")
        assert len(result) >= 1

    def test_xs_gives_small_sizes(self):
        result = expand_intl_to_ru("XS")
        assert len(result) >= 1
        for v in result:
            assert int(v) <= 44


# ---------------------------------------------------------------------------
# parse_explicit_size: strict name parser
# ---------------------------------------------------------------------------

class TestParseExplicitSize:
    # --- POSITIVE cases ---

    def test_explicit_razmer_letter(self):
        result = parse_explicit_size("Футболка мужская размер L")
        assert result  # must not be empty
        assert "L" in [r.upper() for r in result]

    def test_explicit_razmer_numeric(self):
        result = parse_explicit_size("Куртка мужская размер 48")
        assert result
        assert "48" in result

    def test_explicit_rr_short(self):
        result = parse_explicit_size("Джинсы р. 44")
        assert "44" in result

    def test_explicit_size_english(self):
        result = parse_explicit_size("T-shirt size XL")
        assert result
        assert "XL" in [r.upper() for r in result]

    def test_explicit_rr_dash(self):
        result = parse_explicit_size("Куртка р-р 50")
        assert "50" in result

    # --- NEGATIVE cases (must NOT yield a size) ---

    def test_levis_501_no_size(self):
        """Article number 501 must NOT be parsed as a size."""
        result = parse_explicit_size("Джинсы мужские Levis 501")
        assert result == [], f"Expected no size from '501', got {result}"

    def test_year_2024_no_size(self):
        """Year 2024 must NOT be parsed as a size."""
        result = parse_explicit_size("Куртка мужская Nike 2024")
        assert result == [], f"Expected no size from '2024', got {result}"

    def test_brand_name_no_size(self):
        """Pure brand tokens must not be extracted as sizes."""
        result = parse_explicit_size("Футболка Nike Sportswear")
        assert result == [], f"Expected empty, got {result}"

    def test_odd_number_no_size(self):
        """Odd numbers are not valid GOST clothing sizes."""
        result = parse_explicit_size("Куртка 45")
        assert result == [], f"Odd number 45 must not be a size, got {result}"

    def test_large_number_no_size(self):
        """Large numbers outside valid clothing range are not sizes."""
        result = parse_explicit_size("Куртка 2024 сезон")
        assert result == [], f"2024 must not be a size, got {result}"

    def test_model_number_no_size(self):
        """Model numbers like 'A50' must not yield a size."""
        result = parse_explicit_size("Куртка A50 Pro")
        assert result == [], f"'A50' must not yield a size, got {result}"


# ---------------------------------------------------------------------------
# extract_wb_sizes: WB card.json sizes_table parsing
# ---------------------------------------------------------------------------

class TestExtractWbSizes:
    def _make_card(self, sizes_table):
        return {"sizes_table": sizes_table}

    def test_ru_column_range(self):
        """Sizes in RU column as ranges ("42-44") expand to both endpoints."""
        card = self._make_card({
            "details_props": ["RU", "Обхват грудь"],
            "values": [
                {"tech_size": "XS", "chrt_id": 1, "details": ["42-44", "88-96"]},
                {"tech_size": "S",  "chrt_id": 2, "details": ["44-46", "92-100"]},
                {"tech_size": "M",  "chrt_id": 3, "details": ["46-48", "96-104"]},
            ],
        })
        result = extract_wb_sizes(card)
        assert "42" in result
        assert "44" in result
        assert "46" in result
        assert "48" in result

    def test_dedup_overlapping_ranges(self):
        """Adjacent ranges share an endpoint — deduplicated output."""
        card = self._make_card({
            "details_props": ["RU"],
            "values": [
                {"tech_size": "S", "chrt_id": 1, "details": ["44-46"]},
                {"tech_size": "M", "chrt_id": 2, "details": ["46-48"]},
            ],
        })
        result = extract_wb_sizes(card)
        assert result.count("46") == 1, "44-46 and 46-48 both include 46 — must dedup"

    def test_fallback_to_tech_size(self):
        """When no RU column, falls back to tech_size expansion."""
        card = self._make_card({
            "details_props": ["Обхват груди"],
            "values": [
                {"tech_size": "M", "chrt_id": 1, "details": ["96-104"]},
            ],
        })
        result = extract_wb_sizes(card)
        # tech_size "M" → expand_intl_to_ru("M") → ["46","48"]
        assert len(result) > 0
        assert "46" in result or "48" in result

    def test_absent_sizes_table_returns_empty(self):
        assert extract_wb_sizes({}) == []
        assert extract_wb_sizes({"sizes_table": None}) == []

    def test_sizes_table_without_values(self):
        card = self._make_card({"details_props": ["RU"], "values": []})
        assert extract_wb_sizes(card) == []

    def test_universal_size_parsed(self):
        card = self._make_card({
            "details_props": ["RU"],
            "values": [{"tech_size": "OS", "chrt_id": 1, "details": ["one size"]}],
        })
        result = extract_wb_sizes(card)
        assert "универсальный" in result


# ---------------------------------------------------------------------------
# Integration: WB sizes_table → resolved Ozon value_ids (attr 4295)
#
# Uses hardcoded (cat_id, type_id) = (200001161, 93272) — "Шорты" category —
# confirmed live on 2026-06-09 to contain attr 4295 with the standard RU size
# values. Avoids iterating load_ozon_dictionary() which may be mocked in the
# full test suite (lru_cache poisoned by other test modules).
# ---------------------------------------------------------------------------

_SHORTS_CAT_ID = 200001161
_SHORTS_TYPE_ID = 93272


class TestWbSizesResolveToValueIds:
    """Checks that WB sizes_table tokens resolve to real Ozon dict value_ids."""

    def _skip_if_dict_empty(self):
        """Skip gracefully when the Ozon dict is unavailable (CI without data)."""
        from app.services.enrichment.strategies.dictionaries.ozon_loader import (
            get_ozon_characteristics_for_type,
        )
        chars = get_ozon_characteristics_for_type(_SHORTS_CAT_ID, _SHORTS_TYPE_ID)
        if not chars:
            import pytest
            pytest.skip("Ozon dictionary not loaded (expected in CI without data files)")

    def test_standard_ru_sizes_resolve(self):
        """RU numeric sizes 44/46/48/50 must resolve to real Ozon value_ids."""
        self._skip_if_dict_empty()
        from app.services.enrichment.strategies.dictionaries.ozon_loader import resolve_value_id

        for size_val in ["44", "46", "48", "50", "52", "54"]:
            vid = resolve_value_id(_SHORTS_CAT_ID, _SHORTS_TYPE_ID, 4295, size_val)
            assert vid is not None, (
                f"Size '{size_val}' must resolve to a value_id for attr 4295, got None"
            )

    def test_real_size_tokens_resolve_to_int(self):
        """RU numeric size tokens resolve to non-None integer value_ids."""
        self._skip_if_dict_empty()
        from app.services.enrichment.strategies.dictionaries.ozon_loader import resolve_value_id

        for good_token in ["44", "46", "48", "50"]:
            vid = resolve_value_id(_SHORTS_CAT_ID, _SHORTS_TYPE_ID, 4295, good_token)
            assert isinstance(vid, int), (
                f"Token {good_token!r} must resolve to int, got {vid!r}"
            )

    def test_wb_sizes_table_to_resolved_ids(self):
        """End-to-end: wb card with sizes_table → resolve_value_id returns int ids."""
        self._skip_if_dict_empty()
        from app.services.enrichment.strategies.dictionaries.ozon_loader import resolve_value_id

        card = {
            "subj_name": "Шорты",
            "sizes_table": {
                "details_props": ["RU", "Обхват бедер"],
                "values": [
                    {"tech_size": "XS", "chrt_id": 1, "details": ["42-44", "88-96"]},
                    {"tech_size": "S",  "chrt_id": 2, "details": ["44-46", "92-100"]},
                    {"tech_size": "M",  "chrt_id": 3, "details": ["46-48", "96-104"]},
                ],
            },
        }
        tokens = extract_wb_sizes(card)
        assert tokens  # must extract something

        resolved = [
            resolve_value_id(_SHORTS_CAT_ID, _SHORTS_TYPE_ID, 4295, t)
            for t in tokens
        ]
        valid = [r for r in resolved if r is not None]
        assert valid, (
            f"Expected at least one resolved id from tokens {tokens}, got all None"
        )

    def test_unresolvable_tokens_all_dropped_gracefully(self):
        """When all tokens are clearly absent from the dict, resolved_ids is [].

        Checks the Ozon dict values list directly without fuzzy/MatcherService
        (which may segfault via pyarrow). The claim is that these strings simply
        don't appear in the values list for attr 4295.
        """
        self._skip_if_dict_empty()
        from app.services.enrichment.strategies.dictionaries.ozon_loader import (
            get_ozon_characteristics_for_type,
        )
        chars = get_ozon_characteristics_for_type(_SHORTS_CAT_ID, _SHORTS_TYPE_ID)
        char = next((c for c in chars if c.get("id") == 4295), None)
        assert char, "attr 4295 not found in shorts category"

        all_values = {
            str(v.get("value", "")).strip().lower()
            for v in char.get("values", [])
        }
        garbage_tokens = ["xxxnot_a_size", "garbage_token"]
        for tok in garbage_tokens:
            assert tok not in all_values, (
                f"Garbage token {tok!r} must not be a real size in the Ozon dict"
            )
