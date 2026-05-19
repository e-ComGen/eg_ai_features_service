"""Tests for app/services/enrichment/strategies/dictionaries/ozon_loader.py

All tests use mocks — no real API calls, no Playwright, no file I/O on the
real filesystem.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DICT = {
    "502": {
        "name": "Смартфоны",
        "path": ["Электроника", "Смартфоны"],
        "characteristics": [
            {"key": "brand", "name": "Бренд"},
            {"key": "color", "name": "Цвет"},
            {"key": "display_size", "name": "Диагональ экрана"},
        ],
    }
}


def _reset_cache() -> None:
    """Clear lru_cache on load_ozon_dictionary between tests."""
    from app.services.enrichment.strategies.dictionaries.ozon_loader import (
        load_ozon_dictionary,
    )
    load_ozon_dictionary.cache_clear()


# ---------------------------------------------------------------------------
# Tests: load_ozon_dictionary
# ---------------------------------------------------------------------------


class TestLoadOzonDictionary:
    def setup_method(self):
        _reset_cache()

    def test_load_when_no_file_returns_empty(self, tmp_path):
        """load_ozon_dictionary returns {} when ozon_dictionary.json does not exist."""
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                load_ozon_dictionary,
            )
            result = load_ozon_dictionary()
        assert result == {}

    def test_load_parses_valid_json(self, tmp_path):
        """load_ozon_dictionary correctly parses ozon_dictionary.json."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                load_ozon_dictionary,
            )
            result = load_ozon_dictionary()
        assert "502" in result
        assert result["502"]["name"] == "Смартфоны"
        assert result["502"]["path"] == ["Электроника", "Смартфоны"]

    def test_load_cached(self, tmp_path):
        """Repeated calls to load_ozon_dictionary do not re-read the file."""
        dict_path = tmp_path / "ozon_dictionary.json"
        dict_path.write_text(json.dumps(SAMPLE_DICT), encoding="utf-8")

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                load_ozon_dictionary,
            )
            original_read_text = Path.read_text
            call_count = {"n": 0}

            def counting_read_text(self, **kwargs):
                call_count["n"] += 1
                return original_read_text(self, **kwargs)

            with patch.object(Path, "read_text", counting_read_text):
                load_ozon_dictionary()
                load_ozon_dictionary()
                load_ozon_dictionary()

            # File should be read exactly once due to lru_cache
            assert call_count["n"] == 1


# ---------------------------------------------------------------------------
# Tests: get_ozon_characteristics_for_category
# ---------------------------------------------------------------------------


class TestGetOzonCharacteristicsForCategory:
    def setup_method(self):
        _reset_cache()

    def test_get_characteristics_unknown_category_empty(self, tmp_path):
        """Returns [] for a category_id that is not in the dictionary."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                get_ozon_characteristics_for_category,
            )
            result = get_ozon_characteristics_for_category(99999)
        assert result == []

    def test_get_characteristics_known_category(self, tmp_path):
        """Returns the correct characteristics list for a known category_id."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                get_ozon_characteristics_for_category,
            )
            result = get_ozon_characteristics_for_category(502)
        assert len(result) == 3
        assert result[0] == {"key": "brand", "name": "Бренд"}
        assert result[1] == {"key": "color", "name": "Цвет"}
        assert result[2] == {"key": "display_size", "name": "Диагональ экрана"}

    def test_get_characteristics_returns_empty_when_file_missing(self, tmp_path):
        """Returns [] when ozon_dictionary.json is absent (not yet built)."""
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                get_ozon_characteristics_for_category,
            )
            result = get_ozon_characteristics_for_category(502)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: get_ozon_category_name
# ---------------------------------------------------------------------------


class TestGetOzonCategoryName:
    def setup_method(self):
        _reset_cache()

    def test_returns_name_for_known_category(self, tmp_path):
        """get_ozon_category_name returns the category name string."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                get_ozon_category_name,
            )
            assert get_ozon_category_name(502) == "Смартфоны"

    def test_returns_none_for_unknown_category(self, tmp_path):
        """get_ozon_category_name returns None for an unknown category_id."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                get_ozon_category_name,
            )
            assert get_ozon_category_name(0) is None
