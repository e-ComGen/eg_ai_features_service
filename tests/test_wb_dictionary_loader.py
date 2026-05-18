"""Tests for app/services/enrichment/strategies/dictionaries/loader.py

All tests use mocks — no real API calls are made.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DICT = {
    "12345": {
        "name": "Кроссовки мужские",
        "path": ["Обувь", "Мужская обувь", "Кроссовки"],
        "characteristics": [
            {"id": 14177419, "name": "Цвет"},
            {"id": 14177421, "name": "Материал верха"},
        ],
    }
}


def _reset_cache():
    """Clear lru_cache on load_wb_dictionary between tests."""
    from app.services.enrichment.strategies.dictionaries.loader import load_wb_dictionary
    load_wb_dictionary.cache_clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadWbDictionary:
    def setup_method(self):
        _reset_cache()

    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        """load_wb_dictionary returns {} when wb_dictionary.json does not exist."""
        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                load_wb_dictionary,
            )
            result = load_wb_dictionary()
        assert result == {}

    def test_returns_parsed_json_when_file_exists(self, tmp_path):
        """load_wb_dictionary parses and returns the JSON file contents."""
        (tmp_path / "wb_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                load_wb_dictionary,
            )
            result = load_wb_dictionary()
        assert "12345" in result
        assert result["12345"]["name"] == "Кроссовки мужские"

    def test_load_dictionary_is_cached(self, tmp_path):
        """Repeated calls to load_wb_dictionary do not re-read the file."""
        dict_path = tmp_path / "wb_dictionary.json"
        dict_path.write_text(json.dumps(SAMPLE_DICT), encoding="utf-8")

        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                load_wb_dictionary,
            )
            with patch("builtins.open", wraps=open) as mock_open:
                # We patch Path.read_text instead to count file reads
                original_read_text = Path.read_text

                call_count = {"n": 0}

                def counting_read_text(self, **kwargs):
                    call_count["n"] += 1
                    return original_read_text(self, **kwargs)

                with patch.object(Path, "read_text", counting_read_text):
                    load_wb_dictionary()
                    load_wb_dictionary()
                    load_wb_dictionary()

                # File should be read exactly once due to lru_cache
                assert call_count["n"] == 1


class TestGetWbCharacteristicsForCategory:
    def setup_method(self):
        _reset_cache()

    def test_returns_empty_list_for_unknown_subject(self, tmp_path):
        """get_wb_characteristics_for_category returns [] for an unknown subject_id."""
        (tmp_path / "wb_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                get_wb_characteristics_for_category,
            )
            result = get_wb_characteristics_for_category(99999)
        assert result == []

    def test_returns_characteristics_for_known_subject(self, tmp_path):
        """get_wb_characteristics_for_category returns the characteristics list."""
        (tmp_path / "wb_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                get_wb_characteristics_for_category,
            )
            result = get_wb_characteristics_for_category(12345)
        assert len(result) == 2
        assert result[0] == {"id": 14177419, "name": "Цвет"}
        assert result[1] == {"id": 14177421, "name": "Материал верха"}

    def test_returns_empty_list_when_file_missing(self, tmp_path):
        """get_wb_characteristics_for_category returns [] when dictionary file is absent."""
        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                get_wb_characteristics_for_category,
            )
            result = get_wb_characteristics_for_category(12345)
        assert result == []


class TestGetWbSubjectName:
    def setup_method(self):
        _reset_cache()

    def test_returns_name_for_known_subject(self, tmp_path):
        """get_wb_subject_name returns the subject name string."""
        (tmp_path / "wb_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                get_wb_subject_name,
            )
            assert get_wb_subject_name(12345) == "Кроссовки мужские"

    def test_returns_none_for_unknown_subject(self, tmp_path):
        """get_wb_subject_name returns None for an unknown subject_id."""
        (tmp_path / "wb_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                get_wb_subject_name,
            )
            assert get_wb_subject_name(0) is None
