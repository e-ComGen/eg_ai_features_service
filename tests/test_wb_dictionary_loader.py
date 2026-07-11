"""Tests for app/services/enrichment/strategies/dictionaries/loader.py

All tests use mocks — no real API calls are made.
"""
import json
import gzip
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_DICT = {
    "schema_version": 1,
    "source": "wb-api",
    "generated_at": "2024-01-01T00:00:00",
    "categories": {
        "12345": {
            "subject_id": 12345,
            "subject_name": "Кроссовки мужские",
            "parent_id": 8995,
            "characteristics": [
                {"id": 14177419, "name": "Цвет"},
                {"id": 14177421, "name": "Материал верха"},
            ],
        }
    },
}


def _reset_cache():
    """Clear lru_cache on load_wb_dictionary between tests."""
    from app.services.enrichment.strategies.dictionaries.loader import load_wb_dictionary
    load_wb_dictionary.cache_clear()


def _write_gzip_dict(path: Path, data: dict) -> None:
    """Write a gzip-compressed JSON file at *path* containing *data*."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadWbDictionary:
    def setup_method(self):
        _reset_cache()

    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        """load_wb_dictionary returns {} when wb_dictionary.json.gz does not exist."""
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
        """load_wb_dictionary parses and returns the unwrapped categories dict."""
        _write_gzip_dict(tmp_path / "wb_dictionary.json.gz", SAMPLE_DICT)
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
        assert result["12345"]["subject_name"] == "Кроссовки мужские"
        # Prove the envelope was unwrapped — no top-level schema_version key
        assert "schema_version" not in result

    def test_load_dictionary_is_cached(self, tmp_path):
        """Repeated calls to load_wb_dictionary do not re-read the file."""
        _write_gzip_dict(tmp_path / "wb_dictionary.json.gz", SAMPLE_DICT)

        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                load_wb_dictionary,
            )
            with patch(
                "app.services.enrichment.strategies.dictionaries.loader.gzip.open",
                wraps=gzip.open,
            ) as mock_gzip_open:
                load_wb_dictionary()
                load_wb_dictionary()
                load_wb_dictionary()

            # File should be opened exactly once due to lru_cache
            assert mock_gzip_open.call_count == 1

    def test_unwraps_envelope_categories_key(self, tmp_path):
        """load_wb_dictionary returns exactly the categories sub-dict from the envelope."""
        _write_gzip_dict(tmp_path / "wb_dictionary.json.gz", SAMPLE_DICT)
        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                load_wb_dictionary,
            )
            result = load_wb_dictionary()
        expected = SAMPLE_DICT["categories"]
        assert result == expected


class TestGetWbCharacteristicsForCategory:
    def setup_method(self):
        _reset_cache()

    def test_returns_empty_list_for_unknown_subject(self, tmp_path):
        """get_wb_characteristics_for_category returns [] for an unknown subject_id."""
        _write_gzip_dict(tmp_path / "wb_dictionary.json.gz", SAMPLE_DICT)
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
        _write_gzip_dict(tmp_path / "wb_dictionary.json.gz", SAMPLE_DICT)
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
        _write_gzip_dict(tmp_path / "wb_dictionary.json.gz", SAMPLE_DICT)
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
        _write_gzip_dict(tmp_path / "wb_dictionary.json.gz", SAMPLE_DICT)
        with patch(
            "app.services.enrichment.strategies.dictionaries.loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.loader import (
                get_wb_subject_name,
            )
            assert get_wb_subject_name(0) is None
