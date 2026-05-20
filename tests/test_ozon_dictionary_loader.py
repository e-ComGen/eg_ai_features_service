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

# schema_version=1 с categories-оберткой (legacy compound-like, но ключи простые)
SAMPLE_DICT_V2 = {
    "schema_version": 1,
    "source": "manual_seed_top30",
    "generated_at": "2026-05-14",
    "categories": {
        "502": {
            "name": "Смартфоны",
            "path": ["Электроника", "Смартфоны"],
            "characteristics": [
                {"id": 9048, "name": "Бренд"},
                {"id": 4180, "name": "Цвет товара"},
                {"id": 5076, "name": "Объём встроенной памяти, ГБ"},
            ],
        }
    },
}

# schema_version=2: compound-ключи "<description_category_id>:<type_id>"
SAMPLE_DICT_V2_COMPOUND = {
    "schema_version": 2,
    "source": "ozon_seller_api",
    "generated_at": "2026-05-20",
    "categories": {
        "200001175:970671325": {
            "description_category_id": 200001175,
            "type_id": 970671325,
            "name": "Смартфоны",
            "path": ["Электроника", "Смартфоны"],
            "characteristics": [
                {"id": 9048, "name": "Бренд", "type": "String",
                 "is_required": True, "is_collection": False, "description": "Бренд товара"},
                {"id": 4180, "name": "Цвет товара", "type": "Option",
                 "is_required": False, "is_collection": False, "description": "Цвет"},
            ],
        },
        "200001175:970671326": {
            "description_category_id": 200001175,
            "type_id": 970671326,
            "name": "Смартфоны (Б/У)",
            "path": ["Электроника", "Смартфоны"],
            "characteristics": [
                {"id": 9048, "name": "Бренд", "type": "String",
                 "is_required": True, "is_collection": False, "description": "Бренд товара"},
            ],
        },
        "300000002:111111111": {
            "description_category_id": 300000002,
            "type_id": 111111111,
            "name": "Ноутбуки",
            "path": ["Электроника", "Ноутбуки"],
            "characteristics": [
                {"id": 7777, "name": "Объём ОЗУ", "type": "Integer",
                 "is_required": True, "is_collection": False, "description": "RAM"},
            ],
        },
    },
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

    def test_load_schema_version_format_unwraps_categories(self, tmp_path):
        """load_ozon_dictionary unwraps 'categories' key from new schema format."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_V2), encoding="utf-8"
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
        # The result should be the categories dict, NOT the top-level object
        assert "502" in result
        assert "schema_version" not in result
        assert "categories" not in result
        assert result["502"]["name"] == "Смартфоны"

    def test_load_schema_version_format_characteristics_have_id(self, tmp_path):
        """Characteristics in new format use 'id' field, not 'key'."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_V2), encoding="utf-8"
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
        chars = result["502"]["characteristics"]
        assert len(chars) == 3
        assert chars[0] == {"id": 9048, "name": "Бренд"}

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


# ---------------------------------------------------------------------------
# Tests: schema_version=2 compound keys ("<cat_id>:<type_id>")
# ---------------------------------------------------------------------------


class TestSchemaV2CompoundKeys:
    """Проверяем поддержку v2 словаря с compound-ключами '<cat_id>:<type_id>'."""

    def setup_method(self):
        _reset_cache()

    def test_get_characteristics_for_type_known_pair(self, tmp_path):
        """get_ozon_characteristics_for_type возвращает характеристики для известной пары."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_V2_COMPOUND), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                get_ozon_characteristics_for_type,
            )
            result = get_ozon_characteristics_for_type(200001175, 970671325)
        assert len(result) == 2
        assert result[0]["id"] == 9048
        assert result[0]["name"] == "Бренд"
        assert result[0]["type"] == "String"
        assert result[0]["is_required"] is True

    def test_get_characteristics_for_type_unknown_pair_returns_empty(self, tmp_path):
        """get_ozon_characteristics_for_type возвращает [] для неизвестной пары."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_V2_COMPOUND), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                get_ozon_characteristics_for_type,
            )
            result = get_ozon_characteristics_for_type(200001175, 999999999)
        assert result == []

    def test_get_characteristics_for_category_fallback_on_compound(self, tmp_path):
        """get_ozon_characteristics_for_category находит данные по category_id без type_id."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_V2_COMPOUND), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                get_ozon_characteristics_for_category,
            )
            result = get_ozon_characteristics_for_category(200001175)
        # Должен вернуть характеристики хотя бы одного из type_id для этой категории
        assert len(result) >= 1
        char_ids = [c["id"] for c in result]
        assert 9048 in char_ids  # Бренд есть в обоих type

    def test_get_characteristics_for_category_unknown_returns_empty(self, tmp_path):
        """get_ozon_characteristics_for_category возвращает [] для неизвестной категории."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_V2_COMPOUND), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                get_ozon_characteristics_for_category,
            )
            result = get_ozon_characteristics_for_category(999999)
        assert result == []

    def test_get_category_name_from_compound_key(self, tmp_path):
        """get_ozon_category_name работает с compound-ключами v2."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_V2_COMPOUND), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import (
                get_ozon_category_name,
            )
            name = get_ozon_category_name(300000002)
        assert name == "Ноутбуки"
