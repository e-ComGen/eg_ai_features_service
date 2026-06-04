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


# ---------------------------------------------------------------------------
# Tests: resolve_value_id — fuzzy + semantic matcher fallback
# ---------------------------------------------------------------------------

# Dictionary with colour values for matcher tests
_DICT_WITH_COLORS = {
    "schema_version": 2,
    "source": "ozon_seller_api",
    "categories": {
        "500:100": {
            "description_category_id": 500,
            "type_id": 100,
            "name": "Тест",
            "path": ["Тест"],
            "characteristics": [
                {
                    "id": 4180,
                    "name": "Цвет",
                    "type": "Option",
                    "is_required": False,
                    "is_collection": False,
                    "description": "Цвет товара",
                    "values": [
                        {"id": 1001, "value": "черный"},
                        {"id": 1002, "value": "белый"},
                        {"id": 1003, "value": "красный"},
                        {"id": 1004, "value": "синий"},
                    ],
                }
            ],
        }
    },
}


def _sentence_transformers_available() -> bool:
    """Check if sentence_transformers is installed without actually importing it."""
    import importlib.util
    return importlib.util.find_spec("sentence_transformers") is not None


_st_skip = pytest.mark.skipif(
    not _sentence_transformers_available(),
    reason="sentence_transformers not installed",
)


@pytest.mark.slow
class TestResolveValueIdMatcherFallback:
    """Tests resolve_value_id fuzzy+semantic fallback via MatcherService.

    Uses a mock MatcherService so no model is loaded — only the wiring between
    resolve_value_id and the matcher is exercised.  Skipped if
    sentence_transformers is not installed.  Marked slow for real-model runs.
    """

    def setup_method(self):
        if not _sentence_transformers_available():
            pytest.skip("sentence_transformers not installed")
        _reset_cache()
        import app.services.enrichment.strategies.dictionaries.ozon_loader as mod
        mod._matcher_instance = None
        mod._matcher_attempted = False

    @staticmethod
    def _make_mock_matcher(return_value: str):
        """Return a MatcherService stub whose find_best_match always returns return_value."""
        from unittest.mock import MagicMock
        m = MagicMock()
        m.find_best_match.return_value = return_value
        return m

    def test_fuzzy_match_yo_vs_ye(self, tmp_path):
        """'Чёрный' (ё) resolves to id 1001 via ё→е normalization, no matcher needed.

        The strong-normalization step (_normalize_token) maps ё→е, so 'Чёрный'
        normalizes to 'черный' and matches the dict entry directly. This is the
        correct, cheaper path — the semantic matcher must NOT be invoked for a
        purely orthographic ё/е difference.
        """
        import json as _json
        import app.services.enrichment.strategies.dictionaries.ozon_loader as mod
        (tmp_path / "ozon_dictionary.json").write_text(
            _json.dumps(_DICT_WITH_COLORS), encoding="utf-8"
        )
        mock_matcher = self._make_mock_matcher("черный")
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            mod._matcher_instance = mock_matcher
            mod._matcher_attempted = True
            from app.services.enrichment.strategies.dictionaries.ozon_loader import resolve_value_id
            result = resolve_value_id(500, 100, 4180, "Чёрный")
        assert result == 1001, f"Expected 1001 (черный), got {result}"
        mock_matcher.find_best_match.assert_not_called()

    def test_semantic_match_paraphrase(self, tmp_path):
        """'чёрного цвета' resolves to id 1001 when matcher returns 'черный'."""
        import json as _json
        import app.services.enrichment.strategies.dictionaries.ozon_loader as mod
        (tmp_path / "ozon_dictionary.json").write_text(
            _json.dumps(_DICT_WITH_COLORS), encoding="utf-8"
        )
        mock_matcher = self._make_mock_matcher("черный")
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_cache()
            mod._matcher_instance = mock_matcher
            mod._matcher_attempted = True
            from app.services.enrichment.strategies.dictionaries.ozon_loader import resolve_value_id
            result = resolve_value_id(500, 100, 4180, "чёрного цвета")
        assert result == 1001, f"Expected 1001 (черный), got {result}"
        mock_matcher.find_best_match.assert_called_once()
