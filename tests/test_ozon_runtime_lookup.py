"""Tests for ozon_runtime_lookup and the truncated-dict fallback in OzonStrategy.

All tests are fully mocked — no real HTTP calls, no filesystem I/O on the
real ozon_dictionary.json.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Sample dict with one truncated attribute and one normal attribute
SAMPLE_DICT_WITH_TRUNCATED = {
    "schema_version": 2,
    "categories": {
        "100:200": {
            "description_category_id": 100,
            "type_id": 200,
            "name": "Одежда",
            "path": ["Одежда"],
            "characteristics": [
                {
                    "id": 85,
                    "name": "Бренд",
                    "type": "String",
                    "is_required": True,
                    "is_collection": False,
                    "description": "Бренд товара",
                    "values": [{"id": 9999, "value": "Nike"}, {"id": 9998, "value": "Adidas"}],
                    "values_truncated": True,
                },
                {
                    "id": 111,
                    "name": "Цвет",
                    "type": "Option",
                    "is_required": False,
                    "is_collection": False,
                    "description": "Цвет",
                    "values": [{"id": 1001, "value": "Красный"}, {"id": 1002, "value": "Синий"}],
                    # NO values_truncated → False by default
                },
            ],
        }
    },
}


def _reset_loader_cache():
    from app.services.enrichment.strategies.dictionaries.ozon_loader import (
        load_ozon_dictionary,
    )
    load_ozon_dictionary.cache_clear()


def _reset_lookup_cache():
    from app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup import (
        clear_lookup_cache,
    )
    clear_lookup_cache()


# ---------------------------------------------------------------------------
# Tests: is_truncated (ozon_loader)
# ---------------------------------------------------------------------------

class TestIsTruncated:
    def setup_method(self):
        _reset_loader_cache()
        _reset_lookup_cache()

    def test_truncated_attribute_returns_true(self, tmp_path):
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_WITH_TRUNCATED), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_loader_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import is_truncated
            assert is_truncated(100, 200, 85) is True

    def test_non_truncated_attribute_returns_false(self, tmp_path):
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_WITH_TRUNCATED), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_loader_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import is_truncated
            assert is_truncated(100, 200, 111) is False

    def test_unknown_category_returns_false(self, tmp_path):
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_WITH_TRUNCATED), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_loader_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import is_truncated
            assert is_truncated(999, 888, 85) is False

    def test_unknown_attribute_in_known_category_returns_false(self, tmp_path):
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_WITH_TRUNCATED), encoding="utf-8"
        )
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_loader_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import is_truncated
            assert is_truncated(100, 200, 999999) is False

    def test_missing_file_returns_false(self, tmp_path):
        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_loader_cache()
            from app.services.enrichment.strategies.dictionaries.ozon_loader import is_truncated
            assert is_truncated(100, 200, 85) is False


# ---------------------------------------------------------------------------
# Tests: search_value (ozon_runtime_lookup)
# ---------------------------------------------------------------------------

class TestSearchValue:
    def setup_method(self):
        _reset_lookup_cache()

    @pytest.mark.asyncio
    async def test_returns_first_result_on_success(self):
        """search_value returns {id, value} for the first API result."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": [
                {"id": 123456, "value": "Adidas", "info": "спортивный бренд"},
                {"id": 123457, "value": "Adidas Originals", "info": ""},
            ],
            "has_next": False,
        }

        from app.services.enrichment.strategies.dictionaries import ozon_runtime_lookup

        with patch.object(
            httpx.AsyncClient, "__aenter__",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await ozon_runtime_lookup.search_value(
                100, 200, 85, "adid",
                client_id="test_client", api_key="test_key",
            )

        assert result == {"id": 123456, "value": "Adidas"}

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_results(self):
        """search_value returns None when API returns empty result list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": [], "has_next": False}

        from app.services.enrichment.strategies.dictionaries import ozon_runtime_lookup

        with patch.object(
            httpx.AsyncClient, "__aenter__",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await ozon_runtime_lookup.search_value(
                100, 200, 85, "xyznotfound",
                client_id="test_client", api_key="test_key",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_404(self):
        """search_value returns None when attribute is not dict-backed (404)."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {}

        from app.services.enrichment.strategies.dictionaries import ozon_runtime_lookup

        with patch.object(
            httpx.AsyncClient, "__aenter__",
            return_value=AsyncMock(post=AsyncMock(return_value=mock_response)),
        ):
            result = await ozon_runtime_lookup.search_value(
                100, 200, 85, "anything",
                client_id="test_client", api_key="test_key",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_network_error(self):
        """search_value returns None gracefully on network errors."""
        from app.services.enrichment.strategies.dictionaries import ozon_runtime_lookup

        with patch.object(
            httpx.AsyncClient, "__aenter__",
            return_value=AsyncMock(
                post=AsyncMock(side_effect=httpx.ConnectError("connection refused"))
            ),
        ):
            result = await ozon_runtime_lookup.search_value(
                100, 200, 85, "anything",
                client_id="test_client", api_key="test_key",
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_credentials(self):
        """search_value returns None immediately when no client_id/api_key available."""
        from app.services.enrichment.strategies.dictionaries import ozon_runtime_lookup

        with patch.dict("os.environ", {}, clear=True):
            # Remove env vars if present
            import os
            os.environ.pop("OZON_CLIENT_ID", None)
            os.environ.pop("OZON_API_KEY", None)

            result = await ozon_runtime_lookup.search_value(
                100, 200, 85, "anything",
                # No client_id / api_key args and no env vars
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_caches_result_to_avoid_duplicate_calls(self):
        """Repeated calls with same args use cache, not a second HTTP call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": [{"id": 42, "value": "Puma", "info": ""}],
            "has_next": False,
        }
        post_mock = AsyncMock(return_value=mock_response)

        from app.services.enrichment.strategies.dictionaries import ozon_runtime_lookup
        _reset_lookup_cache()

        with patch.object(
            httpx.AsyncClient, "__aenter__",
            return_value=AsyncMock(post=post_mock),
        ):
            r1 = await ozon_runtime_lookup.search_value(
                100, 200, 85, "Puma",
                client_id="c", api_key="k",
            )
            r2 = await ozon_runtime_lookup.search_value(
                100, 200, 85, "puma",  # same query, different case → same cache key
                client_id="c", api_key="k",
            )

        # Second call should be served from cache
        assert post_mock.call_count == 1
        assert r1 == {"id": 42, "value": "Puma"}
        assert r2 == {"id": 42, "value": "Puma"}


# ---------------------------------------------------------------------------
# Tests: OzonStrategy.resolve_value_ids_async
# ---------------------------------------------------------------------------

class TestResolveValueIdsAsync:
    def setup_method(self):
        _reset_loader_cache()
        _reset_lookup_cache()

    def _make_attribute_value(self, attr_id: int, value, is_collection=False):
        from app.services.enrichment.base import AttributeValue, Source
        return AttributeValue(
            attribute_id=attr_id,
            value=value,
            confidence=0.9,
            source=Source.DESCRIPTION,
            is_collection=is_collection,
        )

    def _make_context(self, cat_id=100, type_id=200):
        from app.services.enrichment.base import ExtractionContext
        return ExtractionContext(
            product_id=1,
            product_name="Test",
            category_id=cat_id,
            ozon_type_id=type_id,
        )

    @pytest.mark.asyncio
    async def test_static_dict_hit_no_api_call(self, tmp_path):
        """If value exists in cached dict, no API call is made."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_WITH_TRUNCATED), encoding="utf-8"
        )

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_loader_cache()
            from app.services.enrichment.strategies.ozon_strategy import OzonStrategy

            strategy = OzonStrategy()
            av = self._make_attribute_value(85, "Nike")
            ctx = self._make_context()

            # Patch the alias imported in ozon_strategy — should NOT be called
            with patch(
                "app.services.enrichment.strategies.ozon_strategy._runtime_search_value",
                new_callable=AsyncMock,
            ) as mock_search:
                result = await strategy.resolve_value_ids_async(
                    av, ctx, client_id="c", api_key="k"
                )

            mock_search.assert_not_called()
            assert result.value_id == 9999

    @pytest.mark.asyncio
    async def test_truncated_fallback_called_when_static_miss(self, tmp_path):
        """If value not in truncated dict, runtime API search is called."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_WITH_TRUNCATED), encoding="utf-8"
        )

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_loader_cache()
            from app.services.enrichment.strategies.ozon_strategy import OzonStrategy

            strategy = OzonStrategy()
            av = self._make_attribute_value(85, "Puma")  # not in truncated dict
            ctx = self._make_context()

            # Patch the name as imported in ozon_strategy (alias _runtime_search_value)
            with patch(
                "app.services.enrichment.strategies.ozon_strategy._runtime_search_value",
                new_callable=AsyncMock,
                return_value={"id": 77777, "value": "Puma"},
            ) as mock_search:
                result = await strategy.resolve_value_ids_async(
                    av, ctx, client_id="c", api_key="k"
                )

            mock_search.assert_called_once_with(100, 200, 85, "Puma", client_id="c", api_key="k")
            assert result.value_id == 77777

    @pytest.mark.asyncio
    async def test_non_truncated_no_api_fallback(self, tmp_path):
        """If value not in dict and NOT truncated, runtime API is NOT called."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_WITH_TRUNCATED), encoding="utf-8"
        )

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_loader_cache()
            from app.services.enrichment.strategies.ozon_strategy import OzonStrategy

            strategy = OzonStrategy()
            # attr 111 (Цвет) is NOT truncated; "Зелёный" is not in its values list
            av = self._make_attribute_value(111, "Зелёный")
            ctx = self._make_context()

            with patch(
                "app.services.enrichment.strategies.ozon_strategy._runtime_search_value",
                new_callable=AsyncMock,
            ) as mock_search:
                result = await strategy.resolve_value_ids_async(
                    av, ctx, client_id="c", api_key="k"
                )

            mock_search.assert_not_called()
            assert result.value_id is None

    @pytest.mark.asyncio
    async def test_no_type_id_returns_unchanged(self, tmp_path):
        """If context has no ozon_type_id, returns AttributeValue unchanged."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_WITH_TRUNCATED), encoding="utf-8"
        )

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_loader_cache()
            from app.services.enrichment.strategies.ozon_strategy import OzonStrategy
            from app.services.enrichment.base import ExtractionContext

            strategy = OzonStrategy()
            av = self._make_attribute_value(85, "Nike")
            ctx = ExtractionContext(
                product_id=1, product_name="Test",
                category_id=100, ozon_type_id=None,
            )
            result = await strategy.resolve_value_ids_async(av, ctx)
            assert result.value_id is None

    @pytest.mark.asyncio
    async def test_collection_attribute_resolved(self, tmp_path):
        """is_collection=True resolves multiple values into value_ids list."""
        (tmp_path / "ozon_dictionary.json").write_text(
            json.dumps(SAMPLE_DICT_WITH_TRUNCATED), encoding="utf-8"
        )

        with patch(
            "app.services.enrichment.strategies.dictionaries.ozon_loader.DATA_DIR",
            tmp_path,
        ):
            _reset_loader_cache()
            from app.services.enrichment.strategies.ozon_strategy import OzonStrategy

            strategy = OzonStrategy()
            # attr 85 is truncated; "Nike" in dict (id=9999), "Puma" requires API
            av = self._make_attribute_value(85, ["Nike", "Puma"], is_collection=True)
            ctx = self._make_context()

            # Patch the alias imported in ozon_strategy
            with patch(
                "app.services.enrichment.strategies.ozon_strategy._runtime_search_value",
                new_callable=AsyncMock,
                return_value={"id": 77777, "value": "Puma"},
            ):
                result = await strategy.resolve_value_ids_async(
                    av, ctx, client_id="c", api_key="k"
                )

            assert result.value_ids == [9999, 77777]
