"""Tests for the vendored WB→Ozon field map loader and the verified-first
mapping gate in wb_card_source._map_characteristics.

The loader reads the real vendored file
(app/services/enrichment/strategies/dictionaries/data/eg_wb_ozon_field_map.json)
so these tests double as a smoke test that the consolidation produced sane data.
"""
import pytest

from app.services.enrichment.strategies.dictionaries import eg_wb_ozon_field_map
from app.services.enrichment.strategies.dictionaries.eg_wb_ozon_field_map import (
    eg_get_field_map,
    load_eg_field_map,
)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class TestEgGetFieldMap:
    def setup_method(self):
        # Clear the resolution cache so each test resolves fresh.
        eg_wb_ozon_field_map._resolved_cache.clear()

    def test_loader_returns_real_mapping_for_tolstovki(self):
        """Толстовки «Уход за вещами» → 4655 via the exact key path."""
        m = eg_get_field_map("Толстовки", 200000933, 93232)
        assert m, "expected a non-empty verified map for Толстовки"
        # Field names are normalized to lower+strip.
        assert m.get("уход за вещами") == 4655
        # A few more verified ids from the same file.
        assert m.get("состав") == 4604
        assert m.get("пол") == 9163

    def test_fallback_merge_when_subject_unknown(self):
        """Without wb_subject, the loader unions all keys for the cat/type."""
        m = eg_get_field_map(None, 200000933, 93232)
        assert m.get("уход за вещами") == 4655

    def test_missing_cat_type_returns_empty(self):
        assert eg_get_field_map("Толстовки", None, None) == {}
        assert eg_get_field_map(None, 200000933, None) == {}

    def test_unknown_key_returns_empty(self):
        assert eg_get_field_map("НесуществующийСабж", 1, 1) == {}

    def test_file_loads_and_is_nonempty(self):
        data = load_eg_field_map()
        assert isinstance(data, dict)
        assert len(data) > 1000


# ---------------------------------------------------------------------------
# Gate: verified-first preferred over fuzzy
# ---------------------------------------------------------------------------


class _Target:
    """Minimal TargetAttribute stand-in for the mapping gate."""

    def __init__(self, id, name, is_collection=False):
        self.id = id
        self.name = name
        self.is_collection = is_collection
        self.allowed_values = None
        self.semantic_type = None


class _Ctx:
    def __init__(self):
        self.category_id = "200000933"
        self.ozon_type_id = 93232
        self.product_name = "Толстовка мужская"
        self.image_urls = []


class TestVerifiedFirstGate:
    def setup_method(self):
        eg_wb_ozon_field_map._resolved_cache.clear()

    def test_verified_id_preferred_over_fuzzy_miss(self):
        """A WB name that fuzzy (≥88) would miss is mapped via the verified map.

        Target id 4655 is named 'XYZ Care Instructions' so WRatio against the WB
        name 'Уход за вещами' is far below 88 — only the verified field map can
        bridge them. If the verified path works, we get exactly one result on
        attribute_id 4655.
        """
        from app.services.enrichment.sources.wb_card_source import WbCardSource

        targets = [
            _Target(4655, "XYZ Care Instructions"),  # fuzzy would never match
            _Target(31, "Brand"),
        ]
        chars = [{"name": "Уход за вещами", "value": "Машинная стирка"}]
        ctx = _Ctx()
        card = {"subj_name": "Толстовки"}

        src = WbCardSource.__new__(WbCardSource)  # skip __init__ (no deps needed)
        results = src._map_characteristics(
            chars, targets, ctx, "exact", "Толстовка", 100.0, card,
        )

        assert len(results) == 1
        assert results[0].attribute_id == 4655

    def test_no_verified_entry_falls_back_identically(self):
        """When no verified entry exists, behavior matches the old fuzzy gate.

        Exact-name target → resolved by the unchanged exact-match fallback.
        """
        from app.services.enrichment.sources.wb_card_source import WbCardSource

        targets = [_Target(99, "Цвет")]
        chars = [{"name": "Цвет", "value": "Чёрный"}]
        ctx = _Ctx()
        # No card → wb_subject unknown; "Цвет" target id 99 is not a verified id
        # for this type, so the verified path is a no-op and exact-match wins.
        card = None

        src = WbCardSource.__new__(WbCardSource)
        results = src._map_characteristics(
            chars, targets, ctx, "exact", "Товар", 100.0, card,
        )

        assert len(results) == 1
        assert results[0].attribute_id == 99
