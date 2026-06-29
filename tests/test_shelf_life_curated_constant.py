from unittest.mock import MagicMock
from typing import List
from app.services.enrichment.strategies.ozon_strategy import OzonStrategy
from app.services.enrichment.base import TargetAttribute, ExtractionContext, AttributeValue, Source


class TestShelfLifeCuratedConstant:
    """Oracle invariants for shelf life curated constant (category 17028670, attr 5379)."""

    def test_I1_force_websearch_excludes_attr_5379(self):
        """When resolved_category_id matches curated category, attr 5379 is excluded from force_websearch."""
        targets = [
            TargetAttribute(id=5379, name="Срок годности в днях", type="numeric"),
            TargetAttribute(id=9999, name="Другой numeric", type="numeric"),
        ]
        ctx = ExtractionContext(
            product_id=1,
            product_name="Test",
            category_id=17028670,
            resolved_category_id=17028670,
        )
        strategy = OzonStrategy()
        result = strategy.force_websearch_targets(targets, ctx)
        assert 9999 in result
        assert 5379 not in result

    def test_I1b_fallback_to_category_id(self):
        """When resolved_category_id is None, fallback to category_id still excludes attr 5379."""
        targets = [
            TargetAttribute(id=5379, name="Срок годности в днях", type="numeric"),
            TargetAttribute(id=9999, name="Другой numeric", type="numeric"),
        ]
        ctx = ExtractionContext(
            product_id=1,
            product_name="Test",
            category_id=17028670,
            resolved_category_id=None,
        )
        strategy = OzonStrategy()
        result = strategy.force_websearch_targets(targets, ctx)
        assert 9999 in result
        assert 5379 not in result

    def test_I3_context_none_old_behavior(self):
        """When context is None, no filtering occurs and attr 5379 remains in force_websearch."""
        targets = [
            TargetAttribute(id=5379, name="Срок годности в днях", type="numeric"),
        ]
        strategy = OzonStrategy()
        result = strategy.force_websearch_targets(targets, None)
        assert 5379 in result

    def test_I4_regression_17028612_unchanged(self):
        """Regression: category 17028612 (БП) excludes curated attrs 23278 and 23489 from force_websearch."""
        targets = [
            TargetAttribute(id=9999, name="Другой numeric", type="numeric"),
            TargetAttribute(id=23278, name="БП attr 1", type="enum"),
            TargetAttribute(id=23489, name="БП attr 2", type="enum"),
        ]
        ctx = ExtractionContext(
            product_id=1,
            product_name="Test",
            category_id=17028612,
            resolved_category_id=17028612,
        )
        strategy = OzonStrategy()
        result = strategy.force_websearch_targets(targets, ctx)
        assert 23278 not in result
        assert 23489 not in result
        assert 9999 in result

    def test_I1_post_process_returns_548(self):
        """post_process_values returns curated value '548' for attr 5379 when category matches."""
        targets = [
            TargetAttribute(id=5379, name="Срок годности в днях", type="numeric", is_required=True),
        ]
        values: List[AttributeValue] = []
        ctx = ExtractionContext(
            product_id=1,
            product_name="Test",
            category_id=17028670,
            resolved_category_id=17028670,
        )
        strategy = OzonStrategy()
        # Mock resolve_value_ids to be a no-op
        strategy.resolve_value_ids = MagicMock(return_value=None)
        result = strategy.post_process_values(values, targets, ctx)
        found = [v for v in result if v.attribute_id == 5379]
        assert len(found) == 1
        assert found[0].value == "548"
        assert found[0].source == Source.DESCRIPTION

    def test_I1_force_websearch_excludes_attr_5379_alt_dcid(self):
        """When resolved_category_id matches alt dcid 200001282, attr 5379 is excluded from force_websearch."""
        targets = [
            TargetAttribute(id=5379, name="Срок годности в днях", type="numeric"),
            TargetAttribute(id=9999, name="Другой numeric", type="numeric"),
        ]
        ctx = ExtractionContext(
            product_id=1,
            product_name="Test",
            category_id=200001282,
            resolved_category_id=200001282,
        )
        strategy = OzonStrategy()
        result = strategy.force_websearch_targets(targets, ctx)
        assert 9999 in result
        assert 5379 not in result

    def test_I1_post_process_returns_548_alt_dcid(self):
        """post_process_values returns curated value '548' for attr 5379 when alt dcid 200001282 matches."""
        targets = [
            TargetAttribute(id=5379, name="Срок годности в днях", type="numeric", is_required=True),
        ]
        values: List[AttributeValue] = []
        ctx = ExtractionContext(
            product_id=1,
            product_name="Test",
            category_id=200001282,
            resolved_category_id=200001282,
        )
        strategy = OzonStrategy()
        # Mock resolve_value_ids to be a no-op
        strategy.resolve_value_ids = MagicMock(return_value=None)
        result = strategy.post_process_values(values, targets, ctx)
        found = [v for v in result if v.attribute_id == 5379]
        assert len(found) == 1
        assert found[0].value == "548"
        assert found[0].source == Source.DESCRIPTION

    # --- Override invariants I-OV1..I-OV6 (curated_default_override chunk) ---

    def test_I_OV1_override_existing_web_search_1095(self):
        """I-OV1 (main seam): existing web_search 1095 is removed; curated 548/DESCRIPTION
        takes its place. Result must have exactly ONE value for attr 5379."""
        targets = [TargetAttribute(id=5379, name="Срок годности в днях", type="numeric", is_required=True)]
        values: List[AttributeValue] = [
            AttributeValue(attribute_id=5379, value="1095", confidence=0.97, source=Source.WEB_SEARCH),
        ]
        ctx = ExtractionContext(
            product_id=1, product_name="Test", category_id=17028670, resolved_category_id=17028670,
        )
        strategy = OzonStrategy()
        strategy.resolve_value_ids = MagicMock(return_value=None)
        result = strategy.post_process_values(values, targets, ctx)
        found = [v for v in result if v.attribute_id == 5379]
        assert len(found) == 1, f"Expected exactly 1 value for 5379, got {len(found)}: {[v.value for v in found]}"
        assert found[0].value == "548"
        assert found[0].source == Source.DESCRIPTION
        assert found[0].evidence == "category default (authoritative override)"

    def test_I_OV2_override_existing_alt_dcid(self):
        """I-OV2: same authoritative override on alt dcid 200001282; 1095 removed, 548 installed."""
        targets = [TargetAttribute(id=5379, name="Срок годности в днях", type="numeric", is_required=True)]
        values: List[AttributeValue] = [
            AttributeValue(attribute_id=5379, value="1095", confidence=0.97, source=Source.WEB_SEARCH),
        ]
        ctx = ExtractionContext(
            product_id=1, product_name="Test", category_id=200001282, resolved_category_id=200001282,
        )
        strategy = OzonStrategy()
        strategy.resolve_value_ids = MagicMock(return_value=None)
        result = strategy.post_process_values(values, targets, ctx)
        found = [v for v in result if v.attribute_id == 5379]
        assert len(found) == 1
        assert found[0].value == "548"
        assert found[0].source == Source.DESCRIPTION

    def test_I_OV3_non_curated_attr_untouched(self):
        """I-OV3: override touches ONLY curated attrs; non-curated attr 9999 is preserved as-is."""
        targets = [
            TargetAttribute(id=5379, name="Срок годности в днях", type="numeric", is_required=True),
            TargetAttribute(id=9999, name="Другой атрибут", type="numeric"),
        ]
        values: List[AttributeValue] = [
            AttributeValue(attribute_id=9999, value="SomeValue", confidence=0.9, source=Source.WEB_SEARCH),
        ]
        ctx = ExtractionContext(
            product_id=1, product_name="Test", category_id=17028670, resolved_category_id=17028670,
        )
        strategy = OzonStrategy()
        strategy.resolve_value_ids = MagicMock(return_value=None)
        result = strategy.post_process_values(values, targets, ctx)
        noncurated = [v for v in result if v.attribute_id == 9999]
        assert len(noncurated) == 1
        assert noncurated[0].value == "SomeValue"
        curated = [v for v in result if v.attribute_id == 5379]
        assert len(curated) == 1
        assert curated[0].value == "548"

    def test_I_OV4_empty_values_still_filled(self):
        """I-OV4: when values is empty, curated default is added (regression: override path must not break fill)."""
        targets = [TargetAttribute(id=5379, name="Срок годности в днях", type="numeric", is_required=True)]
        values: List[AttributeValue] = []
        ctx = ExtractionContext(
            product_id=1, product_name="Test", category_id=17028670, resolved_category_id=17028670,
        )
        strategy = OzonStrategy()
        strategy.resolve_value_ids = MagicMock(return_value=None)
        result = strategy.post_process_values(values, targets, ctx)
        found = [v for v in result if v.attribute_id == 5379]
        assert len(found) == 1
        assert found[0].value == "548"
        assert found[0].source == Source.DESCRIPTION

    def test_I_OV5_conditional_stays_fill_empty(self):
        """I-OV5: step B conditional defaults remain fill-empty (overridable); existing gabaret
        value is NOT overwritten by the ATX conditional default."""
        ff_attr_id = 99001  # synthetic form_factor attr not in CATEGORY_DEFAULTS
        targets = [
            TargetAttribute(id=ff_attr_id, name="Форм-фактор", type="enum", semantic_type="form_factor"),
            TargetAttribute(id=8415, name="Длина корпуса", type="numeric"),
        ]
        values: List[AttributeValue] = [
            AttributeValue(attribute_id=ff_attr_id, value="ATX", confidence=0.9, source=Source.WEB_SEARCH),
            AttributeValue(attribute_id=8415, value=20.0, confidence=0.95, source=Source.WEB_SEARCH),
        ]
        ctx = ExtractionContext(
            product_id=1, product_name="ATX БП тест", category_id=17028612, resolved_category_id=17028612,
        )
        strategy = OzonStrategy()
        strategy.resolve_value_ids = MagicMock(return_value=None)
        result = strategy.post_process_values(values, targets, ctx)
        length_vals = [v for v in result if v.attribute_id == 8415]
        assert len(length_vals) == 1, f"Expected 1 length value, got {len(length_vals)}"
        assert length_vals[0].value == 20.0, "Conditional default must not overwrite existing gabaret"

    def test_I_OV6_target_gating_no_override_when_not_in_targets(self):
        """I-OV6: if curated attr 5379 is NOT in targets, override is NOT applied even if
        an existing web_search value for 5379 is present."""
        targets = [TargetAttribute(id=9999, name="Другой атрибут", type="numeric")]
        values: List[AttributeValue] = [
            AttributeValue(attribute_id=5379, value="1095", confidence=0.97, source=Source.WEB_SEARCH),
        ]
        ctx = ExtractionContext(
            product_id=1, product_name="Test", category_id=17028670, resolved_category_id=17028670,
        )
        strategy = OzonStrategy()
        strategy.resolve_value_ids = MagicMock(return_value=None)
        result = strategy.post_process_values(values, targets, ctx)
        found_5379 = [v for v in result if v.attribute_id == 5379]
        assert len(found_5379) == 1
        assert found_5379[0].value == "1095", "Override must not fire when attr_id not in targets"
