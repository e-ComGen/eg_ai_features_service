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
