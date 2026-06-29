"""Unit tests for compute_usage (billing aggregation helper).

Covers: mixed generated+cached across products, fully-cached product,
Exception entries, error dicts without debug_info, tokens summation.
"""

import sys
import os

# Allow importing app.billing_usage without a full FastAPI startup.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.billing_usage import compute_usage


# ---------------------------------------------------------------------------
# Fixtures — test data
# ---------------------------------------------------------------------------

def _make_result(
    product_id: str,
    features: dict[str, str],  # feature_name -> "cache" | "llm" | other
    tokens_used: int = 0,
) -> dict:
    """Build a result dict resembling process_product output."""
    debug_info = {
        name: {"source": source, "extra": "x"}
        for name, source in features.items()
    }
    return {
        "product_id": product_id,
        "filled_features": {k: "value" for k in features},
        "debug_info": debug_info,
        "tokens_used": tokens_used,
        "is_cached": False,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestComputeUsageMixed:
    """Mixed generated + cached features across 2–3 products."""

    def test_basic_counts(self):
        results = [
            _make_result("p1", {"color": "llm", "weight": "cache", "size": "llm"}, tokens_used=100),
            _make_result("p2", {"brand": "cache", "material": "llm"}, tokens_used=200),
        ]
        usage = compute_usage(results)
        # p1: 2 generated, 1 cached; p2: 1 generated, 1 cached
        assert usage["generated_count"] == 3
        assert usage["cached_count"] == 2
        assert usage["generated_products"] == 2  # both have >= 1 generated
        assert usage["total_tokens"] == 300

    def test_three_products(self):
        results = [
            _make_result("p1", {"a": "llm", "b": "llm"}, tokens_used=50),
            _make_result("p2", {"c": "cache", "d": "llm"}, tokens_used=75),
            _make_result("p3", {"e": "llm", "f": "cache", "g": "llm"}, tokens_used=25),
        ]
        usage = compute_usage(results)
        # p1: gen=2, cached=0; p2: gen=1, cached=1; p3: gen=2, cached=1
        assert usage["generated_count"] == 5
        assert usage["cached_count"] == 2
        assert usage["generated_products"] == 3
        assert usage["total_tokens"] == 150


class TestFullyCachedProduct:
    """A fully-cached product should NOT count toward generated_products."""

    def test_fully_cached_not_in_generated_products(self):
        results = [
            _make_result("p_gen", {"x": "llm", "y": "llm"}, tokens_used=100),
            _make_result("p_cached", {"a": "cache", "b": "cache"}, tokens_used=0),
        ]
        usage = compute_usage(results)
        assert usage["generated_count"] == 2
        assert usage["cached_count"] == 2
        assert usage["generated_products"] == 1  # p_cached excluded
        assert usage["total_tokens"] == 100

    def test_all_cached_zero_generated_products(self):
        results = [
            _make_result("p1", {"x": "cache"}, tokens_used=0),
            _make_result("p2", {"y": "cache", "z": "cache"}, tokens_used=0),
        ]
        usage = compute_usage(results)
        assert usage["generated_count"] == 0
        assert usage["cached_count"] == 3
        assert usage["generated_products"] == 0
        assert usage["total_tokens"] == 0


class TestExceptionEntries:
    """Exception objects in the results list must be silently ignored."""

    def test_exception_ignored(self):
        results = [
            _make_result("p1", {"a": "llm"}, tokens_used=50),
            ValueError("timeout"),
            RuntimeError("network error"),
            _make_result("p2", {"b": "cache"}, tokens_used=30),
        ]
        usage = compute_usage(results)
        assert usage["generated_count"] == 1
        assert usage["cached_count"] == 1
        assert usage["generated_products"] == 1
        assert usage["total_tokens"] == 80

    def test_all_exceptions(self):
        results = [ValueError("err1"), RuntimeError("err2")]
        usage = compute_usage(results)
        assert usage == {
            "generated_count": 0,
            "generated_products": 0,
            "cached_count": 0,
            "total_tokens": 0,
        }


class TestErrorDictWithoutDebugInfo:
    """Error dicts (no debug_info) contribute 0 features, tokens added if present."""

    def test_error_dict_zero_features(self):
        error_result = {"product_id": "p_err", "error": "LLM failed", "filled_features": {}}
        normal = _make_result("p_ok", {"f1": "llm"}, tokens_used=60)
        results = [normal, error_result]
        usage = compute_usage(results)
        assert usage["generated_count"] == 1
        assert usage["cached_count"] == 0
        assert usage["generated_products"] == 1
        assert usage["total_tokens"] == 60  # error dict has no tokens_used → 0

    def test_error_dict_with_tokens_used(self):
        """If an error dict somehow carries tokens_used, it should be counted."""
        error_result = {
            "product_id": "p_err",
            "error": "partial failure",
            "filled_features": {},
            "tokens_used": 40,
        }
        results = [error_result]
        usage = compute_usage(results)
        assert usage["generated_count"] == 0
        assert usage["generated_products"] == 0
        assert usage["total_tokens"] == 40


class TestTokensSummation:
    """total_tokens = sum of tokens_used across all non-Exception results."""

    def test_tokens_sum(self):
        results = [
            _make_result("p1", {"a": "llm"}, tokens_used=100),
            _make_result("p2", {"b": "llm"}, tokens_used=250),
            _make_result("p3", {"c": "cache"}, tokens_used=0),
        ]
        usage = compute_usage(results)
        assert usage["total_tokens"] == 350

    def test_missing_tokens_used_defaults_to_zero(self):
        result = {"product_id": "p1", "filled_features": {}, "debug_info": {"f": {"source": "llm"}}}
        usage = compute_usage([result])
        assert usage["total_tokens"] == 0

    def test_none_tokens_used_treated_as_zero(self):
        result = {
            "product_id": "p1",
            "filled_features": {},
            "debug_info": {"f": {"source": "llm"}},
            "tokens_used": None,
        }
        usage = compute_usage([result])
        assert usage["total_tokens"] == 0


class TestReturnSchema:
    """Verify the returned dict always has exactly the four expected keys."""

    def test_empty_results(self):
        usage = compute_usage([])
        assert set(usage.keys()) == {"generated_count", "generated_products", "cached_count", "total_tokens"}
        assert all(v == 0 for v in usage.values())

    def test_keys_present(self):
        usage = compute_usage([_make_result("p1", {"x": "llm"}, tokens_used=1)])
        assert set(usage.keys()) == {"generated_count", "generated_products", "cached_count", "total_tokens"}
