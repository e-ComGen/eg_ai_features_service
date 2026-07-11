from __future__ import annotations
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import app.services.eg_seller_audit.fill_scorer as fill_scorer_mod
from app.services.eg_seller_audit.fill_scorer import ProductFillResult, score_product
from app.services.eg_seller_audit.sampler import stratified_sample
from app.services.eg_seller_audit.aggregator import aggregate
from app.services.eg_seller_audit.report import SellerInfo, UnresolvedInfo, mask_seller_display_name
from app.services.eg_seller_audit import catalog_provider


class TestScoreProduct:
    def test_some_required_and_optional_filled(self, monkeypatch):
        chars = [
            {"name": "Цвет", "required": True},
            {"name": "Размер", "required": True},
            {"name": "Материал", "required": False, "popular": True},
            {"name": "Страна", "required": False, "popular": True},
        ]
        monkeypatch.setattr(fill_scorer_mod, "get_wb_characteristics_for_category", lambda sid: chars)
        monkeypatch.setattr(fill_scorer_mod, "is_wb_platform_field", lambda c: False)
        card = {"options": [{"name": "Цвет", "value": "Красный"}, {"name": "Материал", "value": "Хлопок"}]}
        result = score_product(card, 1)
        assert result.required_total == 2
        assert result.required_filled == 1
        assert result.optional_total == 2
        assert result.optional_filled == 1
        expected_combined = (1 + 1) / (2 + 2)
        assert result.combined == pytest.approx(expected_combined)
        assert result.missing_required == ["Размер"]
        assert result.missing_optional == ["Страна"]

    def test_case_insensitive_matching(self, monkeypatch):
        chars = [
            {"name": "Цвет", "required": True},
        ]
        monkeypatch.setattr(fill_scorer_mod, "get_wb_characteristics_for_category", lambda sid: chars)
        monkeypatch.setattr(fill_scorer_mod, "is_wb_platform_field", lambda c: False)
        card = {"options": [{"name": "ЦВЕТ", "value": "Синий"}]}
        result = score_product(card, 1)
        assert result.required_filled == 1
        assert result.missing_required == []

    def test_card_none(self, monkeypatch):
        monkeypatch.setattr(fill_scorer_mod, "get_wb_characteristics_for_category", lambda sid: [{"name": "X", "required": True}])
        result = score_product(None, 1)
        assert result is None

    def test_subject_id_not_found(self, monkeypatch):
        monkeypatch.setattr(fill_scorer_mod, "get_wb_characteristics_for_category", lambda sid: [])
        result = score_product({"options": []}, 1)
        assert result is None

    def test_popular_platform_field_excluded(self, monkeypatch):
        chars = [
            {"name": "RequiredField", "required": True},
            {"name": "PopularPlatform", "required": False, "popular": True},
        ]
        monkeypatch.setattr(fill_scorer_mod, "get_wb_characteristics_for_category", lambda sid: chars)
        monkeypatch.setattr(fill_scorer_mod, "is_wb_platform_field", lambda c: True)
        card = {"options": [{"name": "RequiredField", "value": "SomeValue"}, {"name": "PopularPlatform", "value": "SomeValue"}]}
        result = score_product(card, 1)
        assert result.required_total == 1
        assert result.required_filled == 1
        assert result.optional_total == 0
        assert result.optional_filled == 0


class TestStratifiedSample:
    def test_deterministic_with_same_seed(self):
        products = [
            {"nm_id": 1, "subject_id": 1},
            {"nm_id": 2, "subject_id": 1},
            {"nm_id": 3, "subject_id": 2},
        ]
        result1 = stratified_sample(products, 3, "seed123")
        result2 = stratified_sample(products, 3, "seed123")
        assert result1 == result2

    def test_seed_affects_pick(self):
        products = [
            {"nm_id": 1, "subject_id": 1},
            {"nm_id": 2, "subject_id": 1},
        ]
        chosen_ids = set()
        for i in range(30):
            seed = f"seed_{i}"
            result = stratified_sample(products, 1, seed)
            chosen_ids.add(result[0]["nm_id"])
        assert len(chosen_ids) >= 2

    def test_proportional_quotas(self):
        products = (
            [{"nm_id": i, "subject_id": 1} for i in range(10)]
            + [{"nm_id": i + 100, "subject_id": 2} for i in range(5)]
            + [{"nm_id": 200, "subject_id": 3}]
        )
        result = stratified_sample(products, 8, "test_seed")
        counts = Counter(p["subject_id"] for p in result)
        assert all(counts[sid] >= 1 for sid in [1, 2, 3])
        assert counts[1] > counts[2]
        assert counts[1] > counts[3]

    def test_returns_all_when_size_less_than_or_equal_target(self):
        products = [
            {"nm_id": 3, "subject_id": 1},
            {"nm_id": 1, "subject_id": 2},
            {"nm_id": 2, "subject_id": 1},
        ]
        result = stratified_sample(products, 5, "seed")
        assert len(result) == 3
        assert [p["nm_id"] for p in result] == [1, 2, 3]


class TestAggregate:
    def test_empty_results_all_zeros_and_gate_false(self):
        seller = SellerInfo(id="s1", display_name="Test", display_name_masked="Test")
        unresolved = UnresolvedInfo()
        report = aggregate([], 0, 0, seller, "wb", "seed", "2024-01-01", unresolved)
        assert report.scores.combined_fill_median == 0.0
        assert report.scores.combined_fill_mean == 0.0
        assert report.scores.required_fill_median is None
        assert report.scores.optional_honest_fill_median == 0.0
        assert report.gate.thin_content_ok is False

    def test_required_fill_median_none_when_no_product_has_applicable_required_schema(self):
        """required_fill_median is None when no product has required_total > 0"""
        seller = SellerInfo(id="s1", display_name="Test", display_name_masked="Test")
        unresolved = UnresolvedInfo()
        results = [
            ProductFillResult(nm_id=1, subject_id=1, required_total=0, required_filled=0,
                              optional_total=10, optional_filled=5, combined=0.5,
                              missing_required=[], missing_optional=[]),
            ProductFillResult(nm_id=2, subject_id=1, required_total=0, required_filled=0,
                              optional_total=10, optional_filled=8, combined=0.8,
                              missing_required=[], missing_optional=[]),
        ]
        report = aggregate(results, len(results), 30, seller, "wb", "seed", "2024-01-01", unresolved)
        assert report.scores.required_fill_median is None
        assert isinstance(report.scores.combined_fill_median, float)
        assert report.scores.combined_fill_median > 0.0

    def test_required_fill_median_excludes_inapplicable_products_not_zero_pads_them(self):
        """Inapplicable products (required_total=0) are excluded from required_fill_median, not counted as 0.0"""
        seller = SellerInfo(id="s1", display_name="Test", display_name_masked="Test")
        unresolved = UnresolvedInfo()
        results = [
            ProductFillResult(nm_id=1, subject_id=1, required_total=0, required_filled=0,
                              optional_total=10, optional_filled=5, combined=0.5,
                              missing_required=[], missing_optional=[]),
            ProductFillResult(nm_id=2, subject_id=1, required_total=5, required_filled=5,
                              optional_total=10, optional_filled=10, combined=1.0,
                              missing_required=[], missing_optional=[]),
            ProductFillResult(nm_id=3, subject_id=1, required_total=3, required_filled=3,
                              optional_total=10, optional_filled=10, combined=1.0,
                              missing_required=[], missing_optional=[]),
        ]
        report = aggregate(results, len(results), 30, seller, "wb", "seed", "2024-01-01", unresolved)
        assert report.scores.required_fill_median == 1.0

    def test_by_category_required_fill_none_when_category_has_no_applicable_schema(self):
        """Category required_fill is None when all products in that category have required_total=0"""
        seller = SellerInfo(id="s1", display_name="Test", display_name_masked="Test")
        unresolved = UnresolvedInfo()
        results = [
            ProductFillResult(nm_id=1, subject_id=42, required_total=0, required_filled=0,
                              optional_total=10, optional_filled=5, combined=0.5,
                              missing_required=[], missing_optional=[]),
            ProductFillResult(nm_id=2, subject_id=42, required_total=0, required_filled=0,
                              optional_total=10, optional_filled=8, combined=0.8,
                              missing_required=[], missing_optional=[]),
        ]
        report = aggregate(results, len(results), 30, seller, "wb", "seed", "2024-01-01", unresolved)
        assert report.by_category[0].required_fill is None
        assert isinstance(report.by_category[0].optional_honest_fill, float)
        assert report.by_category[0].optional_honest_fill > 0.0

    def test_gate_true_with_large_sample_and_multiple_categories(self):
        seller = SellerInfo(id="s1", display_name="Test", display_name_masked="Test")
        unresolved = UnresolvedInfo()
        results = [
            ProductFillResult(nm_id=1, subject_id=1, required_total=1, required_filled=1, optional_total=1, optional_filled=1, combined=1.0, missing_required=[], missing_optional=[]),
            ProductFillResult(nm_id=2, subject_id=2, required_total=1, required_filled=1, optional_total=1, optional_filled=1, combined=1.0, missing_required=[], missing_optional=[]),
        ]
        report = aggregate(results, 2, 30, seller, "wb", "seed", "2024-01-01", unresolved)
        assert report.gate.thin_content_ok is True

    def test_gate_false_with_small_sample(self):
        seller = SellerInfo(id="s1", display_name="Test", display_name_masked="Test")
        unresolved = UnresolvedInfo()
        results = [
            ProductFillResult(nm_id=1, subject_id=1, required_total=1, required_filled=1, optional_total=1, optional_filled=1, combined=1.0, missing_required=[], missing_optional=[]),
            ProductFillResult(nm_id=2, subject_id=2, required_total=1, required_filled=1, optional_total=1, optional_filled=1, combined=1.0, missing_required=[], missing_optional=[]),
        ]
        report = aggregate(results, 2, 29, seller, "wb", "seed", "2024-01-01", unresolved)
        assert report.gate.thin_content_ok is False
        assert "sample_size" in report.gate.reason

    def test_distribution_buckets(self):
        seller = SellerInfo(id="s1", display_name="Test", display_name_masked="Test")
        unresolved = UnresolvedInfo()
        results = [
            ProductFillResult(nm_id=i, subject_id=1, required_total=1, required_filled=1, optional_total=1, optional_filled=1, combined=c, missing_required=[], missing_optional=[])
            for i, c in enumerate([0.1, 0.35, 0.55, 0.75, 1.0])
        ]
        report = aggregate(results, 5, 30, seller, "wb", "seed", "2024-01-01", unresolved)
        bucket_counts = {b.bucket: b.n for b in report.scores.distribution}
        assert bucket_counts == {"0-20": 1, "20-40": 1, "40-60": 1, "60-80": 1, "80-100": 1}
        assert sum(b.n for b in report.scores.distribution) == 5

    def test_top_missing_fields_sorted_by_share_descending(self):
        seller = SellerInfo(id="s1", display_name="Test", display_name_masked="Test")
        unresolved = UnresolvedInfo()
        results = [
            ProductFillResult(nm_id=1, subject_id=1, required_total=2, required_filled=0, optional_total=0, optional_filled=0, combined=0.0, missing_required=["A", "B"], missing_optional=[]),
            ProductFillResult(nm_id=2, subject_id=1, required_total=2, required_filled=0, optional_total=0, optional_filled=0, combined=0.0, missing_required=["A"], missing_optional=[]),
            ProductFillResult(nm_id=3, subject_id=1, required_total=2, required_filled=0, optional_total=0, optional_filled=0, combined=0.0, missing_required=["B"], missing_optional=[]),
        ]
        report = aggregate(results, 3, 30, seller, "wb", "seed", "2024-01-01", unresolved)
        shares = [f.missing_share for f in report.top_missing_fields_overall]
        assert shares == sorted(shares, reverse=True)


class TestMaskSellerDisplayName:
    def test_russian_fio_masked(self):
        assert mask_seller_display_name("Иванов Иван Иванович") == "Иванов И.И."

    def test_legal_entity_unchanged(self):
        assert mask_seller_display_name("ООО Ромашка") == "ООО Ромашка"
        assert mask_seller_display_name("Modern Style Shop") == "Modern Style Shop"

    def test_already_masked_unchanged(self):
        assert mask_seller_display_name("Иванов И.И.") == "Иванов И.И."


class TestWbSellerCatalog:
    @pytest.mark.asyncio
    async def test_fetch_products_stops_on_empty_page(self):
        provider = catalog_provider.WbSellerCatalog()
        mock_resp_page1 = MagicMock()
        mock_resp_page1.status_code = 200
        mock_resp_page1.json.return_value = {"products": [{"id": 1, "subjectId": 10, "subjectName": "Cat1"}, {"id": 2, "subjectId": 10, "subjectName": "Cat1"}, {"id": 3, "subjectId": 10, "subjectName": "Cat1"}, {"id": 4, "subjectId": 10, "subjectName": "Cat1"}, {"id": 5, "subjectId": 10, "subjectName": "Cat1"}], "total": 5}
        mock_resp_page2 = MagicMock()
        mock_resp_page2.status_code = 200
        mock_resp_page2.json.return_value = {"products": [], "total": 5}
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=[mock_resp_page1, mock_resp_page2])
        mock_client_cls = MagicMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_sleep = AsyncMock()
        with patch("httpx.AsyncClient", mock_client_cls), patch.object(catalog_provider, "asyncio") as mock_asyncio:
            mock_asyncio.sleep = mock_sleep
            products, total = await provider.fetch_products("test_seller", max_pages=5)
        assert total == 5
        assert len(products) == 5
        assert mock_client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_httpx_fails_falls_back_to_playwright(self):
        provider = catalog_provider.WbSellerCatalog()
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls = MagicMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_sleep = AsyncMock()
        mock_playwright = AsyncMock(return_value=([{"nm_id": 1, "subject_id": 10}], 1))
        with patch("httpx.AsyncClient", mock_client_cls), patch.object(catalog_provider, "asyncio") as mock_asyncio, patch.object(provider, "_fetch_via_playwright", mock_playwright):
            mock_asyncio.sleep = mock_sleep
            products, total = await provider.fetch_products("test_seller", max_pages=5)
        assert products == [{"nm_id": 1, "subject_id": 10}]
        assert total == 1
        mock_playwright.assert_awaited_once()
