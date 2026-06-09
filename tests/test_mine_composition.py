"""Unit tests for upgraded mine_composition (LLM recall booster + verbatim gate).

Covers:
  - regex-first path (no LLM call when regex hits)
  - multi-query broader search (2-3 queries)
  - top-5 URL fetching
  - LLM path triggers only when regex found nothing
  - ACCEPTANCE GATE: rejects is_our_product=False
  - ACCEPTANCE GATE: rejects evidence_quote NOT in page text (hallucination)
  - ACCEPTANCE GATE: accepts blog/review page where quote is verbatim + has composition
  - LEVIS REGRESSION: review page with "99% хлопка, 1% эластан" must now be ACCEPTED
  - budget guard: LLM skipped when llm_calls_so_far >= budget
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.services.enrichment.websearch_producer import (
    WebSearchProducer,
    _parse_composition_llm_response,
    _dedup_list,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_organic_result(link: str, snippet: str = "", title: str = "Test") -> MagicMock:
    r = MagicMock()
    r.link = link
    r.snippet = snippet
    r.title = title
    r.position = 1
    return r


def _make_search_results(urls: list[str]) -> MagicMock:
    results = MagicMock()
    results.organic_results = [_make_organic_result(u) for u in urls]
    return results


def _make_fetch_result(url: str, raw_html: str = "", content: str = "") -> MagicMock:
    fr = MagicMock()
    fr.url = url
    fr.raw_html = raw_html
    fr.content = content
    return fr


def _make_producer_with_serper(serper_mock) -> WebSearchProducer:
    p = WebSearchProducer.__new__(WebSearchProducer)
    p._serper = serper_mock
    p._extractor = None
    p._use_serper = True
    p._legacy_client = None
    p._legacy_model = "gpt-4o"
    return p


def run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# _parse_composition_llm_response
# ---------------------------------------------------------------------------

class TestParseCompositionLlmResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "composition": "99% хлопок, 1% эластан",
            "material_primary": "хлопок",
            "evidence_quote": "99% хлопка, 1% эластан",
            "source_hint": "soberger.ru",
            "is_our_product": True,
            "confidence": 0.85,
        })
        result = _parse_composition_llm_response(raw)
        assert result is not None
        assert result["composition"] == "99% хлопок, 1% эластан"
        assert result["is_our_product"] is True

    def test_markdown_fenced_json(self):
        raw = "```json\n{\"composition\": \"82% cotton\", \"is_our_product\": true, \"evidence_quote\": \"82% cotton\", \"source_hint\": \"site.com\", \"confidence\": 0.8, \"material_primary\": \"cotton\"}\n```"
        result = _parse_composition_llm_response(raw)
        assert result is not None
        assert result["composition"] == "82% cotton"

    def test_empty_string_returns_none(self):
        assert _parse_composition_llm_response("") is None

    def test_invalid_json_returns_none(self):
        assert _parse_composition_llm_response("not json at all") is None

    def test_null_composition(self):
        raw = json.dumps({
            "composition": None,
            "is_our_product": True,
            "evidence_quote": "nothing here",
            "source_hint": "site.com",
            "confidence": 0.3,
            "material_primary": None,
        })
        result = _parse_composition_llm_response(raw)
        assert result is not None
        assert result["composition"] is None


# ---------------------------------------------------------------------------
# _dedup_list
# ---------------------------------------------------------------------------

class TestDedupList:
    def test_removes_duplicates_case_insensitive(self):
        assert _dedup_list(["A", "b", "a", "B"]) == ["A", "b"]

    def test_preserves_order(self):
        assert _dedup_list(["x", "y", "z"]) == ["x", "y", "z"]

    def test_empty(self):
        assert _dedup_list([]) == []


# ---------------------------------------------------------------------------
# mine_composition — no Serper path
# ---------------------------------------------------------------------------

class TestMineCompositionNoSerper:
    def test_returns_empty_when_no_serper(self):
        p = WebSearchProducer.__new__(WebSearchProducer)
        p._use_serper = False
        p._serper = None
        p._extractor = None
        result = run(p.mine_composition("Champion hoodie", brand="Champion"))
        assert result == []

    def test_returns_empty_when_no_product_name(self):
        p = WebSearchProducer.__new__(WebSearchProducer)
        p._use_serper = True
        p._serper = MagicMock()
        p._extractor = None
        result = run(p.mine_composition("", brand="Champion"))
        assert result == []


# ---------------------------------------------------------------------------
# mine_composition — broader search (2-3 query variants)
# ---------------------------------------------------------------------------

class TestMineCompositionBroaderSearch:
    def test_issues_multiple_queries_for_latin_brand(self):
        """Latin brand → 3 queries (RU состав, RU материал, EN composition)."""
        serper = MagicMock()
        empty_results = MagicMock()
        empty_results.organic_results = []
        serper.search = AsyncMock(return_value=empty_results)

        p = _make_producer_with_serper(serper)

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[])):
            run(p.mine_composition("Champion Reverse Weave Hoodie", brand="Champion"))

        call_queries = [c.args[0] for c in serper.search.call_args_list]
        # Should have exactly 3 calls for Latin brand
        assert len(call_queries) == 3
        assert any("состав" in q for q in call_queries)
        assert any("материал" in q for q in call_queries)
        assert any("material composition" in q for q in call_queries)

    def test_issues_two_queries_for_cyrillic_brand(self):
        """Non-Latin brand → only 2 queries (RU только)."""
        serper = MagicMock()
        empty_results = MagicMock()
        empty_results.organic_results = []
        serper.search = AsyncMock(return_value=empty_results)

        p = _make_producer_with_serper(serper)

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[])):
            run(p.mine_composition("Толстовка Спортмастер", brand="Спортмастер"))

        call_queries = [c.args[0] for c in serper.search.call_args_list]
        assert len(call_queries) == 2

    def test_deduplicates_urls_across_queries(self):
        """Same URL appearing in multiple query results is fetched only once."""
        serper = MagicMock()
        shared_url = "https://example.com/product"
        results_with_url = _make_search_results([shared_url])
        serper.search = AsyncMock(return_value=results_with_url)

        p = _make_producer_with_serper(serper)

        fetched_urls: list[list[str]] = []

        async def mock_fetch(urls):
            fetched_urls.append(list(urls))
            return []

        with patch("app.services.url_fetcher.fetch_all_results", new=mock_fetch):
            run(p.mine_composition("Champion hoodie", brand="Champion"))

        # fetch_all_results called once with deduped URLs
        assert len(fetched_urls) == 1
        assert fetched_urls[0].count(shared_url) == 1

    def test_fetches_up_to_5_urls(self):
        """Collect up to 5 unique URLs across queries."""
        serper = MagicMock()
        call_count = [0]

        def make_results(_call_idx):
            urls = [
                f"https://site{_call_idx * 3 + i}.com/p"
                for i in range(3)
            ]
            return _make_search_results(urls)

        async def fake_search(query, num_results=5):
            idx = call_count[0]
            call_count[0] += 1
            return make_results(idx)

        serper.search = fake_search

        p = _make_producer_with_serper(serper)

        fetched_urls: list[list[str]] = []

        async def mock_fetch(urls):
            fetched_urls.append(list(urls))
            return []

        with patch("app.services.url_fetcher.fetch_all_results", new=mock_fetch):
            run(p.mine_composition("Champion hoodie", brand="Champion"))

        if fetched_urls:
            assert len(fetched_urls[0]) <= 5


# ---------------------------------------------------------------------------
# mine_composition — regex-first path (no LLM call)
# ---------------------------------------------------------------------------

class TestMineCompositionRegexFirst:
    def test_regex_hit_returns_without_llm_call(self):
        """When regex finds composition, LLM must NOT be called."""
        page_html = (
            "<html><head><title>Champion Reverse Weave Hoodie</title></head>"
            "<body><p>Состав: 80% хлопок, 20% полиэстер</p></body></html>"
        )
        serper = MagicMock()
        serper.search = AsyncMock(return_value=_make_search_results(["https://kixbox.ru/champion"]))

        p = _make_producer_with_serper(serper)
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        fetch_result = _make_fetch_result("https://kixbox.ru/champion", raw_html=page_html)

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[fetch_result])):
            result = run(p.mine_composition(
                "Champion Reverse Weave Hoodie",
                brand="Champion",
                llm_provider=mock_llm,
            ))

        # LLM must NOT have been called
        mock_llm.complete.assert_not_called()
        assert result
        assert any("хлопок" in r for r in result)

    def test_regex_hit_returns_composition(self):
        """Verify the actual composition string returned on regex hit."""
        page_html = (
            "<div><h1>Champion Hoodie</h1>"
            "<p>Материал: 79% хлопок, 21% полиэстер</p></div>"
        )
        serper = MagicMock()
        serper.search = AsyncMock(return_value=_make_search_results(["https://kixbox.ru/champ"]))
        p = _make_producer_with_serper(serper)
        fetch_result = _make_fetch_result("https://kixbox.ru/champ", raw_html=page_html)

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[fetch_result])):
            result = run(p.mine_composition("Champion Hoodie", brand="Champion"))

        assert result
        joined = " ".join(result)
        assert "хлопок" in joined

    def test_regex_skips_brand_mismatched_page(self):
        """Brand/model mismatch on a page → regex skipped for that page."""
        # Page mentions Nike, not Champion
        page_html = (
            "<html><head><title>Nike Hoodie</title></head>"
            "<body><p>Состав: 80% хлопок, 20% полиэстер</p></body></html>"
        )
        serper = MagicMock()
        serper.search = AsyncMock(return_value=_make_search_results(["https://nike.com/hoodie"]))
        p = _make_producer_with_serper(serper)
        fetch_result = _make_fetch_result("https://nike.com/hoodie", raw_html=page_html)

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[fetch_result])):
            result = run(p.mine_composition("Champion Reverse Weave", brand="Champion"))

        # Should be empty (Nike page rejected for Champion product)
        assert result == []


# ---------------------------------------------------------------------------
# mine_composition — LLM recall path
# ---------------------------------------------------------------------------

class TestMineCompositionLlmPath:
    def _make_llm_response(self, payload: dict) -> MagicMock:
        resp = MagicMock()
        resp.content = json.dumps(payload)
        return resp

    def _valid_llm_payload(self, page_text: str) -> dict:
        """Valid payload where evidence_quote is a verbatim substring of page_text."""
        quote = "99% хлопка, 1% эластан"
        assert quote in page_text, "Test setup error: quote must be in page_text"
        return {
            "composition": "99% хлопок, 1% эластан",
            "material_primary": "хлопок",
            "evidence_quote": quote,
            "source_hint": "soberger.ru",
            "is_our_product": True,
            "confidence": 0.85,
        }

    def test_llm_triggered_when_regex_finds_nothing(self):
        """LLM is called when regex finds no composition."""
        page_text = "Levi's 501 jeans — это иконический фасон. Настоящий деним."
        page_html = f"<html><body><p>Levi's 501 — это иконический фасон.</p></body></html>"

        serper = MagicMock()
        serper.search = AsyncMock(return_value=_make_search_results(["https://soberger.ru/levis"]))
        p = _make_producer_with_serper(serper)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=self._make_llm_response({
            "composition": None,
            "is_our_product": True,
            "evidence_quote": "",
            "source_hint": "soberger.ru",
            "confidence": 0.3,
            "material_primary": None,
        }))

        fetch_result = _make_fetch_result("https://soberger.ru/levis", raw_html=page_html)

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[fetch_result])):
            run(p.mine_composition(
                "Джинсы Levis 501",
                brand="Levis",
                llm_provider=mock_llm,
            ))

        mock_llm.complete.assert_called_once()

    def test_acceptance_gate_rejects_is_our_product_false(self):
        """is_our_product=False → rejected, returns [].

        Page contains no regex-extractable composition (just a description),
        so the LLM path is taken. LLM returns is_our_product=False → gate rejects.
        """
        # No composition label/% in page → regex finds nothing → LLM is called
        page_html = (
            "<html><body>"
            "<p>Levi's 501 — классические джинсы из плотного денима. "
            "Фирменный крой и посадка прямого силуэта.</p>"
            "</body></html>"
        )
        serper = MagicMock()
        serper.search = AsyncMock(return_value=_make_search_results(["https://soberger.ru/levis"]))
        p = _make_producer_with_serper(serper)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=self._make_llm_response({
            "composition": "99% хлопок, 1% эластан",
            "material_primary": "хлопок",
            "evidence_quote": "плотного денима",  # verbatim in page, has material-ish word
            "source_hint": "soberger.ru",
            "is_our_product": False,   # <-- NOT our product
            "confidence": 0.4,
        }))

        fetch_result = _make_fetch_result("https://soberger.ru/levis", raw_html=page_html)

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[fetch_result])):
            result = run(p.mine_composition(
                "Levi's 501",
                brand="Levis",
                llm_provider=mock_llm,
            ))

        assert result == []

    def test_acceptance_gate_rejects_hallucinated_evidence_quote(self):
        """evidence_quote NOT verbatim in page text → hallucination guard fires → []."""
        page_html = (
            "<html><body>"
            "<p>Levi's 501 — classic denim jeans.</p>"
            "</body></html>"
        )
        serper = MagicMock()
        serper.search = AsyncMock(return_value=_make_search_results(["https://soberger.ru/levis"]))
        p = _make_producer_with_serper(serper)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=self._make_llm_response({
            "composition": "99% хлопок, 1% эластан",
            "material_primary": "хлопок",
            "evidence_quote": "INVENTED: 99% cotton, 1% elastane — NOT IN PAGE",  # hallucinated
            "source_hint": "soberger.ru",
            "is_our_product": True,
            "confidence": 0.9,
        }))

        fetch_result = _make_fetch_result("https://soberger.ru/levis", raw_html=page_html)

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[fetch_result])):
            result = run(p.mine_composition(
                "Levi's 501",
                brand="Levis",
                llm_provider=mock_llm,
            ))

        assert result == [], "Hallucinated evidence_quote must be rejected"

    def test_acceptance_gate_accepts_blog_page_with_verbatim_quote(self):
        """Blog/review page (not a PDP) with verbatim quote must be ACCEPTED.

        This is the Levi's 501 regression: soberger.ru is a fake-vs-original
        article, not a PDP, but it contains the correct composition verbatim.
        Old gate rejected it. New gate accepts it.
        """
        composition_sentence = "99% хлопка, 1% эластан"
        page_html = (
            "<html><head><title>Levi's 501 оригинал или подделка</title></head>"
            "<body>"
            "<p>В этой статье мы разберёмся как отличить оригинальные джинсы Levi's 501.</p>"
            f"<p>Состав ткани оригинала: {composition_sentence} — именно такой указан на ярлыке.</p>"
            "<p>Никаких других материалов в оригинале нет.</p>"
            "</body></html>"
        )
        serper = MagicMock()
        serper.search = AsyncMock(return_value=_make_search_results(["https://soberger.ru/levis501"]))
        p = _make_producer_with_serper(serper)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=self._make_llm_response({
            "composition": "99% хлопок, 1% эластан",
            "material_primary": "хлопок",
            "evidence_quote": composition_sentence,  # VERBATIM from page
            "source_hint": "soberger.ru",
            "is_our_product": True,
            "confidence": 0.85,
        }))

        fetch_result = _make_fetch_result(
            "https://soberger.ru/levis501",
            raw_html=page_html,
        )

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[fetch_result])):
            result = run(p.mine_composition(
                "Levi's 501 Original Джинсы",
                brand="Levis",
                llm_provider=mock_llm,
            ))

        assert result, "Blog/review page with verbatim quote must be ACCEPTED"
        assert any("хлопок" in r or "хлопка" in r for r in result)

    def test_acceptance_gate_rejects_quote_without_composition_signal(self):
        """evidence_quote exists verbatim in page but has no % or material word → rejected."""
        page_html = (
            "<html><body>"
            "<p>Levi's 501 — great jeans, very popular</p>"
            "</body></html>"
        )
        serper = MagicMock()
        serper.search = AsyncMock(return_value=_make_search_results(["https://site.com/levis"]))
        p = _make_producer_with_serper(serper)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=self._make_llm_response({
            "composition": "denim",
            "material_primary": None,
            "evidence_quote": "great jeans, very popular",  # verbatim but NO % or material word
            "source_hint": "site.com",
            "is_our_product": True,
            "confidence": 0.5,
        }))

        fetch_result = _make_fetch_result("https://site.com/levis", raw_html=page_html)

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[fetch_result])):
            result = run(p.mine_composition(
                "Levi's 501",
                brand="Levis",
                llm_provider=mock_llm,
            ))

        assert result == []

    def test_budget_guard_skips_llm(self):
        """When llm_calls_so_far >= budget, LLM is NOT called."""
        page_html = (
            "<html><body>"
            "<p>Champion hoodie — great product</p>"
            "</body></html>"
        )
        serper = MagicMock()
        serper.search = AsyncMock(return_value=_make_search_results(["https://example.com/champ"]))
        p = _make_producer_with_serper(serper)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        fetch_result = _make_fetch_result("https://example.com/champ", raw_html=page_html)

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[fetch_result])):
            result = run(p.mine_composition(
                "Champion Reverse Weave Hoodie",
                brand="Champion",
                llm_provider=mock_llm,
                llm_calls_budget=3,
                llm_calls_so_far=3,  # at budget limit
            ))

        mock_llm.complete.assert_not_called()
        assert result == []


# ---------------------------------------------------------------------------
# LEVIS REGRESSION — the key acceptance test
# ---------------------------------------------------------------------------

class TestLevisRegression:
    """Validate the fix for the Levi's 501 false-reject.

    soberger.ru was rejected by the OLD gate because it was a review article,
    not a product listing. The correct composition "99% хлопка, 1% эластан"
    was present on the page but thrown away.

    The NEW gate (verbatim quote + is_our_product) ACCEPTS this page.
    """

    def test_levis_501_review_page_accepted(self):
        """Review article with verbatim composition quote must be accepted."""
        composition_sentence = "99% хлопка, 1% эластан"
        page_html = (
            "<html>"
            "<head><title>Как отличить оригинальные Levi's 501 от подделки</title></head>"
            "<body>"
            "<h1>Levi's 501 Original — тест подлинности</h1>"
            "<p>Один из способов проверить — состав ткани. "
            f"У оригинальных Levi's 501: {composition_sentence}.</p>"
            "<p>Если состав другой — перед вами подделка.</p>"
            "</body></html>"
        )

        serper = MagicMock()
        serper.search = AsyncMock(return_value=_make_search_results(["https://soberger.ru/levis501-check"]))

        producer = _make_producer_with_serper(serper)

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=MagicMock(content=json.dumps({
            "composition": "99% хлопок, 1% эластан",
            "material_primary": "хлопок",
            "evidence_quote": composition_sentence,
            "source_hint": "soberger.ru",
            "is_our_product": True,
            "confidence": 0.88,
        })))

        fetch_result = _make_fetch_result(
            "https://soberger.ru/levis501-check",
            raw_html=page_html,
        )

        with patch("app.services.url_fetcher.fetch_all_results", new=AsyncMock(return_value=[fetch_result])):
            result = run(producer.mine_composition(
                "Джинсы мужские Levis 501 Original",
                brand="Levis",
                llm_provider=mock_llm,
            ))

        assert result, (
            "REGRESSION: Levi's review page with verbatim composition must be ACCEPTED. "
            f"Got: {result}"
        )
        first = result[0]
        assert "хлопок" in first or "хлопка" in first, f"Expected хлопок in {first!r}"
