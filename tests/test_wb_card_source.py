"""Unit tests for WbCardSource query-building, search retry behavior,
and WB→Ozon attribute-name semantic fallback.

Covers:

  BUG 1 — indeclinable-noun garments ("Худи", "Пальто") break the type-word.
  BUG 2 — permanent 4xx (Serper "Not enough credits" → HTTP 400) must fail fast.
  FEAT  — semantic attr-name fallback (step 5): name-different but meaning-equal
           WB chars (e.g. "Объём чаши") resolve to the correct Ozon target via
           embedding cosine-similarity; unrelated chars DROP; already-matched
           chars bypass the fallback entirely.
"""

import urllib.parse

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.sources.wb_card_source import (
    WbCardSource,
    _build_wb_query,
    _build_wb_article_query,
    _wb_query_type_word,
    _target_type_lemma,
    _type_lemma,
    _lemma,
    _SERPER_MAX_ATTEMPTS,
    _BRAND_LINE_THRESHOLD,
)

import app.services.enrichment.sources.wb_card_source as mod


class FakeScrapflyResult:
    """Mock stand-in for ScrapflyResult (scrapedo_fetch return value)."""

    def __init__(self, success, content, status_code=200, credits_used=10, error=None):
        self.success = success
        self.content = content
        self.status_code = status_code
        self.credits_used = credits_used
        self.error = error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(search_client) -> WbCardSource:
    return WbCardSource(web_search_client=search_client)


# ---------------------------------------------------------------------------
# BUG 1 — indeclinable-noun leaf trust
# ---------------------------------------------------------------------------

def test_hoodie_leaf_trust_query_has_garment_type_not_zipper():
    """'Худи ... на молнии Nike' WITH leaf='Худи' → query LEADS with garment type Худи.

    The bug: the garment type was DROPPED and the type-word became 'молния'
    (zipper). After the fix the query must lead with the garment type 'Худи',
    and the chosen TYPE-WORD must be 'худи', not 'молния'. A residual descriptor
    like 'на молнии' inside the name is harmless once the type leads the query.
    """
    name = "Худи мужское черное на молнии Nike"
    query = _build_wb_query(name, "Nike", cat_leaf="Худи", max_tokens=5)
    low = query.lower()
    assert "худи" in low, f"garment type 'худи' missing from query: {query!r}"
    assert low.startswith("худи"), f"query must lead with garment type: {query!r}"
    # The selected type-word is the garment, NOT the feature word.
    type_word = _wb_query_type_word(name, "Худи")
    assert type_word.lower() == "худи"
    assert "молни" not in type_word.lower()


def test_hoodie_type_word_is_leaf():
    assert _wb_query_type_word(
        "Худи мужское черное на молнии Nike", "Худи"
    ).lower() == "худи"


def test_hoodie_type_lemma_from_leaf():
    # pymorphy parses indeclinable "худи" as a fabricated verb ("худить"), which
    # used to poison the type-gate. The fixed _target_type_lemma keeps the surface
    # form "худи" for non-NOUN top-parses, and — crucially — it must MATCH what the
    # card side (_card_subj_lemmas via _type_lemma) produces for the same word, so a
    # real Nike-hoodie card (subj "Худи") passes the gate.
    target_lemma = _target_type_lemma("Худи мужское черное на молнии Nike", "Худи")
    assert target_lemma == "худи"
    assert target_lemma == _type_lemma("Худи")  # gate symmetry: target == card subj


def test_coat_leaf_trust():
    """Indeclinable 'Пальто' leaf must survive as the type-word."""
    name = "Пальто женское длинное с поясом Zara"
    query = _build_wb_query(name, "Zara", cat_leaf="Пальто", max_tokens=5)
    assert "пальто" in query.lower(), query
    assert _wb_query_type_word(name, "Пальто").lower() == "пальто"


def test_no_leaf_falls_back_to_name_noun_scan():
    """Without a leaf, the leading noun of the name is used (declinable case)."""
    tw = _wb_query_type_word("Куртка мужская зимняя Nike", None)
    assert tw is not None and tw.lower() == "куртка"


# ---------------------------------------------------------------------------
# BUG 1 — regression: declinable nouns already correct stay correct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name, leaf, expected_lemma",
    [
        ("Шорты мужские спортивные Nike Dri-FIT", "Шорты", "шорты"),
        ("Джинсы мужские прямые Wrangler Texas", "Джинсы", "джинсы"),
        ("Юбка плиссированная миди черная", "Юбка", "юбка"),
    ],
)
def test_regression_declinable_type_word_unchanged(name, leaf, expected_lemma):
    query = _build_wb_query(name, None, cat_leaf=leaf, max_tokens=5)
    type_lemmas = {_lemma(t) for t in query.lower().split()}
    assert _lemma(expected_lemma) in type_lemmas, (
        f"expected type '{expected_lemma}' in query {query!r}"
    )
    # And the explicit type-word helper agrees.
    assert _lemma(_wb_query_type_word(name, leaf)) == _lemma(expected_lemma)


# ---------------------------------------------------------------------------
# FIX-13-WB: search transport swapped Serper -> scrape.do + search.wb.ru.
# Permanent-4xx-fail-fast (Serper-specific: _EgPermanentSearchError raised from
# HTTPStatusError) is DROPPED here — scrapedo_fetch never raises, it always
# returns a ScrapflyResult(success=False, ...) on any transport failure, so
# _search's retry-on-empty loop (unchanged) is the only layer left. The old
# test_permanent_400/401/403_* and test_permanent_4xx_no_retry_even_with_zero_layer
# tested a capability that no longer exists at this layer and are removed.
# test_transient_timeout/503/429_still_retries collapse into ONE test below
# (test_scrapedo_persistent_failure_still_retries) since scrape.do abstracts
# the underlying HTTP status away from the caller.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inv13a_search_once_returns_ordered_deduped_nm_ids():
    """INV-13a: mocked scrapedo_fetch (search.wb.ru JSON) -> ordered, deduped top-N nm_id."""
    orig = mod.scrapedo_fetch
    mod.scrapedo_fetch = AsyncMock(return_value=FakeScrapflyResult(
        success=True,
        content='{"products":[{"id":815621985},{"id":823775519},'
                '{"id":815621985},{"id":823776306}]}',
    ))
    try:
        src = mod.WbCardSource()
        result = await src._search_once("poco x6 5g")
        assert result == [815621985, 823775519, 823776306], (
            f"expected ordered dedup top-N; got {result}"
        )
        assert mod.scrapedo_fetch.await_count == 1
    finally:
        mod.scrapedo_fetch = orig


@pytest.mark.asyncio
async def test_inv13b_search_once_parses_json_wrapped_in_html():
    """INV-13b: JSON wrapped in an HTML envelope is still robustly extracted."""
    orig = mod.scrapedo_fetch
    mod.scrapedo_fetch = AsyncMock(return_value=FakeScrapflyResult(
        success=True,
        content='<html><body>junk {"products":[{"id":999888777}]} trailing</body></html>',
    ))
    try:
        src = mod.WbCardSource()
        result = await src._search_once("test query")
        assert result == [999888777]
    finally:
        mod.scrapedo_fetch = orig


@pytest.mark.asyncio
@pytest.mark.parametrize("scrapedo_result", [
    FakeScrapflyResult(success=False, content=None, status_code=None,
                        credits_used=0, error="not configured"),
    FakeScrapflyResult(success=True, content=""),
])
async def test_inv13c_search_once_scrapedo_failure_returns_empty(scrapedo_result):
    """INV-13c: scrapedo_fetch fail (success=False) or empty content -> [] (no exception)."""
    orig = mod.scrapedo_fetch
    mod.scrapedo_fetch = AsyncMock(return_value=scrapedo_result)
    try:
        src = mod.WbCardSource()
        result = await src._search_once("test")
        assert result == []
    finally:
        mod.scrapedo_fetch = orig


@pytest.mark.asyncio
async def test_inv13d_fetch_card_finds_basket_37():
    """INV-13d: extended _ALL_BASKET_NN (01..40) brute-forces to basket-37 (was capped at 21)."""
    src = mod.WbCardSource()
    client = AsyncMock()
    nm_id = 100000000  # vol=1000 -> primary_nn from table is NOT "37" (proves brute-force)
    assert mod._basket_nn_from_table(nm_id) != "37"

    async def fake_try_basket(client, nn, nm_id):
        return {"nm_id": nm_id} if nn == "37" else None

    with patch.object(src, "_try_basket", side_effect=fake_try_basket):
        result = await src._fetch_card(client, nm_id)
    assert result == {"nm_id": nm_id}


@pytest.mark.asyncio
async def test_zero_result_retries_then_succeeds():
    """0 products (attempt 1) -> retry -> attempt 2 returns nm_ids (retry-on-zero preserved)."""
    orig_fetch = mod.scrapedo_fetch
    orig_sleep = mod.asyncio.sleep
    mock_fetch = AsyncMock(side_effect=[
        FakeScrapflyResult(success=True, content='{"products":[]}'),
        FakeScrapflyResult(success=True, content='{"products":[{"id":123456789}]}'),
    ])
    mod.scrapedo_fetch = mock_fetch
    mod.asyncio.sleep = AsyncMock()
    try:
        src = mod.WbCardSource()
        result = await src._search("poco x6 5g")
    finally:
        mod.scrapedo_fetch = orig_fetch
        mod.asyncio.sleep = orig_sleep

    assert result == [123456789]
    assert mock_fetch.await_count == 2


@pytest.mark.asyncio
async def test_zero_result_exhausts_attempts_then_empty():
    """Persistently 0 products -> all _SERPER_MAX_ATTEMPTS used -> []."""
    orig_fetch = mod.scrapedo_fetch
    orig_sleep = mod.asyncio.sleep
    mock_fetch = AsyncMock(return_value=FakeScrapflyResult(
        success=True, content='{"products":[]}',
    ))
    mod.scrapedo_fetch = mock_fetch
    mod.asyncio.sleep = AsyncMock()
    try:
        src = mod.WbCardSource()
        result = await src._search("q")
    finally:
        mod.scrapedo_fetch = orig_fetch
        mod.asyncio.sleep = orig_sleep

    assert result == []
    assert mock_fetch.await_count == _SERPER_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_scrapedo_persistent_failure_still_retries():
    """Persistent scrape.do transport failure (success=False) still exhausts retries, no crash."""
    orig_fetch = mod.scrapedo_fetch
    orig_sleep = mod.asyncio.sleep
    mock_fetch = AsyncMock(return_value=FakeScrapflyResult(
        success=False, content=None, status_code=502, credits_used=0,
        error="Scrape.do HTTP 502",
    ))
    mod.scrapedo_fetch = mock_fetch
    mod.asyncio.sleep = AsyncMock()
    try:
        src = mod.WbCardSource()
        result = await src._search("q")
    finally:
        mod.scrapedo_fetch = orig_fetch
        mod.asyncio.sleep = orig_sleep

    assert result == []
    assert mock_fetch.await_count == _SERPER_MAX_ATTEMPTS


# ---------------------------------------------------------------------------
# Article-anchored WB lookup — STEP 2 tests
# ---------------------------------------------------------------------------

# ---- Unit: _build_wb_article_query ----

def test_article_query_contains_quoted_article():
    """Article is quoted so Serper does not tokenise it."""
    q = _build_wb_article_query("501-0065", "Levi's")
    assert '"501-0065"' in q


def test_article_query_contains_brand_when_present():
    q = _build_wb_article_query("ABC-123", "Nike")
    assert "Nike" in q
    assert "wildberries" in q.lower()


def test_article_query_no_brand_still_has_wildberries():
    q = _build_wb_article_query("ABC-123", None)
    assert '"ABC-123"' in q
    assert "wildberries" in q.lower()
    # No extra spaces from the missing brand
    assert "  " not in q


def test_article_query_blank_brand_treated_as_none():
    q_none = _build_wb_article_query("X1", None)
    q_blank = _build_wb_article_query("X1", "   ")
    assert q_none == q_blank


# ---- Integration: _do_extract with article ----

def _make_context(article=None, product_name="Куртка Nike Resolve", brand="Nike",
                  category_path=None):
    """Build a minimal ExtractionContext for tests."""
    from app.services.enrichment.base import ExtractionContext
    return ExtractionContext(
        product_id=1,
        product_name=product_name,
        category_id=100,
        category_path=category_path or ["Одежда", "Куртки"],
        brand=brand,
        article=article,
    )


def _make_target(attr_id=10, name="Цвет"):
    from app.services.enrichment.base import TargetAttribute
    return TargetAttribute(id=attr_id, name=name, type="text")


def _fake_card(nm_id: int, subj_name: str = "Куртки", imt_name: str = "Nike Resolve",
               options=None) -> dict:
    """Minimal WB card.json dict that passes type-gate for 'куртк*'."""
    return {
        "nm_id": nm_id,
        "subj_name": subj_name,
        "subj_root_name": subj_name,
        "imt_name": imt_name,
        "selling": {"brand_name": "Nike"},
        "options": options or [{"name": "Цвет", "value": "Чёрный"}],
    }


def _query_param(url: str) -> str:
    """Extract+decode the `query` param from a search.wb.ru URL (test helper)."""
    parsed = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    return parsed.get("query", [""])[0]


@pytest.mark.asyncio
async def test_article_path_taken_when_article_present():
    """When context.article is set, the first scrape.do call uses article query.

    Verifies the article-anchored path fires BEFORE the title-based path
    and that the resulting AttributeValues are returned.
    """
    nm_id = 123456789
    orig_fetch = mod.scrapedo_fetch
    mod.scrapedo_fetch = AsyncMock(return_value=FakeScrapflyResult(
        success=True, content='{"products":[{"id":123456789}]}',
    ))
    try:
        src = mod.WbCardSource()
        card = _fake_card(nm_id)
        ctx = _make_context(article="501-0065")
        target = _make_target()

        async def fake_fetch(client, fetched_nm_id):
            return card if fetched_nm_id == nm_id else None

        with patch.object(src, "_fetch_card", side_effect=fake_fetch):
            result = await src._do_extract(ctx, [target])

        first_call_url = mod.scrapedo_fetch.call_args_list[0][0][0]
        assert '"501-0065"' in _query_param(first_call_url), (
            f"First scrape.do query should be article-anchored; got: {first_call_url!r}"
        )
        assert result, "Expected non-empty result from article-path card"
    finally:
        mod.scrapedo_fetch = orig_fetch


@pytest.mark.asyncio
async def test_no_article_uses_title_search_only():
    """When context.article is None, only the title-based path runs.

    The search query must NOT contain a double-quoted article token.
    """
    nm_id = 987654321
    orig_fetch = mod.scrapedo_fetch
    orig_sleep = mod.asyncio.sleep
    mod.scrapedo_fetch = AsyncMock(return_value=FakeScrapflyResult(
        success=True, content='{"products":[{"id":987654321}]}',
    ))
    mod.asyncio.sleep = AsyncMock()
    try:
        src = mod.WbCardSource()
        card = _fake_card(nm_id)
        ctx = _make_context(article=None)
        target = _make_target()

        async def fake_fetch(client, fetched_nm_id):
            return card if fetched_nm_id == nm_id else None

        with patch.object(src, "_fetch_card", side_effect=fake_fetch):
            await src._do_extract(ctx, [target])

        for call in mod.scrapedo_fetch.call_args_list:
            q = _query_param(call[0][0])
            assert '"' not in q, (
                f"Title-search query must not contain quoted article; got: {q!r}"
            )
    finally:
        mod.scrapedo_fetch = orig_fetch
        mod.asyncio.sleep = orig_sleep


@pytest.mark.asyncio
async def test_article_path_falls_back_to_title_when_zero_nm_ids():
    """Article search returns 0 nm_ids → fall back to title search.

    The second search call (title-based) is made, and its result is used.
    """
    title_nm_id = 111222333

    async def side_effect(url, **kwargs):
        if '"' in _query_param(url):
            # Article query — return empty
            return FakeScrapflyResult(success=True, content='{"products":[]}')
        # Title query — return a real card
        return FakeScrapflyResult(
            success=True, content=f'{{"products":[{{"id":{title_nm_id}}}]}}',
        )

    orig_fetch = mod.scrapedo_fetch
    orig_sleep = mod.asyncio.sleep
    mod.scrapedo_fetch = AsyncMock(side_effect=side_effect)
    mod.asyncio.sleep = AsyncMock()
    try:
        src = mod.WbCardSource()
        card = _fake_card(title_nm_id)
        ctx = _make_context(article="NOTFOUND-99")
        target = _make_target()

        async def fake_fetch(client, fetched_nm_id):
            return card if fetched_nm_id == title_nm_id else None

        with patch.object(src, "_fetch_card", side_effect=fake_fetch):
            result = await src._do_extract(ctx, [target])

        # Title search must have been tried (>=2: article attempt + title attempt).
        assert mod.scrapedo_fetch.call_count >= 2, (
            f"Expected >=2 search calls (article + title fallback); "
            f"got {mod.scrapedo_fetch.call_count}"
        )
        # Result comes from the title-path card.
        assert result, "Expected non-empty result from title-path fallback"
    finally:
        mod.scrapedo_fetch = orig_fetch
        mod.asyncio.sleep = orig_sleep


@pytest.mark.asyncio
async def test_article_path_rejects_wrong_type_card_and_falls_back():
    """Article search returns a card of the WRONG type (e.g. шорты vs куртка).

    The wrong-type card must be rejected by the type-gate; flow falls back
    to the title-based path (which in this test also returns nothing, so
    the final result is empty — NOT the wrong card).
    """
    wrong_nm_id = 444555666

    async def side_effect(url, **kwargs):
        if '"' in _query_param(url):
            return FakeScrapflyResult(
                success=True, content=f'{{"products":[{{"id":{wrong_nm_id}}}]}}',
            )
        # Title fallback: no results (simplifies assertion).
        return FakeScrapflyResult(success=True, content='{"products":[]}')

    orig_fetch = mod.scrapedo_fetch
    orig_sleep = mod.asyncio.sleep
    mod.scrapedo_fetch = AsyncMock(side_effect=side_effect)
    mod.asyncio.sleep = AsyncMock()
    try:
        src = mod.WbCardSource()
        # Wrong type: article query returned шорты, but target is куртка.
        wrong_card = _fake_card(
            wrong_nm_id,
            subj_name="Шорты",
            imt_name="Nike Shorts",
            options=[{"name": "Цвет", "value": "Синий"}],
        )

        async def fake_fetch(client, fetched_nm_id):
            return wrong_card if fetched_nm_id == wrong_nm_id else None

        with patch.object(src, "_fetch_card", side_effect=fake_fetch):
            result = await src._do_extract(
                _make_context(article="ART-999"),
                [_make_target()],
            )

        # The wrong card must NOT have been accepted.
        assert result == [], (
            "Wrong-type card from article-path must be rejected; result should be []"
        )
        # Fallback was attempted (at least 2 search calls: article + title).
        assert mod.scrapedo_fetch.call_count >= 2, (
            f"Expected >=2 calls (article + title fallback); "
            f"got {mod.scrapedo_fetch.call_count}"
        )
    finally:
        mod.scrapedo_fetch = orig_fetch
        mod.asyncio.sleep = orig_sleep


# ---------------------------------------------------------------------------
# Semantic attr-name fallback (step 5 of _map_characteristics)
# ---------------------------------------------------------------------------

def _make_extraction_context(**kwargs):
    """Build a minimal ExtractionContext for _map_characteristics unit tests."""
    from app.services.enrichment.base import ExtractionContext
    defaults = dict(
        product_id=1,
        product_name="Блендер мощный",
        category_id=100,
        category_path=[],
        brand=None,
    )
    defaults.update(kwargs)
    return ExtractionContext(**defaults)


def _make_target_attr(attr_id: int, name: str):
    from app.services.enrichment.base import TargetAttribute
    return TargetAttribute(id=attr_id, name=name, type="text")


def _fake_matcher_high_sim(query_name: str, target_names: list[str]) -> int | None:
    """Mock for _semantic_attr_name_match that returns index 0 always (high similarity)."""
    return 0


def _fake_matcher_low_sim(query_name: str, target_names: list[str]) -> int | None:
    """Mock for _semantic_attr_name_match that always returns None (low similarity)."""
    return None


class _FakeSearchClientNoop(MagicMock):
    """Search client that should never be called in _map_characteristics unit tests."""


def _make_src() -> "WbCardSource":
    from app.services.enrichment.sources.wb_card_source import WbCardSource
    return WbCardSource(web_search_client=_FakeSearchClientNoop())


def test_semantic_fallback_maps_meaning_equal_but_name_different_char():
    """WB 'Объём чаши' has no exact/substring/fuzzy≥88 match but the Ozon target
    is 'Объём' — semantic fallback (mocked to return index 0) resolves it.
    """
    import app.services.enrichment.sources.wb_card_source as mod

    src = _make_src()
    ctx = _make_extraction_context()
    targets = [_make_target_attr(42, "Объём")]
    chars = [{"name": "Объём чаши", "value": "2 л"}]

    with (
        patch.object(mod, "_semantic_attr_name_match", side_effect=_fake_matcher_high_sim),
        patch(
            "app.services.enrichment.sources.wb_card_source.get_ozon_characteristics_for_type",
            return_value=[],
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.eg_get_field_map",
            return_value={},
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.resolve_value_id",
            return_value=None,
        ),
    ):
        result = src._map_characteristics(
            chars=chars,
            targets=targets,
            context=ctx,
            mode="exact",
            title="Блендер мощный",
            score=90.0,
        )

    assert len(result) == 1, (
        f"Semantic fallback must resolve 'Объём чаши'→'Объём'; got {result}"
    )
    assert result[0].attribute_id == 42
    assert result[0].value == "2 л"


def test_semantic_fallback_drops_unrelated_char():
    """WB 'Артикул производителя' (low similarity to 'Объём') must be DROPPED,
    not mis-mapped, when the semantic matcher returns None.
    """
    import app.services.enrichment.sources.wb_card_source as mod

    src = _make_src()
    ctx = _make_extraction_context()
    targets = [_make_target_attr(42, "Объём")]
    chars = [{"name": "Артикул производителя", "value": "BL-1234"}]

    with (
        patch.object(mod, "_semantic_attr_name_match", side_effect=_fake_matcher_low_sim),
        patch(
            "app.services.enrichment.sources.wb_card_source.get_ozon_characteristics_for_type",
            return_value=[],
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.eg_get_field_map",
            return_value={},
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.resolve_value_id",
            return_value=None,
        ),
    ):
        result = src._map_characteristics(
            chars=chars,
            targets=targets,
            context=ctx,
            mode="exact",
            title="Блендер мощный",
            score=90.0,
        )

    assert result == [], (
        f"Unrelated char must be DROPPED (semantic fallback returned None); got {result}"
    )


def test_semantic_fallback_not_invoked_when_already_matched():
    """A char 'Объём' that exactly matches target 'Объём' resolves at step 2
    (exact name), so the semantic fallback must NOT be called.
    """
    import app.services.enrichment.sources.wb_card_source as mod

    src = _make_src()
    ctx = _make_extraction_context()
    targets = [_make_target_attr(42, "Объём")]
    # char name matches target name exactly → resolved at step 2, no fallback needed
    chars = [{"name": "Объём", "value": "2 л"}]

    call_tracker = {"called": False}

    def tracking_fallback(wb_char_name, target_names):
        call_tracker["called"] = True
        return None

    with (
        patch.object(mod, "_semantic_attr_name_match", side_effect=tracking_fallback),
        patch(
            "app.services.enrichment.sources.wb_card_source.get_ozon_characteristics_for_type",
            return_value=[],
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.eg_get_field_map",
            return_value={},
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.resolve_value_id",
            return_value=None,
        ),
    ):
        result = src._map_characteristics(
            chars=chars,
            targets=targets,
            context=ctx,
            mode="exact",
            title="Блендер мощный",
            score=90.0,
        )

    assert not call_tracker["called"], (
        "Semantic fallback must NOT be called when char was already matched at steps 1-4"
    )
    assert len(result) == 1, "Exact-match char must still be resolved"
    assert result[0].attribute_id == 42


# ---------------------------------------------------------------------------
# _extract_options: compositions[] field mapping (Task 1 — apparel surface)
# ---------------------------------------------------------------------------

class TestExtractOptionsCompositions:
    """Verify _extract_options handles all WB compositions[] schema variants."""

    def test_plain_value_field(self):
        """compositions[{name, value}] → single 'Состав' entry with pct%."""
        card = {
            "compositions": [
                {"name": "хлопок", "value": "80"},
                {"name": "полиэстер", "value": "20"},
            ]
        }
        opts = WbCardSource._extract_options(card)
        sostav = next((o for o in opts if o["name"] == "Состав"), None)
        assert sostav is not None, "_extract_options must emit 'Состав' from compositions[]"
        assert "хлопок" in sostav["value"]
        assert "полиэстер" in sostav["value"]

    def test_percentage_field(self):
        """compositions[{name, percentage}] — percentage key instead of value."""
        card = {
            "compositions": [
                {"name": "шерсть", "percentage": 90},
                {"name": "эластан", "percentage": 10},
            ]
        }
        opts = WbCardSource._extract_options(card)
        sostav = next((o for o in opts if o["name"] == "Состав"), None)
        assert sostav is not None
        assert "шерсть" in sostav["value"]
        assert "90%" in sostav["value"]

    def test_string_list_compositions(self):
        """compositions[str] variant — list of plain strings."""
        card = {"compositions": ["хлопок 80%", "полиэстер 20%"]}
        opts = WbCardSource._extract_options(card)
        sostav = next((o for o in opts if o["name"] == "Состав"), None)
        assert sostav is not None
        assert "хлопок" in sostav["value"]

    def test_typed_composition_lining(self):
        """compositions[{name, value, type: 'подкладка'}] → 'Материал подкладки' field."""
        card = {
            "compositions": [
                {"name": "хлопок", "value": "100", "type": "основной"},
                {"name": "полиэстер", "value": "100", "type": "подкладка"},
            ]
        }
        opts = WbCardSource._extract_options(card)
        names = [o["name"] for o in opts]
        # Main composition under "Состав" (mapped from "основной" type)
        assert "Состав" in names, f"Expected 'Состав' in {names}"
        # Lining under "Материал подкладки"
        assert "Материал подкладки" in names, f"Expected 'Материал подкладки' in {names}"

    def test_typed_composition_insulation(self):
        """compositions[{name, value, type: 'утеплитель'}] → 'Материал утеплителя'."""
        card = {
            "compositions": [
                {"name": "синтепон", "value": "100", "type": "утеплитель"},
            ]
        }
        opts = WbCardSource._extract_options(card)
        names = [o["name"] for o in opts]
        assert "Материал утеплителя" in names, f"Expected 'Материал утеплителя' in {names}"
        mat_utep = next(o for o in opts if o["name"] == "Материал утеплителя")
        assert "синтепон" in mat_utep["value"]

    def test_options_plus_compositions_combined(self):
        """options[] items AND compositions[] both appear in the output, deduped."""
        card = {
            "options": [
                {"name": "Материал подкладки", "value": "вискоза"},
                {"name": "Пол", "value": "Мужской"},
                {"name": "Страна производства", "value": "Китай"},
            ],
            "compositions": [
                {"name": "хлопок", "value": "80"},
                {"name": "полиэстер", "value": "20"},
            ],
        }
        opts = WbCardSource._extract_options(card)
        names = [o["name"].lower() for o in opts]
        # options[] fields present
        assert "материал подкладки" in names
        assert "пол" in names
        assert "страна производства" in names
        # compositions → Состав
        assert "состав" in names

    def test_compositions_does_not_duplicate_options_lining(self):
        """If 'материал подкладки' is already in options[], typed comp should NOT emit duplicate."""
        card = {
            "options": [
                {"name": "Материал подкладки", "value": "вискоза"},
            ],
            "compositions": [
                {"name": "полиэстер", "value": "100", "type": "подкладка"},
            ],
        }
        opts = WbCardSource._extract_options(card)
        lining_entries = [o for o in opts if o["name"].lower() == "материал подкладки"]
        # Dedup: only ONE entry for lining (the first wins — options[] came before compositions[])
        assert len(lining_entries) == 1, (
            "Dedup must prevent duplicate 'Материал подкладки' entries"
        )
