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

import numpy as np
import httpx
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(search_client) -> WbCardSource:
    return WbCardSource(web_search_client=search_client)


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://google.serper.dev/search")
    response = httpx.Response(status_code, request=request, text="Not enough credits")
    return httpx.HTTPStatusError(
        f"{status_code} error", request=request, response=response
    )


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
# BUG 2 — permanent 4xx fails fast; transient errors still retry
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_permanent_400_fails_fast_single_attempt():
    """Serper HTTP 400 (no credits) → exactly ONE search call, no backoff/retry."""
    client = MagicMock()
    client.search = AsyncMock(side_effect=_http_status_error(400))
    src = _make_source(client)

    result = await src._search("Худи Nike detail.aspx")

    assert result == []
    assert client.search.await_count == 1, (
        f"permanent 4xx must NOT retry; got {client.search.await_count} attempts"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_permanent_4xx_variants_fail_fast(status):
    client = MagicMock()
    client.search = AsyncMock(side_effect=_http_status_error(status))
    src = _make_source(client)

    assert await src._search("q") == []
    assert client.search.await_count == 1


@pytest.mark.asyncio
async def test_transient_timeout_still_retries():
    """A transient timeout exhausts all attempts (retry behavior preserved)."""
    client = MagicMock()
    client.search = AsyncMock(side_effect=httpx.TimeoutException("read timeout"))
    src = _make_source(client)

    # Patch sleep so the test doesn't actually wait on the backoff.
    import app.services.enrichment.sources.wb_card_source as mod
    orig_sleep = mod.asyncio.sleep
    mod.asyncio.sleep = AsyncMock()
    try:
        result = await src._search("q")
    finally:
        mod.asyncio.sleep = orig_sleep

    assert result == []
    assert client.search.await_count == _SERPER_MAX_ATTEMPTS, (
        "transient error must still retry up to _SERPER_MAX_ATTEMPTS"
    )


@pytest.mark.asyncio
async def test_transient_503_still_retries():
    """HTTP 503 (server error) is transient → retried, not failed fast."""
    client = MagicMock()
    client.search = AsyncMock(side_effect=_http_status_error(503))
    src = _make_source(client)

    import app.services.enrichment.sources.wb_card_source as mod
    orig_sleep = mod.asyncio.sleep
    mod.asyncio.sleep = AsyncMock()
    try:
        result = await src._search("q")
    finally:
        mod.asyncio.sleep = orig_sleep

    assert result == []
    assert client.search.await_count == _SERPER_MAX_ATTEMPTS


class _FakeResults:
    """Minimal stand-in for SerperResults (source reads .organic_results)."""

    def __init__(self, organic):
        self.organic_results = organic


class _FakeOrganic:
    """Minimal stand-in for OrganicResult (source reads .link)."""

    def __init__(self, link):
        self.link = link


@pytest.mark.asyncio
async def test_zero_result_retries_then_succeeds():
    """Empty-200 on attempt 1 (0 organic → 0 nm_id) → retry → attempt 2 returns nm_ids.

    This is the core zero-result retry layer: Serper flakes under concurrency and
    returns 0 organic on the first try, then the SAME query returns real WB cards
    on the next. The source must retry-on-zero and surface the recovered nm_ids.
    """
    nike_hoodie_link = (
        "https://www.wildberries.ru/catalog/123456789/detail.aspx"
    )
    client = MagicMock()
    client.search = AsyncMock(side_effect=[
        _FakeResults([]),                            # attempt 1: zero flake
        _FakeResults([_FakeOrganic(nike_hoodie_link)]),  # attempt 2: recovered
    ])
    src = _make_source(client)

    import app.services.enrichment.sources.wb_card_source as mod
    orig_sleep = mod.asyncio.sleep
    mod.asyncio.sleep = AsyncMock()  # skip real backoff
    try:
        result = await src._search("Худи Nike")
    finally:
        mod.asyncio.sleep = orig_sleep

    assert result == [123456789], (
        f"retry-on-zero must recover nm_ids on attempt 2; got {result}"
    )
    assert client.search.await_count == 2, (
        f"expected exactly 2 attempts (1 zero + 1 success); "
        f"got {client.search.await_count}"
    )


@pytest.mark.asyncio
async def test_zero_result_exhausts_attempts_then_empty():
    """Persistently zero (0 organic every attempt) → all attempts used → []."""
    client = MagicMock()
    client.search = AsyncMock(return_value=_FakeResults([]))
    src = _make_source(client)

    import app.services.enrichment.sources.wb_card_source as mod
    orig_sleep = mod.asyncio.sleep
    mod.asyncio.sleep = AsyncMock()
    try:
        result = await src._search("q")
    finally:
        mod.asyncio.sleep = orig_sleep

    assert result == []
    assert client.search.await_count == _SERPER_MAX_ATTEMPTS, (
        "a genuine zero must retry up to _SERPER_MAX_ATTEMPTS"
    )


@pytest.mark.asyncio
async def test_permanent_4xx_no_retry_even_with_zero_layer():
    """4xx fail-fast is preserved on top of the zero-retry layer (single attempt)."""
    client = MagicMock()
    client.search = AsyncMock(side_effect=_http_status_error(400))
    src = _make_source(client)

    import app.services.enrichment.sources.wb_card_source as mod
    orig_sleep = mod.asyncio.sleep
    fake_sleep = AsyncMock()
    mod.asyncio.sleep = fake_sleep
    try:
        result = await src._search("q")
    finally:
        mod.asyncio.sleep = orig_sleep

    assert result == []
    assert client.search.await_count == 1, (
        "permanent 4xx must fail fast even with the zero-retry layer present"
    )
    fake_sleep.assert_not_awaited()  # no backoff burned on permanent 4xx


@pytest.mark.asyncio
async def test_transient_429_still_retries():
    """HTTP 429 (rate limit) is transient → retried (excluded from permanent gate)."""
    client = MagicMock()
    client.search = AsyncMock(side_effect=_http_status_error(429))
    src = _make_source(client)

    import app.services.enrichment.sources.wb_card_source as mod
    orig_sleep = mod.asyncio.sleep
    mod.asyncio.sleep = AsyncMock()
    try:
        result = await src._search("q")
    finally:
        mod.asyncio.sleep = orig_sleep

    assert result == []
    assert client.search.await_count == _SERPER_MAX_ATTEMPTS


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


class _FakeSearchResults:
    def __init__(self, organic):
        self.organic_results = organic


class _FakeSearchOrganic:
    def __init__(self, link):
        self.link = link


def _wb_link(nm_id: int) -> str:
    return f"https://www.wildberries.ru/catalog/{nm_id}/detail.aspx"


@pytest.mark.asyncio
async def test_article_path_taken_when_article_present():
    """When context.article is set, the first Serper call uses article query.

    Verifies the article-anchored path fires BEFORE the title-based path
    and that the resulting AttributeValues are returned.
    """
    nm_id = 123456789

    search_client = MagicMock()
    search_client.search = AsyncMock(
        return_value=_FakeSearchResults([_FakeSearchOrganic(_wb_link(nm_id))])
    )
    src = _make_source(search_client)

    card = _fake_card(nm_id)
    ctx = _make_context(article="501-0065")
    target = _make_target()

    # Patch _fetch_card to return our fake card for nm_id=123456789, else None.
    async def fake_fetch(client, fetched_nm_id):
        return card if fetched_nm_id == nm_id else None

    with patch.object(src, "_fetch_card", side_effect=fake_fetch):
        result = await src._do_extract(ctx, [target])

    # The article query was fired (first search call must contain the article).
    first_call_query = search_client.search.call_args_list[0][0][0]
    assert '"501-0065"' in first_call_query, (
        f"First Serper query should be article-anchored; got: {first_call_query!r}"
    )
    # At least one AttributeValue returned (card had Цвет).
    assert result, "Expected non-empty result from article-path card"


@pytest.mark.asyncio
async def test_no_article_uses_title_search_only():
    """When context.article is None, only the title-based path runs.

    The search query must NOT contain a double-quoted article token.
    """
    nm_id = 987654321

    search_client = MagicMock()
    search_client.search = AsyncMock(
        return_value=_FakeSearchResults([_FakeSearchOrganic(_wb_link(nm_id))])
    )
    src = _make_source(search_client)

    card = _fake_card(nm_id)
    ctx = _make_context(article=None)
    target = _make_target()

    async def fake_fetch(client, fetched_nm_id):
        return card if fetched_nm_id == nm_id else None

    import app.services.enrichment.sources.wb_card_source as mod
    orig_sleep = mod.asyncio.sleep
    mod.asyncio.sleep = AsyncMock()
    try:
        with patch.object(src, "_fetch_card", side_effect=fake_fetch):
            await src._do_extract(ctx, [target])
    finally:
        mod.asyncio.sleep = orig_sleep

    # All search calls must be title-based (no quoted article in any query).
    for call in search_client.search.call_args_list:
        q = call[0][0]
        assert '"' not in q, (
            f"Title-search query must not contain quoted article; got: {q!r}"
        )


@pytest.mark.asyncio
async def test_article_path_falls_back_to_title_when_zero_nm_ids():
    """Article search returns 0 nm_ids → fall back to title search.

    The second search call (title-based) is made, and its result is used.
    """
    title_nm_id = 111222333

    call_count = [0]

    async def side_effect(query, **kwargs):
        call_count[0] += 1
        if '"' in query:
            # Article query — return empty
            return _FakeSearchResults([])
        # Title query — return a real card link
        return _FakeSearchResults([_FakeSearchOrganic(_wb_link(title_nm_id))])

    search_client = MagicMock()
    search_client.search = AsyncMock(side_effect=side_effect)
    src = _make_source(search_client)

    card = _fake_card(title_nm_id)
    ctx = _make_context(article="NOTFOUND-99")
    target = _make_target()

    import app.services.enrichment.sources.wb_card_source as mod
    orig_sleep = mod.asyncio.sleep
    mod.asyncio.sleep = AsyncMock()
    try:
        async def fake_fetch(client, fetched_nm_id):
            return card if fetched_nm_id == title_nm_id else None

        with patch.object(src, "_fetch_card", side_effect=fake_fetch):
            result = await src._do_extract(ctx, [target])
    finally:
        mod.asyncio.sleep = orig_sleep

    # Title search must have been tried (call_count >= 2: article attempt + title attempt).
    assert call_count[0] >= 2, (
        f"Expected >=2 search calls (article + title fallback); got {call_count[0]}"
    )
    # Result comes from the title-path card.
    assert result, "Expected non-empty result from title-path fallback"


@pytest.mark.asyncio
async def test_article_path_rejects_wrong_type_card_and_falls_back():
    """Article search returns a card of the WRONG type (e.g. шорты vs куртка).

    The wrong-type card must be rejected by the type-gate; flow falls back
    to the title-based path (which in this test also returns nothing, so
    the final result is empty — NOT the wrong card).
    """
    wrong_nm_id = 444555666

    call_count = [0]

    async def side_effect(query, **kwargs):
        call_count[0] += 1
        if '"' in query:
            return _FakeSearchResults([_FakeSearchOrganic(_wb_link(wrong_nm_id))])
        # Title fallback: no results (simplifies assertion).
        return _FakeSearchResults([])

    search_client = MagicMock()
    search_client.search = AsyncMock(side_effect=side_effect)
    src = _make_source(search_client)

    # Wrong type: article query returned шорты, but target is куртка.
    wrong_card = _fake_card(
        wrong_nm_id,
        subj_name="Шорты",
        imt_name="Nike Shorts",
        options=[{"name": "Цвет", "value": "Синий"}],
    )

    import app.services.enrichment.sources.wb_card_source as mod
    orig_sleep = mod.asyncio.sleep
    mod.asyncio.sleep = AsyncMock()
    try:
        async def fake_fetch(client, fetched_nm_id):
            return wrong_card if fetched_nm_id == wrong_nm_id else None

        with patch.object(src, "_fetch_card", side_effect=fake_fetch):
            result = await src._do_extract(
                _make_context(article="ART-999"),
                [_make_target()],
            )
    finally:
        mod.asyncio.sleep = orig_sleep

    # The wrong card must NOT have been accepted.
    assert result == [], (
        "Wrong-type card from article-path must be rejected; result should be []"
    )
    # Fallback was attempted (at least 2 search calls: article + title).
    assert call_count[0] >= 2, (
        f"Expected >=2 calls (article + title fallback); got {call_count[0]}"
    )


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
