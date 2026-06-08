"""Unit tests for WbCardSource query-building and search retry behavior.

Covers two diagnosed bugs (no live API — collaborators mocked):

  BUG 1 — indeclinable-noun garments ("Худи", "Пальто") break the type-word:
    pymorphy3 parses them as non-nouns, so the old noun-gate dropped the garment
    type and the name-scan picked a feature word ("молния"). Fix: when a category
    leaf exists, trust the leaf's first significant token UNCONDITIONALLY.

  BUG 2 — permanent 4xx (Serper "Not enough credits" → HTTP 400) must fail fast
    (exactly ONE attempt, no backoff), while transient errors still retry.
"""

import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.enrichment.sources.wb_card_source import (
    WbCardSource,
    _build_wb_query,
    _wb_query_type_word,
    _target_type_lemma,
    _type_lemma,
    _lemma,
    _SERPER_MAX_ATTEMPTS,
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
