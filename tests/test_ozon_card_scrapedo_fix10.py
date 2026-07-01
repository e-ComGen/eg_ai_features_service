"""Unit tests for FIX-10: OzonCardSource fetch transport swap Scrappey -> scrape.do.

Manifest: docs/MANIFEST_ozon_card_scrapedo.md
Covers INV-10a..e:
  a) is_applicable/extract return [] without SCRAPEDO_TOKEN (self-gate preserved).
  b) extract-path fetch calls scrapedo_fetch with the expected Ozon URL and
     render=True/super_proxy=True/geo="ru".
  c) parser-contract unchanged: on a mock Ozon /features/ HTML fixture with >=3
     characteristics, the parsed values are unchanged (parsing itself was NOT touched).
  d) zero references to Scrappey / publisher.scrappey.com / SCRAPPEY_KEY /
     _SCRAPPEY_ENDPOINT remain in ozon_card_source.py (grep = 0). The lowercase
     backward-compat constructor kwarg `scrappey_key` is intentionally kept and is
     NOT one of these forbidden identifiers.
  e) source_type == Source.OZON_CARD (unchanged).

No real network — scrapedo_fetch is monkeypatched at module level everywhere.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.services.enrichment.base import ExtractionContext, Source
from app.services.enrichment.sources import ozon_card_source as ocs
from app.services.providers.scrapfly_client import ScrapflyResult

_OZON_SOURCE_FILE = Path(ocs.__file__)


def _ctx(name="Тестовый Товар Модель X100", brand=""):
    return ExtractionContext(
        product_id=1,
        product_name=name,
        category_id=1,
        category_path=["Электроника", "Тест"],
        brand=brand,
    )


# ---------------------------------------------------------------------------
# INV-10a: self-gate on SCRAPEDO_TOKEN
# ---------------------------------------------------------------------------

def test_inv10a_is_applicable_false_without_token(monkeypatch):
    monkeypatch.delenv("SCRAPEDO_TOKEN", raising=False)
    src = ocs.OzonCardSource()
    assert src._scrapedo_token is None
    assert src.is_applicable(_ctx(), target=None) is False


@pytest.mark.asyncio
async def test_inv10a_extract_returns_empty_without_token(monkeypatch):
    monkeypatch.delenv("SCRAPEDO_TOKEN", raising=False)
    src = ocs.OzonCardSource()
    out = await src.extract(_ctx(), targets=[object()])
    assert out == []


def test_inv10a_backward_compat_kwarg_still_accepted(monkeypatch):
    """Constructing with the legacy scrappey_key kwarg must not raise (back-compat)."""
    monkeypatch.setenv("SCRAPEDO_TOKEN", "fake-token")
    src = ocs.OzonCardSource(scrappey_key="legacy-value-ignored")
    assert src._scrapedo_token == "fake-token"
    assert src.is_applicable(_ctx(), target=None) is True


# ---------------------------------------------------------------------------
# INV-10b + INV-10c: scrapedo_fetch call params + parser contract unchanged
# ---------------------------------------------------------------------------

_FEATURES_HTML_FIXTURE = """
<html><body>
<div id="state-webCharacteristics-123-default-1" data-state='{"characteristics":[
  {"short":[
    {"name":"Цвет","values":[{"text":"Черный","id":101}]},
    {"name":"Вес","values":[{"text":"500 г","id":102}]},
    {"name":"Материал","values":[{"text":"Хлопок","id":103}]}
  ]}
]}'></div>
</body></html>
"""

_SEARCH_HTML_FIXTURE = """
<html><body>
<a href="/product/testovyi-tovar-model-x100-999888/">
  <span class="tsBody">Тестовый Товар Модель X100</span>
</a>
</body></html>
"""


def _make_source(monkeypatch, token="fake-token"):
    monkeypatch.setenv("SCRAPEDO_TOKEN", token)
    return ocs.OzonCardSource()


@pytest.mark.asyncio
async def test_inv10b_fetch_page_calls_scrapedo_fetch_with_ozon_params(monkeypatch):
    """_fetch_page must delegate to scrapedo_fetch(render=True, super_proxy=True, geo='ru')."""
    src = _make_source(monkeypatch)

    fake = AsyncMock(return_value=ScrapflyResult(
        success=True, content="x" * 100, status_code=200, credits_used=1, error=None,
    ))
    monkeypatch.setattr(ocs, "scrapedo_fetch", fake)

    url = "https://www.ozon.ru/product/some-slug-123/features/"
    result = await src._fetch_page(client=None, target_url=url)

    assert result == "x" * 100
    fake.assert_awaited_once_with(url, render=True, super_proxy=True, geo="ru")


@pytest.mark.asyncio
async def test_inv10b_fetch_page_returns_none_on_failure(monkeypatch):
    src = _make_source(monkeypatch)
    fake = AsyncMock(return_value=ScrapflyResult(
        success=False, content=None, status_code=404, credits_used=0, error="not found",
    ))
    monkeypatch.setattr(ocs, "scrapedo_fetch", fake)

    result = await src._fetch_page(client=None, target_url="https://www.ozon.ru/x")
    assert result is None


@pytest.mark.asyncio
async def test_inv10bc_fetch_card_raw_end_to_end_via_scrapedo(monkeypatch):
    """Full _fetch_card_raw (search -> match -> features -> parse) driven entirely by a
    mocked scrapedo_fetch — proves both the transport (scrapedo_fetch called with the
    expected Ozon URLs) and the parser contract (>=3 characteristics correctly extracted
    from the SAME state-webCharacteristics HTML shape as before the swap, INV-10c).
    """
    src = _make_source(monkeypatch)
    # Keep the flow deterministic: go through the internal Ozon search path, not Serper.
    monkeypatch.setattr(ocs, "_OZON_SERPER_FIRST", False)
    monkeypatch.setattr(ocs, "_OZON_SERPER_CARD_FINDING", False)

    calls = []

    async def _fake_scrapedo_fetch(url, *, render=True, super_proxy=True, geo="ru"):
        calls.append({"url": url, "render": render, "super_proxy": super_proxy, "geo": geo})
        if "/search/" in url:
            return ScrapflyResult(success=True, content=_SEARCH_HTML_FIXTURE,
                                   status_code=200, credits_used=1, error=None)
        if "/features/" in url:
            return ScrapflyResult(success=True, content=_FEATURES_HTML_FIXTURE,
                                   status_code=200, credits_used=1, error=None)
        return ScrapflyResult(success=False, content=None, status_code=404,
                               credits_used=0, error="unexpected url")

    monkeypatch.setattr(ocs, "scrapedo_fetch", _fake_scrapedo_fetch)

    ctx = _ctx(name="Тестовый Товар Модель X100", brand="")
    raw = await src._fetch_card_raw(ctx, client=None)

    assert raw["stage"] == "ok", raw
    assert len(raw["raw_chars"]) >= 3, raw["raw_chars"]
    names = {c["name"] for c in raw["raw_chars"]}
    assert names == {"Цвет", "Вес", "Материал"}
    by_name = {c["name"]: c["value"] for c in raw["raw_chars"]}
    assert by_name["Цвет"] == "Черный"
    assert by_name["Вес"] == "500 г"
    assert by_name["Материал"] == "Хлопок"

    # scrapedo_fetch was actually invoked with Ozon URLs and the anti-bot params
    # from the manifest (render/super_proxy=True, geo=ru) for at least one call.
    assert calls, "scrapedo_fetch was never called"
    for c in calls:
        assert "ozon.ru" in c["url"]
        assert c["render"] is True
        assert c["super_proxy"] is True
        assert c["geo"] == "ru"


# ---------------------------------------------------------------------------
# INV-10d: no dead Scrappey references remain in the source file
# ---------------------------------------------------------------------------

def test_inv10d_no_scrappey_references_remain():
    content = _OZON_SOURCE_FILE.read_text(encoding="utf-8")
    forbidden = ["SCRAPPEY", "Scrappey", "publisher.scrappey.com", "_SCRAPPEY_ENDPOINT"]
    hits = {pat: content.count(pat) for pat in forbidden if pat in content}
    assert not hits, f"Dead Scrappey references still present: {hits}"

    # The lowercase backward-compat constructor kwarg is intentionally kept and must
    # NOT be confused with the forbidden (uppercase / capitalized) identifiers above.
    assert "scrappey_key" in content, (
        "backward-compat constructor kwarg 'scrappey_key' should still be accepted"
    )


def test_inv10d_no_httpx_import_left_dangling():
    """httpx/ssl were only needed for the old Scrappey client — must be gone."""
    content = _OZON_SOURCE_FILE.read_text(encoding="utf-8")
    assert "import httpx" not in content
    assert "import ssl" not in content


def test_inv10d_scrapedo_fetch_is_imported():
    assert inspect.isfunction(ocs.scrapedo_fetch) or inspect.iscoroutinefunction(
        ocs.scrapedo_fetch
    )


# ---------------------------------------------------------------------------
# INV-10e: source_type unchanged
# ---------------------------------------------------------------------------

def test_inv10e_source_type_is_ozon_card(monkeypatch):
    monkeypatch.setenv("SCRAPEDO_TOKEN", "fake-token")
    src = ocs.OzonCardSource()
    assert src.source_type == Source.OZON_CARD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
