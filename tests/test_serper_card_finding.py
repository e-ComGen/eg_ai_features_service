"""Тест Serper-assisted card-finding для OzonCard (фича «цвет/размер для RU-обуви»).

Когда внутренний поиск Ozon флачит (no_tiles/low_match — не антибот, а search-качество),
находим точный URL товара через Google (Serper), затем Scrappey тащит /features/.
Гард: Serper-карточка проходит ТОТ ЖЕ match-скоринг → чужой бренд отсекается.
"""
from __future__ import annotations

import asyncio
import types

from app.services.enrichment.base import ExtractionContext
from app.services.enrichment.sources import ozon_card_source as ocs


def _ctx(name="Nike Air Max 90", brand="Nike"):
    return ExtractionContext(
        product_id=1, product_name=name, category_id=15621048,
        category_path=["Обувь", "Кроссовки"], brand=brand,
    )


def _fake_client(organic):
    """Фейковый web_search-клиент: .search() → объект с organic_results."""
    class _Res:
        organic_results = organic

    class _Client:
        async def search(self, *a, **k):
            return _Res()

    return _Client()


def _org(link, title):
    return types.SimpleNamespace(link=link, title=title, snippet="")


# ── _SERPER_SLUG_PID_RE: извлечение slug+pid из URL ──────────────────────────

def test_slug_pid_regex_extracts_both():
    m = ocs._SERPER_SLUG_PID_RE.search("https://www.ozon.ru/product/krossovki-nike-air-max-90-1234567/")
    assert m and m.group(1) == "krossovki-nike-air-max-90" and m.group(2) == "1234567"


def test_slug_pid_regex_no_trailing_slash():
    m = ocs._SERPER_SLUG_PID_RE.search("https://www.ozon.ru/product/adidas-runfalcon-987654321")
    assert m and m.group(2) == "987654321"


def test_card_finding_flag_default_on():
    """Дефолт фичи — ON (только добавляет fallback, гард не пускает мусор)."""
    assert ocs._OZON_SERPER_CARD_FINDING is True


# ── _serper_find_card: выбор первого ozon.ru/product organic ──────────────────

def test_find_card_picks_first_ozon_product(monkeypatch):
    monkeypatch.setattr(
        "app.services.providers.factory.get_web_search_client",
        lambda: _fake_client([
            _org("https://market.yandex.ru/foo", "Яндекс"),
            _org("https://www.ozon.ru/product/krossovki-nike-air-max-90-5550001/", "Nike Air Max 90 — OZON"),
        ]),
    )
    src = ocs.OzonCardSource(scrappey_key="x")
    found = asyncio.run(src._serper_find_card(_ctx()))
    assert found is not None
    assert found["pid"] == "5550001"
    assert found["slug"] == "krossovki-nike-air-max-90"
    assert found["card_url"].endswith("krossovki-nike-air-max-90-5550001/")
    assert "Nike" in found["title"]


def test_find_card_none_when_no_ozon_product(monkeypatch):
    monkeypatch.setattr(
        "app.services.providers.factory.get_web_search_client",
        lambda: _fake_client([_org("https://www.wildberries.ru/catalog/1/detail.aspx", "WB")]),
    )
    src = ocs.OzonCardSource(scrappey_key="x")
    assert asyncio.run(src._serper_find_card(_ctx())) is None


def test_find_card_none_when_client_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.services.providers.factory.get_web_search_client", lambda: None,
    )
    src = ocs.OzonCardSource(scrappey_key="x")
    assert asyncio.run(src._serper_find_card(_ctx())) is None


# ── _try_serper_card: гард по бренду (мусорную карточку не берём) ──────────────

def test_try_serper_card_rejects_wrong_brand(monkeypatch):
    """Serper нашёл карточку ДРУГОГО бренда → гард (_classify_match=skip) → None.

    Запрос Nike, а Google вернул карточку Adidas — _pick_best_match даёт
    brand-mismatch штраф → score ниже порога → не берём (пусто честнее мусора).
    """
    monkeypatch.setattr(
        "app.services.providers.factory.get_web_search_client",
        lambda: _fake_client([
            _org("https://www.ozon.ru/product/krossovki-adidas-runfalcon-7770002/",
                 "Adidas Runfalcon кроссовки мужские — OZON"),
        ]),
    )
    src = ocs.OzonCardSource(scrappey_key="x")
    # Scrappey не должен дёргаться вообще — гард отвергнет до фетча.
    out = asyncio.run(src._try_serper_card(_ctx(name="Nike Air Max 90", brand="Nike"), client=None))
    assert out is None


def test_try_serper_card_disabled_returns_none(monkeypatch):
    """Флаг OFF → метод сразу None (Serper не дёргается)."""
    monkeypatch.setattr(ocs, "_OZON_SERPER_CARD_FINDING", False)
    src = ocs.OzonCardSource(scrappey_key="x")
    out = asyncio.run(src._try_serper_card(_ctx(), client=None))
    assert out is None
