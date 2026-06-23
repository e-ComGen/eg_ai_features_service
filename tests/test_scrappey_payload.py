"""Тест _build_scrappey_payload — антибот-рычаги Scrappey для Ozon (data-driven).

Раньше слался голый {"cmd":"request.get","url":...} (датацентр-IP, без гео) → Ozon-
антибот часто блокал/висел. Теперь payload обогащается env-параметрами; главный —
proxyCountry=Russia (дефолт ON, гео-рычаг). proxy/requestType/session — опц. под замер.
"""
from __future__ import annotations

import importlib

import app.services.enrichment.sources.ozon_card_source as m


def _build(url="https://www.ozon.ru/product/x-1/features/", session=None,
           country="Russia", proxy="", req_type="", monkeypatch=None):
    """Собрать payload при заданном env-конфиге (через monkeypatch модульных констант)."""
    if monkeypatch is not None:
        monkeypatch.setattr(m, "_SCRAPPEY_PROXY_COUNTRY", country)
        monkeypatch.setattr(m, "_SCRAPPEY_PROXY", proxy)
        monkeypatch.setattr(m, "_SCRAPPEY_REQUEST_TYPE", req_type)
    return m._build_scrappey_payload(url, session)


def test_default_is_bare_datacenter_base():
    """Дефолт = проверенный datacenter-base (замер 23.06: 66% vs RU 33%).

    Все антибот-рычаги OFF по умолчанию: payload = {cmd,url} как старое поведение,
    без proxyCountry/proxy/requestType/session. Рычаги — opt-in через env.
    """
    p = m._build_scrappey_payload("https://www.ozon.ru/product/x-1/features/")
    assert p["cmd"] == "request.get"
    assert p["url"].endswith("/features/")
    assert "proxyCountry" not in p  # дефолт datacenter, не Russia
    assert "proxy" not in p
    assert "requestType" not in p
    assert "session" not in p


def test_session_added_only_when_passed():
    """session-ключ появляется только при явной передаче (reuse-режим)."""
    assert "session" not in m._build_scrappey_payload("https://www.ozon.ru/x")
    p = m._build_scrappey_payload("https://www.ozon.ru/x", "deadbeef")
    assert p["session"] == "deadbeef"


def test_empty_country_omits_key(monkeypatch):
    """Пустой proxyCountry → ключ не отправляется (Scrappey-дефолт)."""
    p = _build(country="", monkeypatch=monkeypatch)
    assert "proxyCountry" not in p


def test_custom_proxy_and_request_type(monkeypatch):
    """Кастомный residential proxy + requestType=browser прокидываются."""
    p = _build(proxy="http://u:p@res.proxy:8888", req_type="browser", monkeypatch=monkeypatch)
    assert p["proxy"] == "http://u:p@res.proxy:8888"
    assert p["requestType"] == "browser"
    assert p["proxyCountry"] == "Russia"


def test_module_imports_clean():
    """Sanity: модуль импортируется без heavy-import-сегфолта."""
    importlib.reload(m)
    assert callable(m._build_scrappey_payload)
