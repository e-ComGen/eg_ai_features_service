"""FIX-16: LLM-верификатор идентичности товара — финальный гейт после _pick_best_match.

Manifest: docs/MANIFEST_ozon_llm_identity_verifier_fix16.md

Root cause: токен-гарды (_model_conflict/_brand_conflict/_category_present_in_title)
это whack-a-mole — каждый регекс хрупкий, каждый оставляет класс дыр (цифровые
сиблинги iPhone 15↔14, Pro/Max-варианты на высоком fuzzy, аксессуар-vs-устройство).
FIX-16 ставит дешёвую кросс-семейную LLM ФИНАЛЬНЫМ гейтом на winner-кандидате:
DIFFERENT → abstain (пусто лучше авторитетно неверного), UNKNOWN → fail-safe
(принимает только exact без FIX-15 model-conflict, иначе abstain).

LLM МОКается везде здесь для детерминизма (никаких живых сетевых вызовов) —
живой смоук верификатора запускается отдельно (см. manual_ozon_llm_identity_smoke_fix16.py).

Покрывает INV-16a..h (см. манифест):
  a) моканный verdict "different" -> карта возвращает пусто (abstain).
  b) моканный "same" -> карта отдаёт (как до FIX-16).
  c) FAIL-SAFE: моканный "unknown" + exact + FIX-15-без-конфликта -> отдать;
     "unknown" + brand_line -> abstain; "unknown" + exact + model-conflict -> abstain.
  d) _verify_product_identity: сбой транспорта/пустой/битый JSON/исключение -> "unknown",
     не падает, не "same".
  e) OZON_CARD_LLM_IDENTITY_ENABLED=false -> путь без LLM (гейт не зовётся).
  f) winner-only: LLM зовётся <=1 раза на отданную карту; мемоизация идентичной
     пары не дублирует вызов.
  g) fast-path skip — НЕ РЕАЛИЗОВАН (опциональная оптимизация манифеста). Осознанно
     пропущен: наивный "модель-токены query ⊆ токенов title" ложно засчитал бы
     V7 (Чехол для POCO X6 5G, тот же код x6/5g, без лишнего variant-mod) как "same"
     БЕЗ вызова LLM — ровно тот класс бага (аксессуар-vs-устройство), который FIX-16
     существует чтобы поймать. Каждый winner всегда проходит через реальный/моканный
     verify (см. test_inv16g_no_naive_fastpath_bypasses_verifier ниже).
  h) детерминистика FIX-12/15/_brand_conflict/пороги/_pick_best_match не тронута
     (diff добавляет хелпер + одну ветку применения + флаг) — regression-покрытие
     живёт в своих файлах (test_ozon_model_conflict_fix15.py и т.д.), здесь только
     точечная проверка, что _model_conflict вызывается НЕ мутированным.
"""
from __future__ import annotations

import json

import pytest

import app.services.enrichment.sources.ozon_card_source as ocs_mod
from app.services.enrichment.base import ExtractionContext
from app.services.enrichment.sources.ozon_card_source import (
    OzonCardSource,
    _parse_identity_verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(product_name: str, brand: str = "") -> ExtractionContext:
    return ExtractionContext(
        product_id=1,
        product_name=product_name,
        category_id=500,
        brand=brand or None,
    )


class _FakeResp:
    def __init__(self, content, cost_usd: float = 0.0001) -> None:
        self.content = content
        self.cost_usd = cost_usd


class _FakeProvider:
    """Заглушка DeepSeekProvider для мока транспорта _verify_product_identity."""

    def __init__(self, content=None, exc: Exception | None = None) -> None:
        self._content = content
        self._exc = exc

    async def complete(self, **kwargs):
        if self._exc is not None:
            raise self._exc
        return _FakeResp(self._content)


def _patch_provider(monkeypatch, content=None, exc: Exception | None = None, calls: list | None = None):
    def factory():
        if calls is not None:
            calls.append(1)
        return _FakeProvider(content=content, exc=exc)
    monkeypatch.setattr(ocs_mod, "DeepSeekProvider", factory)


# ---------------------------------------------------------------------------
# _parse_identity_verdict — pure parse layer
# ---------------------------------------------------------------------------

def test_parse_verdict_same():
    assert _parse_identity_verdict(
        '{"verdict":"same","distinguishing":""}'
    ) == ("same", "")


def test_parse_verdict_different_with_distinguishing():
    assert _parse_identity_verdict(
        '{"verdict":"different","distinguishing":"другое поколение"}'
    ) == ("different", "другое поколение")


def test_parse_verdict_json_wrapped_in_noise():
    raw = 'Вот ответ:\n{"verdict":"same","distinguishing":"объём памяти"}\nСпасибо.'
    assert _parse_identity_verdict(raw) == ("same", "объём памяти")


def test_parse_verdict_empty_or_none():
    assert _parse_identity_verdict("") == ("unknown", "")
    assert _parse_identity_verdict("   ") == ("unknown", "")
    assert _parse_identity_verdict(None) == ("unknown", "")


def test_parse_verdict_malformed_json():
    assert _parse_identity_verdict("{not valid json") == ("unknown", "")
    assert _parse_identity_verdict("это не json вообще") == ("unknown", "")


def test_parse_verdict_unexpected_value():
    assert _parse_identity_verdict('{"verdict":"unknown"}') == ("unknown", "")
    assert _parse_identity_verdict('{"verdict":"maybe"}') == ("unknown", "")
    assert _parse_identity_verdict('{"verdict":123}') == ("unknown", "")


def test_parse_verdict_not_a_dict():
    assert _parse_identity_verdict('["same"]') == ("unknown", "")
    assert _parse_identity_verdict('"same"') == ("unknown", "")


def test_parse_verdict_distinguishing_non_string_becomes_empty():
    assert _parse_identity_verdict('{"verdict":"different","distinguishing":123}') == (
        "different", "",
    )


# ---------------------------------------------------------------------------
# INV-16d — _verify_product_identity: transport failure -> "unknown", never raises
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inv16d_empty_query_or_title_no_network_call(monkeypatch):
    calls: list = []
    _patch_provider(monkeypatch, content='{"verdict":"same"}', calls=calls)
    src = OzonCardSource()
    assert await src._verify_product_identity("", "Смартфон POCO X6 5G") == "unknown"
    assert await src._verify_product_identity("POCO X6 5G", "   ") == "unknown"
    assert calls == [], "пустой query/title не должен бить сеть"


@pytest.mark.asyncio
async def test_inv16d_transport_exception_returns_unknown(monkeypatch):
    _patch_provider(monkeypatch, exc=RuntimeError("boom"))
    src = OzonCardSource()
    verdict = await src._verify_product_identity("POCO X6 5G", "Смартфон POCO X6 5G")
    assert verdict == "unknown"


@pytest.mark.asyncio
async def test_inv16d_provider_construction_failure_returns_unknown(monkeypatch):
    def factory():
        raise ValueError("DeepSeek API key is not set.")
    monkeypatch.setattr(ocs_mod, "DeepSeekProvider", factory)
    src = OzonCardSource()
    verdict = await src._verify_product_identity("POCO X6 5G", "Смартфон POCO X6 5G")
    assert verdict == "unknown"


@pytest.mark.asyncio
async def test_inv16d_empty_content_returns_unknown(monkeypatch):
    _patch_provider(monkeypatch, content="")
    src = OzonCardSource()
    verdict = await src._verify_product_identity("POCO X6 5G", "Смартфон POCO X6 5G")
    assert verdict == "unknown"


@pytest.mark.asyncio
async def test_inv16d_malformed_json_content_returns_unknown(monkeypatch):
    _patch_provider(monkeypatch, content="я не уверен, наверное одно и то же")
    src = OzonCardSource()
    verdict = await src._verify_product_identity("POCO X6 5G", "Смартфон POCO X6 5G")
    assert verdict == "unknown"


@pytest.mark.asyncio
async def test_inv16d_never_defaults_to_same_on_failure(monkeypatch):
    """Явная проверка антипаттерна: сбой НИКОГДА не даёт 'same'."""
    for content, exc in [
        (None, RuntimeError("net down")),
        ("", None),
        ("garbage", None),
        ('{"verdict": "nonsense"}', None),
    ]:
        _patch_provider(monkeypatch, content=content, exc=exc)
        src = OzonCardSource()
        verdict = await src._verify_product_identity("iPhone 15", "iPhone 14")
        assert verdict != "same"
        assert verdict == "unknown"


@pytest.mark.asyncio
async def test_inv16d_happy_path_same(monkeypatch):
    _patch_provider(monkeypatch, content='{"verdict":"same","distinguishing":""}')
    src = OzonCardSource()
    assert await src._verify_product_identity("POCO X6 5G", "Смартфон POCO X6 5G 8/256 ГБ") == "same"


@pytest.mark.asyncio
async def test_inv16d_happy_path_different(monkeypatch):
    _patch_provider(
        monkeypatch,
        content='{"verdict":"different","distinguishing":"другая модель (M8 Pro vs X6)"}',
    )
    src = OzonCardSource()
    verdict = await src._verify_product_identity("POCO X6 5G", "Смартфон POCO M8 Pro 5G")
    assert verdict == "different"


# ---------------------------------------------------------------------------
# INV-16f — winner-only + memoization
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inv16f_memoization_same_pair_calls_llm_once(monkeypatch):
    calls: list = []

    async def counting_verify(self, query, title, extra=""):
        calls.append((query, title))
        return "same"

    monkeypatch.setattr(OzonCardSource, "_verify_product_identity", counting_verify)
    src = OzonCardSource()

    v1 = await src._get_identity_verdict("POCO X6 5G", "Смартфон POCO X6 5G")
    v2 = await src._get_identity_verdict("POCO X6 5G", "Смартфон POCO X6 5G")
    assert v1 == v2 == "same"
    assert len(calls) == 1, "идентичная пара (query, title) должна бить LLM ровно один раз"


@pytest.mark.asyncio
async def test_inv16f_different_pairs_each_call_llm(monkeypatch):
    calls: list = []

    async def counting_verify(self, query, title, extra=""):
        calls.append((query, title))
        return "same"

    monkeypatch.setattr(OzonCardSource, "_verify_product_identity", counting_verify)
    src = OzonCardSource()

    await src._get_identity_verdict("POCO X6 5G", "Смартфон POCO X6 5G")
    await src._get_identity_verdict("iPhone 15", "iPhone 15 256GB")
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_inv16f_resolve_gate_calls_verdict_at_most_once(monkeypatch):
    calls: list = []

    async def counting_verify(self, query, title, extra=""):
        calls.append(1)
        return "same"

    monkeypatch.setattr(OzonCardSource, "_verify_product_identity", counting_verify)
    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")

    result = await src._resolve_identity_gate(ctx, "exact", "Смартфон POCO X6 5G 8/256 ГБ")
    assert result is None
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# INV-16a/b/c — _resolve_identity_gate decision logic (mocked verdict)
# ---------------------------------------------------------------------------

def _mock_verdict(monkeypatch, verdict: str):
    async def fake(self, query, title, extra=""):
        return verdict
    monkeypatch.setattr(OzonCardSource, "_verify_product_identity", fake)


@pytest.mark.asyncio
async def test_inv16a_different_abstains(monkeypatch):
    _mock_verdict(monkeypatch, "different")
    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")
    result = await src._resolve_identity_gate(ctx, "exact", "Смартфон POCO M8 Pro 5G")
    assert result == []


@pytest.mark.asyncio
async def test_inv16a_different_abstains_brand_line_too(monkeypatch):
    _mock_verdict(monkeypatch, "different")
    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")
    result = await src._resolve_identity_gate(ctx, "brand_line", "Чехол для POCO X6 5G")
    assert result == []


@pytest.mark.asyncio
async def test_inv16b_same_lets_card_through(monkeypatch):
    _mock_verdict(monkeypatch, "same")
    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")
    result = await src._resolve_identity_gate(ctx, "exact", "Смартфон POCO X6 5G 8/256 ГБ")
    assert result is None, "verdict=same -> None (продолжить extraction как раньше)"


@pytest.mark.asyncio
async def test_inv16c_unknown_exact_no_conflict_accepts(monkeypatch):
    """FAIL-SAFE: unknown + exact + FIX-15 model_conflict==False -> отдать (None)."""
    _mock_verdict(monkeypatch, "unknown")
    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")
    # тот же код (x6/5g), без variant-mod конфликта -> _model_conflict=False
    result = await src._resolve_identity_gate(ctx, "exact", "Смартфон POCO X6 5G 8/256 ГБ")
    assert result is None


@pytest.mark.asyncio
async def test_inv16c_unknown_brand_line_abstains(monkeypatch):
    """FAIL-SAFE: unknown + brand_line (двусмысленная полоса) -> abstain."""
    _mock_verdict(monkeypatch, "unknown")
    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")
    result = await src._resolve_identity_gate(ctx, "brand_line", "Смартфон POCO X6 5G 8/256 ГБ")
    assert result == []


@pytest.mark.asyncio
async def test_inv16c_unknown_exact_with_model_conflict_abstains(monkeypatch):
    """FAIL-SAFE: unknown + exact НО FIX-15 model_conflict=True (сосед-вариант) -> abstain."""
    _mock_verdict(monkeypatch, "unknown")
    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")
    # X6 Pro -- тот же код x6/5g, но лишний variant-mod "pro" -> _model_conflict=True
    result = await src._resolve_identity_gate(ctx, "exact", "Смартфон POCO X6 Pro 5G 12/512")
    assert result == []


# ---------------------------------------------------------------------------
# INV-16e — флаг выключен
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inv16e_disabled_flag_skips_llm_entirely(monkeypatch):
    calls: list = []

    async def counting_verify(self, query, title, extra=""):
        calls.append(1)
        return "different"  # даже если бы вызвался -- вернул бы different

    monkeypatch.setattr(OzonCardSource, "_verify_product_identity", counting_verify)
    monkeypatch.setattr(ocs_mod, "_OZON_CARD_LLM_IDENTITY_ENABLED", False)

    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")
    result = await src._resolve_identity_gate(ctx, "exact", "Смартфон POCO M8 Pro 5G")
    assert result is None, "флаг выключен -> гейт не должен ничего абстейнить"
    assert calls == [], "флаг выключен -> LLM не должна вызываться вообще"


@pytest.mark.asyncio
async def test_inv16e_disabled_flag_mode_skip_also_bypassed(monkeypatch):
    """mode вне (exact, brand_line) -- гейт тоже bypass, недостижимая ветка в проде,
    но метод должен быть безопасен и на этом входе (defensive)."""
    calls: list = []

    async def counting_verify(self, query, title, extra=""):
        calls.append(1)
        return "different"

    monkeypatch.setattr(OzonCardSource, "_verify_product_identity", counting_verify)
    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")
    result = await src._resolve_identity_gate(ctx, "none", "что-то")
    assert result is None
    assert calls == []


# ---------------------------------------------------------------------------
# INV-16g — no naive fast-path: an accessory sharing the model code is NOT
# auto-accepted without hitting the (mocked) verifier.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inv16g_no_naive_fastpath_bypasses_verifier(monkeypatch):
    """V7: 'Чехол для POCO X6 5G' разделяет модельные токены (x6/5g) с query и не
    несёт лишнего variant-mod -- наивный token-subset fast-path засчитал бы это
    'same' без LLM. FIX-16 НЕ реализует такой fast-path: verify всегда вызывается."""
    calls: list = []

    async def counting_verify(self, query, title, extra=""):
        calls.append((query, title))
        return "different"  # правильный вердикт для аксессуара

    monkeypatch.setattr(OzonCardSource, "_verify_product_identity", counting_verify)
    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")
    result = await src._resolve_identity_gate(ctx, "brand_line", "Чехол для POCO X6 5G силиконовый")
    assert len(calls) == 1, "verify должен был вызваться -- нет скрытого fast-path обхода"
    assert result == []


# ---------------------------------------------------------------------------
# INV-16h — _model_conflict used un-mutated (sanity: FIX-15 guard still wired)
# ---------------------------------------------------------------------------

def test_inv16h_model_conflict_still_the_real_fix15_function():
    from app.services.enrichment.sources.ozon_card_source import _model_conflict
    # X6 vs X6 Pro -- тот же код, разный модификатор -> True (FIX-15 rule B)
    assert _model_conflict("POCO X6 5G", "Смартфон POCO X6 Pro 5G 12/512") is True
    # идентичная модель -> False
    assert _model_conflict("POCO X6 5G", "Смартфон POCO X6 5G 8/256 ГБ") is False


# ---------------------------------------------------------------------------
# Oracle V1-V8 (манифест) -- через _resolve_identity_gate с моканным verdict.
# Мод (exact/brand_line) задан так, как реально классифицировал бы _pick_best_match
# на соответствующих парах (см. test_ozon_model_conflict_fix15.py / _classify_match).
# ---------------------------------------------------------------------------

_ORACLE = [
    # (id, query, title, mode, verdict, expect_abstain)
    ("V1", "POCO X6 5G", "Смартфон POCO X6 5G 8/256 ГБ чёрный", "exact", "same", False),
    ("V2", "POCO X6 5G", "Смартфон POCO X6 5G 12/512 ГБ синий", "exact", "same", False),
    ("V3", "POCO X6 5G", "Смартфон POCO M8 Pro 5G", "brand_line", "different", True),
    ("V4", "POCO X6 5G", "Смартфон POCO X6 Pro 5G 12/512", "exact", "different", True),
    ("V5", "iPhone 15", "Смартфон Apple iPhone 14 128 ГБ", "exact", "different", True),
    ("V6", "iPhone 15", "Смартфон Apple iPhone 15 256 ГБ", "exact", "same", False),
    ("V7", "POCO X6 5G", "Чехол для POCO X6 5G силиконовый", "brand_line", "different", True),
    ("V8", "Bosch GSB 13 RE", "Дрель ударная Bosch GSB 13 RE 600 Вт", "exact", "same", False),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,query,title,mode,verdict,expect_abstain", _ORACLE)
async def test_oracle_v1_v8(monkeypatch, case_id, query, title, mode, verdict, expect_abstain):
    _mock_verdict(monkeypatch, verdict)
    src = OzonCardSource()
    ctx = _ctx(query)
    result = await src._resolve_identity_gate(ctx, mode, title)
    if expect_abstain:
        assert result == [], f"{case_id}: ожидался abstain ([])"
    else:
        assert result is None, f"{case_id}: ожидалось продолжение (None)"


# ---------------------------------------------------------------------------
# Mutation self-check: доказать, что INV-16a не пустой -- с ВЫКЛЮЧЕННЫМ гейтом
# (симулирует «убранную ветку different->abstain») карта different НЕ абстейнится.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mutation_self_check_disabled_guard_lets_different_through(monkeypatch):
    _mock_verdict(monkeypatch, "different")
    monkeypatch.setattr(ocs_mod, "_OZON_CARD_LLM_IDENTITY_ENABLED", False)
    src = OzonCardSource()
    ctx = _ctx("POCO X6 5G")
    result = await src._resolve_identity_gate(ctx, "exact", "Смартфон POCO M8 Pro 5G")
    assert result is None, (
        "mutation self-check провалился: с выключенным гейтом identity=different "
        "всё равно абстейнится -- test_inv16a_different_abstains может быть пустым "
        "(не проверяет реальную ветку)"
    )
