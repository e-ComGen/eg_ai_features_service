"""FIX-17: LIVE self-heal WB field-name -> Ozon attribute_id.

Covers the manifest oracle (S1-S4) and invariants INV-17a..g for
``app.services.enrichment.sources.wb_field_map_selfheal`` plus the
integration point in ``WbCardSource._map_characteristics``
(``docs/MANIFEST_wb_selfheal_fieldmap_fix17.md``).

DeepSeek is MOCKED throughout (module-level ``_query_deepseek``) — the
separate live smoke (manual, real DeepSeek call) is run once outside pytest
per the manifest's acceptance criteria, not part of this suite.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.services.enrichment.sources.wb_field_map_selfheal as mod


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_cache_dir(tmp_path, monkeypatch):
    """Redirect the on-disk cache to a throwaway tmp dir for every test."""
    monkeypatch.setattr(mod, "_DATA_DIR", tmp_path / "field_maps")


def _ozon_attrs() -> list[dict]:
    """A small realistic Ozon attribute list for a headphones-ish type."""
    return [
        {"id": 100, "name": "Разъём для наушников", "type": "String", "is_collection": False},
        {"id": 101, "name": "Диагональ экрана", "type": "Numeric", "is_collection": False},
        {"id": 102, "name": "Цвет", "type": "String", "is_collection": True, "values": ["Чёрный"]},
    ]


def _mock_llm_response(mapping: dict) -> str:
    return json.dumps(mapping, ensure_ascii=False)


# ---------------------------------------------------------------------------
# S1 — unmapped WB name maps to the correct existing Ozon attribute
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s1_unmapped_name_maps_to_correct_attr_and_cache_grows():
    ozon_attrs = _ozon_attrs()
    wb_fields = {"тип разъёма наушников": "Type-C"}
    static_map: dict[str, int] = {}

    with patch.object(
        mod, "_query_deepseek",
        AsyncMock(return_value=_mock_llm_response({"тип разъёма наушников": 100})),
    ) as mock_llm:
        result = await mod.self_heal(
            static_map, wb_fields, ozon_attrs,
            wb_subject="Наушники", ozon_cat_id=1, ozon_type_id=1,
        )

    assert mock_llm.await_count == 1
    assert result["тип разъёма наушников"] == 100

    persisted = mod.get_persisted_cache("Наушники", 1, 1)
    assert persisted.get("тип разъёма наушников") == 100, (
        f"self-healed name must be glued into the on-disk cache; got {persisted}"
    )


# ---------------------------------------------------------------------------
# S2 / INV-17b — no suitable Ozon attribute -> None, nothing invented
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s2_no_suitable_attr_returns_none_not_invented():
    ozon_attrs = _ozon_attrs()
    wb_fields = {"абракадабра-поле-xyz": "???"}
    static_map: dict[str, int] = {}

    with patch.object(
        mod, "_query_deepseek",
        AsyncMock(return_value=_mock_llm_response({"абракадабра-поле-xyz": None})),
    ):
        result = await mod.self_heal(
            static_map, wb_fields, ozon_attrs,
            wb_subject="Наушники", ozon_cat_id=1, ozon_type_id=1,
        )

    assert result["абракадабра-поле-xyz"] is None
    assert set(v for v in result.values() if v is not None) == set(), (
        "no Ozon attribute may be fabricated for an unmatched WB name"
    )


@pytest.mark.asyncio
async def test_inv17b_llm_returns_id_not_in_ozon_attrs_is_rejected():
    """Grounding: an attr id the LLM invents that isn't in ozon_attrs -> None."""
    ozon_attrs = _ozon_attrs()
    wb_fields = {"неизвестное поле": "x"}

    with patch.object(
        mod, "_query_deepseek",
        AsyncMock(return_value=_mock_llm_response({"неизвестное поле": 999999})),
    ):
        result = await mod.self_heal(
            {}, wb_fields, ozon_attrs, wb_subject="X", ozon_cat_id=1, ozon_type_id=1,
        )

    assert result["неизвестное поле"] is None


# ---------------------------------------------------------------------------
# S3 / INV-17a — cache-hit (already in static/persisted) never calls the LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s3_already_static_name_skips_llm_entirely():
    ozon_attrs = _ozon_attrs()
    wb_fields = {"диагональ экрана": "6.1"}
    static_map = {"диагональ экрана": 101}  # already resolved statically

    with patch.object(mod, "_query_deepseek", AsyncMock()) as mock_llm:
        result = await mod.self_heal(
            static_map, wb_fields, ozon_attrs,
            wb_subject="Наушники", ozon_cat_id=1, ozon_type_id=1,
        )

    mock_llm.assert_not_awaited()
    assert result["диагональ экрана"] == 101


@pytest.mark.asyncio
async def test_inv17a_persisted_cache_hit_also_skips_llm():
    """A name healed on a PRIOR call is a persisted-cache-hit on the next call."""
    ozon_attrs = _ozon_attrs()
    wb_fields = {"тип разъёма наушников": "Type-C"}

    with patch.object(
        mod, "_query_deepseek",
        AsyncMock(return_value=_mock_llm_response({"тип разъёма наушников": 100})),
    ):
        await mod.self_heal(
            {}, wb_fields, ozon_attrs, wb_subject="Наушники", ozon_cat_id=1, ozon_type_id=1,
        )

    # Second call, fresh static_map (simulating a different product of the
    # same subject/cat/type) — the name is now only in the PERSISTED cache.
    with patch.object(mod, "_query_deepseek", AsyncMock()) as mock_llm_2:
        result = await mod.self_heal(
            {}, wb_fields, ozon_attrs, wb_subject="Наушники", ozon_cat_id=1, ozon_type_id=1,
        )

    mock_llm_2.assert_not_awaited()
    assert result["тип разъёма наушников"] == 100


# ---------------------------------------------------------------------------
# S4 / INV-17d — DeepSeek down -> static-only fallback, never raises, no cache
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_s4_deepseek_exception_static_only_fallback_no_cache_write():
    ozon_attrs = _ozon_attrs()
    wb_fields = {"тип разъёма наушников": "Type-C"}
    static_map = {"цвет": 102}

    # self_heal only ever sees None from _query_deepseek on failure (its own
    # contract, proven separately by test_inv17d_query_deepseek_never_raises_
    # on_provider_exception below) — exercise that fallback path here.
    with patch.object(mod, "_query_deepseek", AsyncMock(return_value=None)):
        result = await mod.self_heal(
            static_map, wb_fields, ozon_attrs,
            wb_subject="Наушники", ozon_cat_id=2, ozon_type_id=2,
        )

    assert result == static_map, "static-only fallback must leave the map untouched"
    assert mod.get_persisted_cache("Наушники", 2, 2) == {}, (
        "an infra failure must NOT write a (meaningless) cache entry"
    )


@pytest.mark.asyncio
async def test_inv17d_query_deepseek_never_raises_on_provider_exception():
    """_query_deepseek's own contract: any Exception from the provider -> None."""
    fake_provider = MagicMock()
    fake_provider.complete = AsyncMock(side_effect=RuntimeError("boom"))
    with patch(
        "app.services.providers.deepseek_provider.DeepSeekProvider",
        return_value=fake_provider,
    ):
        raw = await mod._query_deepseek("prompt", 20)
    assert raw is None


@pytest.mark.asyncio
async def test_inv17d_empty_llm_response_is_static_only_fallback():
    ozon_attrs = _ozon_attrs()
    wb_fields = {"тип разъёма наушников": "Type-C"}

    with patch.object(mod, "_query_deepseek", AsyncMock(return_value=None)):
        result = await mod.self_heal(
            {}, wb_fields, ozon_attrs, wb_subject="Наушники", ozon_cat_id=3, ozon_type_id=3,
        )

    assert result == {}
    assert mod.get_persisted_cache("Наушники", 3, 3) == {}


@pytest.mark.asyncio
async def test_inv17d_unparseable_json_is_static_only_fallback():
    ozon_attrs = _ozon_attrs()
    wb_fields = {"тип разъёма наушников": "Type-C"}

    with patch.object(mod, "_query_deepseek", AsyncMock(return_value="not json at all {{{")):
        result = await mod.self_heal(
            {}, wb_fields, ozon_attrs, wb_subject="Наушники", ozon_cat_id=4, ozon_type_id=4,
        )

    assert result == {}
    assert mod.get_persisted_cache("Наушники", 4, 4) == {}


# ---------------------------------------------------------------------------
# INV-17c — successful heal persists and is reused without the LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_inv17c_heal_persists_and_grows_across_products():
    ozon_attrs = _ozon_attrs()

    with patch.object(
        mod, "_query_deepseek",
        AsyncMock(return_value=_mock_llm_response({"тип разъёма наушников": 100})),
    ):
        await mod.self_heal(
            {}, {"тип разъёма наушников": "Type-C"}, ozon_attrs,
            wb_subject="Наушники", ozon_cat_id=5, ozon_type_id=5,
        )

    # A second product introduces ANOTHER previously-unseen name — the cache
    # must UNION (grow), not overwrite the first entry.
    with patch.object(
        mod, "_query_deepseek",
        AsyncMock(return_value=_mock_llm_response({"цвет корпуса": 102})),
    ):
        await mod.self_heal(
            {}, {"цвет корпуса": "Чёрный"}, ozon_attrs,
            wb_subject="Наушники", ozon_cat_id=5, ozon_type_id=5,
        )

    persisted = mod.get_persisted_cache("Наушники", 5, 5)
    assert persisted.get("тип разъёма наушников") == 100, (
        f"first self-healed entry must survive the second call; got {persisted}"
    )
    assert persisted.get("цвет корпуса") == 102


# ---------------------------------------------------------------------------
# INV-17e — flag off: LLM never called, byte-identical to pre-FIX-17
# ---------------------------------------------------------------------------

def test_inv17e_is_enabled_reads_env_flag(monkeypatch):
    monkeypatch.delenv("WB_CARD_SELFHEAL_ENABLED", raising=False)
    assert mod.is_enabled() is True  # default on

    monkeypatch.setenv("WB_CARD_SELFHEAL_ENABLED", "false")
    assert mod.is_enabled() is False

    monkeypatch.setenv("WB_CARD_SELFHEAL_ENABLED", "0")
    assert mod.is_enabled() is False

    monkeypatch.setenv("WB_CARD_SELFHEAL_ENABLED", "true")
    assert mod.is_enabled() is True


# ---------------------------------------------------------------------------
# INV-17g — atomic write; a corrupt/empty cache file never crashes reads
# ---------------------------------------------------------------------------

def test_inv17g_corrupt_cache_file_is_treated_as_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_DATA_DIR", tmp_path)
    path = mod._cache_path("Наушники", 6, 6)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    assert mod._load_cache("Наушники", 6, 6) is None
    assert mod.get_persisted_cache("Наушники", 6, 6) == {}


def test_inv17g_save_cache_is_atomic_no_leftover_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_DATA_DIR", tmp_path)
    mod._save_cache("Наушники", 7, 7, {"x": 1})

    path = mod._cache_path("Наушники", 7, 7)
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == {"x": 1}
    assert not path.with_suffix(".tmp").exists(), "tmp file must be renamed away, not left behind"


def test_inv17g_empty_missing_cache_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_DATA_DIR", tmp_path)
    assert mod._load_cache("НетТакогоФайла", 8, 8) is None


# ---------------------------------------------------------------------------
# Integration: WbCardSource._map_characteristics <-> self-heal call point
# ---------------------------------------------------------------------------

def _make_wbcard_src():
    from app.services.enrichment.sources.wb_card_source import WbCardSource
    return WbCardSource(web_search_client=MagicMock())


def _ctx(**kwargs):
    from app.services.enrichment.base import ExtractionContext
    defaults = dict(
        product_id=1, product_name="Наушники Type-C", category_id=200,
        category_path=[], brand=None,
    )
    defaults.update(kwargs)
    return ExtractionContext(**defaults)


def _target(attr_id: int, name: str):
    from app.services.enrichment.base import TargetAttribute
    return TargetAttribute(id=attr_id, name=name, type="text")


def test_integration_selfheal_result_wired_into_verified_map():
    """The self-heal result must be usable by the verified-map lookup path
    (step 1 of the char-mapping loop), producing a resolved AttributeValue
    for a WB name the static map + fuzzy/semantic steps could NOT resolve.
    """
    import app.services.enrichment.sources.wb_card_source as wcs

    src = _make_wbcard_src()
    ctx = _ctx(ozon_type_id=201)
    # Deliberately unrelated target name so fuzzy/semantic steps would NOT
    # resolve "Тип разъёма наушников" on their own — only the self-heal path can.
    targets = [_target(100, "Совершенно другое имя атрибута")]
    chars = [{"name": "Тип разъёма наушников", "value": "Type-C"}]

    fake_ozon_chars = [{"id": 100, "name": "Совершенно другое имя атрибута", "type": "String"}]

    with (
        patch(
            "app.services.enrichment.sources.wb_card_source.get_ozon_characteristics_for_type",
            return_value=fake_ozon_chars,
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.eg_get_field_map",
            return_value={},
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.resolve_value_id",
            return_value=None,
        ),
        patch.object(
            wcs.wb_field_map_selfheal, "self_heal_sync",
            return_value={"тип разъёма наушников": 100},
        ) as mock_self_heal,
        patch.object(wcs, "_semantic_attr_name_match", return_value=None),
    ):
        result = src._map_characteristics(
            chars=chars, targets=targets, context=ctx,
            mode="exact", title="Наушники Type-C", score=90.0,
        )

    assert mock_self_heal.call_count == 1
    kwargs = mock_self_heal.call_args.kwargs
    assert kwargs["wb_fields"] == {"тип разъёма наушников": "Type-C"}
    assert len(result) == 1
    assert result[0].attribute_id == 100


def test_integration_selfheal_disabled_flag_never_calls_bridge():
    """INV-17e at the integration point: flag off -> self_heal_sync not called."""
    import app.services.enrichment.sources.wb_card_source as wcs

    src = _make_wbcard_src()
    ctx = _ctx(ozon_type_id=202)
    targets = [_target(100, "Атрибут")]
    chars = [{"name": "Незамапленное поле", "value": "x"}]
    fake_ozon_chars = [{"id": 100, "name": "Атрибут", "type": "String"}]

    with (
        patch(
            "app.services.enrichment.sources.wb_card_source.get_ozon_characteristics_for_type",
            return_value=fake_ozon_chars,
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.eg_get_field_map",
            return_value={},
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.resolve_value_id",
            return_value=None,
        ),
        patch.object(wcs.wb_field_map_selfheal, "is_enabled", return_value=False),
        patch.object(
            wcs.wb_field_map_selfheal, "self_heal_sync",
            side_effect=AssertionError("self_heal_sync must NOT be called when disabled"),
        ),
        patch.object(wcs, "_semantic_attr_name_match", return_value=None),
    ):
        result = src._map_characteristics(
            chars=chars, targets=targets, context=ctx,
            mode="exact", title="X", score=90.0,
        )

    assert result == []  # unresolved char dropped, exactly as pre-FIX-17


def test_integration_selfheal_exception_is_swallowed_extract_survives():
    """INV-17d at the integration point: self_heal_sync raising must not
    break _map_characteristics — it must fall back to the static-only result.
    """
    import app.services.enrichment.sources.wb_card_source as wcs

    src = _make_wbcard_src()
    ctx = _ctx(ozon_type_id=203)
    targets = [_target(100, "Объём")]
    chars = [
        {"name": "Объём", "value": "2 л"},  # resolves at step 2 (exact name)
        {"name": "Незамапленное поле", "value": "x"},
    ]
    fake_ozon_chars = [{"id": 100, "name": "Объём", "type": "String"}]

    with (
        patch(
            "app.services.enrichment.sources.wb_card_source.get_ozon_characteristics_for_type",
            return_value=fake_ozon_chars,
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.eg_get_field_map",
            return_value={},
        ),
        patch(
            "app.services.enrichment.sources.wb_card_source.resolve_value_id",
            return_value=None,
        ),
        patch.object(
            wcs.wb_field_map_selfheal, "self_heal_sync",
            side_effect=RuntimeError("selfheal bridge exploded"),
        ),
        patch.object(wcs, "_semantic_attr_name_match", return_value=None),
    ):
        result = src._map_characteristics(
            chars=chars, targets=targets, context=ctx,
            mode="exact", title="X", score=90.0,
        )

    # extract must NOT raise; the resolvable char ("Объём") still maps.
    assert len(result) == 1
    assert result[0].attribute_id == 100
