"""Tests for IceCatSource and IceCatJudge.

Все тесты используют моки HTTP (aiohttp) — нет реальных запросов к IceCat API.
Нет вызовов реальных LLM или тяжёлых зависимостей.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.icecat_judge import IceCatJudge
from app.services.enrichment.sources.icecat_source import (
    IceCatSource,
    _extract_code_candidates,
    _build_code_candidates,
    closed_brands,
    open_brands,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(**overrides) -> ExtractionContext:
    base = dict(
        product_id=1,
        product_name="Блок питания ASUS ROG STRIX 850G 850W",
        category_id=42,
        brand="ASUS",
    )
    base.update(overrides)
    return ExtractionContext(**base)


def _make_target(attr_id: int, name: str, attr_type: str = "text") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type=attr_type)


def _make_icecat_response(features: list[tuple[str, str]], brand: str = "ASUS") -> dict:
    """Создать фейковый IceCat API ответ с заданными features."""
    groups = []
    for name, value in features:
        groups.append({
            "FeatureGroup": {"Name": {"Value": "Основные"}},
            "Features": [{
                "Feature": {"Name": {"Value": name}},
                "PresentationValue": value,
            }],
        })
    return {
        "data": {
            "GeneralInfo": {
                "Title": f"{brand} ROG STRIX 850G",
                "Brand": brand,
                "ProductId": "12345",
            },
            "FeaturesGroups": groups,
        }
    }


def _make_icecat_source_with_mock_fetch(
    response: dict | str,
    email: str = "test@example.com",
    token: str = "test-token",
) -> IceCatSource:
    """Создать IceCatSource с замоканным _fetch_features.

    Обёртка мокает _search_and_fetch — это точка входа которую вызывает extract().
    """
    source = IceCatSource(email=email, token=token)
    if isinstance(response, str):
        # "403" или "404" → _search_and_fetch возвращает None или "403"
        if response == "403":
            source._search_and_fetch = AsyncMock(return_value="403")
        else:
            source._search_and_fetch = AsyncMock(return_value=None)
    else:
        # Реальный парсинг из словаря
        parsed = source._parse_response(response)
        source._search_and_fetch = AsyncMock(return_value=parsed)
    return source


# ---------------------------------------------------------------------------
# Test 1: Успешный 200 → AttributeValues с правильным attribute_id
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_200_emits_avs_with_correct_mapping():
    """Мокнутый 200 → AVs эмитируются с корректным attribute_id через fuzzy match."""
    features = [
        ("Мощность", "850 Вт"),
        ("Сертификат эффективности", "80+ Gold"),
        ("Форм-фактор", "ATX"),
    ]
    source = _make_icecat_source_with_mock_fetch(_make_icecat_response(features))

    targets = [
        _make_target(101, "Мощность блока питания, Вт"),
        _make_target(102, "Сертификат 80 Plus"),
        _make_target(103, "Форм-фактор"),
    ]
    ctx = _make_context()
    results = await source.extract(ctx, targets)

    assert len(results) > 0, "Должны вернуться AVs при 200 ответе"
    for av in results:
        assert av.source == Source.ICECAT
        assert av.confidence == pytest.approx(0.92, abs=0.01)
        assert av.evidence.startswith("icecat:")

    # Форм-фактор — точное совпадение, должен быть mapped
    attr_ids = {av.attribute_id for av in results}
    assert 103 in attr_ids, "Форм-фактор должен быть смапен (точное совпадение)"


# ---------------------------------------------------------------------------
# Test 2: 403 → возвращает [] + логирует бренд в closed_brands
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_403_returns_empty_and_logs_brand():
    """Мокнутый 403 → graceful [] без исключений, бренд попадает в closed_brands."""
    source = IceCatSource(email="x", token="y")
    # Мокаем _search_and_fetch чтобы вернуть "403" и также обновить closed_brands
    async def _mock_search_403(brand, product_name, context=None):
        closed_brands[brand] += 1
        return "403"
    source._search_and_fetch = _mock_search_403

    targets = [_make_target(101, "Мощность")]
    ctx = _make_context(brand="FullOnlyBrand")

    # Сброс счётчика перед тестом
    closed_brands.clear()

    results = await source.extract(ctx, targets)

    assert results == [], "403 должен вернуть пустой список"
    assert closed_brands.get("FullOnlyBrand", 0) > 0, "Бренд должен быть залогирован в closed_brands"


# ---------------------------------------------------------------------------
# Test 3: 404 → возвращает [] (продукт не найден)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_404_returns_empty():
    """Мокнутый 404 → graceful [] без исключений."""
    source = _make_icecat_source_with_mock_fetch("404")
    targets = [_make_target(101, "Мощность")]
    ctx = _make_context()

    results = await source.extract(ctx, targets)

    assert results == [], "404 должен вернуть пустой список"


# ---------------------------------------------------------------------------
# Test 4: is_applicable=False когда нет бренда
# ---------------------------------------------------------------------------

def test_is_applicable_false_when_no_brand():
    """is_applicable должен вернуть False если brand пустой."""
    source = IceCatSource(email="x", token="y")
    target = _make_target(101, "Мощность")

    ctx_no_brand = _make_context(brand=None)
    assert source.is_applicable(ctx_no_brand, target) is False

    ctx_empty_brand = _make_context(brand="")
    assert source.is_applicable(ctx_empty_brand, target) is False


def test_is_applicable_true_when_brand_and_name():
    """is_applicable должен вернуть True при наличии бренда и достаточной длины имени."""
    source = IceCatSource(email="x", token="y")
    target = _make_target(101, "Мощность")
    ctx = _make_context(brand="ASUS", product_name="ASUS ROG STRIX 850G")
    assert source.is_applicable(ctx, target) is True


# ---------------------------------------------------------------------------
# Test 5: Fuzzy mapping ("Общая мощность" ~ "Мощность блока питания, Вт")
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fuzzy_name_mapping_partial_match():
    """Fuzzy matching должен связать 'Мощность' с 'Мощность блока питания, Вт'."""
    features = [("Мощность", "850 Вт")]
    source = _make_icecat_source_with_mock_fetch(_make_icecat_response(features))

    targets = [
        _make_target(201, "Мощность блока питания, Вт"),
        _make_target(202, "Цвет корпуса"),
    ]
    ctx = _make_context()
    results = await source.extract(ctx, targets)

    # "Мощность" должна смапиться на "Мощность блока питания, Вт"
    assert any(av.attribute_id == 201 for av in results), (
        "Fuzzy match должен связать 'Мощность' c 'Мощность блока питания, Вт'"
    )


# ---------------------------------------------------------------------------
# Test 6: Пустые FeaturesGroups → пустой результат
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_features_groups_returns_empty():
    """Если FeaturesGroups пустые → возвращаем []."""
    empty_response = {"data": {"GeneralInfo": {}, "FeaturesGroups": []}}
    source = _make_icecat_source_with_mock_fetch(empty_response)

    targets = [_make_target(101, "Мощность")]
    ctx = _make_context()
    results = await source.extract(ctx, targets)

    assert results == [], "Пустые FeaturesGroups → пустой результат"


# ---------------------------------------------------------------------------
# Test 7: Judge принимает confidence ≥ 0.85, отклоняет < 0.85
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_icecat_judge_accepts_high_confidence():
    """IceCatJudge принимает значения с confidence ≥ 0.85."""
    judge = IceCatJudge()
    av = AttributeValue(attribute_id=10, value="850 Вт", confidence=0.92, source=Source.ICECAT)
    ctx = _make_context()
    assert await judge.validate(av, ctx) is True


@pytest.mark.asyncio
async def test_icecat_judge_rejects_low_confidence():
    """IceCatJudge отклоняет значения с confidence < 0.85."""
    judge = IceCatJudge()
    av = AttributeValue(attribute_id=10, value="850 Вт", confidence=0.80, source=Source.ICECAT)
    ctx = _make_context()
    assert await judge.validate(av, ctx) is False


# ---------------------------------------------------------------------------
# Test 8: Source.ICECAT зарегистрирован в enum с правильным приоритетом
# ---------------------------------------------------------------------------

def test_source_enum_icecat():
    """Source.ICECAT должен быть зарегистрирован в enum."""
    assert Source.ICECAT == "icecat"
    assert Source.ICECAT in Source


def test_source_priority_icecat():
    """SOURCE_PRIORITY[ICECAT] должен быть > SOURCE_PRIORITY[COMPETITOR_RAG]."""
    from app.services.enrichment.base import SOURCE_PRIORITY
    assert SOURCE_PRIORITY[Source.ICECAT] > SOURCE_PRIORITY[Source.COMPETITOR_RAG]


def test_source_confidence_threshold_icecat():
    """SOURCE_CONFIDENCE_THRESHOLDS[ICECAT] должен быть 0.90."""
    from app.services.enrichment.base import SOURCE_CONFIDENCE_THRESHOLDS
    assert SOURCE_CONFIDENCE_THRESHOLDS[Source.ICECAT] == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Test 9: already_filled → пропустить заполненные атрибуты
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_skips_already_filled():
    """Атрибуты в already_filled не должны попасть в результат."""
    features = [("Форм-фактор", "ATX"), ("Мощность", "850 Вт")]
    source = _make_icecat_source_with_mock_fetch(_make_icecat_response(features))

    targets = [
        _make_target(103, "Форм-фактор"),
        _make_target(104, "Мощность, Вт"),
    ]
    already = [
        AttributeValue(attribute_id=103, value="ATX", confidence=0.95, source=Source.DESCRIPTION)
    ]
    ctx = _make_context()
    results = await source.extract(ctx, targets, already_filled=already)

    # Атрибут 103 (Форм-фактор) уже заполнен → не должен появиться снова
    filled_ids = {av.attribute_id for av in results}
    assert 103 not in filled_ids, "already_filled атрибут не должен быть перезаписан"


# ---------------------------------------------------------------------------
# Test 10: _build_code_candidates — регрессионный тест
# ---------------------------------------------------------------------------

def test_build_code_candidates_asus_rog_strix():
    """ASUS ROG STRIX 850G → должен генерировать ROG-STRIX-850G как один из кандидатов."""
    name = "Блок питания ASUS ROG STRIX 850G 850W 80+ Gold"
    candidates = _build_code_candidates(name, "ASUS")
    assert len(candidates) >= 1, f"Должны быть кандидаты: {candidates}"
    # Дефисованный вариант должен присутствовать
    hyphenated = [c for c in candidates if "-" in c]
    assert len(hyphenated) >= 1, f"Должен быть дефисованный кандидат: {candidates}"
    # Бренд не должен попасть в кандидаты
    assert not any(c.lower() == "asus" for c in candidates), f"Бренд не должен быть в кандидатах: {candidates}"


def test_build_code_candidates_msi_mpg():
    """MSI MPG A850G PCIE5 → первый кандидат — полное имя модели."""
    name = "Блок питания MSI MPG A850G PCIE5 850W 80+ Gold"
    candidates = _build_code_candidates(name, "MSI")
    # Полное имя модели без ватт должно присутствовать
    full_model = [c for c in candidates if "MPG" in c and "A850G" in c]
    assert len(full_model) >= 1, f"Должен быть кандидат с полным именем модели: {candidates}"


def test_build_code_candidates_gigabyte_ud850gm():
    """Gigabyte UD850GM → UD850GM должен быть среди кандидатов."""
    name = "Блок питания Gigabyte UD850GM 850W 80+ Gold Modular"
    candidates = _build_code_candidates(name, "Gigabyte")
    assert any("UD850GM" in c for c in candidates), f"UD850GM должен быть в кандидатах: {candidates}"


def test_build_code_candidates_excludes_generic_words():
    """_build_code_candidates не должен включать стоп-слова как самостоятельные кандидаты."""
    name = "Блок питания Corsair RM750x 750W 80+ Gold Fully Modular"
    candidates = _build_code_candidates(name, "Corsair")
    standalone = [c for c in candidates if c.lower() in ("gold", "fully", "modular", "80+")]
    assert len(standalone) == 0, f"Стоп-слова не должны быть самостоятельными кандидатами: {candidates}"


# Обратная совместимость: _extract_code_candidates — обёртка над _build_code_candidates
def test_extract_code_candidates_backward_compat():
    """_extract_code_candidates (legacy) должен возвращать те же кандидаты что и _build_code_candidates."""
    name = "Блок питания ASUS ROG STRIX 850G 850W ATX"
    result_legacy = _extract_code_candidates(name, "ASUS")
    result_new = _build_code_candidates(name, "ASUS")
    assert result_legacy == result_new, "Legacy wrapper должен давать идентичный результат"


# ---------------------------------------------------------------------------
# Test 11: search_cache — повторный вызов не делает лишних HTTP запросов
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_search_cache_prevents_duplicate_calls():
    """Повторный extract для того же (brand, product_name) не вызывает _search_and_fetch повторно."""
    features = [("Форм-фактор", "ATX")]
    source = IceCatSource(email="x", token="y")
    parsed = source._parse_response(_make_icecat_response(features))
    source._search_and_fetch = AsyncMock(return_value=parsed)

    targets = [_make_target(103, "Форм-фактор")]
    ctx = _make_context()

    # Два вызова extract для одного продукта
    await source.extract(ctx, targets)
    await source.extract(ctx, targets)

    # _search_and_fetch должен быть вызван ровно один раз (второй раз — из кэша)
    assert source._search_and_fetch.call_count == 1, (
        f"_search_and_fetch должен быть вызван 1 раз, вызван {source._search_and_fetch.call_count} раз"
    )


# ---------------------------------------------------------------------------
# Test 12: Поиск с несколькими кандидатами — первый 200 выигрывает
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_candidates_first_200_wins():
    """Перебор кандидатов: первый 200 возвращается, остальные не пробуются."""
    source = IceCatSource(email="x", token="y")

    features_first = [("Форм-фактор", "ATX"), ("Мощность", "850 Вт")]
    features_second = [("Форм-фактор", "SFX")]

    call_count = 0

    async def mock_fetch(brand, code):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "404"  # первый кандидат — не найден
        else:
            return source._parse_response(_make_icecat_response(features_first))

    source._fetch_features = mock_fetch
    targets = [_make_target(103, "Форм-фактор"), _make_target(104, "Мощность, Вт")]
    ctx = _make_context()
    results = await source.extract(ctx, targets)

    assert len(results) > 0, "Должны вернуться AVs когда второй кандидат дал 200"
    assert call_count >= 2, "Должно быть не менее 2 попыток (первый 404, второй 200)"


# ---------------------------------------------------------------------------
# Test 13: Пустой ответ при отсутствии brand → graceful skip
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_extract_returns_empty_when_search_returns_none():
    """Когда _search_and_fetch возвращает None → extract возвращает []."""
    source = IceCatSource(email="x", token="y")
    source._search_and_fetch = AsyncMock(return_value=None)

    targets = [_make_target(101, "Мощность")]
    ctx = _make_context(product_name="Блок питания Unknown Brand XYZ 650W")
    results = await source.extract(ctx, targets)

    assert results == [], "None от _search_and_fetch → пустой результат"


# ---------------------------------------------------------------------------
# Test 14: Pipeline integration — IceCatSource wire-in
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_with_icecat_source():
    """PipelineOrchestrator с IceCatSource → AVs из IceCat появляются в output."""
    from app.services.enrichment.base import LlmJudge
    from unittest.mock import AsyncMock, MagicMock
    from app.services.enrichment.pipeline import PipelineOrchestrator

    def _mock_source(src_type):
        s = MagicMock()
        s.source_type = src_type
        s.is_applicable = MagicMock(return_value=False)
        s.extract = AsyncMock(return_value=[])
        j = MagicMock(spec=LlmJudge)
        j.source = src_type
        j.validate = AsyncMock(return_value=True)
        s.get_judge = MagicMock(return_value=j)
        return s

    # Мокаем IceCatSource чтобы вернуть 1 AV
    icecat_av = AttributeValue(
        attribute_id=103,
        value="ATX",
        confidence=0.92,
        source=Source.ICECAT,
        evidence="icecat:Форм-фактор=ATX",
    )
    icecat_source = MagicMock(spec=IceCatSource)
    icecat_source.source_type = Source.ICECAT
    icecat_source.is_applicable = MagicMock(return_value=True)
    icecat_source.extract = AsyncMock(return_value=[icecat_av])
    icecat_judge = MagicMock(spec=IceCatJudge)
    icecat_judge.source = Source.ICECAT
    icecat_judge.validate = AsyncMock(return_value=True)
    icecat_source.get_judge = MagicMock(return_value=icecat_judge)

    # Мокаем стратегию
    strategy = MagicMock()
    strategy.name = "test"
    strategy.filter_unsupported_attributes = MagicMock(side_effect=lambda t: t)
    strategy.filter_by_dictionary = MagicMock(side_effect=lambda t, c: t)
    strategy.normalize_target_with_context = MagicMock(side_effect=lambda t, c: t)
    strategy.force_websearch_targets = MagicMock(return_value=set())
    strategy.resolve_value_ids = MagicMock(side_effect=lambda v, c: v)
    strategy.post_process_values = MagicMock(side_effect=lambda vals, t, c: vals)
    strategy.validate_value = MagicMock(return_value=MagicMock(is_valid=True, normalized_value=None))
    strategy.build_response_model = MagicMock(side_effect=lambda m, t: m)

    with patch("app.services.enrichment.pipeline.FinishingExtractor") as MockFinishing:
        mock_finisher = MagicMock()
        mock_finisher.extract_missing = AsyncMock(return_value=[])
        MockFinishing.return_value = mock_finisher

        classifier = MagicMock()
        classifier.classify = AsyncMock(return_value={})
        cost_pred = MagicMock()
        cost_pred.is_web_search_worth = AsyncMock(return_value=False)

        pipeline = PipelineOrchestrator(
            description_source=_mock_source(Source.DESCRIPTION),
            knowledge_source=_mock_source(Source.LLM_KNOWLEDGE),
            vision_source=_mock_source(Source.VISION),
            websearch_source=_mock_source(Source.WEB_SEARCH),
            icecat_source=icecat_source,
            classifier=classifier,
            cost_predictor=cost_pred,
            strategy=strategy,
        )

    ctx = _make_context()
    targets = [_make_target(103, "Форм-фактор")]
    results = await pipeline.enrich(ctx, targets)

    assert any(av.source == Source.ICECAT for av in results), (
        f"Ожидаем ICECAT AV в результатах, получено: {results}"
    )


# ---------------------------------------------------------------------------
# Test 15: MPN Lookup — кандидаты все 404, Serper+LLM находит MPN, IceCat 200
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mpn_lookup_finds_mpn_and_icecat_returns_200():
    """Все кандидаты 404 → mocked Serper+LLM возвращают MPN → IceCat 200 → AVs эмитируются."""
    from unittest.mock import AsyncMock, MagicMock
    from app.services.providers.serper_client import SerperResults, OrganicResult

    # Мокаем Serper клиент
    mock_serper = MagicMock()
    serper_results = SerperResults(
        query="ASUS ROG STRIX 850G MPN Part Number",
        organic_results=[
            OrganicResult(
                title="ASUS ROG STRIX 850G PSU",
                link="https://www.asus.com/...",
                snippet="Model: ROG-STRIX-850G | MPN: 90YE00A4-B0NA00 | Part Number: 90YE00A4-B0NA00",
                position=1,
            )
        ],
    )
    mock_serper.search = AsyncMock(return_value=serper_results)

    # Мокаем LLM менеджер — возвращает MPN
    from app.services.enrichment.sources.icecat_source import _MpnResponse
    mock_llm = MagicMock()
    mock_llm.structured_request = AsyncMock(return_value=(_MpnResponse(mpn="90YE00A4-B0NA00"), 10))

    source = IceCatSource(email="x", token="y", serper_client=mock_serper, mpn_llm_manager=mock_llm)

    # _fetch_features: все эвристические кандидаты → 404, MPN кандидат → 200
    features = [("Форм-фактор", "ATX"), ("Мощность", "850 Вт")]
    call_count = {"n": 0}

    async def mock_fetch(brand, code):
        call_count["n"] += 1
        if code == "90YE00A4-B0NA00":
            return source._parse_response(_make_icecat_response(features))
        return "404"

    source._fetch_features = mock_fetch

    targets = [_make_target(103, "Форм-фактор"), _make_target(104, "Мощность, Вт")]
    ctx = _make_context()
    results = await source.extract(ctx, targets)

    assert len(results) > 0, "MPN lookup + IceCat 200 должны вернуть AVs"
    assert any(av.source == Source.ICECAT for av in results), "AVs должны быть из Source.ICECAT"
    mock_serper.search.assert_called_once(), "Serper должен быть вызван один раз"
    mock_llm.structured_request.assert_called_once(), "LLM должен быть вызван один раз"


# ---------------------------------------------------------------------------
# Test 16: MPN Lookup возвращает None → extract возвращает []
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mpn_lookup_returns_none_extract_empty():
    """Все кандидаты 404 И MPN lookup возвращает None → extract возвращает []."""
    from unittest.mock import AsyncMock, MagicMock
    from app.services.providers.serper_client import SerperResults, OrganicResult

    # Serper возвращает результаты, но LLM не находит MPN
    mock_serper = MagicMock()
    serper_results = SerperResults(
        query="ASUS ROG STRIX 850G MPN Part Number",
        organic_results=[
            OrganicResult(
                title="Какой-то сайт",
                link="https://example.com",
                snippet="Блок питания 850W без артикула производителя.",
                position=1,
            )
        ],
    )
    mock_serper.search = AsyncMock(return_value=serper_results)

    from app.services.enrichment.sources.icecat_source import _MpnResponse
    mock_llm = MagicMock()
    # LLM возвращает mpn=None (не нашёл)
    mock_llm.structured_request = AsyncMock(return_value=(_MpnResponse(mpn=None), 5))

    source = IceCatSource(email="x", token="y", serper_client=mock_serper, mpn_llm_manager=mock_llm)
    # Все _fetch_features → 404
    source._fetch_features = AsyncMock(return_value="404")

    targets = [_make_target(103, "Форм-фактор")]
    ctx = _make_context()
    results = await source.extract(ctx, targets)

    assert results == [], "MPN=None → extract должен вернуть []"


# ---------------------------------------------------------------------------
# Test 17: MPN Lookup — Serper недоступен → graceful fallback, extract возвращает []
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mpn_lookup_serper_down_graceful_fallback():
    """Serper бросает исключение → lookup_mpn возвращает None, extract возвращает []."""
    from unittest.mock import AsyncMock, MagicMock

    # Serper кидает ошибку (недоступен)
    mock_serper = MagicMock()
    mock_serper.search = AsyncMock(side_effect=Exception("Connection refused"))

    source = IceCatSource(email="x", token="y", serper_client=mock_serper)
    # Все _fetch_features → 404
    source._fetch_features = AsyncMock(return_value="404")

    targets = [_make_target(103, "Форм-фактор")]
    ctx = _make_context()
    results = await source.extract(ctx, targets)

    assert results == [], "Serper down → graceful [], без исключений"
    mock_serper.search.assert_called_once(), "Serper.search должен быть вызван (и поймать ошибку)"
