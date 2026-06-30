"""PipelineOrchestrator — sequential cost-aware extraction.

Главный entry-point для enrichment. Объединяет все sources, judges и intelligence
в один flow с early-exit (если все targets заполнены — останавливаемся) и
cost gating (CostPredictor перед expensive web search).

Spec: docs/architecture/pipeline.md, section "PipelineOrchestrator".
"""
import asyncio
import logging
import os
import re
from typing import Callable, Optional

from pydantic import BaseModel, Field

from app.services.enrichment.base import (
    Source,
    SOURCE_PRIORITY,
    AttributeValue,
    TargetAttribute,
    ExtractionContext,
    AttributeSource,
)
from app.services.enrichment.sources import (
    DescriptionSource,
    LlmKnowledgeSource,
    VisionSource,
    WebSearchSource,
    CompetitorRagSource,
    WbApparelRagSource,
    IceCatSource,
    PdfDatasheetSource,
    OzonCardSource,
    WbCardSource,
    UgcSource,
    TnvedSource,
    YandexMarketSource,
    ScrapflyOzonSource,
    BarcodeSource,
    RegardSource,
    BestBuySource,
    OnlinerSource,
    BooksSource,
    LamodaScrapflySource,
)
from app.services.enrichment.sources.ozon_card_source import (
    _extract_gender_signal,
    _is_gender_target_name,
)
from app.services.enrichment.sources.image_card_search import (
    find_matching_card as _image_find_matching_card,
)
from app.services.enrichment.size_normalizer import (
    parse_explicit_size,
    expand_intl_to_ru,
)
from app.services.enrichment.sources.wb_card_source import (
    _is_ru_size_target_name,
    _RU_SIZE_ATTR_IDS,
)
# Лемматизатор тип-слова (pymorphy3, морфология опциональна) — переиспользуем тот же
# helper, что и wb_card_source._target_type_lemma/_card_subj_lemmas, чтобы
# «Футболки»↔«Футболка» сходились без новой зависимости и без своей морфологии.
from app.services.enrichment.sources.wb_card_source import _lemma as _type_lemma
from app.services.enrichment.sources.icecat_numeric_normalizer import (
    _detect_attr_unit,
    _parse_float as _icecat_parse_float,
    _format_number as _icecat_format_number,
)
from app.services.enrichment.intelligence import LlmClassifier, CostPredictor
from app.services.providers.factory import get_main_manager
from app.services.enrichment.confidence_aware_judge import ConfidenceAwareJudgeWrapper
from app.strategies.validators.numeric_validator import NumericValidator
from app.judge.judge import HallucinationJudge
from app.judge.judge_profile import JudgeProfile
from app.services.enrichment.prompt_router import classify_target, extract_unit
from app.services.enrichment.strategies.base import MarketplaceStrategy
from app.services.enrichment.strategies.default_strategy import DefaultStrategy
from app.services.enrichment.finishing import FinishingExtractor
from app.services.enrichment.marketplaces.registry import MarketplaceRouter
from app.services.enrichment.judges.wb_card_judge import WbCardJudge

logger = logging.getLogger(__name__)

# YandexMarket is dead via Scrappey (captcha/301 on every URL, no residential proxy
# available) — contributes ZERO data and burns up to 80s/product in timeouts.
# Disable by default; set YANDEX_MARKET_ENABLED=1 to re-enable for testing.
YANDEX_MARKET_ENABLED: bool = os.environ.get("YANDEX_MARKET_ENABLED", "0") == "1"

# Card-protection в финальном _merge: карточные источники (копия из live-карточки
# того же товара) не должны перетираться инференсом (LLM-знания / web-поиск), если
# их confidence лишь незначительно ниже. Band = допустимый зазор.
_CARD_SOURCES = {Source.WB_CARD, Source.OZON_CARD, Source.LAMODA}
# Authoritative sources that override fake-confident inference within the protection
# band. Cards are live per-product listings; IceCat is brand-verified spec data and
# PDF_DATASHEET is the manufacturer's own datasheet — BOTH more authoritative than a
# card, yet previously they fought inference on raw confidence and LOST to llm_knowledge's
# self-assigned 0.95 (e.g. verified IceCat spec beaten by an llm hallucination). Folding
# them into the protection set lets grounded data win the contradiction. A/B-measured.
_AUTHORITATIVE_OVERRIDE_SOURCES = _CARD_SOURCES | {Source.ICECAT, Source.PDF_DATASHEET}
_INFERENCE_SOURCES = {Source.LLM_KNOWLEDGE, Source.WEB_SEARCH}
_CARD_PROTECTION_BAND = 0.10

# Гендер-гард на merge-слое (генеральный, для ВСЕХ источников — не только карточных).
# «External guess» источники: они НЕ видят конкретный товар, а домысливают пол по
# названию/категории/похожим листингам. Гендерное значение «Пол» от ТОЛЬКО таких
# источников при нейтральном имени товара («Кроссовки Ultraboost 22») — спекуляция,
# и его надо отбросить. Товар-специфичные источники (ozon_card/wb_card/vision/
# description) видят реальную карточку/фото/текст этого товара — их пол доверяем.
_GENDER_EXTERNAL_SOURCES = {
    Source.WEB_SEARCH,
    Source.LLM_KNOWLEDGE,
    Source.COMPETITOR_RAG,
}


# FIX 2: тег и confidence для пола, восстановленного из имени товара, когда гард
# опустошил REQUIRED поле «Пол». DESCRIPTION — товар-специфичный сигнал (заголовок
# этого товара). Confidence умеренный: явный сигнал имени, но дефолт-страховка.
_EG_GENDER_FROM_NAME_EVIDENCE = "gender_from_name_required_fallback"
_EG_GENDER_REQUIRED_FALLBACK_CONF = 0.75


# Brand-identity guard на merge-слое (генеральный, безопасность-критичный).
# Бренд — это ИДЕНТИЧНОСТЬ товара, его НЕЛЬЗЯ угадывать. «Guess/identity-unsafe»
# источники домысливают бренд по фото/похожим листингам/общим знаниям, а не видят
# реальную идентичность ЭТОГО товара: vision→«HUGO», web_search→«LEGO»,
# llm_knowledge→«Великобритания» (страна!) на Nike/Levi's/Adidas. Бренд-значение
# ТОЛЬКО от таких источников отбрасывается ДО merge. Авторитетные (карточка/опис/
# IceCat/PDF/ТНВЭД) и brand-from-name остаются — заполняют пустой/правильный таргет.
_BRAND_GUESS_SOURCES = {
    Source.VISION,
    Source.WEB_SEARCH,
    Source.LLM_KNOWLEDGE,
    Source.COMPETITOR_RAG,
}


def _apply_brand_source_guard(
    all_values: list[AttributeValue],
    targets: list[TargetAttribute],
) -> list[AttributeValue]:
    """Дроп brand-таргет кандидатов от guess/identity-unsafe источников.

    Бренд — идентичность: vision/web_search/llm_knowledge/competitor_rag НЕ видят
    реальную идентичность товара, а домысливают её (HUGO/LEGO/Великобритания на
    Nike/Levi's/Adidas). Их кандидаты на brand-таргет (детект по id==31 ИЛИ имени
    «Бренд/Brand/Торговая марка») выбрасываются ДО merge. Авторитетные источники
    (ozon_card/wb_card/description/icecat/pdf_datasheet/tnved) и brand-from-name
    (source=DESCRIPTION, добавляется ПОСЛЕ merge) проходят. Не-brand таргеты — без
    изменений.
    """
    brand_attr_ids: set[int] = {
        t.id for t in targets
        if t.id == _BRAND_TARGET_ATTR_ID or _is_brand_target_name(t.name)
    }
    if not brand_attr_ids:
        return all_values

    out: list[AttributeValue] = []
    for v in all_values:
        if v.attribute_id in brand_attr_ids and v.source in _BRAND_GUESS_SOURCES:
            logger.info(
                "[Pipeline] brand-guard: дроп '%s'='%s' (source=%s) — guess-источник "
                "не видит идентичность товара",
                v.attribute_id, v.value, v.source.value,
            )
            continue
        out.append(v)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Objective-spec predicate + corroboration gate helpers (Stage 4.9)
# ──────────────────────────────────────────────────────────────────────────────

# Semantic types that unambiguously tag an attribute as objective-spec (when set).
_OBJECTIVE_SPEC_SEMANTIC_TYPES: frozenset[str] = frozenset({
    # Material / fabric / composition
    "material", "material_composition", "composition", "fabric",
    "fabric_composition",
    # Physical / numeric measurements
    "weight", "net_weight", "gross_weight", "dimensions",
    "temperature", "thermal", "frequency", "power", "voltage",
    "capacity", "volume", "resolution",
    # Connectivity / interface / audio
    "connectivity", "interface", "audio_config", "channel_config",
    # Boolean feature presence
    "feature_bool", "boolean_feature",
})

# Name substrings (lower-cased) that identify objective-spec attributes when
# semantic_type is None or missing. Matches case-insensitively via str.lower().
#
# Rationale for each group:
#   Material/composition: «Материал», «Состав», «Подкладка», «Материал верха», etc.
#   Connectivity/interface: «Тип подключения», «Интерфейс», «Разъём», «Порт»
#   Audio/video configuration: «Звуковая схема», «Каналы», «Разрядность»
#   Wireless feature flags: «True Wireless», «Bluetooth», «Wi-Fi» (exact feature)
#
# NOT included: color, brand, size/gender (subjective or identity attrs) — those
# stay on Gate B LLM-verify path since wrong values are less dangerous than mud.
_OBJECTIVE_SPEC_NAME_FRAGMENTS: tuple[str, ...] = (
    # Material / composition group
    "материал",
    "состав",
    "подкладк",
    # Tech connectivity / interface
    "интерфейс",
    "тип подключения",
    "разъём",
    "разъем",
    # Audio / video configuration
    "звуковая схема",
    "звуковая система",
    "акустическая система",
    "количество каналов",
    "аудиоканал",
    "конфигурация каналов",
    # Wireless / binary feature flags
    "true wireless",
    "активное шумоподавление",
    "шумоподавлен",
)

# Sources considered AUTHORITATIVE for corroboration purposes.
# These must NOT include LLM_KNOWLEDGE, WEB_SEARCH, or VISION — two guess-prone
# sources agreeing with each other is NOT independent corroboration.
_AUTHORITATIVE_SOURCES: frozenset[Source] = frozenset({
    Source.WB_CARD,
    Source.OZON_CARD,
    Source.LAMODA,
    Source.ICECAT,
    Source.PDF_DATASHEET,
    Source.DESCRIPTION,
})


def _is_objective_spec_attr(target: TargetAttribute) -> bool:
    """True when the attribute is in the OBJECTIVE-SPEC class.

    Objective-spec attributes have factual ground truths that an LLM can be
    confidently wrong about (e.g. True Wireless=true on over-ear headphones,
    Звуковая схема=2.0 on a mono speaker, Бязь fabric on a Nike tee).  For
    these attrs, LLM self-judgment is insufficient — corroboration from an
    authoritative source is required instead (Stage 4.9 gate).

    Three orthogonal signals trigger the class, checked in order:
      1. semantic_type is in _OBJECTIVE_SPEC_SEMANTIC_TYPES (explicit tag).
      2. TargetAttribute.type is "bool" (binary feature-presence: Да/Нет).
      3. TargetAttribute.type is "numeric" (physical measurement with unit).
      4. Name (lower-cased) contains any substring from
         _OBJECTIVE_SPEC_NAME_FRAGMENTS (catches material/connectivity/audio
         attrs regardless of whether semantic_type was populated).

    Judgment calls (deliberately NOT included):
      - "color", "brand", "model" — wrong but less dangerous than mud; Gate B
        LLM-verify is appropriate (LLM *does* know the brand/color from name).
      - Long free-text description attrs — not factual-spec, stay on Gate B.
      - Short lifestyle enums (style, occasion) — subjective, Gate B is fine.
    """
    # Signal 1: explicit semantic_type tag
    st = (target.semantic_type or "").lower()
    if st and st in _OBJECTIVE_SPEC_SEMANTIC_TYPES:
        return True

    # Signal 2 & 3: structural type signals
    if target.type in ("bool", "numeric"):
        return True

    # Signal 4: name-based heuristic (covers attrs where semantic_type is None)
    name_low = target.name.lower()
    for fragment in _OBJECTIVE_SPEC_NAME_FRAGMENTS:
        if fragment in name_low:
            return True

    return False


def _normalize_for_corroboration(value: object) -> str:
    """Normalise a fill value for source-corroboration equality checks.

    Rules (deterministic, no LLM):
      - Convert to str and strip whitespace.
      - Lower-case.
      - Replace ё→е (Russian letter equivalence).
      - Collapse internal whitespace sequences to a single space.
      - Boolean aliases: True / «Да» / «yes» / «1» → «да»;
                         False / «Нет» / «no» / «0» → «нет».
    """
    raw = str(value).strip()
    # Boolean aliases
    lower_raw = raw.lower()
    if lower_raw in {"true", "да", "yes", "1"}:
        return "да"
    if lower_raw in {"false", "нет", "no", "0"}:
        return "нет"
    # General normalisation
    norm = raw.lower()
    norm = norm.replace("ё", "е")
    norm = " ".join(norm.split())
    return norm


def _web_search_grounded_in_evidence(value: str, evidence: str | None) -> bool:
    """Gate A: self-consistency check for WEB_SEARCH fills (single scalar value).

    The web_search `evidence` field is the REAL fetched page snippet, so we can
    ground-check the chosen value against it deterministically. Returns True when
    the value is present (any of its significant tokens) in the evidence text.

    Algorithm:
      1. Normalise both value and evidence (ё→е, lower-case, collapse whitespace).
      2. Tokenise the value via _matcher_token_re (same tokeniser used in brand-matching).
         Filter tokens shorter than 3 chars — too short to disambiguate (e.g. "нт").
      3. Require AT LEAST ONE significant value-token to appear literally in the
         normalised evidence text.  This is conservative: a single matching token
         is sufficient to keep the fill (protects the ~113 correct fills while
         catching clear mismatches like "Бязь" vs "100% хлопок").

    Edge cases:
      - evidence is None or empty → return True (no evidence to contradict; don't drop).
      - value normalises to a boolean ("да"/"нет") → skip check (booleans are not
        grounded by evidence text presence).

    For LIST-valued fills use _filter_list_value_by_evidence instead — that helper
    checks each element individually and returns the grounded subset (or None to drop).
    """
    if not evidence:
        return True  # nothing to check against — conservative, do not drop

    norm_value = _normalize_for_corroboration(value)
    if norm_value in ("да", "нет"):
        return True  # boolean fills don't appear as tokens in evidence text

    norm_evidence = evidence.lower().replace("ё", "е")

    # Tokenise the normalised value
    tokens = _matcher_token_re.findall(norm_value)
    significant = [t for t in tokens if len(t) >= 3]
    if not significant:
        return True  # value too short to check — conservative, keep

    return any(tok in norm_evidence for tok in significant)


def _filter_list_value_by_evidence(
    elements: list,
    evidence: str | None,
) -> list | None:
    """Gate A for LIST-valued WEB_SEARCH fills.

    Checks each list element against the evidence individually via
    _web_search_grounded_in_evidence.  Returns:
      - the filtered list (only grounded elements) when ≥1 element passes, OR
      - None when ALL elements fail (caller should drop the whole fill).

    Conservative: an element with no significant tokens (too short / boolean)
    is kept (delegates to _web_search_grounded_in_evidence's own conservative
    edge-case handling).
    """
    if not evidence:
        return elements  # no evidence → keep all, conservative

    grounded = [
        el for el in elements
        if _web_search_grounded_in_evidence(str(el), evidence)
    ]
    return grounded if grounded else None


def _apply_gender_guard(
    all_values: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
) -> list[AttributeValue]:
    """Генеральный гендер-гард на merge-слое для поля «Пол» (gender-target).

    Применяется к кандидатам ВСЕХ источников ДО merge (а не только к карточным,
    как card-level страховки). Каждый AttributeValue несёт один .source, поэтому
    «поддерживающие источники» для значения — это множество source'ов всех
    кандидатов на тот же attribute_id, несущих этот же (нормализованный) элемент.

    Правила (только для gender-target атрибутов, детект по имени таргета):
      1. Берём гендер ИМЕНИ товара (_extract_gender_signal по product_name).
      2. ЯВНЫЙ гендер имени (male/female): дропаем элемент, чей пол КОНФЛИКТУЕТ
         с именем (male vs female). unisex/нейтрал — не конфликт, не трогаем.
      3. НЕЙТРАЛЬНОЕ имя (None): дропаем гендерный элемент, если ВСЕ источники,
         его подтверждающие, — external-guess (_GENDER_EXTERNAL_SOURCES). Если
         хоть один товар-специфичный источник несёт его — оставляем.
      4. Если после дропа значение опустело — кандидат выбрасывается целиком
         (поле остаётся незаполненным, дефолт НЕ навязываем).

    Не-gender targets и кандидаты на них проходят сквозь без изменений.
    """
    # attribute_id → True если это gender-target (по имени таргета).
    # required_gender_ids — подмножество, помеченное is_required: для них поле НЕ
    # должно молча опустеть, если есть уверенный сигнал пола из имени (FIX 2).
    gender_attr_ids: set[int] = set()
    required_gender_ids: set[int] = set()
    for t in targets:
        if _is_gender_target_name(t.name):
            gender_attr_ids.add(t.id)
            if getattr(t, "is_required", False):
                required_gender_ids.add(t.id)
    if not gender_attr_ids:
        return all_values

    name_gender = _extract_gender_signal(context.product_name or "")
    # Канон Ozon-значения для пола из ИМЕНИ (только взрослые male/female — детские
    # каноны из имени _extract_gender_signal не различает, поэтому их не навязываем).
    _eg_name_gender_canon = {"male": "Мужской", "female": "Женский"}.get(name_gender)

    # Для правила 3: какие источники подтверждают каждый (attribute_id, элемент).
    support: dict[tuple[int, str], set] = {}
    for v in all_values:
        if v.attribute_id not in gender_attr_ids:
            continue
        for el in _norm_elements(v.value):
            support.setdefault((v.attribute_id, el), set()).add(v.source)

    def _drop_element(attr_id: int, norm_el: str) -> bool:
        """True если этот элемент-значение «Пол» надо выбросить."""
        el_gender = _extract_gender_signal(norm_el)
        if el_gender is None or el_gender == "unisex":
            return False  # нейтральное/унисекс значение — не спекуляция, не трогаем
        # Правило 2: явный пол имени, конфликт male vs female.
        if name_gender is not None and name_gender != "unisex":
            return el_gender != name_gender
        # Правило 3: нейтральное имя — дропаем только если все источники external.
        srcs = support.get((attr_id, norm_el), set())
        return bool(srcs) and srcs.issubset(_GENDER_EXTERNAL_SOURCES)

    out: list[AttributeValue] = []
    # attribute_id required-gender целей, для которых ХОТЬ ОДИН кандидат уцелел.
    required_filled: set[int] = set()
    for v in all_values:
        if v.attribute_id not in gender_attr_ids:
            out.append(v)
            continue
        if isinstance(v.value, list):
            kept_idx = [
                i for i, el in enumerate(v.value)
                if not _drop_element(v.attribute_id, str(el).strip().lower())
            ]
            if not kept_idx:
                logger.info(
                    "[Pipeline] gender-guard: дроп кандидата '%s' (source=%s, "
                    "имя-пол=%s) — все элементы отсеяны",
                    v.value, v.source.value, name_gender,
                )
                continue  # поле опустело → кандидат выбрасывается
            if v.attribute_id in required_gender_ids:
                required_filled.add(v.attribute_id)
            if len(kept_idx) != len(v.value):
                new_value = [v.value[i] for i in kept_idx]
                new_ids = None
                if isinstance(v.value_ids, list) and len(v.value_ids) == len(v.value):
                    new_ids = [v.value_ids[i] for i in kept_idx]
                out.append(v.model_copy(update={"value": new_value, "value_ids": new_ids}))
            else:
                out.append(v)
        else:
            if _drop_element(v.attribute_id, str(v.value).strip().lower()):
                logger.info(
                    "[Pipeline] gender-guard: дроп '%s'='%s' (source=%s, имя-пол=%s)",
                    v.attribute_id, v.value, v.source.value, name_gender,
                )
                continue
            if v.attribute_id in required_gender_ids:
                required_filled.add(v.attribute_id)
            out.append(v)

    # FIX 2: REQUIRED поле «Пол» не должно молча опустеть из-за гард-дропа, когда
    # есть УВЕРЕННЫЙ пол из имени товара («Футболка мужская» → «Мужской»). Если
    # required gender-target опустел И имя дало явный взрослый пол (male/female) —
    # подставляем его канон. Имя нейтрально/неизвестно → НЕ выдумываем (оставляем
    # пусто). Детский пол из имени мы не различаем, поэтому его не навязываем.
    if _eg_name_gender_canon is not None:
        for attr_id in required_gender_ids:
            if attr_id in required_filled:
                continue
            out.append(AttributeValue(
                attribute_id=attr_id,
                value=_eg_name_gender_canon,
                confidence=_EG_GENDER_REQUIRED_FALLBACK_CONF,
                source=Source.DESCRIPTION,
                evidence=_EG_GENDER_FROM_NAME_EVIDENCE,
            ))
            logger.info(
                "[Pipeline] gender-guard: REQUIRED поле %s опустело после дропа — "
                "подставлен пол из имени '%s'",
                attr_id, _eg_name_gender_canon,
            )
    return out


# Brand-from-name резолвер на POST-merge слое (генеральный, без хардкода брендов).
# Имя товара часто содержит бренд («Толстовка худи Champion Reverse Weave»,
# «Джинсы мужские Levi's 501»), но поле «Бренд» либо ПУСТО, либо забито МУСОРОМ —
# чужим allowed-enum значением, к которому enum-matcher «прилип» (HUGO вместо
# Levi's, LEGO вместо Adidas). Заголовок авторитетен для бренда: если ровно один
# из allowed-брендов присутствует в имени как слово(сочетание) — фиксируем его.
# Граница 3 символа отсекает ложняки на 1-2 буквенных брендах.
_BRAND_TARGET_ATTR_ID = 31
_BRAND_MIN_LEN = 3
# Токенайзер для brand-матчинга (зеркало matcher._TOKEN_RE — не импортируем сам
# matcher, чтобы не тянуть torch/sentence_transformers в pipeline-модуль).
_matcher_token_re = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)
# Тег источника для заполненного из имени бренда. DESCRIPTION — товар-специфичный
# сигнал (имя/заголовок этого товара), наивысший приоритет на merge.
_BRAND_FROM_NAME_EVIDENCE = "brand_from_name"


def _is_brand_target_name(name: str) -> bool:
    """True если имя таргета — поле «Бренд» (brand). Без хардкода attribute_id.

    Детект по имени: «бренд» / «brand» / «торгов* марк*» (торговая марка) как
    подстрока (имена коротки и однозначны — «Бренд», «Бренд в одежде»,
    «Торговая марка»). Сам id==31 матчится отдельно в _apply_brand_from_name.
    """
    low = name.lower()
    if "бренд" in low or "brand" in low:
        return True
    return bool(re.search(r"торгов\w*\s+марк", low))


def _brand_norm_tokens(text: str) -> list[str]:
    """Токены строки для brand-матчинга: ё→е, lowercase, latin/cyrillic/digits.

    Переиспользует _TOKEN_RE матчера (`[а-яёa-z0-9]+`): пунктуация и апострофы
    становятся разделителями, поэтому «Levi's»→['levi','s'], «be quiet!»→
    ['be','quiet'], «The North Face»→['the','north','face'].
    """
    return _matcher_token_re.findall(text.lower().replace("ё", "е"))


# RE для удаления апострофов и дефисов ВНУТРИ слова перед токенизацией.
# Нужен для «Levi's»→'levis', «Dri-FIT»→'drifit' canon-формы.
_punct_collapse_re = re.compile(r"['’\-]")


def _brand_canon_tokens(text: str) -> list[str]:
    """Canon-токены: апостроф/дефис убираются ПЕРЕД токенизацией.

    «Levi's» → 'levis' → ['levis'] — та же форма, что и title-токен 'Levis'.
    «Dri-FIT» → 'Drifit' → ['drifit'].
    Используется как ЗАПАСНОЕ сравнение в _brand_in_name/_brand_match_span:
    если нормальные токены не совпали — пробуем canon-форму обеих сторон.
    """
    stripped = _punct_collapse_re.sub("", text)
    return _brand_norm_tokens(stripped)


def _brand_in_name(brand: str, name_tokens: list[str]) -> bool:
    """True если бренд присутствует в имени как непрерывная цепочка токенов.

    Многословные бренды («The North Face») матчатся как contiguous-подпоследо-
    вательность токенов имени — без ложняков на разбросанных совпадениях.
    Бренд короче _BRAND_MIN_LEN символов (после нормализации, склейка токенов)
    игнорируется.

    ДОПОЛНИТЕЛЬНО: если нормальные токены не совпали — пробуем canon-форму
    (апостроф/дефис убраны перед токенизацией), чтобы «Levi's» ↔ «Levis»
    совпадали: _brand_canon_tokens('Levi's')=['levis'] vs name_token 'levis' ✓.
    """
    b_tokens = _brand_norm_tokens(brand)
    if not b_tokens:
        return False
    if sum(len(t) for t in b_tokens) < _BRAND_MIN_LEN:
        return False
    n = len(b_tokens)
    for i in range(len(name_tokens) - n + 1):
        if name_tokens[i:i + n] == b_tokens:
            return True
    # Canon-fallback: апостроф/дефис-инсенситивный матч.
    # «Levi's»→['levis'] vs name_token 'levis'. Длина canon-токенов МОЖЕТ
    # отличаться от нормальных → ищем canon-бренд в canon-имени.
    b_canon = _brand_canon_tokens(brand)
    if b_canon != b_tokens:
        name_canon = [_punct_collapse_re.sub("", t) for t in name_tokens]
        nc = len(b_canon)
        for i in range(len(name_canon) - nc + 1):
            if name_canon[i:i + nc] == b_canon:
                return True
    return False


def _brand_match_span(brand: str, name_tokens: list[str]) -> Optional[tuple[int, int]]:
    """Token-span (start, end-exclusive) ПЕРВОГО вхождения бренда в имя, иначе None.

    Зеркало _brand_in_name, но возвращает позицию матча — нужно для
    containment-collapse (вложенные матчи: «NORTH» ⊂ «The North Face»,
    «Original» ⊂ «Adidas Originals» дают пересекающиеся/вложенные спаны).
    Бренд < _BRAND_MIN_LEN символов игнорируется (как в _brand_in_name).

    ДОПОЛНИТЕЛЬНО: canon-fallback (апостроф/дефис-инсенситивный матч).
    Позиция спана возвращается ВСЕГДА в координатах ИСХОДНЫХ name_tokens.
    """
    b_tokens = _brand_norm_tokens(brand)
    if not b_tokens:
        return None
    if sum(len(t) for t in b_tokens) < _BRAND_MIN_LEN:
        return None
    n = len(b_tokens)
    for i in range(len(name_tokens) - n + 1):
        if name_tokens[i:i + n] == b_tokens:
            return (i, i + n)
    # Canon-fallback: апостроф/дефис-инсенситивный матч.
    b_canon = _brand_canon_tokens(brand)
    if b_canon != b_tokens:
        name_canon = [_punct_collapse_re.sub("", t) for t in name_tokens]
        nc = len(b_canon)
        for i in range(len(name_canon) - nc + 1):
            if name_canon[i:i + nc] == b_canon:
                # Возвращаем спан в исходных координатах: canon-токены могут
                # меньше нормальных (слияние апостроф-частей), поэтому конец
                # спана = start + len(name_tokens), но мы матчим nc canon-токенов
                # — каждый соответствует ОДНОМУ исходному name_token (apострофы
                # убираются из исходного токена, не склеивают два токена). OK.
                return (i, i + nc)
    return None


def _is_gender_noise_token(token: str) -> bool:
    """True если ОДИНОЧНЫЙ токен — гендер-слово (Мужская/Женские/...) → шум.

    Переиспользует _extract_gender_signal (грамматические стемы пола, генерально,
    без хардкода): одиночный токен с явным/унисекс гендер-сигналом — это
    атрибут «Пол» из заголовка, а не бренд.
    """
    return _extract_gender_signal(token) is not None


def _category_type_words(category_path: list[str]) -> set[str]:
    """Нормализованные основы слов категории-leaf — слова «типа товара».

    leaf «Футболки» → {'футболк'}; «Джинсы мужские» → {'джинс','мужск'}.
    Используется для дропа noise-матча, чей токен совпадает с типом товара
    (футболка/куртка/джинсы). Грубый стем = первые 5 символов нормализованного
    слова: покрывает словоформы (футболк-а/-и, куртк-а/-у) без лемматизатора.
    Generic, без хардкода списка одежды — берётся из category_path товара.
    """
    if not category_path:
        return set()
    leaf = category_path[-1]
    words: set[str] = set()
    for w in _brand_norm_tokens(leaf):
        if len(w) >= _BRAND_MIN_LEN:
            words.add(w[:5])
    return words


def _is_type_noise_token(token: str, type_words: set[str]) -> bool:
    """True если одиночный токен совпадает с типом товара (категория-leaf) → шум."""
    if not type_words:
        return False
    norm = _brand_norm_tokens(token)
    if len(norm) != 1:
        return False
    return norm[0][:5] in type_words


# Окончания русских КАЧЕСТВЕННЫХ/ОТНОСИТЕЛЬНЫХ прилагательных (общие падежные формы).
# Используется для дропа шумовых «брендов» вида «Спортивные», «Чёрные», «Прямые».
# Список ГЕНЕРАЛЬНЫЙ (морфология, не хардкод слов): прилагательные во всех падежах/
# числах/родах рус. языка оканчиваются на эти суффиксы. Наречия/существительные
# редко на них заканчиваются — ложняки минимальны.
_RU_ADJ_ENDINGS = (
    "ые", "ие", "ая", "яя", "ое", "ее",
    "ый", "ий", "ой",
    "ому", "ему",
    "ыми", "ими",
    "ых", "их",
    "ую", "юю",
    "ого", "его",
    "ым", "им",
)


def _is_adjective_noise_token(token: str) -> bool:
    """True если ОДИНОЧНЫЙ токен — русское прилагательное по морфологии → шум.

    Генеральное правило (не хардкод слов): кириллический токен длиной ≥6 символов,
    оканчивающийся на типичное русское падежное/родовое окончание прилагательного
    (-ые/-ие/-ая/-яя/-ое/-ее/-ый/-ий/-ой/-ому/-ыми/-ых и т.д.).
    Латинские токены — НИКОГДА не прилагательные (бренды Nike/Adidas/Dri-FIT латинские).
    Минимальная длина 6 уберегает от ложняков на коротких словах («дне», «тые»).
    Примеры:
      «спортивные» → True  (шум-описатель, не бренд)
      «чёрные»     → True
      «прямые»     → True
      «nike»       → False (латиница)
      «wrangler»   → False (латиница)
      «адидас»     → False (нет прил. окончания)
    """
    # Только кириллические токены могут быть прилагательными рус. языка
    if not token or not all(c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюя" for c in token.lower()):
        return False
    # Нормализуем ё→е перед проверкой окончаний
    t = token.lower().replace("ё", "е")
    if len(t) < 6:
        return False
    return t.endswith(_RU_ADJ_ENDINGS)


# Минимальная длина бренда в символах для NEEDLE-пути (менее строгая, чем dict-путь).
# Dict-путь: ≥3 символов (много шумовых 1-2 буквенных токенов в 123K dict).
# Needle-путь: ≥2 символов (context.brand — известный бренд из метаданных товара,
# аббревиатуры LG/HP/JBL легитимны). Суммарная длина токенов кандидата ≥ 2.
_NEEDLE_BRAND_MIN_LEN = 2


def _needle_brand_in_name(brand: str, name_tokens: list[str]) -> bool:
    """Needle-вариант _brand_in_name с пониженным минимумом длины (≥2 символа).

    Используется ТОЛЬКО в NEEDLE-пути, где brand = context.brand (известный бренд
    из метаданных, не случайная строка из 123K словаря). Поэтому 2-символьные
    аббревиатуры (LG, HP, JBL) допустимы — ложняки на 2-символьных токенах
    маловероятны, если кандидат пришёл из надёжного источника.
    Логика аналогична _brand_in_name, но без _BRAND_MIN_LEN > 2 ограничения.
    """
    b_tokens = _brand_norm_tokens(brand)
    if not b_tokens:
        return False
    # Needle min: ≥ 2 суммарно (блок однобуквенных, пропускаем LG / HP).
    if sum(len(t) for t in b_tokens) < _NEEDLE_BRAND_MIN_LEN:
        return False
    n = len(b_tokens)
    for i in range(len(name_tokens) - n + 1):
        if name_tokens[i:i + n] == b_tokens:
            return True
    # Canon-fallback: апостроф/дефис-инсенситивный матч (аналогично _brand_in_name).
    b_canon = _brand_canon_tokens(brand)
    if b_canon != b_tokens:
        name_canon = [_punct_collapse_re.sub("", t) for t in name_tokens]
        nc = len(b_canon)
        for i in range(len(name_canon) - nc + 1):
            if name_canon[i:i + nc] == b_canon:
                return True
    return False


def _all_category_words(category_path: list[str]) -> set[str]:
    """Нормализованные 5-символьные стеммы ВСЕХ слов category_path (не только leaf).

    Расширяет _category_type_words на полную цепочку категорий: для пути
    ['Электроника', 'Умные колонки', 'колонка'] вернёт стеммы всех слов
    из всех узлов — не только leaf. Используется в NEEDLE-гарде чтобы
    отвергнуть кандидатов типа «колонка», «книга», «машина», «печь» — слова
    из ЛЮБОГО узла категории-пути, не только листового.
    Граница ≥ _BRAND_MIN_LEN символов (3) — те же правила, что и type_words.
    """
    words: set[str] = set()
    for node in (category_path or []):
        for w in _brand_norm_tokens(node):
            if len(w) >= _BRAND_MIN_LEN:
                words.add(w[:5])
    return words


def _is_category_noun_brand(brand: str, cat_words: set[str]) -> bool:
    """True если кандидат-бренд совпадает с категорийным существительным.

    Генеральное правило (без хардкода): кандидат отвергается, если КАЖДЫЙ его
    нормализованный токен имеет 5-символьный стемм, совпадающий с одним из
    cat_words (слов из category_path). Односимвольные токены игнорируются.
    Примеры (cat_path=['Электроника','Умная колонка']):
      'колонка' → stem='колон' ∈ cat_words → True (отвергнуть)
      'книга'   → stem='книга' ∈ {'книга'} при path=['Электронная книга'] → True
      'Яндекс'  → stem='яндек' ∉ cat_words → False (пропустить)
      'LG'      → stem='lg' ∉ cat_words → False (пропустить)
    Многословные кандидаты отвергаются только если ВСЕ токены = категорийные
    (консервативно: «Умная колонка» как бренд — пограничный случай, не блокируем).
    """
    if not cat_words:
        return False
    b_tokens = _brand_norm_tokens(brand)
    # Игнорируем короткие токены (1-2 символа) при проверке — не категорийное слово.
    meaningful = [t for t in b_tokens if len(t) >= _BRAND_MIN_LEN]
    if not meaningful:
        return False
    return all(t[:5] in cat_words for t in meaningful)


def _disambiguate_brand_matches(
    matches: list[str],
    name_tokens: list[str],
    type_words: set[str],
) -> list[str]:
    """Бренд-aware фильтр матчей ДО подсчёта двусмысленности (генеральный).

    Шаги (в порядке):
      1. Containment-collapse: если token-спан одного матча ВЛОЖЕН в спан другого
         (NORTH ⊂ The North Face, Original ⊂ Adidas Originals) — дроп короткого,
         оставляем максимальный по покрытию.
      2. Дроп noise-матчей, чей единственный токен = ГЕНДЕР-слово (Мужская/Женские)
         или = ТИП товара из category-leaf (футболка/куртка/джинсы).
      3. Leftmost-tiebreak: если после шагов 1-2 уцелело ≥2 «настоящих» бренда
         («Nike Sportswear Club»→{Nike,Sportswear,Club}, «Wrangler Texas»→
         {Wrangler,Texas}) — выбираем ОДИН по самому раннему вхождению в имени.
         RU marketplace-заголовки кладут настоящий бренд первым в описательной
         части («<Тип> <пол> <БРЕНД> <модель>...»), поэтому leftmost = бренд.
    Возвращает дедупнутый список «настоящих» брендов (после шага 3 — ≤1 элемент при
    наличии хотя бы одного матча). НЕ хардкодит бренды/одежду.
    """
    # Уникальные матчи со спанами (по нормализованной форме — дедуп написаний).
    spanned: dict[tuple[str, ...], tuple[str, tuple[int, int]]] = {}
    for b in matches:
        span = _brand_match_span(str(b), name_tokens)
        if span is None:
            continue
        key = tuple(_brand_norm_tokens(str(b)))
        # Оставляем матч с максимальным покрытием на случай дублей.
        prev = spanned.get(key)
        if prev is None or (span[1] - span[0]) > (prev[1][1] - prev[1][0]):
            spanned[key] = (str(b), span)

    items = list(spanned.values())

    # --- Шаг 1: containment-collapse ---
    kept: list[tuple[str, tuple[int, int]]] = []
    for brand, (s, e) in items:
        contained = False
        for other_brand, (os_, oe) in items:
            if (os_, oe) == (s, e) and other_brand == brand:
                continue
            # строго БОЛЬШИЙ спан, полностью покрывающий текущий
            if os_ <= s and e <= oe and (oe - os_) > (e - s):
                contained = True
                break
        if not contained:
            kept.append((brand, (s, e)))

    # --- Шаг 2: дроп гендер/тип/прилагательного-шума (только одиночные токены) ---
    # Прилагательное-шум: «Спортивные», «Чёрные», «Прямые» — реальные enum-бренды
    # в «Бренд в одежде», но НЕ торговые марки. Генеральное морфо-правило:
    # кириллический токен с типичным рус. прилагательным окончанием → шум.
    # Bias to SAFETY: если после дропа НЕ осталось НИ ОДНОГО НЕ-прилагательного
    # матча — возвращаем пустой список (empty > wrong).
    real: list[tuple[str, tuple[int, int]]] = []
    adj_only_kept: list[tuple[str, tuple[int, int]]] = []  # только прилагательные
    for brand, (s, e) in kept:
        if e - s == 1:
            tok = name_tokens[s]
            if _is_gender_noise_token(tok) or _is_type_noise_token(tok, type_words):
                continue
            if _is_adjective_noise_token(tok):
                adj_only_kept.append((brand, (s, e)))
                continue
        real.append((brand, (s, e)))
    # Safety: если real пуст, но adj_only_kept не пуст → НЕ возвращаем прилагательное.
    # empty better than wrong descriptor adjective as brand.
    if not real and adj_only_kept:
        logger.info(
            "[Pipeline] brand-from-name: все уцелевшие матчи — прилагательные %s "
            "→ пустой результат (empty > wrong descriptor)",
            [b for b, _ in adj_only_kept],
        )
        return []

    # --- Шаг 3: leftmost-tiebreak при ≥2 уцелевших «настоящих» брендах ---
    # Заголовок RU-маркетплейса: «<Тип> <пол> <БРЕНД> <модель>...» — настоящий бренд
    # стоит ПЕРВЫМ в описательной части после типа/пола. Шум типа/пола уже отсеян на
    # шаге 2, поэтому самый ранний по позиции токена матч — это бренд. Резолвим в
    # ОДИН бренд по наименьшей стартовой позиции спана.
    if len(real) >= 2:
        brand, _span = min(real, key=lambda item: item[1][0])
        logger.info(
            "[Pipeline] brand-from-name: %d брендов уцелело после фильтра — "
            "leftmost-tiebreak выбрал '%s'",
            len(real), brand,
        )
        return [brand]
    return [b for b, _span in real]


def _resolve_brand_value_id(
    brand: str,
    attr_id: int,
    brand_id_fn: Optional[Callable[[int], dict[str, int]]],
) -> Optional[int]:
    """Словарный value_id выбранного бренда ТОЧНЫМ матчем, иначе None.

    Берёт {value:id}-карту словаря через brand_id_fn(attr_id) и ищет brand
    exact-ом (case + ё/е-инсенситивно). Без fuzzy/частичных совпадений — owner
    чувствителен к неверным id, поэтому привязываем id ТОЛЬКО при точном
    совпадении строки бренда со словарным ключом. brand_id_fn недоступен/упал/
    нет ключа → None (не выдумываем id).
    """
    if brand_id_fn is None:
        return None
    try:
        pairs = brand_id_fn(attr_id) or {}
    except Exception as exc:
        logger.warning(
            "[Pipeline] brand-from-name: {value:id}-карта брендов для attr %s "
            "недоступна: %s", attr_id, exc,
        )
        return None
    if not pairs:
        return None
    # Exact lookup: сперва как есть, затем по нормализованному ключу (ё→е, lower).
    if brand in pairs:
        return pairs[brand]

    def _norm(s: str) -> str:
        return s.lower().replace("ё", "е")

    target = _norm(brand)
    for val, vid in pairs.items():
        if _norm(val) == target:
            return vid
    return None


def _derive_context_brand(
    context: ExtractionContext,
    targets: list[TargetAttribute],
    brand_options_fn: Optional[Callable[[int], list[str]]] = None,
) -> Optional[str]:
    """Подобрать бренд-кандидат из названия, когда context.brand пуст (Баг 3 eg_importer).

    Продавец оставил колонку «Бренд» пустой, но бренд стоит в начале названия
    («Nike Air Max 90»). Без context.brand brand-gated источники (ozon_card/regard/
    onliner/bestbuy/IceCat) не находят donor-карточку и брэнд-гейтят её → Пол/Цвет/
    Размер каскадят в no_data, хотя бренд явно в имени. Бэкфиллим context.brand
    ВЫСОКОТОЧНО — той же логикой, что _apply_brand_from_name: ровно ОДИН словарный
    бренд, присутствующий в имени (containment-дизамбигуация + дроп категория/
    гендер/прилагательное-шума). Ноль матчей или двусмысленность → None: пустой
    бренд честнее, чем мусорная донор-карточка не того бренда.

    Не заполняет сам таргет «Бренд» (это делает _apply_brand_from_name POST-merge);
    только проставляет context.brand как КОНТЕКСТ для источников ДО их запуска.
    """
    if (context.brand or "").strip():
        return None  # бренд задан продавцом — не трогаем
    brand_targets = [
        t for t in targets
        if t.id == _BRAND_TARGET_ATTR_ID or _is_brand_target_name(t.name)
    ]
    if not brand_targets:
        return None
    name_tokens = _brand_norm_tokens(context.product_name or "")
    if not name_tokens:
        return None
    type_words = _category_type_words(context.category_path)
    _BRAND_MAX_INLINE = 100  # зеркало _apply_brand_from_name: ≤100 → вероятно truncated-срез
    for t in brand_targets:
        options = list(t.allowed_values or [])
        if brand_options_fn is not None and len(options) <= _BRAND_MAX_INLINE:
            try:
                full_opts = list(brand_options_fn(t.id) or [])
                if len(full_opts) > len(options):
                    options = full_opts
            except Exception as exc:  # словарь недоступен — не падаем, пропускаем
                logger.warning(
                    "[Pipeline] derive-context-brand: словарь брендов attr %s недоступен: %s",
                    t.id, exc,
                )
        if not options:
            continue
        matches = [b for b in options if _brand_in_name(str(b), name_tokens)]
        real = _disambiguate_brand_matches(matches, name_tokens, type_words)
        if len(real) == 1:
            return str(real[0])
    return None


_COLOR_FROM_NAME_EVIDENCE = "color_from_name"
# Окончания рус. прилагательных (длиннейшие первыми) — стем цвета: черные↔черный.
_COLOR_ADJ_ENDINGS = (
    "ого", "его", "ыми", "ими", "ому", "ему",
    "ый", "ий", "ой", "ая", "яя", "ое", "ее", "ые", "ие",
    "ым", "им", "ых", "их", "ую", "юю", "ом", "ем",
    # Соединительные гласные сложных цветов: «черн-О-белый», «темн-О-синий»,
    # «син-Е-зелёный» — чтобы комбинирующая форма стемилась к базовому цвету
    # (черно→черн, темно→темн) и сложный цвет ловился/детектился как двусмысленный.
    "о", "е",
)


def _color_stem(word: str) -> str:
    """Стем цвета-прилагательного: ё→е, lower, срез согласующего окончания.

    «чёрные»/«чёрный»/«чёрная»→«черн»; «синие»/«синий»→«син». Срезаем только если
    остаётся ≥3 символов основы (не курочим короткие слова).
    """
    w = word.strip().lower().replace("ё", "е")
    for end in _COLOR_ADJ_ENDINGS:
        if len(w) - len(end) >= 3 and w.endswith(end):
            return w[:-len(end)]
    return w


def _apply_color_from_name(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
) -> list[AttributeValue]:
    """Заполнить ПУСТОЙ «Цвет товара» цветом из НАЗВАНИЯ (расцветка продавца).

    Цвет — per-SKU вариантная ось; единственный надёжный источник — само название
    («…чёрные»→чёрный). Доноры (WbCard/ozon_card) отдают палитру ЧУЖОЙ расцветки
    (дропается `_drop_multivalue_color_premerge`). Матчим токены имени против
    allowed_values цвета по СТЕМУ (черные↔черный); ровно один матч → заполняем
    ВАЛИДНЫМ словарным значением (value_id добьёт resolve_value_ids). Ноль или
    двусмысленность (черно-белые → 2 цвета) → не трогаем (пусто честнее). Только
    пустой таргет — одиночный grounded-цвет донора не перетираем.
    """
    color_targets = [t for t in targets if _is_color_target(t) and t.allowed_values]
    if not color_targets:
        return merged
    name_tokens = _brand_norm_tokens(context.product_name or "")
    if not name_tokens:
        return merged
    name_stems = {_color_stem(t) for t in name_tokens}

    filled_ids = {
        v.attribute_id for v in merged
        if v.value not in (None, "", []) and not (isinstance(v.value, str) and not v.value.strip())
    }
    additions: list[AttributeValue] = []
    for t in color_targets:
        if t.id in filled_ids:
            continue
        matches: list[str] = []
        for val in t.allowed_values:
            val_words = _brand_norm_tokens(str(val))
            if not val_words:
                continue
            val_stems = [_color_stem(w) for w in val_words]
            if all(s in name_stems for s in val_stems):
                matches.append(str(val))
        uniq = list(dict.fromkeys(matches))
        if len(uniq) == 1:
            logger.info("[Pipeline] color-from-name: attr=%s ← %r (из названия)", t.id, uniq[0])
            additions.append(AttributeValue(
                attribute_id=t.id, value=uniq[0], confidence=0.8,
                source=Source.DESCRIPTION, evidence=_COLOR_FROM_NAME_EVIDENCE,
            ))
        elif len(uniq) > 1:
            logger.info(
                "[Pipeline] color-from-name: attr=%s двусмысленно %s — не трогаем", t.id, uniq,
            )
    return merged + additions


def _apply_brand_from_name(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
    brand_options_fn: Optional[Callable[[int], list[str]]] = None,
    brand_id_fn: Optional[Callable[[int], dict[str, int]]] = None,
) -> list[AttributeValue]:
    """POST-merge brand-from-name резолвер для поля «Бренд» (генеральный).

    Для каждого brand-таргета (детект по id==31 ИЛИ имени «Бренд/Brand/Торговая
    марка» — allowed_values НЕ требуется): если в ИМЕНИ товара присутствует РОВНО
    ОДИН из СЛОВАРНЫХ брендов (как непрерывная цепочка токенов, бренд ≥3 символов)
    — это B:
      • поле НЕ заполнено → заполняем B (source=DESCRIPTION, evidence=brand_from_name);
      • поле заполнено значением != B → ПЕРЕЗАПИСЫВАЕМ на B (заголовок авторитетен).
    Если в имени НЕТ ни одного словарного бренда, либо их НЕСКОЛЬКО (двусмысленно)
    — поле НЕ трогаем (анти-мусор: не угадываем).

    Источник полного списка брендов (по убыванию приоритета):
      1. target.allowed_values, если не пусто (мелкий enum, список в таргете);
      2. brand_options_fn(attr_id) — ПОЛНЫЙ словарный список (Бренд — огромный
         truncated enum, его allowed_values НЕ переносятся в target; список даёт
         стратегия через словарь). Это закрывает реальный кейс «Бренд» (id 31).
      3. NEEDLE fallback (только когда dict пуст): context.brand присутствует в имени
         — принимаем как единственный кандидат. value_id резолвится async-путём
         (resolve_value_ids_async → search_value API) — здесь остаётся None.
    Если все источники пусты — таргет пропускается (нечего матчить).

    НИКОГДА не пишет бренд, отсутствующий И в имени, И в списке брендов: B всегда
    выбирается из списка И присутствует в имени. В needle-режиме — context.brand
    проверяется на вхождение в name_tokens теми же правилами (_brand_in_name).

    value_id: «Бренд» (id 31) — часто truncated enum (>100k значений), статический
    словарь держит лишь первые 5000, поэтому sync resolve_value_ids в _finalize
    нередко НЕ находит id выбранного бренда → value_id=None → бренд дропается на
    required-enum (drain C). Чтобы этого избежать, прямо здесь привязываем словарный
    value_id для выбранного B через brand_id_fn (та же {value:id}-карта словаря,
    что отдаёт стратегия) ТОЧНЫМ матчем (case/ё-insensitive, без fuzzy — owner
    чувствителен к неверным id). Если id не резолвится — оставляем None (не
    выдумываем), последующий resolve_value_ids ещё раз попробует.
    """
    # brand-таргеты (allowed_values НЕ требуется — для truncated «Бренд» он пуст).
    brand_targets: dict[int, TargetAttribute] = {}
    for t in targets:
        if t.id == _BRAND_TARGET_ATTR_ID or _is_brand_target_name(t.name):
            brand_targets[t.id] = t
    if not brand_targets:
        return merged

    name_tokens = _brand_norm_tokens(context.product_name or "")
    if not name_tokens:
        return merged

    # Слова «типа товара» из category-leaf (футболк-/куртк-/джинс-) — для дропа
    # noise-матчей, маскирующихся под бренд в enum «Бренд в одежде и обуви».
    type_words = _category_type_words(context.category_path)

    # Какое каноническое B подобрать для каждого brand-таргета (ровно один матч).
    resolved_brand: dict[int, str] = {}
    # Привязанный словарный value_id выбранного бренда (exact-матч), если нашёлся.
    resolved_brand_id: dict[int, int] = {}
    # Порог: если allowed_values содержит ≤ _BRAND_MAX_INLINE брендов, список может
    # быть усечённым вырезом из truncated enum (напр. eval передаёт first-50 из 123k).
    # В таком случае ПРЕДПОЧИТАЕМ ПОЛНЫЙ словарь через brand_options_fn.
    # Маленькие НЕ-brand enum'ы (Тип/Пол/Сезон: ≤50 опций) не затрагиваются, поскольку
    # brand_targets содержит только brand-таргеты (детект по id==31 или имени).
    _BRAND_MAX_INLINE = 100  # если allowed_values ≤ 100 → вероятно truncated-срез

    def _try_needle_brand(attr_id: int) -> None:
        """NEEDLE-фоллбэк: принять context.brand, если он валиден и есть в имени.

        Срабатывает когда словарь брендов пуст/недоступен ИЛИ непуст, но НИ ОДИН
        словарный бренд не найден в имени (усечённый срез без нужного бренда —
        иначе бренд silent-пустой). Гарды: токен-матч ≥2 симв.; категорийное
        слово / гендер / прилагательное — отвергаем. value_id=None →
        resolve_value_ids_async добьёт через Ozon search_value (truncated path).
        """
        ctx_brand = (context.brand or "").strip()
        if not (ctx_brand and _needle_brand_in_name(ctx_brand, name_tokens)):
            return
        all_cat_words = _all_category_words(context.category_path)
        if _is_category_noun_brand(ctx_brand, all_cat_words):
            logger.info(
                "[Pipeline] brand-from-name NEEDLE: attr %s, context.brand=%r "
                "ОТВЕРГНУТ — совпадает с категорийным словом (category_path=%s)",
                attr_id, ctx_brand, context.category_path,
            )
            return
        if _is_gender_noise_token(ctx_brand) or _is_adjective_noise_token(ctx_brand):
            logger.info(
                "[Pipeline] brand-from-name NEEDLE: attr %s, context.brand=%r "
                "ОТВЕРГНУТ — гендерное/прилагательное слово, не бренд",
                attr_id, ctx_brand,
            )
            return
        logger.info(
            "[Pipeline] brand-from-name NEEDLE: attr %s, context.brand=%r в имени — "
            "принят напрямую (нет словарного матча, value_id=None → async-resolve)",
            attr_id, ctx_brand,
        )
        resolved_brand[attr_id] = ctx_brand

    for attr_id, t in brand_targets.items():
        # Полный список брендов: target.allowed_values (мелкий enum) ИЛИ словарь.
        options = list(t.allowed_values or [])

        # Если allowed_values коротко (≤ _BRAND_MAX_INLINE) — это почти наверняка
        # усечённый срез из truncated-enum (Бренд: 123k записей). Запрашиваем
        # ПОЛНЫЙ список через brand_options_fn; если он возвращает больше опций —
        # используем его (гарантирует нахождение Nike/Adidas/... вне first-50 среза).
        if brand_options_fn is not None and len(options) <= _BRAND_MAX_INLINE:
            try:
                full_opts = list(brand_options_fn(attr_id) or [])
                if len(full_opts) > len(options):
                    options = full_opts
            except Exception as exc:  # словарь недоступен — не падаем, пропускаем
                logger.warning(
                    "[Pipeline] brand-from-name: словарный список брендов для attr %s "
                    "недоступен: %s", attr_id, exc,
                )

        if options:
            matches = [b for b in options if _brand_in_name(str(b), name_tokens)]
            # Бренд-aware дизамбигуация ДО подсчёта: containment-collapse + дроп
            # гендер/тип-шума (фейковые «бренды» в enum «Бренд в одежде»: футболка,
            # Мужская, NORTH ⊂ The North Face). Дедуп по норм-форме внутри.
            real = _disambiguate_brand_matches(matches, name_tokens, type_words)
            if len(real) == 1:
                chosen = real[0]
                resolved_brand[attr_id] = chosen
                # Привязка словарного value_id выбранного бренда ТОЧНЫМ матчем. «Бренд» —
                # truncated enum: sync resolve_value_ids в _finalize часто не находит id
                # (нет в первых 5000 словаря) → бренд дропается. Берём id из той же
                # {value:id}-карты словаря, что и список опций. Без fuzzy — owner
                # чувствителен к неверным id; только exact (case/ё-insensitive).
                vid = _resolve_brand_value_id(chosen, attr_id, brand_id_fn)
                if vid is not None:
                    resolved_brand_id[attr_id] = vid
            elif len(real) > 1:
                logger.info(
                    "[Pipeline] brand-from-name: %d брендов в имени '%s' для attr %s — "
                    "двусмысленно, не трогаем",
                    len(real), context.product_name, attr_id,
                )
            else:
                # options непусты, но НИ ОДИН словарный бренд не найден в имени —
                # вероятно усечённый срез без нужного бренда. NEEDLE по context.brand
                # (те же гарды) спасает от silent-пустого бренда (drain при truncated
                # allowed_values 101..N, когда brand_options_fn не запрашивается).
                _try_needle_brand(attr_id)
        else:
            # options пусто (truncated enum полностью вне первых 5000) → NEEDLE.
            _try_needle_brand(attr_id)

    if not resolved_brand:
        return merged

    out: list[AttributeValue] = []
    seen_attr: set[int] = set()
    for v in merged:
        b = resolved_brand.get(v.attribute_id)
        if b is None:
            out.append(v)
            continue
        seen_attr.add(v.attribute_id)
        cur = str(v.value).strip()
        if cur.lower().replace("ё", "е") == b.lower().replace("ё", "е"):
            out.append(v)  # уже корректный бренд
            continue
        logger.info(
            "[Pipeline] brand-from-name: перезапись attr %s '%s'→'%s' (из имени)",
            v.attribute_id, v.value, b,
        )
        out.append(v.model_copy(update={
            "value": b,
            # Словарный value_id выбранного бренда (exact). None → пере-резолвится
            # стратегией в _finalize. Привязка здесь спасает truncated «Бренд» от
            # drain-C дропа (sync resolve_value_ids не находит id вне первых 5000).
            "value_id": resolved_brand_id.get(v.attribute_id),
            "confidence": 0.95,
            "source": Source.DESCRIPTION,
            "evidence": _BRAND_FROM_NAME_EVIDENCE,
        }))

    # Незаполненные brand-таргеты с найденным B — создаём значение.
    for attr_id, b in resolved_brand.items():
        if attr_id in seen_attr:
            continue
        logger.info(
            "[Pipeline] brand-from-name: заполнение attr %s='%s' (из имени)",
            attr_id, b,
        )
        out.append(AttributeValue(
            attribute_id=attr_id,
            value=b,
            value_id=resolved_brand_id.get(attr_id),  # exact словарный id, иначе None
            confidence=0.95,
            source=Source.DESCRIPTION,
            evidence=_BRAND_FROM_NAME_EVIDENCE,
        ))
    return out


# ---------------------------------------------------------------------------
# Brand-from-title LLM: constrained LLM extraction for brand when deterministic
# parser misses it (two-word category nouns: "Умная колонка Яндекс"→Яндекс).
# ---------------------------------------------------------------------------
_BRAND_FROM_TITLE_LLM_EVIDENCE = "brand_from_title_llm"
_BRAND_FROM_TITLE_LLM_CONF = 0.88

# Safety: maximum brand options to pass in the LLM prompt.
# Huge brand dicts (123k) will be truncated; deterministic path handles those
# via dict scan. LLM path is primarily for the "title has brand but needle
# grabbed the category noun" case — a small working set is sufficient and safe.
_BRAND_FROM_TITLE_LLM_MAX_OPTIONS = 200


async def _apply_brand_from_title_llm(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
    brand_options_fn: Optional[Callable[[int], list[str]]] = None,
    brand_id_fn: Optional[Callable[[int], dict[str, int]]] = None,
) -> list[AttributeValue]:
    """POST-merge ASYNC brand filler: LLM picks brand from TITLE.

    Runs ONLY for brand targets that are STILL EMPTY after _apply_brand_from_name.
    Scenario: "Умная колонка Яндекс Станция Мини 2" — the positional parser set
    context.brand="колонка" (category noun, correctly rejected by needle guard).
    The dict scan also misses Яндекс when it's absent from the first 5000 brands.
    This LLM call bridges that gap.

    Two sub-paths depending on whether the brand target has an allowed-values list:

    ENUM-CONSTRAINED (enum_free=False, original behaviour):
      The prompt receives the official allowed-brand list and must return one of
      those values or null. Value not in the options list → drop.
      value_id is resolved via brand_id_fn; survives drain-C (required-enum drop).

    ENUM-FREE (enum_free=True, new path):
      Brand target has no allowed_values — it is a free-text field. Applies to
      categories where «Бренд» accepts arbitrary strings (Умная колонка, Электронная
      книга, Стиральная машина, Микроволновая печь, …). Uses an unconstrained prompt
      that asks "what brand name literally appears in the title?". All the same
      safety guards apply (title-anchored, category-noun, gender/adj noise); only the
      enum-match check is skipped (there is no enum). value_id stays None by design
      (free-text) — drain-C (_drop_unresolved_required_enums) skips targets without
      allowed_values, so None value_id is safe here.

    Shared safety constraints (fail-closed — «пусто честнее мусора»):
      1. TITLE-ANCHORED: returned value must pass _needle_brand_in_name token check.
      2. NULL allowed: LLM returns null → field stays empty. No forced fills.
      3. CATEGORY-NOUN GUARD: _is_category_noun_brand rejects category nouns.
      4. GENDER/ADJ NOISE GUARD: _is_gender_noise_token / _is_adjective_noise_token.
      5. Source=DESCRIPTION, evidence="brand_from_title_llm": bypasses
         _BRAND_GUESS_SOURCES guard (reads the title, not world knowledge).
      6. General LLM brand guessing (LLM_KNOWLEDGE source) remains DROPPED.
    """
    # Lazy import to avoid circular deps and keep LLM import at call site only.
    from pydantic import BaseModel as _PydanticBase, Field as _Field

    brand_targets: dict[int, TargetAttribute] = {}
    for t in targets:
        if t.id == _BRAND_TARGET_ATTR_ID or _is_brand_target_name(t.name):
            brand_targets[t.id] = t
    if not brand_targets:
        return merged

    # Only act on targets STILL EMPTY after prior deterministic brand fill.
    filled_attr_ids: set[int] = {v.attribute_id for v in merged}

    name = (context.product_name or "").strip()
    if not name:
        return merged
    name_tokens = _brand_norm_tokens(name)
    if not name_tokens:
        return merged

    all_cat_words = _all_category_words(context.category_path)

    # Collect (attr_id, options_list, enum_free) triples that still need filling.
    # enum_free=True  → brand target has NO allowed_values (free-text field).
    #                   Use unconstrained title-extraction prompt; skip enum-match guard.
    # enum_free=False → normal enum-constrained path (existing behaviour, unchanged).
    pending: list[tuple[int, list[str], bool]] = []
    for attr_id, t in brand_targets.items():
        if attr_id in filled_attr_ids:
            continue  # already filled by deterministic path

        options = list(t.allowed_values or [])

        # Prefer the full dict list for truncated enums (same logic as _apply_brand_from_name).
        _BRAND_MAX_INLINE = 100
        if brand_options_fn is not None and len(options) <= _BRAND_MAX_INLINE:
            try:
                full_opts = list(brand_options_fn(attr_id) or [])
                if len(full_opts) > len(options):
                    options = full_opts
            except Exception as exc:
                logger.warning(
                    "[Pipeline] brand-from-title-llm: brand list unavailable for attr %s: %s",
                    attr_id, exc,
                )

        enum_free = not options
        # enum_free targets: run unconstrained title-extraction (all safety guards kept).
        # enum-constrained targets: run original allowed-list prompt.
        pending.append((attr_id, options, enum_free))

    if not pending:
        return merged

    # One LLM call per pending brand target (usually exactly 1).
    out = list(merged)
    llm = get_main_manager()

    def _norm_cmp(s: str) -> str:
        return s.strip().lower().replace("ё", "е")

    for attr_id, options, enum_free in pending:

        if enum_free:
            # ── FREE-TEXT brand path ────────────────────────────────────────────
            # No allowed-list to constrain against; prompt asks for the exact
            # manufacturer/brand name that literally appears in the title.

            class _BrandFreeTextResponse(_PydanticBase):
                brand: Optional[str] = _Field(
                    None,
                    description=(
                        "The manufacturer/brand name that literally appears in the "
                        "product title as a word or phrase (its exact spelling from "
                        "the title), or null if no clear brand name is present."
                    ),
                )

            system_prompt = (
                "You are a brand-extraction assistant. Your ONLY job is to find the "
                "manufacturer or brand name that is literally written in the product title.\n\n"
                "RULES (non-negotiable):\n"
                "1. Return the brand ONLY if it appears verbatim in the title as a word or phrase.\n"
                "2. Do NOT infer brands from world knowledge or context. Read the title literally.\n"
                "3. Category nouns are NOT brands: колонка, книга, машина, печь, телефон, "
                "принтер, монитор, планшет, холодильник, пылесос and similar descriptive words "
                "describe the product type, NOT the maker → return null for those.\n"
                "4. Return the brand spelled EXACTLY as it appears in the title.\n"
                "5. If the title contains no identifiable brand name → return null.\n"
                "6. Null is always safer than a wrong answer."
            )
            user_text = (
                f"Product title: {name}\n\n"
                "What manufacturer or brand name appears literally in this title? "
                "Return its exact spelling from the title, or null if none is present."
            )

            try:
                parsed, _ = await llm.structured_request(
                    system_prompt=system_prompt,
                    user_text=user_text,
                    response_model=_BrandFreeTextResponse,
                )
            except Exception as exc:
                logger.warning(
                    "[Pipeline] brand-from-title-llm (free-text): LLM call failed for attr %s: %s",
                    attr_id, exc,
                )
                continue

            context.llm_calls_so_far += 1

            if parsed is None or parsed.brand is None:
                logger.info(
                    "[Pipeline] brand-from-title-llm (free-text): attr %s — LLM returned null "
                    "for title=%r",
                    attr_id, name[:60],
                )
                continue

            brand_raw = str(parsed.brand).strip()
            if not brand_raw:
                continue

            # Safety guard 1 (free-text): must appear verbatim in the title.
            # Use _needle_brand_in_name (min-len=2) — brand comes from LLM reading the title,
            # not from world-knowledge, so 2-char abbreviations (LG, HP) are acceptable.
            if not _needle_brand_in_name(brand_raw, name_tokens):
                logger.info(
                    "[Pipeline] brand-from-title-llm (free-text): attr %s, '%s' NOT found in "
                    "title tokens → drop (title-anchored constraint)",
                    attr_id, brand_raw,
                )
                continue

            # Safety guard 2 (free-text): category-noun filter.
            if _is_category_noun_brand(brand_raw, all_cat_words):
                logger.info(
                    "[Pipeline] brand-from-title-llm (free-text): attr %s, '%s' is a category "
                    "noun → drop",
                    attr_id, brand_raw,
                )
                continue

            # Safety guard 3 (free-text): adjective/gender noise.
            b_tokens_ft = _brand_norm_tokens(brand_raw)
            if len(b_tokens_ft) == 1 and (
                _is_gender_noise_token(b_tokens_ft[0]) or _is_adjective_noise_token(b_tokens_ft[0])
            ):
                logger.info(
                    "[Pipeline] brand-from-title-llm (free-text): attr %s, '%s' is "
                    "gender/adj noise → drop",
                    attr_id, brand_raw,
                )
                continue

            # All free-text guards passed.
            # value_id stays None — free-text brand has no enum to bind.
            # drain-C (_drop_unresolved_required_enums) checks `not target.allowed_values`
            # first and skips the whole target → None value_id is safe here.
            logger.info(
                "[Pipeline] brand-from-title-llm (free-text): attr %s ← '%s' from title=%r",
                attr_id, brand_raw, name[:60],
            )
            out.append(AttributeValue(
                attribute_id=attr_id,
                value=brand_raw,
                value_id=None,
                confidence=_BRAND_FROM_TITLE_LLM_CONF,
                source=Source.DESCRIPTION,
                evidence=_BRAND_FROM_TITLE_LLM_EVIDENCE,
            ))

        else:
            # ── ENUM-CONSTRAINED brand path (original behaviour, unchanged) ────
            # Truncate to a safe size to keep prompt concise.
            options_for_prompt = options[:_BRAND_FROM_TITLE_LLM_MAX_OPTIONS]
            options_str = ", ".join(f'"{o}"' for o in options_for_prompt)

            class _BrandFromTitleResponse(_PydanticBase):
                brand: Optional[str] = _Field(
                    None,
                    description=(
                        "Exact brand value from the allowed list that is PRESENT in the title, "
                        "or null if no allowed brand appears in the title."
                    ),
                )

            system_prompt = (
                "You are a brand-extraction assistant. Your ONLY job is to identify which brand "
                "from the provided ALLOWED LIST is explicitly present in the product title.\n\n"
                "RULES (non-negotiable):\n"
                "1. Return a brand ONLY if it literally appears in the title as a word or phrase.\n"
                "2. Do NOT guess or infer brands from world knowledge. If unsure → return null.\n"
                "3. The returned value must be EXACTLY one of the allowed values (spelling must match).\n"
                "4. If zero or multiple allowed brands appear in the title → return null.\n"
                "5. Category nouns (колонка, книга, машина, печь, телефон, etc.) are NOT brands → null.\n"
                "6. Return null rather than an incorrect brand. Empty is safer than wrong."
            )
            user_text = (
                f"Product title: {name}\n\n"
                f"Allowed brands: {options_str}\n\n"
                "Which single brand from the allowed list appears in this title? "
                "Return its exact spelling from the list, or null."
            )

            try:
                parsed, _ = await llm.structured_request(
                    system_prompt=system_prompt,
                    user_text=user_text,
                    response_model=_BrandFromTitleResponse,
                )
            except Exception as exc:
                logger.warning(
                    "[Pipeline] brand-from-title-llm: LLM call failed for attr %s: %s",
                    attr_id, exc,
                )
                continue

            context.llm_calls_so_far += 1

            if parsed is None or parsed.brand is None:
                logger.info(
                    "[Pipeline] brand-from-title-llm: attr %s — LLM returned null for title=%r",
                    attr_id, name[:60],
                )
                continue

            brand_raw = str(parsed.brand).strip()
            if not brand_raw:
                continue

            # Safety guard 1: returned value must be in the options list (enum constraint).
            norm_raw = _norm_cmp(brand_raw)
            matched_option: Optional[str] = None
            for opt in options:
                if _norm_cmp(opt) == norm_raw:
                    matched_option = opt
                    break
            if matched_option is None:
                logger.info(
                    "[Pipeline] brand-from-title-llm: attr %s, LLM returned '%s' "
                    "NOT in options list → drop (enum constraint)",
                    attr_id, brand_raw,
                )
                continue

            # Safety guard 2: must appear in the title (title-anchored, use needle min-len=2
            # since these are known brands from the official enum, not arbitrary strings).
            if not _needle_brand_in_name(matched_option, name_tokens):
                logger.info(
                    "[Pipeline] brand-from-title-llm: attr %s, '%s' NOT found in title tokens "
                    "→ drop (title-anchored constraint)",
                    attr_id, matched_option,
                )
                continue

            # Safety guard 3: category-noun filter (same as needle path).
            if _is_category_noun_brand(matched_option, all_cat_words):
                logger.info(
                    "[Pipeline] brand-from-title-llm: attr %s, '%s' is a category noun → drop",
                    attr_id, matched_option,
                )
                continue

            # Safety guard 4: adjective/gender noise.
            b_tokens = _brand_norm_tokens(matched_option)
            if len(b_tokens) == 1 and (
                _is_gender_noise_token(b_tokens[0]) or _is_adjective_noise_token(b_tokens[0])
            ):
                logger.info(
                    "[Pipeline] brand-from-title-llm: attr %s, '%s' is gender/adj noise → drop",
                    attr_id, matched_option,
                )
                continue

            # All guards passed — resolve value_id and emit.
            vid = _resolve_brand_value_id(matched_option, attr_id, brand_id_fn)
            logger.info(
                "[Pipeline] brand-from-title-llm: attr %s ← '%s' (vid=%s) from title=%r",
                attr_id, matched_option, vid, name[:60],
            )
            out.append(AttributeValue(
                attribute_id=attr_id,
                value=matched_option,
                value_id=vid,
                confidence=_BRAND_FROM_TITLE_LLM_CONF,
                source=Source.DESCRIPTION,
                evidence=_BRAND_FROM_TITLE_LLM_EVIDENCE,
            ))

    return out


_TYPE_FROM_CATEGORY_EVIDENCE = "type_from_category"


def _apply_type_from_category(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
    value_id_fn: Optional[Callable[[int, str], Optional[int]]] = None,
) -> list[AttributeValue]:
    """POST-merge guard: заполняет ПУСТОЙ обязательный enum из category-leaf (генеральный).

    Зеркало _apply_brand_from_name, но для «типа товара»: category leaf — это И ЕСТЬ
    тип товара (leaf «Футболки» → enum-значение «Футболка»). Без хардкода поля «Тип»
    или имени категории — срабатывает только по ТОЧНОМУ enum-матчу:

    Для КАЖДОГО REQUIRED enum-таргета (allowed_values непуст), который сейчас ПУСТ или
    нерезолвнут (value_id=None):
      • нормализуем category-leaf = context.category_path[-1] (lower+strip+лемма через
        тот же _lemma/pymorphy3, что и wb_card_source — без новой зависимости);
      • если нормализованный leaf ТОЧНО (case + морфология, БЕЗ fuzzy) совпадает ровно с
        одной из allowed-опций таргета — заполняем таргет этой опцией (строка как в
        словаре) + её value_id (через value_id_fn тем же путём, что brand-from-name).
      • нет точного совпадения → НЕ трогаем (drop-guard ниже опустошит мусор).

    Это GENERAL, не per-category: для не-«Тип» enum (Цвет/Материал…) leaf не совпадёт с
    их опциями → harmless no-op. Никогда НЕ перезаписывает уже резолвнутое значение
    (value_id присутствует) — empty > wrong, resolved > derived.

    Ordering: вызывается в _finalize_async ПОСЛЕ brand-from-name/gender/llm_resolve_tail,
    но ДО _drop_unresolved_required_enums — чтобы корректно выведенный тип заполнился и
    НЕ был сдроплен.
    """
    if not context.category_path:
        return merged
    leaf = (context.category_path[-1] or "").strip()
    if not leaf:
        return merged
    leaf_lemma = _type_lemma(leaf)
    if not leaf_lemma:
        return merged

    def _norm(s: str) -> str:
        return _type_lemma(str(s).strip()) if str(s).strip() else ""

    # Существующие значения по attr_id: пусто/None-value_id → кандидат на заполнение.
    by_attr: dict[int, AttributeValue] = {}
    for v in merged:
        by_attr.setdefault(v.attribute_id, v)

    out = list(merged)
    for t in targets:
        # Scope: ТОЛЬКО required enum (conservative). Не enum / optional → no-op.
        if not t.is_required or not t.allowed_values:
            continue
        existing = by_attr.get(t.id)
        # Уже резолвнутое значение (value_id есть) — НИКОГДА не трогаем.
        if existing is not None and existing.value_id is not None:
            continue
        # Ровно-один ТОЧНЫЙ (норм/лемма) матч leaf среди allowed-опций.
        matches = [opt for opt in t.allowed_values if _norm(opt) == leaf_lemma]
        if len(matches) != 1:
            continue
        chosen = matches[0]
        vid: Optional[int] = None
        if value_id_fn is not None:
            try:
                vid = value_id_fn(t.id, chosen)
            except Exception as exc:  # резолвер упал — не падаем, оставляем None
                logger.warning(
                    "[Pipeline] type-from-category: value_id для attr %s '%s' "
                    "недоступен: %s", t.id, chosen, exc,
                )
                vid = None
        if existing is not None:
            # Перезаписываем ПУСТОЕ/нерезолвнутое значение того же attr_id.
            logger.info(
                "[Pipeline] type-from-category: заполнение attr %s='%s' из category-leaf '%s'",
                t.id, chosen, leaf,
            )
            out = [
                v.model_copy(update={
                    "value": chosen,
                    "value_id": vid,
                    "confidence": 0.9,
                    "source": Source.DESCRIPTION,
                    "evidence": _TYPE_FROM_CATEGORY_EVIDENCE,
                }) if v is existing else v
                for v in out
            ]
        else:
            logger.info(
                "[Pipeline] type-from-category: заполнение пустого attr %s='%s' из "
                "category-leaf '%s'", t.id, chosen, leaf,
            )
            out.append(AttributeValue(
                attribute_id=t.id,
                value=chosen,
                value_id=vid,
                confidence=0.9,
                source=Source.DESCRIPTION,
                evidence=_TYPE_FROM_CATEGORY_EVIDENCE,
            ))
    return out


_SIZE_FROM_NAME_EVIDENCE = "size_from_name"
_SIZE_FROM_NAME_CONF = 0.80  # lower than card sources (0.93/0.85) — name is a weaker signal


def _apply_size_from_name(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
    value_id_fn: Optional[Callable[[int, str], Optional[int]]] = None,
) -> list[AttributeValue]:
    """LOW-PRIORITY FALLBACK: fill «Российский размер» from an explicit size token in the name.

    Called ONLY when attr 4295/4298 is still EMPTY after all other sources
    (Lever 1 WB sizes_table, card characteristics, IceCat, LLM, etc.).

    Rules (fail-closed — «пусто честнее мусора»):
      1. parse_explicit_size(name) must return a non-empty list of tokens.
         Returns [] when unsure → we leave the field empty.
      2. Each token is expanded via expand_intl_to_ru() (letter → list of RU candidates,
         range "42-44" → ["42","44"], numeric "46" → ["46"]).
         Ambiguous letter→RU mappings emit ALL candidates because is_collection=True —
         the Ozon dict will validate each; only resolve_value_id hits survive.
      3. resolve via value_id_fn; drop candidates where value_id_fn returns None.
      4. If NO candidate resolves → emit nothing (empty).

    NEVER overwrites an already-filled (value present) size attribute — fallback only.
    """
    # Find size target(s)
    size_targets = [
        t for t in targets
        if t.id in _RU_SIZE_ATTR_IDS or _is_ru_size_target_name(t.name)
    ]
    if not size_targets:
        return merged

    # Check which size attrs are already filled
    filled_attr_ids: set[int] = {
        v.attribute_id for v in merged
        if v.attribute_id in {t.id for t in size_targets}
    }

    # Parse size tokens from name
    name = (context.product_name or "").strip()
    if not name:
        return merged

    raw_tokens = parse_explicit_size(name)
    if not raw_tokens:
        return merged

    # Expand all raw tokens into RU numeric candidates
    all_candidates: list[str] = []
    seen_c: set[str] = set()
    for tok in raw_tokens:
        for candidate in expand_intl_to_ru(tok):
            if candidate not in seen_c:
                seen_c.add(candidate)
                all_candidates.append(candidate)
    if not all_candidates:
        return merged

    out = list(merged)
    for target in size_targets:
        if target.id in filled_attr_ids:
            continue  # already filled — fallback does not overwrite

        if value_id_fn is None:
            continue

        resolved_ids: list[int] = []
        for candidate in all_candidates:
            try:
                vid = value_id_fn(target.id, candidate)
            except Exception:
                vid = None
            if vid is not None:
                resolved_ids.append(vid)

        # Dedup
        seen_ids: set[int] = set()
        unique_ids: list[int] = []
        for vid in resolved_ids:
            if vid not in seen_ids:
                seen_ids.add(vid)
                unique_ids.append(vid)
        resolved_ids = unique_ids

        if not resolved_ids:
            logger.debug(
                "[Pipeline] size-from-name: tokens=%s → 0 resolved value_ids for attr %s "
                "(all unresolvable — leaving empty)",
                all_candidates, target.id,
            )
            continue

        logger.info(
            "[Pipeline] size-from-name: attr %s ← name=%r tokens=%s resolved_ids=%s",
            target.id, name[:60], all_candidates, resolved_ids,
        )
        out.append(AttributeValue(
            attribute_id=target.id,
            value=all_candidates,
            confidence=_SIZE_FROM_NAME_CONF,
            source=Source.DESCRIPTION,
            evidence=_SIZE_FROM_NAME_EVIDENCE,
            is_collection=True,
            value_id=None,
            value_ids=resolved_ids,
        ))
    return out


# ---------------------------------------------------------------------------
# Lever 1: "Название" — fill the marketplace product-title target verbatim from
# the input product name.
#
# The marketplace schema has a product-title attribute (name == "Название", or
# synonyms like "Наименование товара") that is often left empty in the gap.
# We already RECEIVE this value as context.product_name — fill it verbatim.
# Conservative guards:
#   - Only matches the STANDALONE title field ("Название" / "Наименование")
#     NOT any compound attr with "название" embedded ("название цвета",
#     "название бренда", etc.).
#   - Only fills EMPTY targets — never overwrites.
#   - Source=DESCRIPTION (the input product name is product-specific data).
#   - Confidence 0.97 (verbatim, no inference) — above DESCRIPTION threshold.
# ---------------------------------------------------------------------------
_TITLE_FROM_INPUT_CONF = 0.97
_TITLE_FROM_INPUT_EVIDENCE = "название:from_input"

# Exact/bare standalone title field names (lower-stripped).
# The match is: attr name lowercased == one of these strings EXACTLY,
# OR attr name lowercased startswith one of these AND has no meaningful suffix.
# We use an exact-match set — cheap and safe.
_TITLE_FIELD_NAMES: frozenset[str] = frozenset({
    "название",
    "наименование",
    "наименование товара",
    "название товара",
    "полное наименование",
    "полное название",
})

# Substrings that disqualify an attr name — prevent matching "название цвета" etc.
_TITLE_FIELD_DISQUALIFIERS: tuple[str, ...] = (
    "цвет",
    "бренд",
    "марк",     # торговая марка
    "модел",    # название модели
    "серии",
    "серия",
    "вариант",
    "типа",     # название типа
)


def _is_title_target(name: str) -> bool:
    """True if the attribute is the standalone product-title field.

    Matches 'Название', 'Наименование товара' etc. while rejecting compounds
    like 'Название цвета', 'Название модели'.
    """
    low = name.lower().strip()
    # Exact match first (fastest, most conservative)
    if low in _TITLE_FIELD_NAMES:
        return True
    # Starts-with check: "название" or "наименование" as the first word
    if not (low.startswith("название") or low.startswith("наименование")):
        return False
    # Disqualifier check: reject compound names
    for dq in _TITLE_FIELD_DISQUALIFIERS:
        if dq in low:
            return False
    return True


def _apply_title_from_input(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
) -> list[AttributeValue]:
    """POST-merge verbatim filler for the product-title target attribute.

    Fills STILL-EMPTY 'Название' / 'Наименование товара' targets verbatim
    with context.product_name (the input product name received from the
    marketplace request). Never overwrites an already-filled attr.

    Source=DESCRIPTION (input name is product-specific, not inferred).
    Confidence=0.97 — verbatim, no inference.
    """
    product_name = (context.product_name or "").strip()
    if not product_name:
        return merged

    filled_attr_ids: set[int] = {v.attribute_id for v in merged}
    title_targets = [
        t for t in targets
        if _is_title_target(t.name) and t.id not in filled_attr_ids
    ]
    if not title_targets:
        return merged

    out = list(merged)
    for target in title_targets:
        logger.info(
            "[Pipeline] title-from-input: attr=%s '%s' ← '%s'",
            target.id, target.name, product_name[:80],
        )
        out.append(AttributeValue(
            attribute_id=target.id,
            value=product_name,
            confidence=_TITLE_FROM_INPUT_CONF,
            source=Source.DESCRIPTION,
            evidence=_TITLE_FROM_INPUT_EVIDENCE,
        ))
    return out


# ---------------------------------------------------------------------------
# Model-from-title: verbatim filler for free-text MODEL / title-template-model
# fields ("Модель", "Название модели для шаблона наименования"). These are
# explicitly DISQUALIFIED from _apply_title_from_input (they are not the full
# title), yet the model designation IS verbatim-present in the product name.
# Free-text only (never forces an enum), still-empty only → zero mud risk.
# ---------------------------------------------------------------------------
_MODEL_FROM_TITLE_EVIDENCE = "model_from_title"
_MODEL_FROM_TITLE_CONF = 0.9


def _is_model_title_target(name: str) -> bool:
    """True for a free-text model-designation / model-name-for-title field.

    Matches 'Модель', 'Модель наушников', 'Название модели для шаблона …'.
    Rejects color/material compounds ('Модель цвета', 'Название модели цвета').
    """
    low = name.lower().strip()
    if any(dq in low for dq in ("цвет", "материал", "размер")):
        return False
    if "название модели" in low:
        return True
    return low == "модель" or low.startswith("модель ")


def _strip_leading_brand(product_name: str, brand: str) -> str:
    """Drop a leading brand token from the name ('Apple AirPods Pro 2' → 'AirPods Pro 2').

    Only strips when the name starts with the brand; returns the original name
    when stripping would leave it empty or the brand is absent. Stays verbatim
    (a substring of the real product name) — no fabrication.
    """
    name = (product_name or "").strip()
    b = (brand or "").strip()
    if b and name.lower().startswith(b.lower()):
        rest = name[len(b):].strip(" -—|")
        return rest or name
    return name


def _apply_model_from_title(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
) -> list[AttributeValue]:
    """POST-merge verbatim filler for still-empty free-text MODEL fields.

    'Название модели для шаблона …' ← full product_name (it IS the title model).
    'Модель …' ← product_name with the leading brand stripped (cleaner designation).
    Skips enum targets (has allowed_values) and already-filled targets.
    Source=DESCRIPTION, conf=0.9 — verbatim, no inference.
    """
    product_name = (context.product_name or "").strip()
    if not product_name:
        return merged

    filled_attr_ids: set[int] = {v.attribute_id for v in merged}
    out = list(merged)
    for t in targets:
        if t.id in filled_attr_ids:
            continue
        if t.allowed_values:  # never force a value into a constrained enum
            continue
        if not _is_model_title_target(t.name):
            continue
        low = t.name.lower()
        value = product_name if "название модели" in low else _strip_leading_brand(
            product_name, context.brand or ""
        )
        if not value:
            continue
        logger.info(
            "[Pipeline] model-from-title: attr=%s '%s' ← '%s'",
            t.id, t.name, value[:80],
        )
        out.append(AttributeValue(
            attribute_id=t.id,
            value=value,
            confidence=_MODEL_FROM_TITLE_CONF,
            source=Source.DESCRIPTION,
            evidence=_MODEL_FROM_TITLE_EVIDENCE,
        ))
    return out


# ---------------------------------------------------------------------------
# Spec-from-title: deterministic filler for OPTIONAL enum attrs whose allowed
# value is literally present (whole token sequence) in the product title.
# Mirrors _apply_brand_from_name / _apply_type_from_category in spirit.
# ---------------------------------------------------------------------------
_SPEC_FROM_TITLE_EVIDENCE = "spec_from_title"
_SPEC_FROM_TITLE_CONF = 0.90
# Allowed values shorter than this (in total normalised token chars) are skipped —
# they produce too many false positives on common short tokens (e.g. "SSD" length 3
# is the minimum; single tokens < 3 chars are filtered by MIN sum).
_SPEC_MIN_VALUE_LEN = 3


def _spec_value_in_title(
    value: str,
    title_tokens: list[str],
) -> bool:
    """True if `value` appears as a contiguous normalised token sequence in `title_tokens`.

    Normalisation mirrors _brand_norm_tokens: ё→е, lowercase, only
    [а-яёa-z0-9]+ tokens. Allowed values shorter than _SPEC_MIN_VALUE_LEN
    chars (total over all tokens) are rejected to avoid accidental hits on
    common abbreviations / punctuation remnants.
    """
    v_tokens = _brand_norm_tokens(value)
    if not v_tokens:
        return False
    total_len = sum(len(t) for t in v_tokens)
    if total_len < _SPEC_MIN_VALUE_LEN:
        return False
    n = len(v_tokens)
    for i in range(len(title_tokens) - n + 1):
        if title_tokens[i : i + n] == v_tokens:
            return True
    return False


def _apply_spec_from_title(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
    value_id_fn: Optional[Callable[[int, str], Optional[int]]] = None,
) -> list[AttributeValue]:
    """POST-merge filler: fill still-EMPTY optional enum attrs from the product title.

    For each OPTIONAL enum target (has allowed_values) that is still EMPTY after
    all real sources, scan the normalised product title. If EXACTLY ONE allowed
    value appears as a whole-token/phrase match → fill it with
    source=DESCRIPTION, evidence=spec_from_title.

    Mud guards (fail-closed — «пусто честнее мусора»):
      1. Whole-word/phrase match only (contiguous normalised tokens, NOT arbitrary
         substring). Uses the same _brand_norm_tokens normalisation (ё→е,
         case-insensitive, latin+cyrillic+digits only).
      2. Min value length: ≥ _SPEC_MIN_VALUE_LEN (3) total chars across tokens.
         Skips trivially short values that could hit by coincidence.
      3. Ambiguity guard: if ≥2 different allowed values of the same attr both
         match the title → SKIP that attr entirely (empty > wrong).
      4. Only fills targets still EMPTY (no existing AttributeValue for that attr_id).
      5. Only fills OPTIONAL targets (is_required==False). Required fields are
         handled by _apply_type_from_category + mandatory pipeline stages.
      6. Only enum targets (allowed_values non-empty). Free-text fields are never
         touched — they have no allowed_values to match against.
      7. Never touches brand targets (detected by _is_brand_target_name / id==31) —
         those are handled by _apply_brand_from_name with dedicated logic.

    Called in _finalize_async AFTER _apply_size_from_name and BEFORE the
    _drop_unresolved_* guards, so filled values get a chance at value_id resolution.
    """
    if not (context.product_name or "").strip():
        return merged

    title_tokens = _brand_norm_tokens(context.product_name)
    if not title_tokens:
        return merged

    # attribute_ids that are already filled (any value present → skip)
    filled_ids: set[int] = {v.attribute_id for v in merged}

    out = list(merged)
    for t in targets:
        # Scope: ONLY optional, ONLY enum (has allowed_values), ONLY empty.
        if t.is_required:
            continue
        if not t.allowed_values:
            continue
        if t.id in filled_ids:
            continue
        # Brand attrs: leave to _apply_brand_from_name
        if t.id == _BRAND_TARGET_ATTR_ID or _is_brand_target_name(t.name):
            continue

        # Find which allowed values match the title as whole-token sequence.
        hitting: list[str] = [
            v for v in t.allowed_values
            if _spec_value_in_title(v, title_tokens)
        ]

        if not hitting:
            continue

        if len(hitting) >= 2:
            # Ambiguity guard: multiple hits → skip (empty > wrong).
            logger.info(
                "[Pipeline] spec-from-title: AMBIG attr=%s '%s' — %d values hit title"
                " %s → skip",
                t.id, t.name, len(hitting), hitting[:4],
            )
            continue

        chosen = hitting[0]
        vid: Optional[int] = None
        if value_id_fn is not None:
            try:
                vid = value_id_fn(t.id, chosen)
            except Exception as exc:
                logger.warning(
                    "[Pipeline] spec-from-title: value_id для attr %s '%s' "
                    "недоступен: %s", t.id, chosen, exc,
                )
                vid = None

        logger.info(
            "[Pipeline] spec-from-title: attr=%s '%s' ← '%s' (vid=%s) из title=%r",
            t.id, t.name, chosen, vid, (context.product_name or "")[:60],
        )
        out.append(AttributeValue(
            attribute_id=t.id,
            value=chosen,
            value_id=vid,
            confidence=_SPEC_FROM_TITLE_CONF,
            source=Source.DESCRIPTION,
            evidence=_SPEC_FROM_TITLE_EVIDENCE,
        ))
    return out


# ---------------------------------------------------------------------------
# Lever 2 (GENERAL): verbatim numeric-spec extractor from fetched text.
#
# Generalizes the original "срок службы" extractor to cover ANY numeric target
# attribute (Время автономной работы, Время зарядки, Мощность, Срок службы, …).
#
# For each STILL-EMPTY numeric/free-text target, the extractor:
#   1. Derives a distinctive keyword phrase from the attr name (lowercased, trimmed
#      of trailing unit suffix such as ", ч" / ", лет").
#   2. Determines the expected unit via _detect_attr_unit (icecat normalizer).
#   3. Builds a regex that requires BOTH the keyword AND a number with the matching
#      unit within a tight window (≤ 30 chars).
#   4. Searches product_description and then evidence strings.
#   5. Fills VERBATIM (number from text), Source=DESCRIPTION, conf=0.97.
#
# Cross-attribute bleed guard (critical, mud-sensitive):
#   Each attr gets its OWN keyword from its name — "зарядка 2 часа" will NOT fill
#   "время разговора" because the keyword for "время разговора" ("время разговора")
#   must appear in the window around the number, not just any keyword.
#
# Special case: "срок службы / срок эксплуатации" uses the original dedicated regex
# (Pattern A + B bidirectional) since these attrs often have no unit in their name
# and use "лет/год/года" as the unit word.
#
# Confidence 0.97 (> DESCRIPTION threshold 0.95) — verbatim + grounded match.
# ---------------------------------------------------------------------------
_NUMSPEC_EVIDENCE_PREFIX = "numspec:"
_NUMSPEC_CONF = 0.97

# --- Service-life kept as a special case (backward-compat for existing tests) ---

_SERVICE_LIFE_EVIDENCE_PREFIX = "срок_службы:verbatim:"
_SERVICE_LIFE_CONF = _NUMSPEC_CONF

# Match "срок службы/эксплуатации" (with punctuation) followed (within 0-25 chars)
# by a number + year-word, OR a number followed by "срок службы/эксплуатации".
# Captures the numeric string (group 1) in both orderings.
#
# Pattern A:  срок службы/эксплуатации ... N лет/год/года
# Pattern B:  N лет/год ... срок службы (value before phrase — rarer but occurs)
#
# Case-insensitive; allows spaces, colon, dash between phrase and number.
_SERVICE_LIFE_RE = re.compile(
    r"срок\s+(?:службы|эксплуатации)\W{0,10}(\d+(?:[.,]\d+)?)\s*(?:лет|год(?:а)?)\b"
    r"|"
    r"(\d+(?:[.,]\d+)?)\s*(?:лет|год(?:а)?)\W{0,30}срок\s+(?:службы|эксплуатации)",
    re.IGNORECASE,
)


def _is_service_life_target(name: str) -> bool:
    """True if the attribute name matches 'срок службы' or 'срок эксплуатации'."""
    low = name.lower()
    return "срок службы" in low or "срок эксплуатации" in low


def _extract_service_life_number(text: str) -> Optional[str]:
    """Extract the first verbatim numeric value from a 'срок службы' phrase in text.

    Returns the number as a clean string (comma→dot, no trailing zeros), or None
    if no matching phrase is found.

    Verbatim-only: the number MUST literally appear adjacent to a 'срок службы/
    эксплуатации' phrase in the real text. No guessing.
    """
    if not text:
        return None
    m = _SERVICE_LIFE_RE.search(text)
    if m is None:
        return None
    # Group 1: number comes after the phrase; Group 2: number comes before
    raw = m.group(1) or m.group(2)
    if not raw:
        return None
    # Normalise: comma→dot decimal separator, strip trailing zeros
    raw = raw.replace(",", ".")
    try:
        val = float(raw)
    except ValueError:
        return None
    # Format: integer if whole number, else keep decimal
    if val == int(val):
        return str(int(val))
    # Up to 1 decimal place (срок службы usually whole years)
    return f"{val:.1f}".rstrip("0").rstrip(".")


# --- General numeric keyword extractor ---

# Unit display tokens used in regex matching (value side). Each entry:
#   canonical_unit → list of regex-ready alternatives (sorted longest-first).
_UNIT_REGEX_TOKENS: dict[str, str] = {
    "min":   r"(?:мин(?:ут[аы]?)?\.?|min(?:utes?)?)",
    "hz":    r"(?:гц|hz|герц)",
    "ms":    r"(?:мс|ms|мс\.?)",
    "kg":    r"(?:кг|kg|килограмм(?:ов?)?)",
    "g":     r"(?:гр?\.?|граммов?|gram(?:m?s?)?)",
    "w":     r"(?:вт|w|ватт(?:ов?)?)",
    "v":     r"(?:в(?:ольт(?:ов?)?)?|v(?:olts?)?)",
    "mah":   r"(?:мач|mah|мa\.?ч\.?)",
    "cm":    r"(?:см|cm|сантиметр(?:ов?)?)",
    "mm":    r"(?:мм|mm|миллиметр(?:ов?)?)",
    "m":     r"(?:метр(?:ов?)?|(?<!\w)м(?!\w)|(?<!\w)m(?!\w))",
    "inch":  r"(?:дюйм(?:ов?)?|inch(?:es?)?|\")",
    "cd/m2": r"(?:кд/м[²2]|cd/m[²2])",
    "khz":   r"(?:кгц|khz)",
    "ghz":   r"(?:ггц|ghz)",
    "kwh":   r"(?:квтч|kwh)",
    "ft":    r"(?:фут(?:ов?)?|ft|feet|foot)",
}

# Keyword extraction: strip unit suffixes from attr name to get the distinctive phrase.
# e.g. "Время автономной работы, ч" → "время автономной работы"
_ATTR_NAME_UNIT_SUFFIX_RE = re.compile(
    r",?\s*(?:ч(?:асов?|\.)?|лет|год(?:а)?|мес(?:яц(?:ев?)?)?\.?|"
    r"мин(?:ут[аы]?)?\.?|гц|hz|мс|ms|кг|kg|г,?|гр\.?|вт|w|в\b|v\b|мач|mah|"
    r"см|cm|мм|mm|м\b|m\b|дюйм(?:ов?)?|inch(?:es?)?|кд/м[²2]|cd/m[²2]|"
    r"кгц|khz|ггц|ghz|квтч|kwh)\s*$",
    re.IGNORECASE,
)


def _attr_keyword(attr_name: str) -> str:
    """Derive a distinctive keyword phrase from attr name (strip unit suffix, lowercase)."""
    stripped = _ATTR_NAME_UNIT_SUFFIX_RE.sub("", attr_name).strip().lower()
    # Also strip trailing punctuation
    return stripped.rstrip(",;:.").strip()


def _build_numspec_regex(keyword: str, unit: str) -> Optional[re.Pattern[str]]:
    """Build a regex matching keyword + number + unit (or unit + number + keyword).

    Returns None if we can't build a meaningful pattern (unknown unit or empty keyword).

    The pattern requires BOTH keyword AND number+unit within ≤30 chars of each other.
    This is the cross-attribute bleed guard: only fires when the attr's OWN keyword
    phrase is present near the number.
    """
    unit_rx = _UNIT_REGEX_TOKENS.get(unit)
    if not unit_rx or not keyword:
        return None
    kw_rx = re.escape(keyword)
    num_rx = r"\d+(?:[.,]\d+)?"
    # Bridge between keyword and number (Pattern A: forward).
    # Chars between keyword and number: non-digit, non-newline, non-sentence-end.
    # Non-greedy: prevents consuming leading digits of the number.
    bridge_fwd = r"[^\d\n.!?]{0,30}?"
    # Bridge for Pattern B (backward: number unit ... keyword).
    # Must NOT cross sentence boundary (period/newline): "90 мин. Время разговора"
    # must NOT match for "время разговора" attr.
    bridge_bwd = r"[^\d\n.!?]{0,20}?"
    # Pattern A: keyword ... number unit (forward)
    # Pattern B: number unit ... keyword (backward, same sentence only)
    pattern = (
        rf"{kw_rx}{bridge_fwd}({num_rx})\s*{unit_rx}\b"
        rf"|"
        rf"({num_rx})\s*{unit_rx}\b{bridge_bwd}{kw_rx}"
    )
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


def _extract_numspec_number(text: str, pattern: re.Pattern[str]) -> Optional[tuple[str, str]]:
    """Search text for the attr-specific pattern; return (raw_number, snippet) or None.

    Verbatim-only: number MUST appear adjacent to the attr's own keyword.
    """
    if not text:
        return None
    m = pattern.search(text)
    if m is None:
        return None
    raw = m.group(1) or m.group(2)
    if not raw:
        return None
    val = _icecat_parse_float(raw)
    if val is None:
        return None
    # Format using icecat normalizer for consistency (drops trailing zeros, int for whole)
    # Use unit from the pattern context — we don't have it here; use generic formatting.
    raw_norm = raw.replace(",", ".")
    try:
        fval = float(raw_norm)
    except ValueError:
        return None
    if fval == int(fval):
        formatted = str(int(fval))
    else:
        formatted = raw_norm.rstrip("0").rstrip(".")
    start = max(0, m.start() - 10)
    end = min(len(text), m.end() + 10)
    snippet = text[start:end].strip()
    return formatted, snippet


def _apply_service_life_from_text(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
) -> list[AttributeValue]:
    """Backward-compat wrapper: delegates to _apply_numeric_spec_from_text.

    POST-merge verbatim filler for 'Срок службы, лет' / 'Срок эксплуатации'.
    Kept as a thin wrapper so existing callers and tests continue to work.
    """
    return _apply_numeric_spec_from_text(merged, targets, context)


def _apply_numeric_spec_from_text(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
) -> list[AttributeValue]:
    """POST-merge verbatim filler for EMPTY numeric/free-text target attributes.

    Generalised form of the original "срок службы" extractor. Handles:
      - Срок службы / Срок эксплуатации (special dedicated regex — bidirectional)
      - ANY other numeric target whose name encodes a known unit (via _detect_attr_unit)
        AND whose keyword phrase is BOTH distinctive AND present next to the number
        in the fetched text.

    For each STILL-EMPTY numeric target:
      1. If _is_service_life_target → use the dedicated _SERVICE_LIFE_RE (backward compat).
      2. Otherwise: derive keyword from attr name, detect expected unit from name via
         _detect_attr_unit, build keyword+unit regex, search text pools.

    Cross-attribute bleed guard (critical): the attr's OWN keyword phrase must appear
    in the window around the number. "зарядка 2 часа" will NOT fill "время разговора"
    because "время разговора" must literally be present near any number it fills.

    Text pools searched (priority order):
      1. context.product_description
      2. Evidence strings of accumulated AttributeValues (web_search snippets, etc.)

    Source=DESCRIPTION, evidence="numspec:<keyword>:verbatim:<snippet>", conf=0.97.
    """
    filled_attr_ids: set[int] = {v.attribute_id for v in merged}

    # Candidates: empty numeric/free-text targets (no allowed_values constraint)
    candidate_targets = [
        t for t in targets
        if t.id not in filled_attr_ids and not t.allowed_values
    ]
    if not candidate_targets:
        return merged

    # Build text corpus: description first (highest authority), then evidence strings
    text_pools: list[str] = []
    if context.product_description:
        text_pools.append(context.product_description)
    for v in merged:
        ev = (v.evidence or "").strip()
        if ev and len(ev) >= 10:
            text_pools.append(ev)

    if not text_pools:
        return merged

    out = list(merged)
    for target in candidate_targets:
        # --- Special case: service-life uses dedicated bidirectional regex ---
        if _is_service_life_target(target.name):
            found_number: Optional[str] = None
            found_snippet: str = ""
            for text in text_pools:
                number = _extract_service_life_number(text)
                if number is not None:
                    found_number = number
                    m = _SERVICE_LIFE_RE.search(text)
                    if m:
                        start = max(0, m.start() - 10)
                        end = min(len(text), m.end() + 10)
                        found_snippet = text[start:end].strip()
                    break
            if found_number is None:
                continue
            evidence = _SERVICE_LIFE_EVIDENCE_PREFIX + found_snippet[:150]
            logger.info(
                "[Pipeline] numspec service-life: attr=%s '%s' ← '%s' (evidence=%r)",
                target.id, target.name, found_number, found_snippet[:60],
            )
            out.append(AttributeValue(
                attribute_id=target.id,
                value=found_number,
                confidence=_NUMSPEC_CONF,
                source=Source.DESCRIPTION,
                evidence=evidence,
            ))
            continue

        # --- General case: keyword + unit proximity ---
        unit = _detect_attr_unit(target.name)
        if unit is None:
            # No known unit encodable from this attr name → skip (conservative)
            continue

        keyword = _attr_keyword(target.name)
        if len(keyword) < 4:
            # Keyword too short → too many false positives → skip
            continue

        pattern = _build_numspec_regex(keyword, unit)
        if pattern is None:
            continue

        found_result: Optional[tuple[str, str]] = None
        for text in text_pools:
            result = _extract_numspec_number(text, pattern)
            if result is not None:
                found_result = result
                break

        if found_result is None:
            continue

        num_str, snippet = found_result
        evidence = f"{_NUMSPEC_EVIDENCE_PREFIX}{keyword}:verbatim:{snippet[:120]}"
        logger.info(
            "[Pipeline] numspec: attr=%s '%s' ← '%s' (keyword=%r, unit=%s, "
            "evidence=%r)",
            target.id, target.name, num_str, keyword, unit, snippet[:60],
        )
        out.append(AttributeValue(
            attribute_id=target.id,
            value=num_str,
            confidence=_NUMSPEC_CONF,
            source=Source.DESCRIPTION,
            evidence=evidence,
        ))

    return out




# ---------------------------------------------------------------------------
# Lever 3: "Гарантия" ↔ "Гарантийный срок" alias cross-fill.
#
# COMPATIBILITY VERIFIED (2026-06-12, ozon dictionary):
#   - "Гарантия"         (id=10400): type=String, allowed_values=[]  → free-text
#   - "Гарантийный срок" (id=4385):  type=String, allowed_values=[]  → free-text
#   - "Гарантийный срок" (id=8802):  type=String, allowed_values=[]  → free-text
# Both are String free-text with no enum constraint — genuinely compatible aliases.
# "Гарантия на товар, мес." (id=4164, Integer) is NOT an alias — different format.
#
# Cross-fill rule (general, no per-category hardcode):
#   If ONE alias is filled and its NAME-EQUIVALENT counterpart is empty and
#   BOTH have no allowed_values (same format: free-text) → copy the value verbatim.
#
# Alias map is a small general dict of equivalent name pairs (case-insensitive).
# ---------------------------------------------------------------------------
_WARRANTY_ALIAS_EVIDENCE = "warranty_alias_cross_fill:verbatim"
_WARRANTY_ALIAS_CONF = 0.97  # verbatim copy, same confidence as service-life

# Canonical alias groups: each group = frozenset of lowercased name fragments
# that identify mutually equivalent warranty-duration attrs.
# «Гарантия» alone: matches "гарантия" WITHOUT "товар"/"мес"/"лет" qualifiers
# (to exclude "Гарантия на товар, мес." which has different semantics/type).
_WARRANTY_ALIAS_GROUPS: list[frozenset[str]] = [
    frozenset({"гарантийный срок", "гарантия"}),
    frozenset({"срок гарантии", "гарантийный срок"}),
]


def _warranty_alias_key(name: str) -> Optional[frozenset[str]]:
    """Return the alias group a target name belongs to, or None.

    Matching is conservative:
      - "гарантийный срок" → matches (contains exact phrase)
      - "гарантия"         → matches ONLY when the name is EXACTLY "гарантия"
        or contains it WITHOUT extra qualifier words "товар", "мес", "лет".
        This prevents matching "Гарантия на товар, мес." (different type).
      - General: no per-category or per-attr-id hardcode; matches by name only.
    """
    low = name.lower().strip()
    for group in _WARRANTY_ALIAS_GROUPS:
        for fragment in group:
            if fragment in low:
                # Extra guard for bare "гарантия": reject if qualifier words present
                if fragment == "гарантия":
                    has_qualifier = any(
                        q in low for q in ("товар", "мес", "лет", "год", "внутренний", "дополнительн")
                    )
                    if has_qualifier:
                        continue
                return group
    return None


def _is_free_text_target(target: TargetAttribute) -> bool:
    """True if the target is a free-text field (no allowed_values constraint)."""
    return not target.allowed_values


def _apply_warranty_alias_cross_fill(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
) -> list[AttributeValue]:
    """POST-merge alias cross-fill for Гарантия ↔ Гарантийный срок.

    Both "Гарантия" and "Гарантийный срок" are String free-text fields in the
    Ozon dictionary (type=String, no allowed_values). They are genuine aliases —
    same semantics, same format, same unit (months/years as free text).

    Rule (general, verbatim-only):
      For each alias group where:
        (a) exactly ONE target in the group is currently FILLED with a value, AND
        (b) at least ONE other target in the same group is EMPTY, AND
        (c) BOTH the filled and empty targets have no allowed_values (free-text),
      → copy the filled value to the empty target verbatim.

    Safety (fail-closed — «пусто честнее мусора»):
      - Only fires for free-text targets (no enum constraint mismatch possible).
      - Never copies when the value could be an enum-incompatible value.
      - If MULTIPLE targets in the group are filled → ambiguous, no cross-fill.
      - Source=DESCRIPTION (verbatim from product data, no inference).
      - Evidence tag: warranty_alias_cross_fill:verbatim.
    """
    # Index targets by alias group
    # For each group, find the (attr_id → target) mapping for members of that group
    targets_by_id: dict[int, TargetAttribute] = {t.id: t for t in targets}

    # Group targets into alias buckets
    alias_buckets: dict[int, list[TargetAttribute]] = {}  # bucket_idx → [targets]
    target_bucket: dict[int, int] = {}  # attr_id → bucket_idx
    for t in targets:
        group = _warranty_alias_key(t.name)
        if group is None:
            continue
        # Use id(group) as bucket key (each frozenset object is unique per group)
        bucket_idx = id(group)
        alias_buckets.setdefault(bucket_idx, []).append(t)
        target_bucket[t.id] = bucket_idx

    if not alias_buckets:
        return merged

    # Build current fill state by attr_id (best confidence wins)
    filled: dict[int, AttributeValue] = {}
    for v in merged:
        prev = filled.get(v.attribute_id)
        if prev is None or v.confidence > prev.confidence:
            filled[v.attribute_id] = v

    out = list(merged)
    for bucket_idx, bucket_targets in alias_buckets.items():
        # Partition into filled + empty, restricting to free-text only
        filled_in_bucket: list[tuple[TargetAttribute, AttributeValue]] = []
        empty_in_bucket: list[TargetAttribute] = []
        for t in bucket_targets:
            if not _is_free_text_target(t):
                continue  # enum target — skip (type mismatch possible)
            if t.id in filled:
                filled_in_bucket.append((t, filled[t.id]))
            else:
                empty_in_bucket.append(t)

        # Safety: only when EXACTLY ONE filled in bucket (no ambiguity)
        if len(filled_in_bucket) != 1 or not empty_in_bucket:
            continue

        source_target, source_av = filled_in_bucket[0]
        value_to_copy = source_av.value

        for dest_target in empty_in_bucket:
            if not _is_free_text_target(dest_target):
                continue  # already guarded above, belt-and-suspenders
            logger.info(
                "[Pipeline] warranty-alias cross-fill: attr=%s '%s' ← attr=%s '%s' "
                "value=%r (verbatim copy)",
                dest_target.id, dest_target.name,
                source_target.id, source_target.name,
                value_to_copy,
            )
            out.append(AttributeValue(
                attribute_id=dest_target.id,
                value=value_to_copy,
                confidence=_WARRANTY_ALIAS_CONF,
                source=Source.DESCRIPTION,
                evidence=_WARRANTY_ALIAS_EVIDENCE,
            ))
    return out


# ---------------------------------------------------------------------------
# Lever 3a: Boolean (Да/Нет) — stated-only verbatim fill.
#
# Fills STILL-EMPTY boolean target attributes with "Да" ONLY when the feature is
# EXPLICITLY stated in the fetched text. Never fills "Да" by default or from absence.
# Never fills "Нет" (absence of mention ≠ "Нет" — «пусто честнее мусора»).
#
# For each empty boolean target (allowed_values contains "Да"/"Нет"):
#   1. Derive a distinctive keyword phrase from the attr name.
#   2. Search product_description + evidence strings for the keyword.
#   3. If found → fill "Да". If not found → leave empty.
#
# Keyword matching is CONSERVATIVE: requires the attr's own keyword phrase to appear
# verbatim in the text. No category/product-type defaulting.
#
# Source=DESCRIPTION, evidence="bool:stated:<keyword>", conf=0.97.
# ---------------------------------------------------------------------------
_BOOL_STATED_CONF = 0.97
_BOOL_STATED_EVIDENCE_PREFIX = "bool:stated:"

# Per-keyword overrides for boolean attrs: maps distinctive keyword(s) from attr name
# to the exact text phrase(s) that indicate "Да". Each entry is a list of alternatives
# (any one match → "Да"). General: derived from attr name if no override.
# Order matters: more-specific overrides first. All lowercase.
_BOOL_KEYWORD_OVERRIDES: list[tuple[str, list[str]]] = [
    # Управление со смартфона / через приложение
    ("управление со смартфон", ["управление со смартфон", "через приложени", "мобильн.*приложени"]),
    ("управление через приложени", ["управление через приложени", "через приложени", "мобильн.*приложени"]),
    ("мобильн.*приложени", ["мобильн.*приложени", "через приложени"]),
    # Серийный номер
    ("серийн.*номер", ["серийн.*номер", "serial number", "серийный"]),
    # Bluetooth
    ("bluetooth", ["bluetooth", "блютус"]),
    # Wi-Fi
    ("wi-fi", ["wi-fi", "wifi", "вай-фай"]),
    # NFC
    ("nfc", ["nfc"]),
    # USB
    ("usb", ["usb"]),
    # GPS
    ("gps", ["gps", "глонасс"]),
    # Подсветка
    ("подсветк", ["подсветк"]),
    # Таймер
    ("таймер", ["таймер"]),
]


def _bool_stated_keywords(attr_name: str) -> list[re.Pattern[str]]:
    """Return regex patterns that indicate "Да" for a boolean attr.

    Checks _BOOL_KEYWORD_OVERRIDES first; falls back to using the attr name
    (lowercased, stripped of noise) as the keyword.
    """
    low = attr_name.lower().strip()
    # Check overrides
    for key, phrases in _BOOL_KEYWORD_OVERRIDES:
        if re.search(key, low):
            return [re.compile(p, re.IGNORECASE) for p in phrases]
    # Fallback: use attr name itself as keyword (stripped of trailing type hints)
    keyword = _attr_keyword(attr_name)
    if len(keyword) < 4:
        return []
    return [re.compile(re.escape(keyword), re.IGNORECASE)]


def _is_bool_target(target: TargetAttribute) -> bool:
    """True if the target is a boolean Да/Нет field."""
    if not target.allowed_values:
        return False
    lower_vals = {v.lower() for v in target.allowed_values}
    return "да" in lower_vals and "нет" in lower_vals


def _apply_boolean_stated_from_text(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
) -> list[AttributeValue]:
    """POST-merge verbatim filler for EMPTY boolean (Да/Нет) target attributes.

    Fills "Да" ONLY when the feature is EXPLICITLY stated in the fetched text.
    NEVER defaults to "Да" based on category, product type, or absence of mention.
    Never fills "Нет" (we cannot infer absence).

    For each empty boolean target:
      1. Derive keyword from attr name (or use override patterns).
      2. Search product_description + evidence strings.
      3. Keyword found → "Да". Not found → leave empty.

    Guards (fail-closed — «пусто честнее мусора»):
      - Only fires for EMPTY targets (never overwrites).
      - Only fires when keyword is EXPLICITLY present in text.
      - Keyword patterns come from attr name — NO per-category defaults.
      - Source=DESCRIPTION, conf=0.97.
    """
    filled_attr_ids: set[int] = {v.attribute_id for v in merged}
    bool_targets = [
        t for t in targets
        if _is_bool_target(t) and t.id not in filled_attr_ids
    ]
    if not bool_targets:
        return merged

    # Build text corpus
    text_pools: list[str] = []
    if context.product_description:
        text_pools.append(context.product_description)
    for v in merged:
        ev = (v.evidence or "").strip()
        if ev and len(ev) >= 10:
            text_pools.append(ev)

    if not text_pools:
        return merged

    combined_text = "\n".join(text_pools)

    out = list(merged)
    for target in bool_targets:
        patterns = _bool_stated_keywords(target.name)
        if not patterns:
            continue

        # "Да" only when ANY pattern matches EXPLICITLY in text
        matched_keyword: Optional[str] = None
        for pat in patterns:
            if pat.search(combined_text):
                matched_keyword = pat.pattern
                break

        if matched_keyword is None:
            continue  # not stated → leave empty (пусто честнее мусора)

        evidence = f"{_BOOL_STATED_EVIDENCE_PREFIX}{_attr_keyword(target.name)}"
        logger.info(
            "[Pipeline] bool-stated: attr=%s '%s' ← 'Да' (keyword=%r stated in text)",
            target.id, target.name, matched_keyword[:60],
        )
        out.append(AttributeValue(
            attribute_id=target.id,
            value="Да",
            confidence=_BOOL_STATED_CONF,
            source=Source.DESCRIPTION,
            evidence=evidence,
        ))
    return out


# ---------------------------------------------------------------------------
# Lever 3b: "Комплектация" — verbatim list from product description.
#
# Extracts the "в комплекте: …" / "комплектация: …" section from the product
# description verbatim. Never defaults or infers.
#
# Source=DESCRIPTION, evidence="комплектация:verbatim", conf=0.97.
# ---------------------------------------------------------------------------
_KOMPLEKTATSIYA_CONF = 0.97
_KOMPLEKTATSIYA_EVIDENCE = "комплектация:verbatim"

# Regex to find "комплектация:" or "в комплекте:" section and capture list content.
# Captures everything up to: blank line, next section header (word + colon + space),
# or end of string. Max 400 chars captured.
_KOMPLEKTATSIYA_RE = re.compile(
    r"(?:комплектаци[яи]|в\s+комплект[еи])\s*:\s*([^\n]{3,400})",
    re.IGNORECASE,
)

# Attr name keywords that identify the "Комплектация" field
_KOMPLEKTATSIYA_ATTR_KEYWORDS: tuple[str, ...] = (
    "комплектаци",
    "в комплект",
    "состав комплект",
)


def _is_komplektatsiya_target(name: str) -> bool:
    """True if the attribute is a 'Комплектация' (kit contents) free-text field."""
    low = name.lower().strip()
    return any(kw in low for kw in _KOMPLEKTATSIYA_ATTR_KEYWORDS)


def _apply_komplektatsiya_from_text(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
) -> list[AttributeValue]:
    """POST-merge verbatim filler for 'Комплектация' (kit contents) from description.

    Extracts "комплектация: …" / "в комплекте: …" section verbatim from the
    product description. Never infers contents or defaults.

    Guards (fail-closed — «пусто честнее мусора»):
      - Only fires for EMPTY targets whose name indicates 'комплектация'.
      - Text section MUST contain the phrase 'комплектация:' or 'в комплекте:'.
      - Verbatim-only: captured text directly from description.
      - Source=DESCRIPTION, conf=0.97.
    """
    filled_attr_ids: set[int] = {v.attribute_id for v in merged}
    komplek_targets = [
        t for t in targets
        if _is_komplektatsiya_target(t.name) and not t.allowed_values
        and t.id not in filled_attr_ids
    ]
    if not komplek_targets:
        return merged

    # Only search product_description (description section — not evidence snippets)
    desc = (context.product_description or "").strip()
    if not desc:
        return merged

    m = _KOMPLEKTATSIYA_RE.search(desc)
    if m is None:
        return merged

    value = m.group(1).strip()
    if not value:
        return merged

    out = list(merged)
    for target in komplek_targets:
        logger.info(
            "[Pipeline] комплектация-verbatim: attr=%s '%s' ← '%s'",
            target.id, target.name, value[:80],
        )
        out.append(AttributeValue(
            attribute_id=target.id,
            value=value,
            confidence=_KOMPLEKTATSIYA_CONF,
            source=Source.DESCRIPTION,
            evidence=_KOMPLEKTATSIYA_EVIDENCE,
        ))
    return out


def _merge_winner(
    challenger: AttributeValue, incumbent: AttributeValue
) -> AttributeValue:
    """Выбирает победителя для одного attribute_id между двумя кандидатами.

    Спец-правило ТОЛЬКО для пары карточка-vs-инференс: карточный источник
    удерживает атрибут, если card_conf >= other_conf - _CARD_PROTECTION_BAND.
    Для всех прочих пар — обычное правило: highest-conf, tie-break по SOURCE_PRIORITY.
    Поведение симметрично (order-independent).
    """
    # Card-protection band: только карточка-vs-инференс
    card, inference = None, None
    if challenger.source in _AUTHORITATIVE_OVERRIDE_SOURCES and incumbent.source in _INFERENCE_SOURCES:
        card, inference = challenger, incumbent
    elif incumbent.source in _AUTHORITATIVE_OVERRIDE_SOURCES and challenger.source in _INFERENCE_SOURCES:
        card, inference = incumbent, challenger
    if card is not None:
        if card.confidence >= inference.confidence - _CARD_PROTECTION_BAND:
            return card
        return inference

    # Обычное правило для всех прочих пар (без изменений).
    if challenger.confidence > incumbent.confidence:
        return challenger
    if (
        challenger.confidence == incumbent.confidence
        and SOURCE_PRIORITY[challenger.source] > SOURCE_PRIORITY[incumbent.source]
    ):
        return challenger
    return incumbent


def _is_collection_value(v: AttributeValue) -> bool:
    """True если значение надо мерджить как коллекцию (union), а не winner-takes-all."""
    return bool(v.is_collection) or isinstance(v.value, list)


# ---------------------------------------------------------------------------
# Placeholder values that must NEVER fill an enum field (they are non-values).
# Checked case-insensitively, stripped of surrounding whitespace.
# ---------------------------------------------------------------------------
_PLACEHOLDER_VALUES: frozenset[str] = frozenset({
    "нет",
    "-",
    "—",
    "none",
    "n/a",
    "без",
    "не указано",
    "нет данных",
})


_ENUM_GUARD_EXEMPT_MARKERS = ("тн вэд", "тнвэд", "еаэс", "бренд")
# Порог для членов МНОГОЗНАЧНОГО словарного поля. Используем token_set_ratio,
# а НЕ WRatio: члены — многословные фразы («Автоматическое центрирование тостов»),
# где WRatio раздувает score на общем хвосте («…тостов») и пропускает мусор
# («подогрев готовых тостов»→86), одновременно роняя легит-вариант
# («автоцентрирование тостов»→81<85). token_set_ratio награждает общие токены без
# учёта порядка/лишних слов: экстра-подъем=100, автоцентрирование=81 (легит),
# подогрев=52, выдвижной=39 — чистое разделение на пороге 80.
_ENUM_MEMBER_TOKENSET_THRESHOLD = 80


def _match_collection_member(
    raw: object,
    norm_options: "list[tuple[str, str]]",
) -> "Optional[str]":
    """Сматчить ОДИН член многозначного поля с каноном словаря или None.

    norm_options — [(normalized_allowed, canonical_allowed)]. Стратегии:
    1. Нормализованный exact (ё→е, латиница→кириллица — как словарь).
    2. rapidfuzz token_set_ratio ≥ порога (переставленные/усечённые фразы).
    Ниже порога → None («пусто честнее мусора», не форсим ближайший).
    """
    from app.services.enrichment.strategies.dictionaries.ozon_loader import _normalize_token

    if raw is None:
        return None
    key = str(raw).strip().lower()
    if not key:
        return None
    norm_key = _normalize_token(key)
    if not norm_key:
        return None
    for n, canon in norm_options:
        if n and n == norm_key:
            return canon
    try:
        from rapidfuzz import fuzz, process
        best = process.extractOne(
            norm_key, [n for n, _ in norm_options], scorer=fuzz.token_set_ratio
        )
        if best and best[1] >= _ENUM_MEMBER_TOKENSET_THRESHOLD:
            return norm_options[best[2]][1]
    except Exception as exc:  # pragma: no cover — rapidfuzz всегда есть
        logger.warning("enum-member rapidfuzz failed: %s", exc)
    return None


def _apply_enum_membership_guard(
    values: "list[AttributeValue]",
    targets_by_id: "dict[int, TargetAttribute]",
) -> "list[AttributeValue]":
    """Дропнуть/каноникализировать enum-значения, не принадлежащие allowed_values.

    Работает ДО resolve_value_ids: ленивый Ozon-search присвоил бы вне-словарному
    мусору («выдвижной лоток») какой-нибудь id, и он бы протёк. LLM-гейт
    _enforce_allowed_values действует только на LLM-слое — IceCat/marketplace/donor
    его ОБХОДЯТ, поэтому фильтруем здесь, для любого источника.

    - Скаляр: маппим через _map_enum_value (exact→норм→fuzzy-порог); None → дроп.
    - Список (Ⓜ️ многозначное поле): каноникализируем КАЖДЫЙ член, дропаем
      вне-словарные, дедуп с сохранением порядка; пустой результат → дроп поля.
    Исключения: ТН ВЭД (free-text код, режет TnvedSource), бренд (free-text),
    поля без allowed_values, и значения с evidence-сентинелом tnved_resolver:.
    «пусто честнее мусора».
    """
    from app.services.enrichment.strategies.ozon_strategy import _map_enum_value

    out: list[AttributeValue] = []
    for v in values:
        tgt = targets_by_id.get(v.attribute_id)
        if (
            tgt is None
            or not tgt.allowed_values
            or (v.evidence or "").startswith("tnved_resolver:")
            or any(m in (tgt.name or "").lower() for m in _ENUM_GUARD_EXEMPT_MARKERS)
        ):
            out.append(v)
            continue
        canon_map = {str(a).strip().lower(): str(a) for a in tgt.allowed_values}
        allowed_list = [str(a) for a in tgt.allowed_values]
        if isinstance(v.value, list):
            from app.services.enrichment.strategies.dictionaries.ozon_loader import _normalize_token
            norm_options = [(_normalize_token(str(a).strip().lower()), str(a))
                            for a in tgt.allowed_values]
            kept: list = []
            seen: set = set()
            for el in v.value:
                c = _match_collection_member(el, norm_options)
                if c is None:
                    logger.info(
                        "[Pipeline] enum-membership дроп (член списка): attr=%s "
                        "элемент=%r source=%s — вне allowed_values (%d значений)",
                        v.attribute_id, el, v.source.value, len(allowed_list),
                    )
                    continue
                if c.lower() in seen:
                    continue
                seen.add(c.lower())
                kept.append(c)
            if not kept:
                logger.info(
                    "[Pipeline] enum-membership дроп: attr=%s — ни один член %r "
                    "не в словаре (%d значений)",
                    v.attribute_id, v.value, len(allowed_list),
                )
                continue
            if kept != list(v.value):
                # состав/форма изменились → сбрасываем value_ids под пере-резолв ниже
                v = v.model_copy(update={"value": kept, "value_ids": None})
            out.append(v)
            continue
        canon = _map_enum_value(v.value, canon_map, allowed_list)
        if canon is None:
            logger.info(
                "[Pipeline] enum-membership дроп: attr=%s value=%r source=%s — "
                "не принадлежит allowed_values (%d значений)",
                v.attribute_id, v.value, v.source.value, len(allowed_list),
            )
            continue
        if canon != v.value:
            v = v.model_copy(update={"value": canon})  # каноническая форма словаря
        out.append(v)
    return out


def _is_placeholder_value(raw: object) -> bool:
    """True если значение — заведомый плейсхолдер/пустышка.

    Используется ДО/ВО ВРЕМЯ резолюции: такие значения не несут информации
    и никогда не должны заполнять enum-поле. Проверяется case-insensitive,
    strip whitespace. Список хранится в _PLACEHOLDER_VALUES (константа модуля).
    """
    return str(raw).strip().lower() in _PLACEHOLDER_VALUES


# ТН ВЭД (EAEU customs code) attribute-name markers. The customs-code field is
# free-text (any valid 10-digit code) even though Ozon ships a small sample
# allowed_values list — so it must be exempted from the enum drop-guards, which
# otherwise silently dropped every TnvedSource-validated code (value_id=None).
_TNVED_ATTR_NAME_MARKERS = ("тн вэд", "тнвэд", "еаэс")

# Open-vocabulary required fields (thousands of values / free-text) — never
# LLM-pick from the live list (hallucination risk). «Бренд» gets a deterministic
# "Нет бренда" fallback when the brand extractors found nothing (genuinely
# brandless goods); the others are left to their dedicated free-text levers.
_OZON_OPEN_VOCAB_REQUIRED = frozenset({
    "бренд", "производитель", "модель", "название модели", "партномер", "артикул",
})


def _drop_unresolved_optional_enums(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
) -> list[AttributeValue]:
    """Дроп OPTIONAL enum-значений, не резолвнувшихся в словарный value_id.

    Генеральный conservative finalize-cleanup. Запускается ПОСЛЕ полной резолюции
    value_id (детерминированный resolve_value_ids + llm_resolve_tail). LLM иногда
    кладёт в специфичный enum-таргет generic-«Да» или значение чужого атрибута
    (Монитор «Покрытие экрана»=«1,07 миллиардов цветов», «Крепление VESA»=«Да»).
    Matcher честно отказывает (value_id=None), но RAW-текст остаётся как «filled» →
    fake-fill, который Ozon отклонит на загрузке (enum-поле требует value_id).
    Лучше пусто чем враньё: дропаем такие значения, поднимая качество value_id.

    СТРОГИЙ scope (чтобы не навредить):
    - ТОЛЬКО таргеты С allowed_values (enum). Free-text — без value_id by design,
      их НИКОГДА не трогаем.
    - ТОЛЬКО OPTIONAL (is_required == False). Required-enum покрывается отдельным
      гардом _drop_unresolved_required_enums — вызывается в _finalize_async ПОСЛЕ
      llm_resolve_tail, когда оба optional и required получают одинаковый дроп.
    - Дроп ТОЛЬКО когда value_id None/пуст ПОСЛЕ всей резолюции.
    - is_collection: дропаем только нерезолвнутые элементы, резолвнутые оставляем.
      Если ВСЕ элементы нерезолвнуты → дроп всего поля.
    """
    return _drop_unresolved_enums(merged, targets, required_only=False)


def _drop_unresolved_required_enums(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
) -> list[AttributeValue]:
    """Дроп REQUIRED enum-значений, не резолвнувшихся в словарный value_id.

    Симметричный гард для обязательных enum-полей. REQUIRED enum с value_id=None
    — мусор, который Ozon отклонит так же как optional. Пустое поле честнее:
    Ozon сообщает об отсутствующем обязательном поле, а не отклоняет всю карточку
    из-за неверного enum-значения. Применяется ПОСЛЕ optional-гарда (обе функции
    вызываются из _finalize_async через _drop_unresolved_enum_garbage).

    Типичные случаи:
      - «Тип» (garment type) ← LLM/WbCard кладёт «без карманов»/«открытые»
        (feature value, не garment type) → matcher отказывает, value_id=None.
      - «Тип» ← «нет» (плейсхолдер принят как значение) → value_id=None.

    Scope:
    - ТОЛЬКО таргеты С allowed_values (enum).
    - ТОЛЬКО REQUIRED (is_required == True).
    - Дроп ТОЛЬКО когда value_id None ПОСЛЕ ВСЕЙ резолюции (вкл. llm_resolve_tail).
    - НЕ трогает значения с корректным value_id (brand-from-name, gender-fill и т.п.
      уже получили value_id до этого этапа → они в безопасности).
    """
    return _drop_unresolved_enums(merged, targets, required_only=True)


def _is_color_target(target: TargetAttribute) -> bool:
    """True если таргет — основной атрибут «Цвет» (НЕ «Название цвета» free-text).

    «Цвет товара» — словарный enum конкретной расцветки SKU; «Название цвета» —
    свободное маркетинговое описание (не трогаем). Детект по semantic_type=="color"
    ИЛИ имени с «цвет» без «название».
    """
    if (target.semantic_type or "").lower() == "color":
        return True
    low = (target.name or "").lower()
    return "цвет" in low and "название" not in low


def _is_multivalue_color_value(value, value_ids) -> bool:
    """True если значение цвета — палитра/мульти-цвет (НЕ один цвет SKU).

    Формы палитры от разных источников:
      • список ≥2 (llm_knowledge: «белый/чёрный/серый…»);
      • строка с ≥2 частями через `;`/`,`/`/` (WB/marketplace _map_characteristics
        join-ит мультизначение: «коричневый; темно-коричневый; белый; зеленый…» —
        реальный кейс WbCard для «Nike Air Max 90 чёрные», донор другой расцветки);
      • value_ids ≥2 (ozon_card api-resolve: скаляр value + 10/40 dict-id).
    Один цвет SKU ни одной из этих форм не имеет.
    """
    if isinstance(value, list) and len(value) >= 2:
        return True
    if isinstance(value_ids, list) and len(value_ids) >= 2:
        return True
    if isinstance(value, str):
        parts = [p for p in re.split(r"[;,/]", value) if p.strip()]
        if len(parts) >= 2:
            return True
    return False


def _drop_multivalue_color_premerge(
    all_values: list[AttributeValue],
    targets: list[TargetAttribute],
) -> list[AttributeValue]:
    """ДО merge выкинуть палитру-цвета (мульти), чтобы ОДИНОЧНЫЙ grounded-цвет
    (из названия товара / точного донора) выиграл merge и заполнился.

    Без этого высоко-conf палитра (WbCard conf 0.93) вытесняет одиночный цвет в
    merge, а post-merge гард потом дропает палитру → теряем ОБА (пусто вместо
    верного цвета из названия). Дроп палитры до merge решает: палитра уходит,
    одиночный цвет остаётся единственным кандидатом и наполняет таргет.
    """
    color_ids = {t.id for t in targets if _is_color_target(t)}
    if not color_ids:
        return all_values
    out: list[AttributeValue] = []
    for v in all_values:
        if v.attribute_id in color_ids and _is_multivalue_color_value(v.value, v.value_ids):
            logger.info(
                "[Pipeline] pre-merge drop-multivalue-color: attr=%s src=%s value=%r "
                "value_ids=%s — палитра донора, не цвет этого SKU",
                v.attribute_id, v.source, v.value, v.value_ids,
            )
            continue
        out.append(v)
    return out


# Color-identity guard (ALLOWLIST): «Цвет товара» — per-SKU расцветка продавца. НИ ОДИН
# источник, кроме собственного текста товара, не знает колорвей ЭТОГО SKU: доноры
# (ozon_card/wb_card) дают цвет ПОХОЖЕГО листинга (PUMA Flyer Runner без цвета в имени →
# ozon_card «белый»); web_search/vision/llm_knowledge/competitor_rag — гадают по вебу/фото.
# eg_importer: «гадать колорвей нельзя, пусто честнее мусора». Поэтому для цвета оставляем
# ТОЛЬКО source=DESCRIPTION — это и color-from-name (добавляется post-merge), и verbatim-
# цвет из описания товара. Всё остальное (доноры/веб/vision/llm) дропается ДО merge.
_COLOR_ALLOWED_SOURCES = {Source.DESCRIPTION}


def _apply_color_source_guard(
    all_values: list[AttributeValue],
    targets: list[TargetAttribute],
) -> list[AttributeValue]:
    """Гейт «Цвет товара» на per-SKU источники (ALLOWLIST=DESCRIPTION), дроп ДО merge.

    Цвет — per-SKU данные продавца; знают его только имя/описание самого товара.
    Доноры (ozon_card/wb_card) копируют цвет похожего листинга, web/vision/llm гадают
    колорвей — всё это не цвет ЭТОГО SKU → дроп. Остаётся source=DESCRIPTION: color-from-
    name (добавляется ПОСЛЕ merge, гард его не видит → не трогает) + verbatim из описания.
    Не-цвет таргеты — без изменений.
    """
    color_ids = {t.id for t in targets if _is_color_target(t)}
    if not color_ids:
        return all_values
    out: list[AttributeValue] = []
    for v in all_values:
        if v.attribute_id in color_ids and v.source not in _COLOR_ALLOWED_SOURCES:
            logger.info(
                "[Pipeline] color-guard: дроп цвета attr=%s='%s' (source=%s) — не per-SKU "
                "(колорвей знают только имя/описание товара)",
                v.attribute_id, v.value, getattr(v.source, "value", v.source),
            )
            continue
        out.append(v)
    return out


def _drop_ungrounded_color_guess(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
) -> list[AttributeValue]:
    """Дроп МУЛЬТИ-цвет «Цвет товара» (СПИСОК ≥2) от ЛЮБОГО источника → no_data.

    «Цвет товара» (Ozon) = ОСНОВНОЙ/доминирующий цвет ОДНОГО SKU. Мульти-цвет
    коллекция — это НЕ цвет этого товара, а:
      • llm_knowledge — галлюцинированный разброс ходовых цветов («Nike Air Max 90»
        → белый/чёрный/серый/красный/синий);
      • ozon_card — палитра ВСЕХ расцветок модели от донор-карточки (мульти-
        вариантный листинг: «Adidas Runfalcon» → черный/белый/синий/серый/красный;
        «PUMA» → 39 цветов). Выровненная по value↔value_ids, поэтому reconcile её
        не ловит, но к ЭТОМУ SKU она не привязана.
    Из названия конкретную расцветку не вывести → честнее no_data, чем залить пачку
    неверных цветов (eg_importer: «grounded-цвет для ЭТОГО товара либо no_data»).
    Одиночный (grounded primary) цвет НЕ трогаем.
    """
    color_ids = {t.id for t in targets if _is_color_target(t)}
    if not color_ids:
        return merged
    out: list[AttributeValue] = []
    for v in merged:
        # Мульти-цвет в ЛЮБОЙ форме: список value (≥2) ИЛИ скаляр value со списком
        # value_ids (≥2). Донор ozon_card отдаёт value="черный" (СКАЛЯР) + value_ids
        # на 10/40 цветов — основной цвет одного SKU столько id иметь не может.
        # Поэтому условие вешаем на ДЛИНУ value_ids, а не на форму value.
        if v.attribute_id in color_ids and _is_multivalue_color_value(v.value, v.value_ids):
            logger.info(
                "[Pipeline] drop-multivalue-color: дроп мульти-цвета attr=%s src=%s "
                "value=%r value_ids=%s — палитра/разброс, не цвет этого SKU",
                v.attribute_id, v.source, v.value, v.value_ids,
            )
            continue
        out.append(v)
    return out


def _reconcile_enum_value_ids(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
) -> list[AttributeValue]:
    """Инвариант: value_ids обязаны 1:1 соответствовать элементам value.

    Донор (ozon_card/competitor_rag мульти-вариантной карточки) иногда приклеивает
    ВСЮ палитру категории к одному цвет-значению: value='черный', а value_ids — 15
    реальных id РАЗНЫХ цветов (белый/серый/синий…). eg_importer зипует value↔ids и
    заливает неверные цвета. Аналогично скаляр НЕ должен нести список value_ids.

    При рассинхроне (len(value_ids) != числу элементов value, либо скаляр со
    списком >1 id) сбрасываем value_ids/value_id в None — последующий авторитетный
    resolve_value_ids пересоберёт их СТРОГО из value-текста (черный → [61574]).
    Выровненные коллекции (len совпадает) не трогаем.
    """
    out: list[AttributeValue] = []
    for v in merged:
        vids = v.value_ids if isinstance(v.value_ids, list) else None
        if vids is None:
            out.append(v)
            continue
        elems = v.value if isinstance(v.value, list) else [v.value]
        scalar_stray = (not v.is_collection) and len(vids) > 1
        if len(vids) != len(elems) or scalar_stray:
            logger.info(
                "[Pipeline] reconcile-enum-ids: attr=%s value=%r — рассинхрон "
                "(%d value_ids на %d значений), сброс под пере-резолв",
                v.attribute_id, v.value, len(vids), len(elems),
            )
            out.append(v.model_copy(update={"value_ids": None, "value_id": None}))
        else:
            out.append(v)
    return out


def _drop_unresolved_enums(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    *,
    required_only: bool,
) -> list[AttributeValue]:
    """Общая реализация дропа нерезолвнутых enum-значений.

    required_only=False → обрабатывает ТОЛЬКО optional (is_required=False).
    required_only=True  → обрабатывает ТОЛЬКО required (is_required=True).

    Не вызывайте напрямую — используйте публичные обёртки:
      _drop_unresolved_optional_enums / _drop_unresolved_required_enums.
    """
    targets_by_id = {t.id: t for t in targets}
    out: list[AttributeValue] = []
    for v in merged:
        target = targets_by_id.get(v.attribute_id)
        # Вне scope → пропускаем как есть: нет таргета, не enum, или
        # is_required не совпадает с режимом вызова.
        if target is None or not target.allowed_values:
            out.append(v)
            continue
        # ТН ВЭД exemption: the customs-code attr (22232) carries a sample
        # allowed_values list (~50 codes) but is effectively FREE-TEXT — any
        # valid 10-digit ЕАЭС code is acceptable, not only those samples. A
        # TnvedSource-validated code legitimately resolves to value_id=None, so
        # the enum drop-guard must NOT treat it as "unresolved garbage" (that
        # silently killed ТН ВЭД on ~all products). Garbage ТН ВЭД from other
        # sources is already removed earlier by the TNVED_SOURCE_FIX filter.
        if any(m in (target.name or "").lower() for m in _TNVED_ATTR_NAME_MARKERS):
            out.append(v)
            continue
        if required_only and not target.is_required:
            out.append(v)
            continue
        if not required_only and target.is_required:
            out.append(v)
            continue

        if _is_collection_value(v) and isinstance(v.value, list):
            ids = v.value_ids if isinstance(v.value_ids, list) else []
            n_resolved = len(ids)
            if n_resolved == 0:
                logger.info(
                    "[Pipeline] drop-unresolved-enum: дроп %s enum-коллекции "
                    "attr=%s value=%r — все элементы без value_id",
                    "required" if required_only else "optional",
                    v.attribute_id, v.value,
                )
                continue  # все элементы нерезолвнуты → дроп поля
            if n_resolved < len(v.value):
                new_value = v.value[:n_resolved]
                logger.info(
                    "[Pipeline] drop-unresolved-enum: частичный дроп %s "
                    "enum-коллекции attr=%s оставлено %d/%d",
                    "required" if required_only else "optional",
                    v.attribute_id, n_resolved, len(v.value),
                )
                out.append(v.model_copy(update={"value": new_value}))
            else:
                out.append(v)
        else:
            if v.value_id is None:
                logger.info(
                    "[Pipeline] drop-unresolved-enum: дроп %s enum attr=%s "
                    "value=%r — нет словарного value_id после резолюции",
                    "required" if required_only else "optional",
                    v.attribute_id, v.value,
                )
                continue
            out.append(v)
    return out


def _norm_elements(value) -> list[str]:
    """Нормализованные (strip+lower) элементы значения.

    Скаляр → один элемент; список → поэлементно. Используется для consensus-подсчёта
    и поэлементного дедупа коллекций (вместо str(list) по всей строке).
    """
    if isinstance(value, list):
        out: list[str] = []
        for el in value:
            s = str(el).strip().lower()
            if s:
                out.append(s)
        return out
    s = str(value).strip().lower()
    return [s] if s else []


def _collection_card_protected(
    a: AttributeValue, b: AttributeValue
) -> Optional[AttributeValue]:
    """Card-protection для коллекций (аналог _merge_winner band, но для union).

    Если на attribute_id один кандидат — КАРТОЧНЫЙ источник (WB_CARD/OZON_CARD),
    а другой — ИНФЕРЕНС (web_search/llm_knowledge) с confidence НИЖЕ карточной более
    чем на _CARD_PROTECTION_BAND, то инференс-мусор НЕ подмешиваем в union: возвращаем
    карточный кандидат как есть.

    Возвращает:
      - карточный AttributeValue, если инференс отсекается по band;
      - None, если защита неприменима (нет карточно-vs-инференс пары, либо инференс
        в пределах band) → обычный union выполняется выше по стеку.

    Сохраняет пользу union: card+card, card+уверенный-инференс (в пределах band),
    чистый инференс (карточки нет) — None → нормальный union.
    """
    card, inference = None, None
    if a.source in _AUTHORITATIVE_OVERRIDE_SOURCES and b.source in _INFERENCE_SOURCES:
        card, inference = a, b
    elif b.source in _AUTHORITATIVE_OVERRIDE_SOURCES and a.source in _INFERENCE_SOURCES:
        card, inference = b, a
    if card is None:
        return None
    # Инференс отсекается ТОЛЬКО если его conf заметно ниже карточной.
    if card.confidence >= inference.confidence + _CARD_PROTECTION_BAND:
        return card
    return None


def _merge_collection(
    a: AttributeValue, b: AttributeValue
) -> AttributeValue:
    """Объединяет два коллекционных кандидата на один attribute_id.

    Card-protection: если один источник карточный (WB/Ozon), а другой —
    низко-confidence инференс (web_search/llm_knowledge, conf ниже карточной более
    чем на band), инференс НЕ подмешивается (защита от enum-галлюцинаций вроде
    Материал=Бязь от web_search поверх карточного значения). Иначе — обычный UNION.

    UNION дедуплицированных (регистронезависимо) элементов обоих источников.
    Порядок: первое вхождение сохраняется. value_ids объединяются параллельно
    значениям (best-effort: если оба источника несут ids — мерджим, иначе сбрасываем,
    чтобы их корректно дорезолвил resolve_value_ids в _finalize). confidence = max.
    """
    protected = _collection_card_protected(a, b)
    if protected is not None:
        return protected

    base = a if a.confidence >= b.confidence else b

    merged_values: list = []
    merged_ids: list = []
    seen: set[str] = set()
    have_ids = True  # ids валидны только если ОБА источника дали ids на все элементы

    for src in (a, b):
        vals = src.value if isinstance(src.value, list) else [src.value]
        ids = src.value_ids if isinstance(src.value_ids, list) else None
        if ids is None or len(ids) != len(vals):
            have_ids = False
        for i, el in enumerate(vals):
            norm = str(el).strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            merged_values.append(el)
            merged_ids.append(ids[i] if (ids is not None and i < len(ids)) else None)

    if not have_ids or any(x is None for x in merged_ids):
        merged_ids = None  # дорезолвит resolve_value_ids в _finalize

    return base.model_copy(update={
        "value": merged_values,
        "value_ids": merged_ids,
        "value_id": None,
        "is_collection": True,
        "confidence": max(a.confidence, b.confidence),
    })


# Backfill allowed_values из живого словаря Ozon (list_values API) для enum-полей,
# у которых caller не прислал options. Бэг (eg_importer, тостер): «Количество
# отделений» — закрытый словарь {1,2,3,4}, но options не пришёл → enum-гейт нечем
# было активировать → verbatim «8» из описания протекало. Локального словаря Ozon
# у движка для многих категорий нет, но API отдаёт точный список. Тот же механизм,
# что у ТН ВЭД. После бэкфилла classify_target→enum и _enforce_allowed_values сам
# режет out-of-list значения.
_OZON_API_ENUM_OPTIONS = os.getenv("OZON_API_ENUM_OPTIONS_ENABLED", "1").lower() not in (
    "0", "false", "no",
)
# Кап на размер списка: закрытые словари малы (цвет ~57, страна ~250). Если API
# отдал ≥ капа — это усечённый/огромный справочник (бренд, ТН ВЭД), его НЕЛЬЗЯ
# использовать как allowlist (выкинули бы валидные значения вне первой страницы).
_ENUM_OPTIONS_CAP = 512
_ENUM_OPTIONS_CONCURRENCY = 6


def _is_enum_options_candidate(target: TargetAttribute) -> bool:
    """Стоит ли пытаться тянуть словарь из API для target с пустым allowed_values.

    Пропускаем заведомо свободные поля: ТН ВЭД (свой резолвер), url/model_name/
    dimensions и числовые-с-единицей (Вт/см/мм/г/м) — они не словарные.
    """
    if target.allowed_values:
        return False
    if "тн вэд" in target.name.lower():
        return False  # резолвится TnvedSource, не словарный allowlist
    if classify_target(target) in ("url", "model_name", "dimensions"):
        return False
    if (extract_unit(target.name) or "").strip():
        return False  # свободное числовое поле с единицей измерения
    return True


async def _backfill_allowed_values_from_api(
    targets: list[TargetAttribute],
    context: ExtractionContext,
) -> list[TargetAttribute]:
    """Заполнить allowed_values из Ozon list_values для enum-полей без options.

    Возвращает НОВЫЙ список targets (model_copy с обновлённым allowed_values там,
    где словарь найден). Безопасно при любом сбое: при отсутствии type_id/кред/сети
    возвращает targets как есть. Усечённые/огромные словари (≥ _ENUM_OPTIONS_CAP)
    НЕ применяются — иначе резали бы валидные значения вне первой страницы.
    """
    if not _OZON_API_ENUM_OPTIONS:
        return targets
    type_id = context.ozon_type_id
    if type_id is None or not (os.getenv("OZON_CLIENT_ID") and os.getenv("OZON_API_KEY")):
        return targets
    candidates = [t for t in targets if _is_enum_options_candidate(t)]
    if not candidates:
        return targets

    from app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup import (
        list_values, resolve_description_category_id,
    )

    # category_id из шаблона caller'а бывает устаревшим → values-API «not found».
    # Берём живой description_category_id (родитель type_id) из дерева Ozon; если
    # дерево недоступно/тип не найден — фоллбэк на присланный category_id.
    _live_dcid = await resolve_description_category_id(type_id)
    eff_cat = _live_dcid or context.category_id
    if _live_dcid:
        context.resolved_category_id = _live_dcid

    sem = asyncio.Semaphore(_ENUM_OPTIONS_CONCURRENCY)

    async def _fetch(t: TargetAttribute) -> tuple[int, Optional[list[str]]]:
        async with sem:
            try:
                vals = await list_values(
                    eff_cat, type_id, t.id, max_values=_ENUM_OPTIONS_CAP,
                )
            except Exception as exc:  # pragma: no cover — сеть не должна ронять стейдж
                logger.warning("[Pipeline] list_values options-backfill ошибка attr=%s: %s", t.id, exc)
                return t.id, None
        if not vals or len(vals) >= _ENUM_OPTIONS_CAP:
            return t.id, None  # не словарь / усечён → не применяем как allowlist
        opts = [str(v.get("value", "")) for v in vals if v.get("value") not in (None, "")]
        return t.id, (opts or None)

    results = dict(await asyncio.gather(*(_fetch(t) for t in candidates)))

    out: list[TargetAttribute] = []
    filled = 0
    for t in targets:
        opts = results.get(t.id)
        if opts:
            out.append(t.model_copy(update={"allowed_values": opts}))
            filled += 1
        else:
            out.append(t)
    if filled:
        logger.info(
            "[Pipeline] options-backfill из Ozon API: %d enum-полей закрыто словарём "
            "(cat=%s type=%s)", filled, context.category_id, type_id,
        )
    return out


class PipelineOrchestrator:
    """Sequential cost-aware pipeline.

    Order: Description → IceCat → CompetitorRAG (fallback) → Classifier → (per-attr routing) → Knowledge → Vision → CostPredictor → WebSearch
    Early-exit at each stage if all targets filled with high confidence.
    IceCat вставлен сразу после Description: brand-verified спеки без LLM, самый авторитетный источник.
    CompetitorRAG запускается ТОЛЬКО как фолбэк когда IceCat вернул < 5 атрибутов (403/404 бренд).
    Если IceCat закрыл ≥ 5 атрибутов — RAG пропускается (экономия Qdrant I/O и compute).
    """

    def __init__(
        self,
        description_source: Optional[DescriptionSource] = None,
        knowledge_source: Optional[LlmKnowledgeSource] = None,
        vision_source: Optional[VisionSource] = None,
        websearch_source: Optional[WebSearchSource] = None,
        competitor_rag_source: Optional[CompetitorRagSource] = None,
        icecat_source: Optional[IceCatSource] = None,
        pdf_datasheet_source: Optional[PdfDatasheetSource] = None,
        ozon_card_source: Optional[OzonCardSource] = None,
        wb_card_source: Optional[WbCardSource] = None,
        yandex_market_source: Optional[YandexMarketSource] = None,
        ugc_source: Optional[UgcSource] = None,
        tnved_source: Optional[TnvedSource] = None,
        scrapfly_ozon_source: Optional[ScrapflyOzonSource] = None,
        barcode_source: Optional[BarcodeSource] = None,
        regard_source: Optional[RegardSource] = None,
        bestbuy_source: Optional[BestBuySource] = None,
        onliner_source: Optional[OnlinerSource] = None,
        books_source: Optional[BooksSource] = None,
        lamoda_scrapfly_source: Optional[LamodaScrapflySource] = None,
        gtin_resolver: Optional[bool] = None,
        classifier: Optional[LlmClassifier] = None,
        cost_predictor: Optional[CostPredictor] = None,
        strategy: Optional[MarketplaceStrategy] = None,
    ):
        self._strategy: MarketplaceStrategy = strategy or DefaultStrategy()
        _strat = self._strategy  # передаём стратегию в sources для build_response_model
        self._sources: dict[Source, AttributeSource] = {
            Source.DESCRIPTION: description_source or DescriptionSource(strategy=_strat),
            Source.LLM_KNOWLEDGE: knowledge_source or LlmKnowledgeSource(strategy=_strat),
            Source.VISION: vision_source or VisionSource(strategy=_strat),
            Source.WEB_SEARCH: websearch_source or WebSearchSource(strategy=_strat),
        }
        # CompetitorRagSource хранится отдельно (не в _sources) — у него особый порядок вызова
        # и нет смысла включать в classifier routing (он не LLM-based).
        # Передаём None → source не создаётся автоматически (нет готового индекса по умолчанию).
        self._competitor_rag: Optional[CompetitorRagSource] = competitor_rag_source
        # WbApparelRagSource — OPT-IN apparel-RAG из WB, ВЫКЛЮЧЕН по умолчанию.
        # Активен ТОЛЬКО когда WB_APPAREL_RAG_ENABLED=1 И явный competitor_rag не передан.
        # Шарит Source.COMPETITOR_RAG → ездит на том же stage/judge без новых веток.
        # Сам source дормантен пока wb_apparel_rag.qdrant не построен (gate внутри).
        if (
            self._competitor_rag is None
            and os.environ.get("WB_APPAREL_RAG_ENABLED", "0") == "1"
        ):
            try:
                self._competitor_rag = WbApparelRagSource()
                logger.info(
                    "[Pipeline] WB_APPAREL_RAG_ENABLED=1 → WbApparelRagSource registered "
                    "(dormant until wb_apparel_rag.qdrant exists)"
                )
            except Exception as e:  # noqa: BLE001 — не ломаем pipeline, если что-то не так
                logger.warning("[Pipeline] WbApparelRagSource init failed: %s", e)
                self._competitor_rag = None
        # IceCatSource хранится отдельно: brand-verified, HTTP, без LLM.
        # None → IceCat stage пропускается.
        self._icecat: Optional[IceCatSource] = icecat_source
        # PdfDatasheetSource: datasheet PDF от производителя через Gemini native PDF.
        # None → PDF stage пропускается.
        self._pdf_datasheet: Optional[PdfDatasheetSource] = pdf_datasheet_source
        # OzonCardSource: копия характеристик из живой Ozon-карточки через Scrappey.
        # None → Ozon card stage пропускается.
        self._ozon_card: Optional[OzonCardSource] = ozon_card_source
        # WbCardSource: копия характеристик из WB basket-API (бесплатно, без anti-bot).
        # None → WB card stage пропускается.
        self._wb_card: Optional[WbCardSource] = wb_card_source
        # YandexMarketSource: копия характеристик с live market.yandex.ru-карточки через Scrappey.
        # Запускается ПОСЛЕ WbCard/OzonCard (card=N fallback для apparel и generic-named products).
        # None → Yandex Market stage пропускается.
        self._yandex_market: Optional[YandexMarketSource] = yandex_market_source
        # UgcSource: отзывы и Q&A с Ozon/WB для compat/physical attrs.
        # None → UGC stage пропускается.
        self._ugc: Optional[UgcSource] = ugc_source
        # TnvedSource: per-category резолвер ТН ВЭД ЕАЭС с кэшем.
        # Создаётся ОДИН раз → кэш переживает все товары батча.
        # None → по умолчанию создаём инстанс (всегда нужен для Ozon).
        self._tnved: TnvedSource = tnved_source or TnvedSource()
        # GTINResolver: name→EAN via Serper (Stage 0.54, before IceCat).
        # Активен при GTIN_RESOLVE_ENABLED=1 ИЛИ явный gtin_resolver=True.
        # Флаг хранится как bool — сам resolver импортируется лениво в run().
        if gtin_resolver is not None:
            self._gtin_resolver_enabled: bool = gtin_resolver
        else:
            self._gtin_resolver_enabled = os.environ.get("GTIN_RESOLVE_ENABLED", "0") == "1"
        # BarcodeSource: verbatim EAN/barcode extractor — zero LLM, zero cost.
        # Always created (cheap singleton, no external deps).
        self._barcode: BarcodeSource = barcode_source or BarcodeSource()
        # RegardSource: verbatim electronics specs from regard.ru (plain httpx).
        # Fires Stage 0.57 (after IceCat, before CompetitorRAG) for electronics
        # categories where IceCat is weak (CPU/RAM/storage/GPU specs).
        # None → regard stage skipped.
        self._regard: Optional[RegardSource] = regard_source
        # BestBuySource: verbatim specs from Best Buy Developer API (free, official).
        # Fires Stage 0.58 — after Regard, before CompetitorRAG.
        # English values translated EN→RU before enum-match.
        # Graceful no-op when BESTBUY_API_KEY absent.
        # None → BestBuy stage skipped.
        self._bestbuy: Optional[BestBuySource] = bestbuy_source
        # OnlinerSource: verbatim specs from Onliner.by (public REST + JSON-LD).
        # Fires Stage 0.59 — after BestBuy, before CompetitorRAG.
        # Russian values, no translation needed.
        # None → Onliner stage skipped.
        self._onliner: Optional[OnlinerSource] = onliner_source
        # BooksSource: verbatim book metadata from Open Library + Google Books APIs.
        # Fires Stage 0.56 — after IceCat (0.55), before Regard (0.57).
        # ISBN-gated: fires ONLY when context.ean starts with 978/979 (book ISBN-13).
        # Cost: 2–4 free API calls (Open Library, optionally Google Books). Zero Serper.
        # None → Books stage skipped.
        self._books: Optional[BooksSource] = books_source
        # ScrapflyOzonSource: last-resort Ozon card gap-filler via Scrapfly.
        # Fires ONLY when OzonCardSource (Scrappey) returned 0 results AND
        # SCRAPFLY_OZON_FALLBACK_ENABLED=true AND there are still-empty targets.
        # Default: auto-create with env-based config (dormant when flag is off).
        self._scrapfly_ozon: Optional[ScrapflyOzonSource] = (
            scrapfly_ozon_source if scrapfly_ozon_source is not None
            else ScrapflyOzonSource()
        )
        # LamodaScrapflySource: last-resort clothing attribute gap-filler via Scrapfly.
        # Gated by LLM "is clothing?" classifier + LAMODA_SCRAPFLY_ENABLED=true.
        # Default: auto-create with env-based config (dormant when flag is off).
        self._lamoda_scrapfly: Optional[LamodaScrapflySource] = (
            lamoda_scrapfly_source if lamoda_scrapfly_source is not None
            else LamodaScrapflySource()
        )
        self._judges: dict[Source, ConfidenceAwareJudgeWrapper] = {
            src: ConfidenceAwareJudgeWrapper(s.get_judge())
            for src, s in self._sources.items()
        }
        # Judge для CompetitorRag (если source передан)
        if self._competitor_rag is not None:
            self._judges[Source.COMPETITOR_RAG] = ConfidenceAwareJudgeWrapper(
                self._competitor_rag.get_judge()
            )
        # Judge для IceCat (если source передан)
        if self._icecat is not None:
            self._judges[Source.ICECAT] = ConfidenceAwareJudgeWrapper(
                self._icecat.get_judge()
            )
        # Judge для PdfDatasheet (если source передан)
        if self._pdf_datasheet is not None:
            self._judges[Source.PDF_DATASHEET] = ConfidenceAwareJudgeWrapper(
                self._pdf_datasheet.get_judge()
            )
        # Judge для OzonCard (если source передан)
        if self._ozon_card is not None:
            self._judges[Source.OZON_CARD] = ConfidenceAwareJudgeWrapper(
                self._ozon_card.get_judge()
            )
        # Judge для WbCard (если source передан)
        if self._wb_card is not None:
            self._judges[Source.WB_CARD] = ConfidenceAwareJudgeWrapper(
                self._wb_card.get_judge()
            )
        # Judge для RegardSource: emits Source.WB_CARD — register only when
        # WB_CARD judge not already set (WbCardSource takes precedence).
        if self._regard is not None and Source.WB_CARD not in self._judges:
            self._judges[Source.WB_CARD] = ConfidenceAwareJudgeWrapper(
                self._regard.get_judge()
            )
        # BestBuySource / OnlinerSource also emit Source.WB_CARD — same judge applies.
        # Register fallback only if WB_CARD judge still not set (all three share the judge).
        if self._bestbuy is not None and Source.WB_CARD not in self._judges:
            self._judges[Source.WB_CARD] = ConfidenceAwareJudgeWrapper(
                self._bestbuy.get_judge()
            )
        if self._onliner is not None and Source.WB_CARD not in self._judges:
            self._judges[Source.WB_CARD] = ConfidenceAwareJudgeWrapper(
                self._onliner.get_judge()
            )
        # BooksSource also emits Source.WB_CARD — register fallback only if judge not set.
        if self._books is not None and Source.WB_CARD not in self._judges:
            self._judges[Source.WB_CARD] = ConfidenceAwareJudgeWrapper(
                self._books.get_judge()
            )
        # Judge для YandexMarket (если source передан).
        # YandexMarketSource эмитит Source.OZON_CARD — переиспользуем тот же judge
        # (см. module docstring yandex_market_source.py).
        # Важно: judge регистрируется только если OZON_CARD judge ещё не задан —
        # чтобы не перезаписать OzonCardSource judge когда оба переданы.
        if self._yandex_market is not None and Source.OZON_CARD not in self._judges:
            self._judges[Source.OZON_CARD] = ConfidenceAwareJudgeWrapper(
                self._yandex_market.get_judge()
            )
        # Judge для UGC (если source передан)
        if self._ugc is not None:
            self._judges[Source.UGC] = ConfidenceAwareJudgeWrapper(
                self._ugc.get_judge()
            )
        # ScrapflyOzonSource emits Source.OZON_CARD — reuse OZON_CARD judge.
        # Register ONLY when OZON_CARD judge not already set (ScrapflyOzon is a
        # last-resort path; OzonCardSource judge takes precedence when present).
        if (
            self._scrapfly_ozon is not None
            and Source.OZON_CARD not in self._judges
        ):
            self._judges[Source.OZON_CARD] = ConfidenceAwareJudgeWrapper(
                self._scrapfly_ozon.get_judge()
            )
        # LamodaScrapflySource emits Source.LAMODA — register its judge.
        if self._lamoda_scrapfly is not None:
            self._judges[Source.LAMODA] = ConfidenceAwareJudgeWrapper(
                self._lamoda_scrapfly.get_judge()
            )
        # Judge для Source.YANDEX_MARKET (из MarketplaceRouter).
        # Яндекс.Маркет — card-like источник (pre-moderated listing), поэтому
        # переиспользуем WbCardJudge — тот же, что и для Lamoda/WB карточек.
        if Source.YANDEX_MARKET not in self._judges:
            self._judges[Source.YANDEX_MARKET] = ConfidenceAwareJudgeWrapper(
                WbCardJudge()
            )
        # Judge для WEB_MARKETPLACE (GenericMarketplace pool).
        # Grounding already guards mud; WbCardJudge reused for consistency.
        self._judges[Source.WEB_MARKETPLACE] = ConfidenceAwareJudgeWrapper(
            WbCardJudge()
        )
        # MarketplaceRouter — полиморфный пул маркетплейсов (Stage 4.65).
        self._marketplace_router = MarketplaceRouter()
        self._classifier = classifier or LlmClassifier()
        self._cost_predictor = cost_predictor or CostPredictor()
        self._finisher = FinishingExtractor(sources=list(self._sources.values()))

    async def enrich(
        self, context: ExtractionContext, targets: list[TargetAttribute]
    ) -> list[AttributeValue]:
        """Run full pipeline. Returns merged final attributes (one per attribute_id)."""
        # Шаг 0a: убираем атрибуты которые marketplace не поддерживает
        targets = self._strategy.filter_unsupported_attributes(targets)
        # Шаг 0b: убираем атрибуты неизвестные словарю (Ozon: только словарные char_id)
        targets = self._strategy.filter_by_dictionary(targets, context)
        # Шаг 0c: обогащаем оставшиеся targets метаданными из словаря (name, type, description)
        targets = [
            self._strategy.normalize_target_with_context(t, context)
            for t in targets
        ]
        # Шаг 0c.5: бэкфилл allowed_values из живого словаря Ozon (list_values API)
        # для enum-полей, которым caller не прислал options (eg_importer тостер:
        # «Количество отделений» — закрытый {1,2,3,4}, но options пуст → enum-гейт
        # молчал → verbatim «8» из описания протекало). После — _enforce_allowed_values
        # сам режет out-of-list. Тот же API-механизм, что у ТН ВЭД.
        targets = await _backfill_allowed_values_from_api(targets, context)

        # Шаг 0d: бэкфилл context.brand из названия, когда продавец оставил «Бренд»
        # пустым (Баг 3 eg_importer). Brand-gated источники (ozon_card/regard/onliner/
        # bestbuy/IceCat) без context.brand не находят/брэнд-гейтят donor-карточку →
        # Пол/Цвет/Размер каскадят в no_data, хотя бренд стоит в начале имени. Высоко-
        # точно: ровно один словарный бренд из имени; иначе пусто (честнее мусора).
        if not (context.brand or "").strip():
            derived = _derive_context_brand(
                context, targets,
                brand_options_fn=lambda attr_id: self._strategy.brand_value_options(attr_id, context),
            )
            if derived:
                logger.info(
                    "[Pipeline] context.brand бэкфилл из имени: %r (продавец оставил «Бренд» пустым)",
                    derived,
                )
                context.brand = derived

        all_values: list[AttributeValue] = []
        # filled_so_far — накапливаем high-confidence AVs для skip-filled кооперации
        filled_so_far: list[AttributeValue] = []
        # Track whether OzonCardSource (Scrappey) returned ≥1 values this run.
        # ScrapflyOzonSource gate (a): skip if Scrappey already got the card.
        _ozon_card_obtained: bool = False

        # Stage 0: DescriptionSource (always first, cheapest)
        new_avs = await self._run_stage(Source.DESCRIPTION, context, targets)
        all_values += new_avs
        filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
        remaining = self._remaining_targets(targets, all_values)
        if not remaining:
            all_values += await self._run_finishing(context, targets, all_values)
            all_values += await self._generate_annotation(context, targets, all_values)
            return await self._finalize_async(all_values, targets, context)

        # Stage 0.46: BarcodeSource — verbatim EAN/barcode from description text.
        # Zero cost (no LLM, no network). Runs after Stage 0 so description text
        # is already in context; fills barcode-type attributes from verbatim digits.
        if remaining:
            new_avs = await self._run_barcode_stage(context, remaining, already_filled=filled_so_far)
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.45: WbCardSource — копия характеристик с похожего WB-товара
        # через бесплатный basket-API (БЕЗ Scrappey credits). Запускаем ПЕРВЫМ
        # — нулевая стоимость, ~300ms latency. WB-чары мапятся на тот же
        # Ozon-словарь (мы заполняем для Ozon, но WB — отличный источник).
        if self._wb_card is not None and remaining:
            new_avs = await self._run_wb_card_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.5: OzonCardSource — копия характеристик с похожего Ozon-товара.
        # Запускаем ПЕРВЫМ (до IceCat, PDF, LLM) — на eval-аудите парсер достаёт
        # 95.5% chars (21/22) из /features/ страницы и mapping на Ozon dict
        # gives direct attr_id resolution. Конкурирующие sources при таком
        # порядке дополняют OzonCard на attrs которые тот пропустил (outlier
        # товары, OzonCard match=skip), а не наоборот — OzonCard «съел» 14
        # потенциальных fills у других sources в предыдущей версии порядка.
        if self._ozon_card is not None and remaining:
            new_avs = await self._run_ozon_card_stage(
                context, remaining, already_filled=filled_so_far,
            )
            if new_avs:
                _ozon_card_obtained = True
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.52: YandexMarketSource — копия характеристик с live market.yandex.ru.
        # Запускается ПОСЛЕ WbCard + OzonCard (card=N fallback): нет смысла тратить
        # Scrappey-кредиты если WB/Ozon card уже закрыл нужные атрибуты. Особенно
        # полезен для apparel (generic-named, без SKU) где WB/Ozon дают card=N.
        # Cost-gated: только при remaining > 0 (т.е. когда предыдущие card-источники
        # не закрыли все targets).
        # DISABLED: YANDEX_MARKET_ENABLED=False — dead via Scrappey (captcha/301 every
        # URL, no residential), zero data contribution, burns up to 80s/product.
        if YANDEX_MARKET_ENABLED and self._yandex_market is not None and remaining:
            new_avs = await self._run_yandex_market_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.53: ImageCardSearch — reverse-image Serper /lens fallback.
        # Runs ONLY when: image_urls present AND all text card-sources returned card=N
        # (remaining still has unfilled targets). Cost: 1 Serper /lens call (~$0.005).
        # Gating: image_urls[0] required; fail-closed gate inside find_matching_card.
        if context.image_urls and remaining:
            new_avs = await self._run_image_card_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.54: GTINResolver — поиск EAN через Serper (имя→штрихкод).
        # Обогащает context.ean перед IceCat (Stage 0.55) чтобы тот мог использовать
        # GTIN lookup (_fetch_features_by_gtin) вместо нестабильного brand+name поиска.
        # Активен только при GTIN_RESOLVE_ENABLED=1 и отсутствующем context.ean.
        # НЕ заполняет AttributeValue — только мутирует context.ean.
        if self._gtin_resolver_enabled and not context.ean:
            try:
                from app.services.enrichment.sources.gtin_resolver import resolve_gtin
                _resolved_ean = await resolve_gtin(context)
                if _resolved_ean:
                    context.ean = _resolved_ean
                    logger.info(
                        "[Pipeline] Stage 0.54 GTINResolver: context.ean = %s "
                        "(product=%r brand=%r)",
                        _resolved_ean, context.product_name, context.brand,
                    )
            except Exception as _gtin_exc:
                logger.warning(
                    "[Pipeline] Stage 0.54 GTINResolver failed (non-fatal): %s",
                    _gtin_exc,
                )

        # Stage 0.55: IceCatSource — brand-verified спеки без LLM (IceCat Open API).
        # Дополняет attrs которые OzonCard не закрыл (brand_line skip, outlier товары).
        # При 403/404 (неизвестный бренд) возвращает [].
        # Vision Stage 0.52 (mpn/ean enrich) был отключён в v10b — Vision видит фото
        # чужой brand_line карточки от OzonCard, MPN с неё неточен. Vision запускается
        # только Stage 3 (classifier-routed) для реально визуальных attrs.
        icecat_filled_count = 0
        if self._icecat is not None:
            new_avs = await self._run_icecat_stage(context, remaining, already_filled=filled_so_far)
            icecat_filled_count = len([v for v in new_avs if v.is_confident()])
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.56: BooksSource — verbatim book metadata from Open Library + Google Books.
        # ISBN-gated: fires ONLY when context.ean is a book ISBN-13 (prefix 978/979).
        # Cost: 2–4 free API calls; zero Serper. Category-specific, safe to skip for non-books.
        if self._books is not None and remaining:
            new_avs = await self._run_books_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.57: RegardSource — verbatim electronics specs from regard.ru.
        # Fires after IceCat: fills CPU/RAM/storage/GPU specs IceCat missed.
        # Only when remaining > 0 (skip-guard inside source: ≥80% filled → []).
        # Cost: 1 Serper + 1 plain httpx GET (~$0.001 + free).
        if self._regard is not None and remaining:
            new_avs = await self._run_regard_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.58: BestBuySource — verbatim specs from Best Buy API (free, official).
        # Fires after Regard: rich structured EN specs, EN→RU translated before enum-match.
        # UPC lookup when EAN present; manufacturer+search otherwise.
        # Cost: 1 Best Buy API call (free, rate-limited). No-op when BESTBUY_API_KEY absent.
        if self._bestbuy is not None and remaining:
            new_avs = await self._run_bestbuy_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.59: OnlinerSource — verbatim RU specs from Onliner.by (public REST + JSON-LD).
        # Fires after BestBuy: 80+ clean RU name/value pairs from additionalProperty.
        # Cost: 1 search httpx GET + 1 page httpx GET (free, ~200ms).
        if self._onliner is not None and remaining:
            new_avs = await self._run_onliner_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.6: PdfDatasheetSource — official manufacturer datasheet PDF (Gemini native).
        # Запускается ПОСЛЕ IceCat: дополняет / перекрывает atрибуты не найденные через IceCat.
        # Skip-guard внутри source: если ≥80% targets уже filled, source сам возвращает [].
        if self._pdf_datasheet is not None and remaining:
            new_avs = await self._run_pdf_datasheet_stage(
                context, remaining, already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 0.7: CompetitorRagSource — дешёвый RAG без LLM (0 API calls)
        # Запускается всегда когда source задан — Qdrant query ~100ms на товар.
        # Раньше был skip-guard "< 5 IceCat fills" но это мешало измерять реальный
        # эффект RAG (icecat avg = 5.35 на БП → RAG никогда не вызывался).
        if self._competitor_rag is not None:
            new_avs = await self._run_rag_stage(context, remaining, already_filled=filled_so_far)
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 1: Classifier — 1 LLM call for routing decisions
        routing = await self._classifier.classify(context, remaining)

        # Stage 2: LlmKnowledgeSource — детерминированный добор на ОСТАТОЧНЫХ таргетах.
        # Запускаем LK, если LLM_KNOWLEDGE присутствует в routing ВООБЩЕ (не только [0]).
        # Раньше условие было routing[t.id][0] == LLM_KNOWLEDGE — недетерминированный
        # классификатор то ставил LK первым, то нет → apparel-поля (Стиль/Назначение/
        # Особенности/Рисунок) непостоянно доходили до LK между товарами.
        # Безопасность: knowledge_targets берётся из remaining (незаполненные), и сам
        # source ещё раз фильтрует already_filled. Карточные значения (WB/Ozon, conf≥0.90)
        # уже исключены из remaining → LK только ДОБИРАЕТ пустые, не перетирает.
        knowledge_targets = [
            t for t in remaining
            if Source.LLM_KNOWLEDGE in routing.get(t.id, [])
            and self._sources[Source.LLM_KNOWLEDGE].is_applicable(context, t)
        ]
        if knowledge_targets:
            new_avs = await self._run_stage(
                Source.LLM_KNOWLEDGE, context, knowledge_targets,
                already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 3: VisionSource — attrs with VISION in suggested AND image_urls present
        vision_targets = [
            t for t in remaining
            if Source.VISION in routing.get(t.id, [])
            and self._sources[Source.VISION].is_applicable(context, t)
        ]
        if vision_targets:
            new_avs = await self._run_stage(
                Source.VISION, context, vision_targets,
                already_filled=filled_so_far,
            )
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

        # Stage 4 gate: CostPredictor — check if web search is worth running
        #
        # Force-websearch attrs bypass CostPredictor AND the classifier's routing decision.
        # They are sent to WebSearch even if:
        #   - classifier didn't suggest WEB_SEARCH for them, or
        #   - they were already filled by LLM_KNOWLEDGE (web is more authoritative for these).
        #
        # Groups:
        #  - force_ws: targets whose id is in strategy.force_websearch_targets(targets)
        #              → always run WebSearch, regardless of routing / prior fill
        #  - optional_ws: remaining targets that classifier routed to WEB_SEARCH
        #                 → CostPredictor decides as before
        force_attr_ids = self._strategy.force_websearch_targets(targets, context)
        ws_applicable = self._sources[Source.WEB_SEARCH].is_applicable

        # Force targets: all targets in the force list (not just unfilled ones — we want
        # the web value to potentially override the LLM-knowledge value in _merge if
        # it has higher confidence).
        force_ws = [
            t for t in targets  # from ALL original targets, not just remaining
            if t.id in force_attr_ids
            and ws_applicable(context, t)
        ]

        # Optional targets: unfilled, routed to WEB_SEARCH, not in force list
        optional_ws = [
            t for t in remaining
            if t.id not in force_attr_ids
            and Source.WEB_SEARCH in routing.get(t.id, [])
            and ws_applicable(context, t)
        ]

        # Decide on optional targets via CostPredictor
        approved_optional: list = []
        if optional_ws:
            worth = await self._cost_predictor.is_web_search_worth(context, optional_ws)
            if worth:
                approved_optional = optional_ws

        # Deduplicate by id (force list may overlap with optional)
        seen_ids: set[int] = set()
        websearch_targets: list[TargetAttribute] = []
        for t in force_ws + approved_optional:
            if t.id not in seen_ids:
                seen_ids.add(t.id)
                websearch_targets.append(t)

        if not websearch_targets:
            all_values += await self._run_finishing(context, targets, all_values)
            all_values += await self._generate_annotation(context, targets, all_values)
            return await self._finalize_async(all_values, targets, context)

        if force_ws:
            logger.debug(
                "[Pipeline] force-websearch attrs: %s",
                [t.id for t in force_ws],
            )

        # Stage 4: WebSearchSource — передаём все накопленные high-conf AVs
        all_values += await self._run_stage(
            Source.WEB_SEARCH, context, websearch_targets,
            already_filled=filled_so_far,
        )
        filled_so_far = self._merge_high_conf(filled_so_far, all_values)

        # Stage 4.5: UgcSource — отзывы и Q&A с Ozon/WB для compat/physical attrs
        # (длина кабеля как у покупателя, шум, совместимость материнской платой).
        # Запускается последним перед finishing — после всех structured sources.
        if self._ugc is not None:
            remaining_for_ugc = self._remaining_targets(targets, all_values)
            if remaining_for_ugc:
                new_avs = await self._run_ugc_stage(
                    context, remaining_for_ugc, already_filled=filled_so_far,
                )
                all_values += new_avs

        # Stage 4.6: ScrapflyOzonSource — last-resort Ozon card gap-filler via Scrapfly.
        # Fires ONLY when:
        #   (a) _ozon_card_obtained=False (Scrappey got nothing this run), AND
        #   (b) there are still-empty target attributes (remaining > 0), AND
        #   (c) SCRAPFLY_OZON_FALLBACK_ENABLED=true (env flag).
        # Cost: 60 credits/product (30 search + 30 features). Default: OFF.
        if self._scrapfly_ozon is not None:
            remaining_for_scrapfly = self._remaining_targets(targets, all_values)
            if remaining_for_scrapfly and not _ozon_card_obtained:
                new_avs = await self._run_scrapfly_ozon_stage(
                    context,
                    remaining_for_scrapfly,
                    already_filled=filled_so_far,
                    ozon_card_obtained=_ozon_card_obtained,
                )
                all_values += new_avs
                filled_so_far = self._merge_high_conf(filled_so_far, new_avs)

        # Stage 4.65: MarketplaceRouter — полиморфный пул маркетплейсов.
        # Яндекс.Маркет (always-on) + Lamoda/специалисты по типу товара.
        # Гейт: LAMODA_SCRAPFLY_ENABLED=true (тот же флаг, что ранее у LamodaScrapflySource).
        # Default: OFF.
        from app import config as _cfg_mp  # local import — избегаем циклической зависимости
        if _cfg_mp.LAMODA_SCRAPFLY_ENABLED:
            remaining_for_mp = self._remaining_targets(targets, all_values)
            if remaining_for_mp:
                new_avs = await self._run_marketplace_router_stage(
                    context,
                    remaining_for_mp,
                    already_filled=filled_so_far,
                )
                all_values += new_avs
                filled_so_far = self._merge_high_conf(filled_so_far, new_avs)

        # Stage 4.7: TnvedSource — per-category резолвер ТН ВЭД ЕАЭС (OZON ONLY).
        # Запускается после всех товарных sources: кэш по category_id уже тёплый
        # если несколько товаров одной категории обрабатываются параллельно.
        # WB ТН ВЭД резолвится из собственного WB-словаря в _apply_wb_api_resolve.
        if self._strategy.name == "ozon":
            new_avs = await self._run_tnved_stage(context, targets, already_filled=filled_so_far)
            all_values += new_avs

        # Stage 4.8: SafeEnumFillSource — gated LLM fill for still-empty short
        # optional enum attrs.  Off by default (SAFE_LLM_ENUM_FILL_ENABLED flag).
        # Gate A: verbatim value in web_search summary (zero extra LLM calls).
        # Gate B: adversarial verifier LLM call (batched per product, 1 call).
        # The reverted "force-route short enums to llm_knowledge" is what this
        # replaces, with an actual mud-gate rather than self-reported confidence.
        from app import config as _cfg  # local import avoids circular dep at module level
        if _cfg.SAFE_LLM_ENUM_FILL_ENABLED:
            remaining_for_safe_enum = self._remaining_targets(targets, all_values)
            if remaining_for_safe_enum:
                new_avs = await self._run_safe_enum_fill_stage(
                    context,
                    remaining_for_safe_enum,
                    already_filled=filled_so_far,
                )
                all_values += new_avs
                filled_so_far = self._merge_high_conf(filled_so_far, new_avs)

        # Stage 4.9: inference-source adversarial post-merge pass (FIX #1 + FIX B).
        # Covers Source.LLM_KNOWLEDGE (world-knowledge, no verbatim anchor) and
        # Source.WEB_SEARCH (wrong-product attribution leaks past Gate A's verbatim
        # check — e.g. Бязь on a Nike tee from a web snippet about a different product).
        # Gate B's «correct for THIS exact product» check catches both cases.
        # Skip fills already verbatim-anchored (evidence tag "safe_enum:verbatim_gate").
        #
        # OFF by default. This is an ABSENCE-based drop gate: it KEEPS a spec fill only
        # if an authoritative source independently produced the same value, else DROPS.
        # A/B on Ozon (2026-06-17) showed this is too blunt — it drops correct-but-
        # uncorroborated specs (8 ГБ RAM, M.2 SSD) along with real mud (Чипсет=AMD on an
        # Intel laptop): −8.8 pp opt_honest. The CONTRADICTION-based corroboration-override
        # (Stage 4.93) supersedes it for mud removal without coverage loss. Kept behind the
        # flag for the aggressive "empty > wrong" mode when explicitly requested.
        if os.environ.get("LLM_KNOWLEDGE_ADVERSARIAL_ENABLED", "0") == "1":
            # Force-websearch targets are explicitly trusted: exempt their WEB_SEARCH fills
            # from the adversarial gate.  Dropping them + re-firing in finishing would be a
            # real Serper double-charge and contradicts the intent of the force list.
            _force_avs = [
                v for v in all_values
                if v.attribute_id in force_attr_ids and v.source == Source.WEB_SEARCH
            ]
            # TnvedSource fills are deterministically validated (10-digit code, per-category
            # LLM prompt) — exempt from adversarial gate which cannot verify customs codes
            # from product title alone (same reason as UniversalGate's _is_tnved_attr skip).
            _tnved_avs = [
                v for v in all_values
                if (v.evidence or "").startswith("tnved_resolver:")
            ]
            _exempt_ids = {id(v) for v in _force_avs} | {id(v) for v in _tnved_avs}
            _non_force_avs = [v for v in all_values if id(v) not in _exempt_ids]
            gated = await self._run_llm_knowledge_adversarial_pass(
                _non_force_avs, targets, context, filled_so_far,
            )
            all_values = gated + _force_avs + _tnved_avs

        # Stage 5: Finishing pass — focused re-extraction for empty required attributes
        all_values += await self._run_finishing(context, targets, all_values)

        # Stage 5.5: Аннотация generation — generative step, not extraction.
        # Runs AFTER all sources and finishing so it can use the full set of filled attrs.
        all_values += await self._generate_annotation(context, targets, all_values)

        # Stage 6: детерминированный resolve_value_ids (в _finalize) + LLM-резолвер
        # ХВОСТА — батч-вызов на нерезолвнутые enum-value_id (семантика/перевод).
        return await self._finalize_async(all_values, targets, context)

    async def _generate_annotation(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        all_values: list[AttributeValue],
    ) -> list[AttributeValue]:
        """Stage 5.5: генерация поля «Аннотация» из уже собранных характеристик.

        Аннотация — генерируемое маркетинговое описание, не извлекаемое.
        Запускается ПОСЛЕ всех источников. Judge не нужен — это генерация, не извлечение.
        Пропускается если Аннотация уже заполнена или не входит в targets.
        """
        # Detect Аннотация target (case-insensitive)
        annotation_target = next(
            (t for t in targets if t.name.lower() == "аннотация"),
            None,
        )
        if annotation_target is None:
            return []

        # Skip if already filled with high confidence
        already_filled_ids = {v.attribute_id for v in all_values if v.is_confident()}
        if annotation_target.id in already_filled_ids:
            return []

        # Build characteristics summary from filled values
        filled_by_id: dict[int, AttributeValue] = {}
        for v in all_values:
            prev = filled_by_id.get(v.attribute_id)
            if prev is None or v.confidence > prev.confidence:
                filled_by_id[v.attribute_id] = v

        target_names: dict[int, str] = {t.id: t.name for t in targets}
        char_lines = []
        for attr_id, av in filled_by_id.items():
            if attr_id == annotation_target.id:
                continue
            name = target_names.get(attr_id, str(attr_id))
            char_lines.append(f"  {name}: {av.value}")

        chars_block = "\n".join(char_lines) if char_lines else "  (нет данных)"

        class _AnnotationResponse(BaseModel):
            annotation: str = Field(..., description="Маркетинговое описание товара 2-4 предложения")

        system_prompt = (
            "Ты маркетолог. Составь маркетинговое описание товара 2-4 предложения "
            "на основе предоставленных характеристик. Текст должен быть живым, "
            "продающим, без перечислений через запятую. Только текст описания, без заголовков."
        )
        user_text = (
            f"Товар: {context.product_name}\n"
            f"Бренд: {context.brand or 'неизвестен'}\n"
            f"Категория: {' / '.join(context.category_path) or 'н/д'}\n\n"
            f"Характеристики:\n{chars_block}\n\n"
            f"Составь маркетинговое описание товара 2-4 предложения."
        )

        try:
            llm = get_main_manager()
            parsed, _ = await llm.structured_request(
                system_prompt=system_prompt,
                user_text=user_text,
                response_model=_AnnotationResponse,
            )
        except Exception as e:
            logger.warning("[Pipeline] annotation generation failed: %s", e, exc_info=True)
            return []

        if parsed is None or not parsed.annotation.strip():
            return []

        context.llm_calls_so_far += 1
        return [
            AttributeValue(
                attribute_id=annotation_target.id,
                value=parsed.annotation.strip(),
                confidence=0.9,
                source=Source.LLM_KNOWLEDGE,
                evidence="generated from collected attributes",
                is_collection=False,
            )
        ]

    def _track_a_corroboration_filter(
        self,
        all_values: list[AttributeValue],
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Finishing re-gate: deterministic Track-A corroboration filter ($0, no LLM).

        Called once at the start of _finalize_async after all Finishing stages.
        Drops objective-spec fills from guess sources (LLM_KNOWLEDGE, WEB_SEARCH,
        COMPETITOR_RAG) that lack authoritative corroboration.
        Mirror of Stage 4.9 Track A, but runs on the FINAL all_values
        (authoritative_fills maximally populated).
        """
        _ADVERSARIAL_SOURCES = {Source.LLM_KNOWLEDGE, Source.WEB_SEARCH, Source.COMPETITOR_RAG}

        # Build authoritative fills set
        authoritative_fills: set[tuple[int, str]] = set()
        for v in all_values:
            if v.source in _AUTHORITATIVE_SOURCES:
                norm = _normalize_for_corroboration(v.value)
                authoritative_fills.add((v.attribute_id, norm))

        # Build target lookup
        target_by_id = {t.id: t for t in targets}

        # Filter values
        result: list[AttributeValue] = []
        for v in all_values:
            if v.source in _ADVERSARIAL_SOURCES:
                target = target_by_id.get(v.attribute_id)
                if target is not None and _is_objective_spec_attr(target):
                    if not (v.evidence or "").startswith("safe_enum:verbatim_gate"):
                        norm_val = _normalize_for_corroboration(v.value)
                        if (v.attribute_id, norm_val) not in authoritative_fills:
                            logger.info(
                                "[Pipeline] spec-corroboration DROP (finishing re-gate): "
                                "attr=%s value=%r source=%s norm=%r "
                                "— empty>wrong for objective-spec attr",
                                v.attribute_id, v.value, v.source.value, norm_val,
                            )
                            continue
            result.append(v)

        return result

    async def _finalize_async(
        self,
        all_values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Async финализация: детерминированный _finalize + LLM-резолвер ХВОСТА value_id.

        _finalize остаётся синхронным и неизменным (merge + post-process +
        детерминированный resolve_value_ids + validation). Затем один батч-LLM-вызов
        добивает нерезолвнутые enum-value_id (семантика/перевод). Для не-Ozon
        стратегий llm_resolve_tail — no-op. Используется во ВСЕХ точках выхода enrich.

        Placeholder guard (ДО финализации): скалярные значения-плейсхолдеры («нет»,
        «-», «n/a» и т.п.) удаляются из enum-кандидатов ДО resolve, чтобы они не
        попали в LLM-хвост и не заняли required-поле.

        Enum garbage guard (ПОСЛЕ полной резолюции value_id): любой enum-таргет
        (optional ИЛИ required), чьё значение не резолвнулось в словарный value_id,
        дропается — Ozon отклонит такую карточку на загрузке. Пустое честнее.
        Порядок гарантирует безопасность: brand-from-name и gender-fill получают
        value_id ДО этого этапа (в _finalize и llm_resolve_tail), поэтому их
        корректно-резолвнутые значения НЕ затрагиваются.
        """
        # Finishing re-gate (Track A): drop objective-spec fills from guess sources
        # that have no authoritative corroboration. Runs BEFORE all other finalization
        # so that Finishing-hallucinated values don't reach post_process/resolve.
        # Mirrors Stage 4.9 Track A but at the final all_values (most authoritative).
        all_values = self._track_a_corroboration_filter(all_values, targets)

        # Hard-drop: confidence <= 0.0 is always a zero-signal fill — drop globally
        # before any merge. Catches Vision «ABS пластик» conf=0.0 / evidence=«No material
        # listed» and any other source that emits a value it itself has no confidence in.
        # «пусто честнее мусора» — owner rule. Applies to ALL sources, ALL targets.
        conf_zero_dropped: list[AttributeValue] = []
        for v in all_values:
            if v.confidence <= 0.0:
                logger.info(
                    "[Pipeline] conf<=0 hard-drop: attr=%s value=%r source=%s conf=%s — "
                    "zero-confidence fill never reaches filled set",
                    v.attribute_id, v.value, v.source.value, v.confidence,
                )
                continue
            conf_zero_dropped.append(v)
        all_values = conf_zero_dropped

        # Placeholder pre-filter: «нет», «-», «n/a» etc. удаляем из enum-таргетов
        # ДО resolve_value_ids, чтобы они не осели как «filled» с value_id=None.
        targets_by_id = {t.id: t for t in targets}
        filtered_values: list[AttributeValue] = []
        for v in all_values:
            tgt = targets_by_id.get(v.attribute_id)
            if tgt is not None and tgt.allowed_values and not isinstance(v.value, list):
                if _is_placeholder_value(v.value):
                    logger.info(
                        "[Pipeline] placeholder-filter: дроп enum attr=%s value=%r "
                        "(плейсхолдер не является enum-значением)",
                        v.attribute_id, v.value,
                    )
                    continue
            filtered_values.append(v)

        # Enum-membership гард (Баг тостер: IceCat смапил «уровней поджаривания=8»
        # → «Количество отделений», у которого закрытый словарь {1,2,3,4}). LLM-гейт
        # _enforce_allowed_values работает только на LLM-слое — IceCat/marketplace/
        # donor-источники его ОБХОДЯТ. Здесь, до resolve_value_ids, дропаем скалярное
        # значение, которое реально не принадлежит allowed_values (тот же _map_enum_value:
        # exact→норм→fuzzy-порог). «пусто честнее мусора». Исключения: ТН ВЭД
        # (allowed_values ~50-code sample, но фактически free-text — режет TnvedSource)
        # и бренд (free-text/truncated). Многозначные (Ⓜ️) словарные поля
        # («Функциональные особенности», «Системы защиты», «Режимы тостера») —
        # тоже закрытые словари: фильтруем КАЖДЫЙ член списка, иначе мусор от
        # не-LLM источников (IceCat/donor) протекает мимо LLM-гейта.
        filtered_values = _apply_enum_membership_guard(filtered_values, targets_by_id)

        # TNVED_SOURCE_FIX_ENABLED: drop LLM_KNOWLEDGE/WEB_SEARCH values for ТН ВЭД
        # attributes UNLESS they came from TnvedSource (sentinel prefix "tnved_resolver:"
        # in evidence field).  web_search routinely returns absurd customs codes (e.g.
        # «цилиндры для контактных линз» on a car stereo) and generic LLM_KNOWLEDGE
        # fares no better.  TnvedSource uses a focused per-category prompt with 10-digit
        # validation and double-checked locking — it is the only trusted ТН ВЭД source.
        # Option (а) chosen over (б) new Source enum: evidence-prefix costs zero refactor,
        # keeps SOURCE_PRIORITY/judges intact, and is reversible via the flag alone.
        from app import config as _cfg_tnved
        if _cfg_tnved.TNVED_SOURCE_FIX_ENABLED:
            _TNVED_NAME_MARKERS = ("тн вэд", "тнвэд", "еаэс")
            _TNVED_GARBAGE_SOURCES = {Source.LLM_KNOWLEDGE, Source.WEB_SEARCH}
            _TNVED_RESOLVER_PREFIX = "tnved_resolver:"
            tnved_fixed: list[AttributeValue] = []
            for v in filtered_values:
                tgt = targets_by_id.get(v.attribute_id)
                if tgt is not None:
                    tname_l = (tgt.name or "").lower()
                    is_tnved_attr = any(m in tname_l for m in _TNVED_NAME_MARKERS)
                    if is_tnved_attr and v.source in _TNVED_GARBAGE_SOURCES:
                        # Keep only TnvedSource fills (identified by evidence sentinel)
                        ev = (v.evidence or "")
                        if not ev.startswith(_TNVED_RESOLVER_PREFIX):
                            logger.info(
                                "[TnvedFix] DROP attr=%s value=%r source=%s "
                                "— not from TnvedSource (evidence=%r)",
                                v.attribute_id, v.value, v.source.value, ev[:60],
                            )
                            continue
                tnved_fixed.append(v)
            filtered_values = tnved_fixed

        finalized = self._finalize(filtered_values, targets, context)
        resolved = await self._strategy.llm_resolve_tail(finalized, targets, context)

        # Brand-from-title LLM: AFTER brand-from-name (deterministic, already ran inside
        # _finalize) and AFTER llm_resolve_tail, but BEFORE drop-guards. Only fires when
        # brand targets are STILL EMPTY — deterministic path wins when it succeeds.
        # Exempt from _BRAND_GUESS_SOURCES guard: emits Source.DESCRIPTION (title-read),
        # not Source.LLM_KNOWLEDGE (world-knowledge). Title-anchored + enum-constrained.
        resolved = await _apply_brand_from_title_llm(
            resolved, targets, context,
            brand_options_fn=lambda attr_id: self._strategy.brand_value_options(attr_id, context),
            brand_id_fn=lambda attr_id: self._strategy.brand_value_id_options(attr_id, context),
        )

        # Type-from-category: ПОСЛЕ brand-from-name/gender/llm_resolve_tail, но ДО
        # drop-guard. Category leaf — это И ЕСТЬ тип товара; заполняем ПУСТОЙ/нерезолвнутый
        # required enum точным enum-матчем leaf-леммы, привязывая словарный value_id (тем
        # же resolve_value_ids-путём). Заполненный тип получает value_id → НЕ дропается ниже.
        resolved = _apply_type_from_category(
            resolved, targets, context,
            value_id_fn=lambda attr_id, val: self._strategy.resolve_value_ids(
                AttributeValue(
                    attribute_id=attr_id, value=val, confidence=0.9,
                    source=Source.DESCRIPTION,
                ),
                context,
            ).value_id,
        )

        # Size-from-name: LOW-PRIORITY FALLBACK for «Российский размер» (4295/4298).
        # Fills ONLY when the attr is still empty after all earlier stages.
        # parse_explicit_size is strict (empty > wrong) — safe to run unconditionally.
        resolved = _apply_size_from_name(
            resolved, targets, context,
            value_id_fn=lambda attr_id, val: self._strategy.resolve_value_ids(
                AttributeValue(
                    attribute_id=attr_id, value=val, confidence=_SIZE_FROM_NAME_CONF,
                    source=Source.DESCRIPTION,
                ),
                context,
            ).value_id,
        )

        # Spec-from-title: deterministic filler for OPTIONAL enum targets still empty
        # after all real sources. Fills only when EXACTLY ONE allowed value is present
        # in the product title as a whole-token sequence (ambiguity → skip).
        # Runs BEFORE drop-guards so filled values get value_id resolution.
        resolved = _apply_spec_from_title(
            resolved, targets, context,
            value_id_fn=lambda attr_id, val: self._strategy.resolve_value_ids(
                AttributeValue(
                    attribute_id=attr_id, value=val, confidence=_SPEC_FROM_TITLE_CONF,
                    source=Source.DESCRIPTION,
                ),
                context,
            ).value_id,
        )

        # Lever 1: "Название" — fill marketplace product-title attr from input name.
        # Verbatim fill from context.product_name. Never overwrites. Source=DESCRIPTION.
        resolved = _apply_title_from_input(resolved, targets, context)

        # Lever 1b: free-text MODEL / title-template-model fields ← product name
        # (verbatim; «Модель» strips leading brand). Closes Модель/Название модели
        # which _apply_title_from_input deliberately skips.
        resolved = _apply_model_from_title(resolved, targets, context)

        # Lever 2: General verbatim numeric-spec extractor.
        # Covers "Срок службы, лет" (special regex) + all other numeric attrs whose
        # name encodes a known unit AND whose keyword phrase appears near the number.
        # Cross-attribute bleed guard: each attr requires its OWN keyword in window.
        # Source=DESCRIPTION, conf=0.97, verbatim-only.
        resolved = _apply_numeric_spec_from_text(resolved, targets, context)

        # Lever 3a: Boolean Да/Нет — stated-only verbatim fill.
        # Fills "Да" ONLY when feature keyword EXPLICITLY present in fetched text.
        # NEVER defaults to "Да" from category/absence. Source=DESCRIPTION, conf=0.97.
        resolved = _apply_boolean_stated_from_text(resolved, targets, context)

        # Lever 3b: "Комплектация" — verbatim list from product description.
        # Extracts "комплектация: …" / "в комплекте: …" section verbatim.
        # Source=DESCRIPTION, conf=0.97.
        resolved = _apply_komplektatsiya_from_text(resolved, targets, context)

        # Lever 3 (original): "Гарантия" ↔ "Гарантийный срок" alias cross-fill.
        # Both are free-text String fields (no enum constraint). Verbatim copy only.
        # Fires ONLY when one alias is filled and its counterpart is empty.
        resolved = _apply_warranty_alias_cross_fill(resolved, targets)

        # ПОСЛЕ полной резолюции value_id (детерминированный + LLM-хвост):
        # 1. Дроп OPTIONAL enum-значений без value_id (fake-fill, Ozon отклонит).
        # 2. Дроп REQUIRED enum-значений без value_id — мусор в обязательном поле
        #    хуже пустоты: Ozon отвергает карточку, а не просто помечает поле пустым.
        #    Correctly-resolved required values (value_id is set) НЕ затрагиваются.
        # Stage 4.95 — required-enum mud-gate. Plugs the hole left by the Stage-4.9
        # inference gate (which covers only llm_knowledge / web_search / competitor_rag):
        # a REQUIRED enum filled by DescriptionSource or Vision with a value that is NOT
        # verbatim-present in the product's own text is an LLM inference, not data. Such
        # values are adversarially verified (fail-closed) and retracted when baseless —
        # e.g. Пол=«Женский» hallucinated for headphones. Structural attrs (Тип/Бренд) and
        # authoritative card sources are exempt; verbatim-grounded values pass untouched.
        resolved = await self._run_required_enum_mud_gate(resolved, targets, context)

        # Universal verification gate (UNIVERSAL_VERIFY_ENABLED=1): applies
        # evidence-self-admission, fast numeric reject, unit-sanity, and selective
        # LLM hallucination judge to ALL sources after merge.
        resolved = await self._run_universal_verification_gate(resolved, targets, context)

        # value_ids ↔ value reconciliation: сбрасываем рассинхрон ПЕРЕД авторитетным
        # резолвом, чтобы он пересобрал ids строго из value-текста (донор ozon_card
        # иногда приклеивает ВСЮ палитру категории — 15 value_id «черный/белый/…» — к
        # одному цвет-значению; зальётся как неверные цвета).
        resolved = _reconcile_enum_value_ids(resolved, targets)

        # Authoritative value resolution via the marketplace's OWN live API —
        # runs AFTER the gates and JUST BEFORE the drop-guards so real value_ids
        # prevent the enum drop-guards from discarding these fields. Each
        # marketplace has its own endpoints/value-model (Ozon: numeric value_id;
        # WB: canonical dictionary string).
        if self._strategy.name == "ozon":
            resolved = await self._apply_ozon_api_resolve(resolved, targets, context)
        elif self._strategy.name == "wb":
            resolved = await self._apply_wb_api_resolve(resolved, targets, context)

        resolved = _drop_ungrounded_color_guess(resolved, targets)
        resolved = _drop_unresolved_optional_enums(resolved, targets)
        resolved = _drop_unresolved_required_enums(resolved, targets)
        return resolved

    # ──────────────────────────────────────────────────────────────────────────
    # Universal verification gate (Stage 4.92)
    # ──────────────────────────────────────────────────────────────────────────

    # Non-authoritative sources that must pass LLM hallucination judge
    # when the value is NOT verbatim in grounding text.
    _UNVERIFIED_SOURCES: frozenset[Source] = frozenset({
        Source.VISION,
        Source.LLM_KNOWLEDGE,
        Source.WEB_SEARCH,
        Source.COMPETITOR_RAG,
        Source.SAFE_ENUM_FILL,
    })

    # Evidence self-admission patterns — drop without LLM cost.
    # Expanded to catch vision-style hedges like «No information about pedal type in vision»,
    # and LLM self-hedges like «closest in list», «skipping», «not confident», «inferred».
    _EVIDENCE_NO_INFO_RE = re.compile(
        r"no information|not specified|not visible|cannot\s+(?:be\s+)?determin|"
        r"uncertain|unknown|not mentioned|no\s+\S+\s+(?:information|data)|"
        r"не указан|нет данных|не найдено|не\s+вид(?:но|ен)|невозможно\s+определить|"
        r"information not found|"
        # LLM self-admission of guessing / approximation
        r"skipping|skip\b|closest\s+in\s+(?:the\s+)?list|not\s+in\s+(?:the\s+)?list|"
        r"no\s+\w+\s+match|not\s+confident|guess(?:ed|ing)?|inferred|"
        r"не\s+уверен|ближайш|нет\s+в\s+списке|предположительно",
        re.IGNORECASE,
    )

    # Attribute name patterns that indicate country-of-origin fields.
    # Filling country from inference sources is unreliable → DROP.
    _COUNTRY_ATTR_RE = re.compile(
        r"страна|country|изготовител|производител|origin",
        re.IGNORECASE,
    )

    # Unit groups: values whose string contains a unit from group A are invalid
    # for fields whose expected unit is in group B, and vice versa.
    # Format: list of frozensets (each set = one "incompatible-with-others" unit class).
    _UNIT_CONFLICT_GROUPS: tuple[frozenset[str], ...] = (
        frozenset({"г", "кг", "мг", "lb", "oz", "фунт"}),     # weight
        frozenset({"шт", "штук", "pcs", "piece"}),               # count
        frozenset({"мл", "л", "ml", "l"}),                        # volume
        # Linear units added back: unit_sanity fires ONLY when the TARGET field
        # itself expects a weight/count/volume unit. If the field is "Вес, г" and
        # the value is "50 см" — that is always wrong, any source.
        frozenset({"см", "мм", "м", "km", "дюйм", "in", "ft", "mm", "cm"}),  # linear
    )

    # Boolean string values — in a numeric/dimensions field these are always wrong.
    _BOOLEAN_VALS: frozenset[str] = frozenset(
        {"да", "нет", "true", "false", "yes", "no"}
    )

    # Count-field names that classify_target() may not catch (type='text', no unit suffix)
    # but which only accept integer counts — not boolean strings.
    _COUNT_FIELD_RE = re.compile(
        r"\b(?:число|количество|кол-во|кол\.?\s*во|count|number\s+of|qty)\b",
        re.IGNORECASE,
    )

    # Sub-name / descriptive fields where a value equal to the WHOLE product
    # title is a lazy placeholder echo, never a real attribute value. The main
    # name/merge fields (Название, Наименование, Название модели…, Объединить…,
    # Озон.Видео: название) legitimately hold the title and are deliberately NOT
    # listed here — blanket-dropping value==title would cut real coverage.
    _TITLE_ECHO_FIELD_RE = re.compile(
        r"аннотаци|название\s+вкуса|название\s+цвета|название\s+аромат|"
        r"название\s+запах|модель\s+тс\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _norm_for_title_echo(s: str) -> str:
        """Normalize for title-echo comparison: lowercase, punctuation→space,
        collapse whitespace. Cyrillic-safe (\\w matches Unicode letters)."""
        return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", str(s).lower())).strip()

    async def _run_universal_verification_gate(
        self,
        resolved: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Universal anti-hallucination gate (Stage 4.92).

        Applies 4 checks to every resolved AttributeValue after merge:

        Step 1 — Evidence self-admission (free): if v.evidence matches the
          "no information / not specified" regex → DROP. The source itself
          admitted it had nothing.

        Step 2 — Fast numeric reject (free): if the target is numeric/dimensions
          AND the value string contains a digit AND that digit is NOT present in
          grounding_text (name + description) → DROP. Catches wb_card returning
          «50 cm» weight, IceCat stale specs, etc.

        Step 3 — Unit sanity (free): if target expected unit is in a weight/count/
          volume group, and the value string contains a unit from a DIFFERENT group →
          DROP. Catches «50 см» in a weight field (г/кг expected).

        Step 4 — Hallucination judge (LLM, batched): only for non-authoritative
          sources (VISION, LLM_KNOWLEDGE, WEB_SEARCH, COMPETITOR_RAG, SAFE_ENUM_FILL)
          whose value is NOT verbatim-present in grounding_text → LLM adversarial
          check. Authoritative sources (WB_CARD, ICECAT, OZON_CARD, PDF_DATASHEET,
          DESCRIPTION, LAMODA) trust-skip the LLM judge but still pass steps 1–3.

        Gated behind UNIVERSAL_VERIFY_ENABLED env flag (default OFF to avoid
        latency regression on existing deployments).

        Anti-regression guarantee: authoritative sources NEVER enter Step 4.
        Steps 1–3 are deterministic and only fire on clear evidence of error.
        """
        if os.getenv("UNIVERSAL_VERIFY_ENABLED", "0").strip().lower() not in (
            "1", "true", "yes", "on",
        ):
            return resolved

        targets_by_id: dict[int, TargetAttribute] = {t.id: t for t in targets}

        # Grounding text = name + description (product-level, always available).
        grounding_base = " ".join(
            s for s in (context.product_name, context.product_description) if s
        )

        out: list[AttributeValue] = []
        # Accumulate candidates for LLM judge batching.
        # key: index in `out` where we will append (or drop) the value.
        llm_candidates: list[tuple[int, AttributeValue, TargetAttribute, str]] = []

        for v in resolved:
            t = targets_by_id.get(v.attribute_id)
            if t is None:
                out.append(v)
                continue

            val_str = str(v.value).strip() if not isinstance(v.value, list) else ""
            evidence_str = (v.evidence or "").strip()

            # Per-value grounding text: name + description + THIS value's own evidence.
            # For web_search/icecat/wb_card the evidence field contains the fetched
            # snippet — the number must be present in THAT text, not only in the title.
            grounding_text_v = " ".join(s for s in (evidence_str, grounding_base) if s)

            # ── Step 1: Evidence self-admission ──────────────────────────────
            if evidence_str and self._EVIDENCE_NO_INFO_RE.search(evidence_str):
                logger.info(
                    "[UniversalGate] DROP attr=%s value=%r source=%s "
                    "step=evidence_self_admission evidence=%r",
                    v.attribute_id, v.value, v.source.value, evidence_str[:80],
                )
                continue

            # ── Step 1b: Country-of-origin from COMPETITOR_RAG → DROP ──
            # COMPETITOR_RAG converges on wrong RU origin (Champion→Россия,
            # Ahmad Tea→Россия) because competing listings copy each other.
            # LLM_KNOWLEDGE is NOT blocked here — Германия/Индонезия from world
            # knowledge is often correct; it will be audited by the judge if enabled.
            if (
                self._COUNTRY_ATTR_RE.search(t.name or "")
                and v.source is Source.COMPETITOR_RAG
            ):
                logger.info(
                    "[UniversalGate] DROP attr=%s value=%r source=%s "
                    "step=country_competitor_rag",
                    v.attribute_id, v.value, v.source.value,
                )
                continue

            # ── Step 1c: Boolean value in numeric/dimensions field → DROP ──
            # Catches IceCat emitting «Да»/«Нет»/«True»/«False»/«Yes»/«No» into
            # fields like «Время зарядки, ч» or «Число портов HDMI».
            # Fires for any source, any scalar value.
            if val_str and val_str.strip().lower() in self._BOOLEAN_VALS:
                _kind_for_bool = classify_target(t)
                _has_unit = bool((extract_unit(t.name) or "").strip())
                _is_count_field = bool(self._COUNT_FIELD_RE.search(t.name or ""))
                if _kind_for_bool in ("numeric", "dimensions") or _has_unit or _is_count_field:
                    logger.info(
                        "[UniversalGate] DROP attr=%s value=%r source=%s "
                        "step=boolean_in_numeric field=%r",
                        v.attribute_id, v.value, v.source.value, t.name,
                    )
                    continue

            # ── Step 1d: Title echo in a sub-name / descriptive field → DROP ──
            # A descriptive or qualified-name field (Аннотация, Название вкуса/
            # цвета/аромата, Модель ТС) whose value is just the WHOLE product
            # title is a lazy placeholder fill, never a real value. Exact
            # normalized match only; the main name/merge fields are NOT in the
            # denylist, so legitimate title fills are untouched.
            if val_str and self._TITLE_ECHO_FIELD_RE.search(t.name or ""):
                prod_title = (context.product_name or "").strip()
                if prod_title and self._norm_for_title_echo(val_str) == \
                        self._norm_for_title_echo(prod_title):
                    logger.info(
                        "[UniversalGate] DROP attr=%s value=%r source=%s "
                        "step=title_echo field=%r",
                        v.attribute_id, v.value, v.source.value, t.name,
                    )
                    continue

            # ── Steps 2 & 3 only for scalar values ───────────────────────────
            if val_str and not isinstance(v.value, list):
                kind = classify_target(t)
                is_numeric_kind = kind in ("numeric", "dimensions")

                # Step 2: Fast numeric reject
                # Apply ONLY to text-extracting sources (DESCRIPTION, WEB_SEARCH) whose
                # numbers must literally appear in the fetched text.
                # EXEMPT: authoritative sources (wb_card, icecat, ozon_card, etc.) — their
                # specs are NOT in the short seller text by design.
                # EXEMPT: LLM_KNOWLEDGE — its numbers are world-knowledge about the named
                # model; correctness is judged by polarity-correct judge in Step 4, not
                # by text-presence.
                _FAST_NUMERIC_SOURCES = {Source.DESCRIPTION, Source.WEB_SEARCH}
                if is_numeric_kind and v.source in _FAST_NUMERIC_SOURCES:
                    # Check if value string contains any digit
                    has_digit = any(c.isdigit() for c in val_str)
                    if has_digit and grounding_text_v:
                        if not NumericValidator.is_value_in_text(val_str, grounding_text_v):
                            logger.info(
                                "[UniversalGate] DROP attr=%s value=%r source=%s "
                                "step=fast_numeric_reject — число не в grounding_text",
                                v.attribute_id, v.value, v.source.value,
                            )
                            continue

                # Step 3: Unit sanity
                expected_unit = (extract_unit(t.name) or "").strip().lower()
                if expected_unit and val_str:
                    val_lower = val_str.lower()
                    # Find which group the expected unit belongs to
                    expected_group: Optional[frozenset[str]] = None
                    for grp in self._UNIT_CONFLICT_GROUPS:
                        if expected_unit in grp:
                            expected_group = grp
                            break
                    if expected_group is not None:
                        # Check if value contains a unit from a DIFFERENT group
                        for grp in self._UNIT_CONFLICT_GROUPS:
                            if grp is expected_group:
                                continue
                            for unit_tok in grp:
                                # Whole-word match to avoid «мл» matching «мл» within «мл/с»
                                if re.search(
                                    r"(?<![а-яёa-z])" + re.escape(unit_tok) + r"(?![а-яёa-z])",
                                    val_lower,
                                    re.IGNORECASE,
                                ):
                                    logger.info(
                                        "[UniversalGate] DROP attr=%s value=%r source=%s "
                                        "step=unit_sanity — unit '%s' conflicts with "
                                        "expected '%s' (field=%r)",
                                        v.attribute_id, v.value, v.source.value,
                                        unit_tok, expected_unit, t.name,
                                    )
                                    # Mark as dropped by NOT appending — break inner loops
                                    val_str = ""  # sentinel: signals unit-sanity DROP
                                    break
                            if not val_str:
                                break
                    if not val_str:
                        continue  # unit-sanity dropped

            # ── Step 4: Hallucination judge (LLM, selective) ─────────────────
            # Only non-authoritative sources that are NOT verbatim in grounding text.
            # EXEMPT: ТН ВЭД / ТНВЭД / ЕАЭС attributes — customs codes derived
            # deterministically from category and confirmed by existing TnvedSource /
            # inference-gate; LLM judge cannot verify them from title alone.
            _tname_l = (t.name or "").lower()
            _is_tnved_attr = (
                "тн вэд" in _tname_l or "тнвэд" in _tname_l or "еаэс" in _tname_l
            )
            if _is_tnved_attr:
                out.append(v)
                continue

            if v.source in self._UNVERIFIED_SOURCES and val_str:
                # Use per-value grounding (name + description + evidence snippet).
                grounding_text = grounding_text_v

                # ── Leaf-aware pre-check (free, deterministic) ────────────────
                # Map attribute to its strategy leaf and ask the leaf's own
                # evaluate_need_for_judgment before hitting the LLM.
                from app.services.enrichment.leaf_mapper import map_to_leaf, leaf_threshold
                leaf_cls = map_to_leaf(t)
                pre_verdict = leaf_cls.evaluate_need_for_judgment(
                    val_str, grounding_text or "", list(t.allowed_values or [])
                )
                logger.debug(
                    "[UniversalGate] attr=%s leaf=%s pre_verdict=%s source=%s",
                    v.attribute_id, leaf_cls.name, pre_verdict, v.source.value,
                )
                if pre_verdict == "REJECT":
                    logger.info(
                        "[UniversalGate] DROP attr=%s value=%r source=%s "
                        "step=leaf_pre_reject leaf=%s",
                        v.attribute_id, v.value, v.source.value, leaf_cls.name,
                    )
                    continue
                if pre_verdict == "ACCEPT":
                    # Leaf says value is clean — skip LLM judge.
                    out.append(v)
                    continue

                # Verbatim check: if value already present in grounding text → trust, skip LLM.
                from app.services.enrichment.sources.safe_enum_fill_source import _verbatim_check
                if grounding_text and _verbatim_check(val_str, grounding_text):
                    out.append(v)
                    continue
                # Queue for batched LLM judge (carry leaf_cls for per-leaf profile)
                slot = len(out)
                out.append(v)  # placeholder — may be removed after judge
                llm_candidates.append((slot, v, t, grounding_text, leaf_cls))
                continue

            out.append(v)

        # ── Batch LLM judge for non-authoritative non-verbatim values ─────────
        # llm_candidates items: (slot, v, t, grounding_text, leaf_cls)
        # Guard: UNIVERSAL_VERIFY_JUDGE_ENABLED (default OFF).
        # When OFF, candidates queued for judge are kept as-is (already in `out`).
        _judge_enabled = os.getenv(
            "UNIVERSAL_VERIFY_JUDGE_ENABLED", "0"
        ).strip().lower() in ("1", "true", "yes", "on")

        if llm_candidates and not _judge_enabled:
            logger.info(
                "[UniversalGate] judge SKIPPED (UNIVERSAL_VERIFY_JUDGE_ENABLED=0): "
                "%d candidates kept as-is",
                len(llm_candidates),
            )
            return out

        if llm_candidates and _judge_enabled:
            from app.services.enrichment.leaf_mapper import leaf_threshold
            llm_mgr = get_main_manager()
            judge = HallucinationJudge(llm_mgr)

            # Run all judge calls concurrently (one per candidate).
            async def _judge_one(
                slot: int,
                v: AttributeValue,
                t: TargetAttribute,
                grounding: str,
                leaf_cls: type,
            ) -> tuple[int, bool]:
                try:
                    # Build per-leaf judge profile (inherits leaf's custom rules).
                    profile = leaf_cls.get_judge_profile()
                    profile.goal = (
                        "Determine whether the extracted value is FACTUALLY WRONG or "
                        "implausible for THIS specific product model. A value that is "
                        "correct/plausible for the named product must be KEPT even if "
                        "not literally restated in the source text — world-knowledge "
                        "about a named model is acceptable. Flag as unsupported ONLY "
                        "when the value CONTRADICTS the source text/evidence, or is a "
                        "clearly wrong spec for this exact model."
                    )
                    # Universal gate baseline rules appended after leaf-specific ones.
                    profile.custom_rules.extend([
                        "Retract (is_supported_by_text=False) ONLY if the value "
                        "contradicts the product name/description/evidence OR is a "
                        "known-wrong spec for this exact model (e.g. wrong storage "
                        "size, wrong connectivity standard, wrong country of origin). "
                        "Do NOT retract correct technical facts about the named product "
                        "merely because they are absent from the short seller text.",
                        "Boolean fills ('Да'/'Нет') from SAFE_ENUM_FILL: retract only "
                        "if the feature is explicitly contradicted by the text or is "
                        "clearly inapplicable to this product type — do NOT retract "
                        "merely because the feature keyword is absent from the text.",
                    ])
                    verdict, _tokens = await judge.execute_audit(
                        text=grounding or context.product_name or "",
                        feature_name=t.name,
                        extracted_value=v.value,
                        profile=profile,
                    )
                    # Apply leaf-specific threshold via process_judgment.
                    judge_result = leaf_cls.process_judgment(verdict, _tokens)
                    keep = judge_result.is_success
                    threshold = leaf_threshold(leaf_cls)
                    if not keep:
                        logger.info(
                            "[UniversalGate] DROP attr=%s value=%r source=%s "
                            "step=hallucination_judge leaf=%s threshold=%s — "
                            "is_supported=%s violates=%s conf=%s analysis=%r",
                            v.attribute_id, v.value, v.source.value,
                            leaf_cls.name, threshold,
                            verdict.is_supported_by_text, verdict.violates_rules,
                            verdict.error_confidence, verdict.analysis[:100],
                        )
                    else:
                        logger.debug(
                            "[UniversalGate] KEEP attr=%s value=%r source=%s "
                            "leaf=%s threshold=%s conf=%s",
                            v.attribute_id, v.value, v.source.value,
                            leaf_cls.name, threshold, verdict.error_confidence,
                        )
                    return slot, keep
                except Exception as exc:
                    logger.warning(
                        "[UniversalGate] judge failed for attr=%s source=%s: %s — keeping",
                        v.attribute_id, v.source.value, exc,
                    )
                    return slot, True  # fail-open on judge error (keep)

            judgements: list[tuple[int, bool]] = await asyncio.gather(
                *[_judge_one(slot, v, t, gt, lc) for slot, v, t, gt, lc in llm_candidates]
            )

            # Build set of slots to drop
            drop_slots: set[int] = {slot for slot, keep in judgements if not keep}
            if drop_slots:
                out = [v for i, v in enumerate(out) if i not in drop_slots]

        return out

    async def _run_required_enum_mud_gate(
        self,
        resolved: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Adversarial mud-gate for REQUIRED enum fills from inference sources.

        Complements the Stage-4.9 inference gate (llm_knowledge / web_search /
        competitor_rag) by covering Source.DESCRIPTION and Source.VISION — the
        non-authoritative sources whose required-enum fills otherwise reach the
        output ungated (the DescriptionSource Пол=«Женский»-for-headphones leak).

        For each REQUIRED enum value from a gated source that is NOT verbatim-present
        in the product's own text (name + description = Gate A), run the shared
        adversarial verifier (Gate B). Retract every value the verifier does not
        CONFIRM (fail-closed: empty > wrong). Structural attrs (Тип 8229, Бренд
        31/85) and authoritative card sources are never touched.

        Gated behind REQUIRED_ENUM_MUD_GATE_ENABLED (default on).
        """
        if os.getenv("REQUIRED_ENUM_MUD_GATE_ENABLED", "true").strip().lower() not in (
            "1", "true", "yes", "on",
        ):
            return resolved

        from app.services.enrichment.sources.safe_enum_fill_source import (
            run_adversarial_verify,
            _verbatim_check,
        )

        # Structural required enums are derived (Тип from cat_key) or identity (Бренд),
        # never world-knowledge guesses — they have their own dedicated guards.
        # +22232 ТН ВЭД: customs codes follow deterministically from product
        # category (not a world-knowledge guess), so vision/LLM deriving one is
        # legitimate — never mud-gate it. Live-confirmed: the gate was retracting
        # a CORRECT ТН ВЭД from vision → required 85% instead of 95%.
        _EXEMPT_ATTR_IDS = {8229, 31, 85, 22232}
        # Non-authoritative sources NOT already covered by the Stage-4.9 inference gate.
        _GATED_SOURCES = {Source.DESCRIPTION, Source.VISION}

        targets_by_id = {t.id: t for t in targets}
        source_text = " ".join(
            s for s in (context.product_name, context.product_description) if s
        )

        proposals: list[tuple[int, str, str]] = []
        candidate_ids: set[int] = set()
        for v in resolved:
            t = targets_by_id.get(v.attribute_id)
            if t is None or not t.is_required or not t.allowed_values:
                continue
            _name_l = (t.name or "").lower()
            if (v.attribute_id in _EXEMPT_ATTR_IDS
                    or "тн вэд" in _name_l or "тнвэд" in _name_l or "еаэс" in _name_l):
                continue
            if v.source not in _GATED_SOURCES:
                continue
            if isinstance(v.value, list):
                continue  # scalar required enums only
            val_str = str(v.value).strip()
            if not val_str:
                continue
            # Gate A: value literally present in the product's own text → grounded, keep.
            if source_text and _verbatim_check(val_str, source_text):
                continue
            proposals.append((v.attribute_id, t.name, val_str))
            candidate_ids.add(v.attribute_id)

        if not proposals:
            return resolved

        logger.info(
            "[Pipeline] required-enum mud-gate: product=%s verifying %d ungrounded "
            "required-enum fill(s) from %s",
            context.product_id, len(proposals),
            ", ".join(sorted({
                v.source.value for v in resolved
                if v.attribute_id in candidate_ids and v.source in _GATED_SOURCES
            })),
        )

        try:
            confirmed_ids = await run_adversarial_verify(
                context, proposals, resolved_attrs=resolved,
            )
        except Exception as exc:
            logger.warning(
                "[Pipeline] required-enum mud-gate failed (fail-closed, retract all): %s",
                exc,
            )
            confirmed_ids = set()

        drop_ids = candidate_ids - confirmed_ids
        if not drop_ids:
            return resolved

        out: list[AttributeValue] = []
        for v in resolved:
            if (
                v.attribute_id in drop_ids
                and v.source in _GATED_SOURCES
                and not isinstance(v.value, list)
            ):
                logger.info(
                    "[Pipeline] required-enum mud-gate DROP: attr=%s value=%r source=%s "
                    "— adversarial retracted (empty>wrong)",
                    v.attribute_id, v.value, v.source.value,
                )
                continue
            out.append(v)
        return out

    def _finalize(
        self,
        all_values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Merge + strategy post-process + strategy validation. Used at every early-exit point."""
        # Brand-identity guard ДО merge: бренд — идентичность, его нельзя угадывать.
        # Дроп кандидатов на «Бренд» от guess-источников (vision/web_search/
        # llm_knowledge/competitor_rag): HUGO/LEGO/Великобритания на Nike/Levi's/
        # Adidas. Остаются только авторитетные (карточка/опис/IceCat/PDF/ТНВЭД), а
        # brand-from-name ниже заполнит опустевший/правильный таргет из имени.
        all_values = _apply_brand_source_guard(all_values, targets)

        # Гендер-гард ДО merge: генеральный, для всех источников (web_search/llm/
        # vision/cards). Отсекает гендерные «Пол»-значения, конфликтующие с именем
        # или навеянные только external-guess источниками при нейтральном имени.
        all_values = _apply_gender_guard(all_values, targets, context)

        # Цвет-гард ДО merge: цвет — per-SKU расцветка продавца, guess-источники
        # (web_search/vision/llm_knowledge/competitor_rag) гадают колорвей по вебу/фото —
        # дропаем (eg_importer: гадать колорвей нельзя, пусто честнее мусора).
        all_values = _apply_color_source_guard(all_values, targets)
        # Палитра-цвет ДО merge: донор (WbCard/ozon_card/llm) отдаёт мульти-цвет
        # (палитру расцветок модели), часто НЕ цвета этого SKU. Дропаем до merge,
        # чтобы одиночный grounded-цвет из названия («…чёрные»→чёрный) выиграл merge
        # и заполнился, а не был вытеснен высоко-conf палитрой. [[honest_gap_composition]]
        all_values = _drop_multivalue_color_premerge(all_values, targets)
        merged = self._merge(all_values)

        # Brand-from-name POST-merge: имя товара авторитетно для бренда. Заполняет
        # пустой «Бренд» / перезаписывает мусорный (чужой allowed-enum) ровно-одним
        # allowed-брендом, присутствующим в имени. Идёт ДО resolve_value_ids, чтобы
        # заполненный/перезаписанный бренд получил словарный value_id.
        merged = _apply_brand_from_name(
            merged, targets, context,
            brand_options_fn=lambda attr_id: self._strategy.brand_value_options(attr_id, context),
            brand_id_fn=lambda attr_id: self._strategy.brand_value_id_options(attr_id, context),
        )

        # Color-from-name: цвет per-SKU, надёжный источник — название («…чёрные»→чёрный).
        # Заполняет ПУСТОЙ «Цвет товара» одним словарным цветом из имени (доноры дают
        # палитру чужой расцветки, уже дропнутую). ДО resolve_value_ids — получит value_id.
        merged = _apply_color_from_name(merged, targets, context)

        # Strategy post-processing FIRST (добавляет CategoryDefaults + cross-fills с source=DESCRIPTION).
        # Должно идти ДО resolve_value_ids, иначе свежедобавленные AVs не получат value_id.
        merged = self._strategy.post_process_values(merged, targets, context)

        # Привязываем словарные value_id(s) (Ozon) или no-op для других стратегий.
        # Прогоняем ВСЕ AVs включая только что добавленные CategoryDefaults — иначе они
        # уходят в БД с value_id=NULL и проваливают value_id_resolution метрику.
        merged = [self._strategy.resolve_value_ids(v, context) for v in merged]

        # Strategy validation — drop or normalise individual values
        filtered: list[AttributeValue] = []
        for v in merged:
            target = next((t for t in targets if t.id == v.attribute_id), None)
            if target:
                result = self._strategy.validate_value(target, v.value, context)
                if result.is_valid:
                    if result.normalized_value is not None:
                        v.value = result.normalized_value
                    filtered.append(v)
                # else: drop value (validation failed)
            else:
                filtered.append(v)
        return filtered

    async def _run_rag_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить CompetitorRagSource + его судью. Ошибки не прерывают pipeline."""
        if self._competitor_rag is None:
            return []
        judge_wrapper = self._judges.get(Source.COMPETITOR_RAG)
        try:
            extracted = await self._competitor_rag.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] competitor_rag source failed: %s", e, exc_info=True)
            return []

        if judge_wrapper is None:
            return extracted

        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] competitor_rag judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_pdf_datasheet_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить PdfDatasheetSource + judge. Ошибки не прерывают pipeline."""
        if self._pdf_datasheet is None:
            return []
        judge_wrapper = self._judges.get(Source.PDF_DATASHEET)
        try:
            extracted = await self._pdf_datasheet.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] pdf_datasheet source failed: %s", e, exc_info=True)
            return []

        if judge_wrapper is None:
            return extracted

        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] pdf_datasheet judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_wb_card_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить WbCardSource + его судью. Ошибки не прерывают pipeline."""
        if self._wb_card is None:
            return []
        judge_wrapper = self._judges.get(Source.WB_CARD)
        try:
            extracted = await self._wb_card.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] wb_card source failed: %s", e, exc_info=True)
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] wb_card judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_ugc_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить UgcSource + его судью. Ошибки не прерывают pipeline."""
        if self._ugc is None:
            return []
        judge_wrapper = self._judges.get(Source.UGC)
        try:
            extracted = await self._ugc.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] ugc source failed: %s", e, exc_info=True)
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] ugc judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_scrapfly_ozon_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
        ozon_card_obtained: bool = False,
    ) -> list[AttributeValue]:
        """Run ScrapflyOzonSource — last-resort Ozon card gap-fill via Scrapfly.

        Errors never interrupt the pipeline (return []).
        Gate (a): ozon_card_obtained=True → source returns [] immediately.
        Gate (b): no remaining gaps → source returns [] immediately.
        Gate (c): SCRAPFLY_OZON_FALLBACK_ENABLED env flag → off by default.
        """
        if self._scrapfly_ozon is None:
            return []
        if not self._scrapfly_ozon.is_applicable(context, targets[0] if targets else None):
            return []
        judge_wrapper = self._judges.get(Source.OZON_CARD)
        try:
            extracted = await self._scrapfly_ozon.extract(
                context,
                targets,
                already_filled=already_filled,
                ozon_card_obtained=ozon_card_obtained,
            )
        except Exception as e:
            logger.warning("[Pipeline] scrapfly_ozon source failed: %s", e, exc_info=True)
            return []
        if not extracted:
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] scrapfly_ozon judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_marketplace_router_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Run MarketplaceRouter — полиморфный пул маркетплейсов (Stage 4.65).

        Каждое значение судится по своему source через self._judges.
        Ошибки не прерывают пайплайн (возвращают []).
        """
        if not targets:
            return []
        filled_ids: set[int] = {v.attribute_id for v in (already_filled or [])}
        try:
            raw = await self._marketplace_router.fill_gaps(
                context, targets, already_filled_ids=filled_ids
            )
        except Exception as exc:
            logger.warning(
                "[Pipeline] marketplace_router failed: %s", exc, exc_info=True
            )
            return []
        if not raw:
            return []
        results: list[AttributeValue] = []
        for value in raw:
            judge_wrapper = self._judges.get(value.source)
            if judge_wrapper is None:
                # Нет зарегистрированного судьи — пропускаем as-is
                results.append(value)
                continue
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as exc:
                logger.warning(
                    "[Pipeline] marketplace_router judge failed for attr %s (source=%s): %s",
                    value.attribute_id, value.source, exc,
                )
        return results

    async def _run_lamoda_scrapfly_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Run LamodaScrapflySource — last-resort clothing attribute gap-fill via Lamoda+Scrapfly.

        Errors never interrupt the pipeline (return []).
        LLM gate inside source: non-clothing products return [] immediately (0 credits).
        LAMODA_SCRAPFLY_ENABLED env flag → off by default.
        """
        if self._lamoda_scrapfly is None:
            return []
        if not targets:
            return []
        if not self._lamoda_scrapfly.is_applicable(context, targets[0]):
            return []
        judge_wrapper = self._judges.get(Source.LAMODA)
        try:
            extracted = await self._lamoda_scrapfly.extract(
                context,
                targets,
                already_filled=already_filled,
            )
        except Exception as e:
            logger.warning("[Pipeline] lamoda_scrapfly source failed: %s", e, exc_info=True)
            return []
        if not extracted:
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] lamoda_scrapfly judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_barcode_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
        source_text: Optional[str] = None,
    ) -> list[AttributeValue]:
        """Stage 0.46: BarcodeSource — verbatim EAN/barcode extractor.

        Zero cost (no LLM, no network).  Deterministic checksum validation.
        Errors do not interrupt the pipeline.
        """
        try:
            extracted = await self._barcode.extract(
                context, targets, already_filled=already_filled,
                source_text=source_text,
            )
        except Exception as e:
            logger.warning("[Pipeline] barcode_source failed: %s", e, exc_info=True)
            return []
        # BarcodeJudge is deterministic — always validate
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                valid = await self._barcode.get_judge().validate(value, context)
                if valid:
                    results.append(value)
            except Exception as e:
                logger.warning(
                    "[Pipeline] barcode judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_tnved_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить TnvedSource (per-category кэш). Ошибки не прерывают pipeline."""
        try:
            extracted = await self._tnved.extract(context, targets, already_filled=already_filled)
        except Exception as e:
            logger.warning("[Pipeline] tnved source failed: %s", e, exc_info=True)
            return []
        # TnvedJudge — детерминированный (10 цифр), всегда применяем
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                valid = await self._tnved.get_judge().validate(value, context)
                if valid:
                    results.append(value)
            except Exception as e:
                logger.warning(
                    "[Pipeline] tnved judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _ozon_api_llm_pick(
        self,
        context: ExtractionContext,
        target: TargetAttribute,
        candidates: list[dict],
    ) -> Optional[dict]:
        """LLM picks the single best authoritative value from the live Ozon list.

        candidates: [{"id": int, "value": str}] from the Ozon API. Returns the
        chosen {"id","value"} or None (nothing fits / LLM error). The choice is
        CONSTRAINED to the real category list, so the result always carries a
        valid value_id — unlike a free LLM guess.
        """
        cands = candidates[:150]
        if not cands:
            return None
        listing = "\n".join(f"{i}. {c['value']}" for i, c in enumerate(cands))
        cat_hint = " / ".join(context.category_path) if context.category_path else ""

        class _PickResponse(BaseModel):
            index: int = Field(..., description="индекс выбранного значения из списка; -1 если ничего не подходит")

        user_text = (
            f'Товар: "{context.product_name}".\n'
            f'Поле для заполнения: "{target.name}".\n'
            + (f'Категория: {cat_hint}.\n' if cat_hint else "")
            + "Выбери из списка ОДНО значение, наиболее точно подходящее этому товару.\n"
            "Верни ТОЛЬКО индекс (число). ВАЖНО: если ты не уверен или ни одно "
            "значение точно не подходит — верни -1 (пустое поле честнее неверного).\n\n"
            f"Список:\n{listing}"
        )
        try:
            from app.services.providers.factory import get_main_manager
            llm = get_main_manager()
            parsed, _ = await asyncio.wait_for(
                llm.structured_request(
                    system_prompt=(
                        "Ты эксперт по классификации товаров для маркетплейса Ozon. "
                        "Выбираешь ровно одно значение из списка по индексу, и только "
                        "когда уверен; при сомнении возвращаешь -1."
                    ),
                    user_text=user_text,
                    response_model=_PickResponse,
                ),
                timeout=20,
            )
        except Exception as exc:
            logger.warning("[OzonApiResolve] LLM pick failed for attr %s: %s", target.id, exc)
            return None
        if parsed is None:
            return None
        idx = parsed.index
        if 0 <= idx < len(cands):
            return cands[idx]
        return None

    async def _apply_ozon_api_resolve(
        self,
        resolved: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Authoritative ТН ВЭД / Тип resolution via the live Ozon Seller API.

        The local dictionary cache is stale for these dict-backed fields (ТН ВЭД
        ships a generic 66-code sample; «Тип» ships 4 unrelated values — the same
        on every category), so values never resolve to a real value_id and the
        ТН ВЭД code is otherwise an LLM guess that Ozon's category list rejects.
        For each ТН ВЭД / Тип target still lacking a value_id:
          1. search_value(proposed value | category leaf) — cheap exact/partial.
          2. else LLM-pick from the authoritative live category list.
        Sets value + value_id in place (or appends when the field was empty).
        Creds-gated internally; no-op without OZON_CLIENT_ID/OZON_API_KEY.
        """
        from app import config as _cfg
        if not _cfg.OZON_API_RESOLVE_ENABLED:
            return resolved
        cat_id = context.category_id
        type_id = context.ozon_type_id
        if cat_id is None or type_id is None:
            return resolved
        from app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup import (
            search_value, list_values,
        )

        leaf = (context.category_path[-1] if context.category_path else "") or ""
        by_attr: dict[int, AttributeValue] = {}
        for v in resolved:
            by_attr.setdefault(v.attribute_id, v)

        def _apply_hit(t: TargetAttribute, cur: Optional[AttributeValue], hit: dict) -> None:
            if cur is not None:
                logger.info(
                    "[OzonApiResolve] attr=%s '%s': %r → %r (value_id=%s)",
                    t.id, t.name, str(cur.value)[:40], hit["value"][:40], hit["id"],
                )
                cur.value = hit["value"]
                cur.value_id = hit["id"]
                cur.source = Source.OZON_CARD
                cur.evidence = "ozon_api: authoritative category value"
            else:
                logger.info(
                    "[OzonApiResolve] attr=%s '%s': EMPTY → %r (value_id=%s)",
                    t.id, t.name, hit["value"][:40], hit["id"],
                )
                resolved.append(AttributeValue(
                    attribute_id=t.id,
                    value=hit["value"],
                    confidence=0.95,
                    source=Source.OZON_CARD,
                    value_id=hit["id"],
                    evidence="ozon_api: authoritative category value",
                ))

        # Generalised to ALL required targets still lacking an authoritative
        # value_id — not only ТН ВЭД / Тип. Closes niche required holes (Класс
        # опасности, etc.) via the same authoritative live-list mechanism.
        for t in targets:
            # Цвет — per-SKU расцветка продавца: НЕ дорезолвируем из словаря категории.
            # LLM-pick «белый» из 651 цвета на пустом таргете = гадание колорвея
            # (eg_importer: «гадать колорвей нельзя, пусто честнее мусора»). Цвет из
            # имени уже получил value_id через resolve_value_ids — api-resolve ему не нужен.
            if _is_color_target(t):
                continue
            if not t.is_required:
                continue
            cur = by_attr.get(t.id)
            if cur is not None and cur.value_id is not None:
                continue  # already resolved to an authoritative value_id
            name_l = (t.name or "").lower().strip()

            # Open-vocabulary required fields: never LLM-pick from thousands of
            # values. Бренд → deterministic "Нет бренда" (extractors found none →
            # genuinely brandless). The rest are left to their free-text levers.
            if name_l in _OZON_OPEN_VOCAB_REQUIRED:
                if name_l == "бренд":
                    cur_brand = (
                        str(cur.value).strip()
                        if cur is not None and not isinstance(cur.value, list)
                        else ""
                    )
                    if cur_brand:
                        # An extracted brand (e.g. "Xiaomi") that just lacks a
                        # value_id — resolve IT to the Ozon brand id. NEVER
                        # overwrite a real brand with "Нет бренда".
                        hit = await search_value(cat_id, type_id, t.id, cur_brand)
                        if hit is not None:
                            _apply_hit(t, cur, hit)
                        # no Ozon match → keep the extracted brand as free-text
                    else:
                        # Genuinely brandless → Ozon's standard "Нет бренда".
                        hit = await search_value(cat_id, type_id, t.id, "Нет бренда")
                        if hit is not None:
                            _apply_hit(t, cur, hit)
                continue

            # 1) cheap search by proposed value (or, for empty Тип, category leaf)
            hit = None
            query = None
            if cur is not None and not isinstance(cur.value, list) and str(cur.value).strip():
                query = str(cur.value).strip()
            elif name_l == "тип" and leaf:
                query = leaf
            if query:
                hit = await search_value(cat_id, type_id, t.id, query)

            # 2) constrained LLM pick from the authoritative live list. Empty
            #    list ⇒ field is free-text/not-dict-backed (404) ⇒ skip. The pick
            #    is conservative (returns -1 → no fill) so an ill-fitting required
            #    field stays honestly empty rather than getting a guessed value.
            if hit is None:
                cands = await list_values(cat_id, type_id, t.id)
                if cands:
                    hit = await self._ozon_api_llm_pick(context, t, cands)

            if hit is not None:
                _apply_hit(t, cur, hit)
        return resolved

    async def _apply_wb_api_resolve(
        self,
        resolved: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Authoritative WB value resolution via the live WB Content API.

        WB analogue of _apply_ozon_api_resolve, but WB takes VALUE STRINGS
        validated against its dictionaries (colors/countries/seasons/ТН ВЭД) —
        not numeric value_ids. For each FILLED WB target that is a dictionary-
        backed charc, match the extracted value to the canonical WB dictionary
        string (overwrite value; keep the WB id when present, e.g. country).
        resolve_wb_value short-circuits to None for non-dictionary charcs, so
        calling it per value is cheap. Subject id comes from context.category_id
        (the WB eval puts the WB subjectID there).
        """
        subject_id = context.category_id
        resolve = getattr(self._strategy, "resolve_wb_value", None)
        if subject_id is None or resolve is None:
            return resolved
        targets_by_id = {t.id: t for t in targets}
        for v in resolved:
            t = targets_by_id.get(v.attribute_id)
            if t is None:
                continue
            # WB collections (maxCount>1, e.g. Цвет) arrive as a Python list —
            # resolve each element; scalars resolve as a single-item list.
            is_list = isinstance(v.value, list)
            items = v.value if is_list else [v.value]
            new_items: list = []
            resolved_id = v.value_id
            for item in items:
                val = str(item).strip()
                if not val:
                    new_items.append(item)
                    continue
                try:
                    hit = await resolve(int(subject_id), t.name, val)
                except Exception as exc:
                    logger.warning("[WbApiResolve] resolve failed attr=%s: %s", t.id, exc)
                    new_items.append(item)
                    continue
                if hit and hit.get("value"):
                    canonical = hit["value"]
                    if canonical != val:
                        logger.info(
                            "[WbApiResolve] attr=%s '%s': %r → %r (id=%s)",
                            t.id, t.name, val[:40], canonical[:40], hit.get("id"),
                        )
                    new_items.append(canonical)
                    # Keep the WB id even on an exact-string match (e.g. country
                    # 'Германия'→'Германия' id=15000096): the id is authoritative,
                    # not the string equality.
                    if resolved_id is None and hit.get("id") is not None:
                        resolved_id = hit["id"]
                else:
                    new_items.append(item)
            v.value = new_items if is_list else new_items[0]
            if resolved_id is not None:
                v.value_id = resolved_id
        return resolved

    async def _run_safe_enum_fill_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Stage 4.8: SafeEnumFillSource — gated LLM fill for short optional enums.

        Disabled by default (SAFE_LLM_ENUM_FILL_ENABLED flag).
        Gate A: verbatim value in web_search summary (zero extra LLM calls).
        Gate B: adversarial verifier LLM call (batched per product, 1 call).
        Errors do not interrupt the pipeline (returns []).
        """
        from app.services.enrichment.sources.safe_enum_fill_source import (
            SafeEnumFillSource,
            _is_short_enum,
        )

        # Restrict to short-enum optional targets only
        short_enum_targets = [t for t in targets if _is_short_enum(t)]
        if not short_enum_targets:
            return []

        # Gate A needs the web_search summary for this product.
        # WebSearchSource caches it in _summary_cache; extract from the source instance.
        ws_source = self._sources.get(Source.WEB_SEARCH)
        source_text: Optional[str] = None
        if ws_source is not None:
            summary_cache = getattr(ws_source, "_summary_cache", {})
            source_text = summary_cache.get(context.product_id)

        source = SafeEnumFillSource(llm_manager=None)  # uses default get_main_manager()
        try:
            results = await source.extract(
                context,
                short_enum_targets,
                already_filled=already_filled,
                source_text=source_text,
            )
        except Exception as exc:
            logger.warning(
                "[Pipeline] safe_enum_fill stage failed: %s", exc, exc_info=True
            )
            return []

        return results

    async def _run_llm_knowledge_adversarial_pass(
        self,
        all_values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
        resolved_attrs: list[AttributeValue],
    ) -> list[AttributeValue]:
        """Stage 4.9: deterministic corroboration gate (spec-class) + LLM Gate B (non-spec).

        Two-track approach based on attribute class:

        TRACK A — OBJECTIVE-SPEC attrs (_is_objective_spec_attr == True):
          Material, composition, boolean feature flags (True Wireless),
          numeric measurements, audio/connectivity configs, etc.
          Gate: DETERMINISTIC source-corroboration.
          KEEP the fill ONLY if at least one AUTHORITATIVE source
          (_AUTHORITATIVE_SOURCES: wb_card, ozon_card, icecat, pdf_datasheet,
          description) independently produced the SAME (attribute_id, normalised
          value) for this product.
          Two guess-prone sources (llm_knowledge + web_search) agreeing is NOT
          corroboration.  If not corroborated → DROP (empty > wrong).
          No LLM call needed — deterministic, zero extra cost.

        TRACK B — non-spec attrs (_is_objective_spec_attr == False):
          Lifestyle enums (style, occasion), color names inferred from description,
          OS family, etc.  LLM Gate B: «is this correct for THIS exact product?»
          Retracted fills are DROPPED (empty > wrong). Errors retract all (fail-closed).

        Both tracks only apply to Source.LLM_KNOWLEDGE and Source.WEB_SEARCH fills
        that are NOT verbatim-anchored (safe_enum:verbatim_gate — already passed Gate A).

        Flag: LLM_KNOWLEDGE_ADVERSARIAL_ENABLED (name kept for back-compat; now gates
        both llm_knowledge and web_search fills).
        """
        from app.services.enrichment.sources.safe_enum_fill_source import (
            run_adversarial_verify,
        )

        # Sources that require the gate.
        # COMPETITOR_RAG is consensus from similar products (NOT ground-truth per-product data).
        # It must go through Track A for objective-spec attrs — "most speakers are stereo" would
        # otherwise corroborate a wrong Звуковая схема=2.0 without any ground-truth evidence.
        # For non-spec attrs it goes through Track B LLM Gate B like other guess-prone sources.
        _ADVERSARIAL_SOURCES = {Source.LLM_KNOWLEDGE, Source.WEB_SEARCH, Source.COMPETITOR_RAG}

        target_by_id: dict[int, TargetAttribute] = {t.id: t for t in targets}

        # Build a lookup: authoritative fills already present in all_values,
        # keyed by (attribute_id, normalised_value).  Used for Track A corroboration.
        authoritative_fills: set[tuple[int, str]] = set()
        for v in all_values:
            if v.source in _AUTHORITATIVE_SOURCES:
                norm = _normalize_for_corroboration(v.value)
                authoritative_fills.add((v.attribute_id, norm))

        # Partition fills.
        keep_as_is: list[AttributeValue] = []          # passthrough (other sources / verbatim)
        spec_pending: list[AttributeValue] = []         # Track A: deterministic corroboration
        non_spec_pending: list[AttributeValue] = []     # Track B: LLM Gate B

        for v in all_values:
            if v.source not in _ADVERSARIAL_SOURCES:
                keep_as_is.append(v)
                continue
            # Verbatim-anchored fills already passed Gate A — keep as-is.
            if (v.evidence or "").startswith("safe_enum:verbatim_gate"):
                keep_as_is.append(v)
                continue
            # ── Gate A (WEB_SEARCH only): self-consistency — value token must appear
            # in its own evidence snippet (the REAL fetched page text).
            # llm_knowledge evidence is LLM-self-generated, so this check is only
            # meaningful for web_search where evidence is the actual page snippet.
            if v.source == Source.WEB_SEARCH:
                if isinstance(v.value, list):
                    # LIST-valued fill: check each element; keep only grounded ones.
                    grounded_elements = _filter_list_value_by_evidence(v.value, v.evidence)
                    if grounded_elements is None:
                        # All elements failed → drop the whole fill.
                        logger.info(
                            "[Pipeline] web_search Gate A DROP list (all elements ungrounded): "
                            "attr=%s value=%r evidence=%r",
                            v.attribute_id, v.value, (v.evidence or "")[:120],
                        )
                        continue
                    if len(grounded_elements) < len(v.value):
                        # Some elements dropped → replace value with filtered subset.
                        dropped = [el for el in v.value if el not in grounded_elements]
                        logger.info(
                            "[Pipeline] web_search Gate A PARTIAL DROP list: "
                            "attr=%s dropped=%r kept=%r evidence=%r",
                            v.attribute_id, dropped, grounded_elements, (v.evidence or "")[:120],
                        )
                        v = v.model_copy(update={"value": grounded_elements})
                elif not _web_search_grounded_in_evidence(str(v.value), v.evidence):
                    logger.info(
                        "[Pipeline] web_search Gate A DROP (value not in evidence): "
                        "attr=%s value=%r evidence=%r — value token absent from own evidence",
                        v.attribute_id, v.value, (v.evidence or "")[:120],
                    )
                    continue  # drop: value contradicts (or is absent from) its own evidence
            t = target_by_id.get(v.attribute_id)
            if t is not None and _is_objective_spec_attr(t):
                spec_pending.append(v)
            else:
                non_spec_pending.append(v)

        result: list[AttributeValue] = list(keep_as_is)

        # ── Track A: DETERMINISTIC corroboration for objective-spec attrs ────────
        for v in spec_pending:
            norm_val = _normalize_for_corroboration(v.value)
            corroborated = (v.attribute_id, norm_val) in authoritative_fills
            if corroborated:
                logger.info(
                    "[Pipeline] spec-corroboration PASS: attr=%s value=%r source=%s "
                    "— matched by authoritative source",
                    v.attribute_id, v.value, v.source.value,
                )
                result.append(v)
            else:
                logger.info(
                    "[Pipeline] spec-corroboration DROP (no authoritative match): "
                    "attr=%s value=%r source=%s norm=%r "
                    "— empty>wrong for objective-spec attr",
                    v.attribute_id, v.value, v.source.value, norm_val,
                )

        # ── Track B: LLM Gate B for non-spec attrs ───────────────────────────────
        if not non_spec_pending:
            return result

        proposals: list[tuple[int, str, str]] = []
        for v in non_spec_pending:
            t = target_by_id.get(v.attribute_id)
            attr_name = t.name if t else str(v.attribute_id)
            proposals.append((v.attribute_id, attr_name, str(v.value)))

        logger.info(
            "[Pipeline] inference adversarial pass (Gate B, non-spec): "
            "product=%s, %d fills to verify (sources: %s)",
            context.product_id, len(proposals),
            ", ".join(sorted({str(v.source.value) for v in non_spec_pending})),
        )

        try:
            confirmed_ids = await run_adversarial_verify(
                context,
                proposals,
                resolved_attrs=resolved_attrs,
            )
        except Exception as exc:
            logger.warning(
                "[Pipeline] inference adversarial pass (Gate B) failed (all retracted): %s",
                exc,
            )
            confirmed_ids = set()  # fail-closed: retract all on error

        for v in non_spec_pending:
            if v.attribute_id in confirmed_ids:
                logger.info(
                    "[Pipeline] inference adversarial CONFIRMED (Gate B): "
                    "attr=%s value=%r source=%s",
                    v.attribute_id, v.value, v.source.value,
                )
                result.append(v)
            else:
                logger.info(
                    "[Pipeline] inference adversarial RETRACTED (Gate B, MUD): "
                    "attr=%s value=%r source=%s — not verifiable for this product",
                    v.attribute_id, v.value, v.source.value,
                )

        return result

    async def _run_ozon_card_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить OzonCardSource + его судью. Ошибки не прерывают pipeline."""
        if self._ozon_card is None:
            return []
        judge_wrapper = self._judges.get(Source.OZON_CARD)
        try:
            extracted = await self._ozon_card.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] ozon_card source failed: %s", e, exc_info=True)
            return []

        if judge_wrapper is None:
            return extracted

        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] ozon_card judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_yandex_market_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить YandexMarketSource + его судью. Ошибки не прерывают pipeline."""
        if self._yandex_market is None:
            return []
        # YandexMarketSource эмитит Source.OZON_CARD — используем тот же judge.
        judge_wrapper = self._judges.get(Source.OZON_CARD)
        try:
            extracted = await self._yandex_market.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] yandex_market source failed: %s", e, exc_info=True)
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] yandex_market judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_image_card_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Reverse-image card search via Serper /lens.

        Only fires when context.image_urls is non-empty (gated at call site).
        find_matching_card returns None if no candidate passes the gate → [].
        On success: uses the found card's URL to run the appropriate card source
        (ozon/wb/yandex_market) for full attr extraction — BUT we don't have a
        card-fetch path here. Instead, we emit a minimal AttributeValue set using
        the GateResult metadata (title, visual_score) as evidence, with confidence
        capped by IMAGE_CARD_CONF_CAP. Full attr extraction from the image-found
        card is left for a future iteration; this stage proves the integration
        path end-to-end.

        In practice: finds the best matching card URL by image; the URL is logged
        as a strong signal for downstream use. Currently returns [] (the gate runs
        but we don't yet emit fills) — this is intentionally conservative:
        the gate-validated URL can be used to prime ozon/wb/ym sources in future.
        The stage is wired and observable in logs now.
        """
        if not context.image_urls:
            return []
        image_url = context.image_urls[0]
        try:
            from app.services.enrichment.sources.wb_card_source import _target_type_lemma
            target_type = _target_type_lemma(
                context.product_name or "",
                context.category_path[-1] if context.category_path else None,
            )
        except Exception:
            target_type = None

        try:
            result = await _image_find_matching_card(
                image_url,
                context.product_name or "",
                our_brand=context.brand,
                target_type=target_type,
            )
        except Exception as exc:
            logger.warning("[Pipeline] image_card_stage error: %s", exc, exc_info=True)
            return []

        if result is None:
            logger.info("[Pipeline] image_card_stage: no gate-passing candidate for %s",
                        image_url[:80])
            return []

        logger.info(
            "[Pipeline] image_card_stage: gate-passed candidate url=%s conf=%.2f — "
            "card URL validated; full attr-extraction from image-found card is a future step",
            result.candidate.url[:80], result.match_confidence,
        )
        # Conservative: return [] — we've validated the card exists but don't emit fills yet.
        # The URL is available in logs; wiring actual attribute extraction from the found URL
        # (fetch + parse + map) is straightforward once this stage is proven in live testing.
        return []

    async def _run_books_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Run BooksSource (Stage 0.56) + WB_CARD judge. Errors don't interrupt pipeline.

        ISBN-gated inside BooksSource.extract(): fires only when context.ean is a
        book ISBN-13 (prefix 978/979). Non-book products return [] immediately.
        """
        if self._books is None:
            return []
        judge_wrapper = self._judges.get(Source.WB_CARD)
        try:
            extracted = await self._books.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] books source failed: %s", e, exc_info=True)
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] books judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_regard_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Run RegardSource (Stage 0.57) + WB_CARD judge. Errors don't interrupt pipeline."""
        if self._regard is None:
            return []
        judge_wrapper = self._judges.get(Source.WB_CARD)
        try:
            extracted = await self._regard.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] regard source failed: %s", e, exc_info=True)
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] regard judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_bestbuy_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Run BestBuySource (Stage 0.58) + WB_CARD judge. Errors don't interrupt pipeline."""
        if self._bestbuy is None:
            return []
        judge_wrapper = self._judges.get(Source.WB_CARD)
        try:
            extracted = await self._bestbuy.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] bestbuy source failed: %s", e, exc_info=True)
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] bestbuy judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_onliner_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Run OnlinerSource (Stage 0.59) + WB_CARD judge. Errors don't interrupt pipeline."""
        if self._onliner is None:
            return []
        judge_wrapper = self._judges.get(Source.WB_CARD)
        try:
            extracted = await self._onliner.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] onliner source failed: %s", e, exc_info=True)
            return []
        if judge_wrapper is None:
            return extracted
        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] onliner judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_icecat_stage(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Запустить IceCatSource + его судью. Ошибки не прерывают pipeline."""
        if self._icecat is None:
            return []
        judge_wrapper = self._judges.get(Source.ICECAT)
        try:
            extracted = await self._icecat.extract(
                context, targets, already_filled=already_filled
            )
        except Exception as e:
            logger.warning("[Pipeline] icecat source failed: %s", e, exc_info=True)
            return []

        if judge_wrapper is None:
            return extracted

        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] icecat judge failed for attr %s: %s",
                    value.attribute_id, e,
                )
        return results

    async def _run_stage(
        self,
        src: Source,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        """Run one source + judges, return validated values."""
        source = self._sources[src]
        judge_wrapper = self._judges[src]
        try:
            extracted = await source.extract(context, targets, already_filled=already_filled)
        except Exception as e:
            logger.warning("[Pipeline] %s source failed: %s", src.value, e, exc_info=True)
            return []

        results: list[AttributeValue] = []
        for value in extracted:
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    results.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] %s judge failed for attr %s: %s",
                    src.value, value.attribute_id, e,
                )
        return results

    async def _run_finishing(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        all_values: list[AttributeValue],
    ) -> list[AttributeValue]:
        """Stage 5: focused re-extraction for empty required attributes.

        Финишный проход переиспользует те же sources, но напрямую (минуя _run_stage),
        поэтому судьи здесь НЕ применяются автоматически. Прогоняем результат через
        тех же per-source судей, что и обычные стадии (_run_stage) — иначе finishing
        стал бы backdoor'ом, воскрешающим judge-rejected значения без валидации.
        """
        recovered = await self._finisher.extract_missing(context, targets, all_values)
        if not recovered:
            return []

        validated: list[AttributeValue] = []
        for value in recovered:
            judge_wrapper = self._judges.get(value.source)
            if judge_wrapper is None:
                validated.append(value)
                continue
            try:
                judged = await judge_wrapper.maybe_validate(value, context)
                if judged is not None:
                    validated.append(judged)
            except Exception as e:
                logger.warning(
                    "[Pipeline] finishing judge failed for attr %s (source=%s): %s",
                    value.attribute_id, value.source.value, e,
                )
        return validated

    def _remaining_targets(
        self, all_targets: list[TargetAttribute], collected: list[AttributeValue]
    ) -> list[TargetAttribute]:
        """Targets which don't yet have a confident value."""
        confident_attr_ids = {v.attribute_id for v in collected if v.is_confident()}
        return [t for t in all_targets if t.id not in confident_attr_ids]

    def _merge_high_conf(
        self,
        existing: list[AttributeValue],
        new_avs: list[AttributeValue],
        threshold: float = 0.85,
    ) -> list[AttributeValue]:
        """Обновляет список already_filled: добавляет/заменяет AVs с confidence ≥ threshold.

        Используется для передачи skip-filled контекста следующему source в pipeline.
        """
        by_id: dict[int, AttributeValue] = {v.attribute_id: v for v in existing}
        for v in new_avs:
            if v.confidence < threshold:
                continue
            prev = by_id.get(v.attribute_id)
            if prev is None or v.confidence > prev.confidence:
                by_id[v.attribute_id] = v
        return list(by_id.values())

    def _merge(self, all_values: list[AttributeValue]) -> list[AttributeValue]:
        """Per attribute_id, pick highest confidence; tie-break by SOURCE_PRIORITY.

        Ensemble voting: если ≥2 разных source выдали одинаковое
        normalized value на тот же attribute_id — bump confidence
        +0.10 (cap 0.97). Cross-source agreement = сильный сигнал
        достоверности (LLM сказал, web search подтвердил, etc).
        """
        # Step 1: count unique sources per (attribute_id, normalized element).
        # Для коллекций ключуем ПОЭЛЕМЕНТНО (а не по str(list)), чтобы consensus
        # и дедуп работали по отдельным элементам, а не по строке всего списка.
        sources_per_value: dict[tuple[int, str], set] = {}
        for v in all_values:
            for norm_value in _norm_elements(v.value):
                key = (v.attribute_id, norm_value)
                sources_per_value.setdefault(key, set()).add(v.source)

        # Step 2: apply consensus bonus. Для скаляра — по значению; для коллекции
        # — если ХОТЯ БЫ один элемент подтверждён ≥2 источниками.
        boosted: list[AttributeValue] = []
        for v in all_values:
            n_sources = max(
                (len(sources_per_value[(v.attribute_id, ev)]) for ev in _norm_elements(v.value)),
                default=0,
            )
            if n_sources >= 2 and v.confidence < 0.97:
                new_conf = min(0.97, v.confidence + 0.10)
                boosted.append(v.model_copy(update={"confidence": new_conf}))
            else:
                boosted.append(v)

        # Step 3: per attribute_id.
        #  - Коллекционные (is_collection или value-список): UNION дедуплицированных
        #    элементов всех судьёй-прошедших/уверенных источников (multi-value
        #    значения теряться не должны — Особенности/Декор и т.п.).
        #  - Скалярные: highest-conf wins с card-protection band (_merge_winner),
        #    поведение БЕЗ изменений.
        by_id: dict[int, AttributeValue] = {}
        for v in boosted:
            existing = by_id.get(v.attribute_id)
            if existing is None:
                by_id[v.attribute_id] = v
                continue
            if _is_collection_value(v) or _is_collection_value(existing):
                by_id[v.attribute_id] = _merge_collection(existing, v)
            else:
                by_id[v.attribute_id] = _merge_winner(v, existing)
        return list(by_id.values())
