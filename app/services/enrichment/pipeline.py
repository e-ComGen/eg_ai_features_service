"""PipelineOrchestrator — sequential cost-aware extraction.

Главный entry-point для enrichment. Объединяет все sources, judges и intelligence
в один flow с early-exit (если все targets заполнены — останавливаемся) и
cost gating (CostPredictor перед expensive web search).

Spec: docs/architecture/pipeline.md, section "PipelineOrchestrator".
"""
import logging
import re
from typing import Optional

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
    IceCatSource,
    PdfDatasheetSource,
    OzonCardSource,
    WbCardSource,
    UgcSource,
    TnvedSource,
)
from app.services.enrichment.sources.ozon_card_source import (
    _extract_gender_signal,
    _is_gender_target_name,
)
from app.services.enrichment.intelligence import LlmClassifier, CostPredictor
from app.services.providers.factory import get_main_manager
from app.services.enrichment.confidence_aware_judge import ConfidenceAwareJudgeWrapper
from app.services.enrichment.strategies.base import MarketplaceStrategy
from app.services.enrichment.strategies.default_strategy import DefaultStrategy
from app.services.enrichment.finishing import FinishingExtractor

logger = logging.getLogger(__name__)

# Card-protection в финальном _merge: карточные источники (копия из live-карточки
# того же товара) не должны перетираться инференсом (LLM-знания / web-поиск), если
# их confidence лишь незначительно ниже. Band = допустимый зазор.
_CARD_SOURCES = {Source.WB_CARD, Source.OZON_CARD}
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
    gender_attr_ids: set[int] = set()
    for t in targets:
        if _is_gender_target_name(t.name):
            gender_attr_ids.add(t.id)
    if not gender_attr_ids:
        return all_values

    name_gender = _extract_gender_signal(context.product_name or "")

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
            out.append(v)
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


def _brand_in_name(brand: str, name_tokens: list[str]) -> bool:
    """True если бренд присутствует в имени как непрерывная цепочка токенов.

    Многословные бренды («The North Face») матчатся как contiguous-подпоследо-
    вательность токенов имени — без ложняков на разбросанных совпадениях.
    Бренд короче _BRAND_MIN_LEN символов (после нормализации, склейка токенов)
    игнорируется.
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
    return False


def _apply_brand_from_name(
    merged: list[AttributeValue],
    targets: list[TargetAttribute],
    context: ExtractionContext,
) -> list[AttributeValue]:
    """POST-merge brand-from-name резолвер для enum-полей «Бренд» (генеральный).

    Для каждого brand-таргета С allowed_values (enum-бренд; free-text бренд — вне
    области, см. ниже): если в ИМЕНИ товара присутствует РОВНО ОДИН из allowed-
    брендов (как непрерывная цепочка токенов, бренд ≥3 символов) — это B:
      • поле НЕ заполнено → заполняем B (source=DESCRIPTION, evidence=brand_from_name);
      • поле заполнено значением != B → ПЕРЕЗАПИСЫВАЕМ на B (заголовок авторитетен).
    Если в имени НЕТ ни одного allowed-бренда, либо их НЕСКОЛЬКО (двусмысленно) —
    поле НЕ трогаем (анти-мусор: не угадываем).

    НИКОГДА не пишет бренд, отсутствующий И в имени, И в allowed_values: B всегда
    выбирается из allowed_values И присутствует в имени. value_id заполняется
    стратегией позже (resolve_value_ids в _finalize), как для прочих значений.

    Free-text brand-таргеты (без allowed_values) пропускаются — там нет enum, к
    которому «прилипает» мусор, и нет канонического написания для перезаписи.
    """
    # brand-таргеты с allowed_values: attr_id → (target, allowed)
    brand_targets: dict[int, TargetAttribute] = {}
    for t in targets:
        if (t.id == _BRAND_TARGET_ATTR_ID or _is_brand_target_name(t.name)) and t.allowed_values:
            brand_targets[t.id] = t
    if not brand_targets:
        return merged

    name_tokens = _brand_norm_tokens(context.product_name or "")
    if not name_tokens:
        return merged

    # Какое каноническое B подобрать для каждого brand-таргета (ровно один матч).
    resolved_brand: dict[int, str] = {}
    for attr_id, t in brand_targets.items():
        matches = [b for b in t.allowed_values if _brand_in_name(str(b), name_tokens)]
        # Дедуп по нормализованной форме (один и тот же бренд в разных написаниях).
        uniq = {tuple(_brand_norm_tokens(str(b))): str(b) for b in matches}
        if len(uniq) == 1:
            resolved_brand[attr_id] = next(iter(uniq.values()))
        elif len(uniq) > 1:
            logger.info(
                "[Pipeline] brand-from-name: %d брендов в имени '%s' для attr %s — "
                "двусмысленно, не трогаем",
                len(uniq), context.product_name, attr_id,
            )

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
            "value_id": None,        # пере-резолвится стратегией в _finalize
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
            confidence=0.95,
            source=Source.DESCRIPTION,
            evidence=_BRAND_FROM_NAME_EVIDENCE,
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
    if challenger.source in _CARD_SOURCES and incumbent.source in _INFERENCE_SOURCES:
        card, inference = challenger, incumbent
    elif incumbent.source in _CARD_SOURCES and challenger.source in _INFERENCE_SOURCES:
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
    - ТОЛЬКО OPTIONAL (is_required == False). Required-enum вне scope.
    - Дроп ТОЛЬКО когда value_id None/пуст ПОСЛЕ всей резолюции.
    - is_collection: дропаем только нерезолвнутые элементы, резолвнутые оставляем.
      Если ВСЕ элементы нерезолвнуты → дроп всего поля.
    """
    targets_by_id = {t.id: t for t in targets}
    out: list[AttributeValue] = []
    for v in merged:
        target = targets_by_id.get(v.attribute_id)
        # Вне scope → пропускаем как есть: нет таргета, не enum, или required.
        if target is None or not target.allowed_values or target.is_required:
            out.append(v)
            continue

        if _is_collection_value(v) and isinstance(v.value, list):
            ids = v.value_ids if isinstance(v.value_ids, list) else []
            n_resolved = len(ids)
            if n_resolved == 0:
                logger.info(
                    "[Pipeline] drop-unresolved-enum: дроп optional enum-коллекции "
                    "attr=%s value=%r — все элементы без value_id",
                    v.attribute_id, v.value,
                )
                continue  # все элементы нерезолвнуты → дроп поля
            if n_resolved < len(v.value):
                # Резолвер (ozon resolve_value_ids) компактит value_ids до списка
                # ТОЛЬКО резолвнутых id (resolved = [i for i in ids if i is not None]),
                # сохраняя порядок — но НЕ удаляет нерезолвнутые элементы из value.
                # Оставляем первые n_resolved элементов value (резолвнутый префикс),
                # дропаем хвост нерезолвнутых. value_ids уже компактный → как есть.
                new_value = v.value[:n_resolved]
                logger.info(
                    "[Pipeline] drop-unresolved-enum: частичный дроп optional "
                    "enum-коллекции attr=%s оставлено %d/%d",
                    v.attribute_id, n_resolved, len(v.value),
                )
                out.append(v.model_copy(update={"value": new_value}))
            else:
                out.append(v)
        else:
            if v.value_id is None:
                logger.info(
                    "[Pipeline] drop-unresolved-enum: дроп optional enum attr=%s "
                    "value=%r — нет словарного value_id после резолюции",
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
    if a.source in _CARD_SOURCES and b.source in _INFERENCE_SOURCES:
        card, inference = a, b
    elif b.source in _CARD_SOURCES and a.source in _INFERENCE_SOURCES:
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
        ugc_source: Optional[UgcSource] = None,
        tnved_source: Optional[TnvedSource] = None,
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
        # UgcSource: отзывы и Q&A с Ozon/WB для compat/physical attrs.
        # None → UGC stage пропускается.
        self._ugc: Optional[UgcSource] = ugc_source
        # TnvedSource: per-category резолвер ТН ВЭД ЕАЭС с кэшем.
        # Создаётся ОДИН раз → кэш переживает все товары батча.
        # None → по умолчанию создаём инстанс (всегда нужен для Ozon).
        self._tnved: TnvedSource = tnved_source or TnvedSource()
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
        # Judge для UGC (если source передан)
        if self._ugc is not None:
            self._judges[Source.UGC] = ConfidenceAwareJudgeWrapper(
                self._ugc.get_judge()
            )
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

        all_values: list[AttributeValue] = []
        # filled_so_far — накапливаем high-confidence AVs для skip-filled кооперации
        filled_so_far: list[AttributeValue] = []

        # Stage 0: DescriptionSource (always first, cheapest)
        new_avs = await self._run_stage(Source.DESCRIPTION, context, targets)
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
            all_values += new_avs
            filled_so_far = self._merge_high_conf(filled_so_far, new_avs)
            remaining = self._remaining_targets(targets, all_values)
            if not remaining:
                all_values += await self._run_finishing(context, targets, all_values)
                all_values += await self._generate_annotation(context, targets, all_values)
                return await self._finalize_async(all_values, targets, context)

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
        force_attr_ids = self._strategy.force_websearch_targets(targets)
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

        # Stage 4.7: TnvedSource — per-category резолвер ТН ВЭД ЕАЭС.
        # Запускается после всех товарных sources: кэш по category_id уже тёплый
        # если несколько товаров одной категории обрабатываются параллельно.
        new_avs = await self._run_tnved_stage(context, targets, already_filled=filled_so_far)
        all_values += new_avs

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
        """
        finalized = self._finalize(all_values, targets, context)
        resolved = await self._strategy.llm_resolve_tail(finalized, targets, context)
        # ПОСЛЕ полной резолюции value_id (детерминированный + LLM-хвост): дроп
        # OPTIONAL enum-значений, оставшихся без словарного value_id (fake-fill,
        # который Ozon отклонит). Required-enum и free-text не трогаем. См. docstring.
        return _drop_unresolved_optional_enums(resolved, targets)

    def _finalize(
        self,
        all_values: list[AttributeValue],
        targets: list[TargetAttribute],
        context: ExtractionContext,
    ) -> list[AttributeValue]:
        """Merge + strategy post-process + strategy validation. Used at every early-exit point."""
        # Гендер-гард ДО merge: генеральный, для всех источников (web_search/llm/
        # vision/cards). Отсекает гендерные «Пол»-значения, конфликтующие с именем
        # или навеянные только external-guess источниками при нейтральном имени.
        all_values = _apply_gender_guard(all_values, targets, context)
        merged = self._merge(all_values)

        # Brand-from-name POST-merge: имя товара авторитетно для бренда. Заполняет
        # пустой «Бренд» / перезаписывает мусорный (чужой allowed-enum) ровно-одним
        # allowed-брендом, присутствующим в имени. Идёт ДО resolve_value_ids, чтобы
        # заполненный/перезаписанный бренд получил словарный value_id.
        merged = _apply_brand_from_name(merged, targets, context)

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
