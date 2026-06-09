"""PipelineOrchestrator — sequential cost-aware extraction.

Главный entry-point для enrichment. Объединяет все sources, judges и intelligence
в один flow с early-exit (если все targets заполнены — останавливаемся) и
cost gating (CostPredictor перед expensive web search).

Spec: docs/architecture/pipeline.md, section "PipelineOrchestrator".
"""
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
    Если оба источника пусты — таргет пропускается (нечего матчить).

    НИКОГДА не пишет бренд, отсутствующий И в имени, И в списке брендов: B всегда
    выбирается из списка И присутствует в имени.

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
    for attr_id, t in brand_targets.items():
        # Полный список брендов: target.allowed_values (мелкий enum) ИЛИ словарь.
        options = list(t.allowed_values or [])
        if not options and brand_options_fn is not None:
            try:
                options = list(brand_options_fn(attr_id) or [])
            except Exception as exc:  # словарь недоступен — не падаем, пропускаем
                logger.warning(
                    "[Pipeline] brand-from-name: словарный список брендов для attr %s "
                    "недоступен: %s", attr_id, exc,
                )
                options = []
        if not options:
            continue
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


def _is_placeholder_value(raw: object) -> bool:
    """True если значение — заведомый плейсхолдер/пустышка.

    Используется ДО/ВО ВРЕМЯ резолюции: такие значения не несут информации
    и никогда не должны заполнять enum-поле. Проверяется case-insensitive,
    strip whitespace. Список хранится в _PLACEHOLDER_VALUES (константа модуля).
    """
    return str(raw).strip().lower() in _PLACEHOLDER_VALUES


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
        yandex_market_source: Optional[YandexMarketSource] = None,
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

        # Stage 0.52: YandexMarketSource — копия характеристик с live market.yandex.ru.
        # Запускается ПОСЛЕ WbCard + OzonCard (card=N fallback): нет смысла тратить
        # Scrappey-кредиты если WB/Ozon card уже закрыл нужные атрибуты. Особенно
        # полезен для apparel (generic-named, без SKU) где WB/Ozon дают card=N.
        # Cost-gated: только при remaining > 0 (т.е. когда предыдущие card-источники
        # не закрыли все targets).
        if self._yandex_market is not None and remaining:
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

        finalized = self._finalize(filtered_values, targets, context)
        resolved = await self._strategy.llm_resolve_tail(finalized, targets, context)

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

        # ПОСЛЕ полной резолюции value_id (детерминированный + LLM-хвост):
        # 1. Дроп OPTIONAL enum-значений без value_id (fake-fill, Ozon отклонит).
        # 2. Дроп REQUIRED enum-значений без value_id — мусор в обязательном поле
        #    хуже пустоты: Ozon отвергает карточку, а не просто помечает поле пустым.
        #    Correctly-resolved required values (value_id is set) НЕ затрагиваются.
        resolved = _drop_unresolved_optional_enums(resolved, targets)
        resolved = _drop_unresolved_required_enums(resolved, targets)
        return resolved

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
