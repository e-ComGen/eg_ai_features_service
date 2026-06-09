"""WbCardSource — копирует характеристики из живой Wildberries-карточки похожего товара.

Старый путь ходил на ``search.wb.ru`` — WB банит серверный IP (0 заполнений).
Новый путь полностью минует anti-bot WB:

  1. **Построить запрос**: ``_build_wb_query`` чистит product_name через
     ``_compress_search_query`` (срез стоп-слов/спеков/бренд-дублей), НО, в
     отличие от Ozon, ОБЯЗАТЕЛЬНО сохраняет ТИП-слово товара («куртка»,
     «футболка») — для WB Serper-поиска тип критичен (без него выдача уходит в
     чужой класс товара). Тип берётся динамически: первое сущ. category leaf или
     ведущее сущ. названия (pymorphy3), без хардкода списков типов.
  2. **Найти nm_id через Serper** (НЕ search.wb.ru). Запрос вида
     ``inurl:catalog detail.aspx <query>`` → ~100% прямых ссылок на карточки WB.
     Из organic-результатов nm_id вытаскивается регэкспом, предпочитая домен
     ``wildberries.ru`` (зеркала .ge/.am/.by дают тот же nm_id — fallback).
     Берём топ-10 уникальных nm_id (многие архивные → 404).
  3. **Скачать card.json с CDN** простым httpx GET (Chrome User-Agent, БЕЗ Scrappey —
     CDN не банится). URL:
     ``https://basket-{NN}.wbbasket.ru/vol{vol}/part{part}/{nm_id}/info/ru/card.json``,
     где ``vol = nm_id // 100000``, ``part = nm_id // 1000``. NN по диапазону vol
     из таблицы ниже; на 404 перебираются остальные NN (01..21).
  4. **Распарсить характеристики**: ``options[]``, ``grouped_options[].options``,
     ``compositions[]`` (Состав) → пары name→value.
  5. **Выбрать лучший кандидат** среди скачанных card.json: ``_pick_best_card``
     сначала применяет ЖЁСТКИЙ ТИП-ГЕЙТ (``_type_compatible``: карточка с
     ``subj_name``/``subj_root_name`` чужого типа — «шорты/жакет» под цель
     «куртка» — отбраковывается, а не штрафуется; если все кандидаты не того
     типа → честный 0), затем отсекает нерелевантные по матч-скору
     (``_pick_best_match``: model-token бонус, type-mismatch penalty по
     ``imt_name``/``subj_name``), затем среди
     релевантных при сопоставимом скоре предпочитает карточку с бОльшим числом
     полезных options (богатую, не пустую). card.json качаем по кандидатам по
     порядку, останавливаясь на _MAX_FETCHED_CARDS успешно скачанных (404 не
     прекращает перебор).
  6. **Маппинг WB→Ozon**: ``_map_characteristics`` мапит русские имена характеристик на
     Ozon-словарь (lowercase + substring + fuzzy WRatio≥88) — финальное API публикации
     у нас Ozon, WB лишь источник данных.

Стоимость: 1 Serper-запрос (~$0.001) + N бесплатных CDN GET'ов. Latency: ~1-3s.
Все сетевые вызовы — с таймаутами и graceful-fallback (возврат [] вместо падения).
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections import OrderedDict
from typing import Any, Optional, Union

import httpx

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.judges.wb_card_judge import WbCardJudge
from app.services.enrichment.prompt_router import filter_already_filled_targets
from app.services.enrichment.sources.ozon_card_source import (
    _compress_search_query,
    _extract_model_tokens,
    _extract_alpha_model_tokens,
    _extract_gender_signal,
    _gender_conflict,
    _is_gender_target_name,
    _is_spec_or_unit_token,
    _normalize_model,
    _LEADING_STOPWORDS,
)
from app.services.enrichment.strategies.dictionaries.ozon_loader import (
    get_ozon_characteristics_for_type,
    resolve_value_id,
)
from app.services.enrichment.strategies.dictionaries.eg_wb_ozon_field_map import (
    eg_get_field_map,
)
from app.services.enrichment.size_normalizer import extract_wb_sizes
from app.services.providers.factory import get_web_search_client

logger = logging.getLogger(__name__)

# Морфология (RU): лемматизация тип-слова товара, чтобы «куртка»↔«куртки»,
# «футболка»↔«футболки» сходились, а «куртка» vs «шорты» — нет. Опциональна:
# при отсутствии pymorphy3 деградируем к сравнению по нормализованным токенам.
try:  # pragma: no cover — морфология опциональна
    import pymorphy3
    _MORPH = pymorphy3.MorphAnalyzer()
except Exception:  # noqa: BLE001
    _MORPH = None

_TYPE_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _lemma(word: str) -> str:
    """Лемма слова (нормальная форма). Fallback — lowercase сам токен."""
    w = word.strip().lower()
    if not w or _MORPH is None:
        return w
    try:
        return _MORPH.parse(w)[0].normal_form
    except Exception:  # noqa: BLE001
        return w


def _type_lemma(word: str) -> str:
    """Лемма ТИП-слова товара, устойчивая к несклоняемым существительным.

    Обычный ``_lemma`` слепо берёт ``parse()[0].normal_form``, а pymorphy3 для
    разговорных/заимствованных НЕСКЛОНЯЕМЫХ существительных («худи», «боди»)
    верхней гипотезой ставит выдуманный ГЛАГОЛ («худи»→«худить», «боди»→«бодить»)
    или мусорный noun («худь»). Это отравляло тип-гейт: цель-лемма «худить» не
    совпадает НИ с одной карточкой-худи (subj «Худи»/«Толстовки») → отбраковка
    всех валидных кандидатов → честный 0 на товаре, где карточек сотни.

    Правило (general, без хардкода списков типов): лемматизируем ТОЛЬКО когда
    верхняя гипотеза pymorphy — существительное (NOUN). Иначе (глагол/иное, как у
    несклоняемых) возвращаем сам токен в нижнем регистре — несклоняемое и так не
    меняет форму, а surface-форма стабильно совпадает с обеих сторон гейта
    (цель «худи» ↔ карточка «худи»). Склоняемые («толстовки»→«толстовка»,
    «куртки»→«куртка») по-прежнему нормализуются корректно (их топ-парс — NOUN).
    """
    w = word.strip().lower()
    if not w or _MORPH is None:
        return w
    try:
        parse = _MORPH.parse(w)
        if parse and "NOUN" in parse[0].tag:
            return parse[0].normal_form
        # Топ-гипотеза не сущ. (несклоняемое слово → выдуманный глагол): surface.
        return w
    except Exception:  # noqa: BLE001
        return w


def _is_noun_lemma(word: str) -> bool:
    """True если слово (по pymorphy) — существительное. Без морфологии — всегда True."""
    if _MORPH is None:
        return True
    try:
        p = _MORPH.parse(word.strip().lower())
        return bool(p) and "NOUN" in p[0].tag
    except Exception:  # noqa: BLE001
        return True


def _target_type_lemma(
    product_name: str,
    cat_leaf: Optional[str],
) -> Optional[str]:
    """Лемма ТИП-слова целевого товара (динамически, без хардкода списков).

    Источник, по приоритету:
      1. category leaf (`category_path[-1]`) — первое существительное-токен
         («Куртки» → «куртка», «Платья женские» → «платье»). Самый надёжный
         сигнал типа: категория задаётся явно.
      2. fallback — ведущее существительное product_name после среза
         стоп-слов («Куртка мужская …» → «куртка»).

    Возвращает лемму типа или None если тип определить нельзя.
    """
    # 1. Категория-leaf: ПЕРВЫЙ значимый токен — БЕЗ noun-гейта.
    #    Leaf — это явный ярлык категории, а не свободный текст: его первое
    #    значимое слово ВСЕГДА тип товара. pymorphy3 ошибочно парсит несклоняемые
    #    («худи»→глагол «худить», «пальто», «боди», «бикини») как НЕ-сущ., поэтому
    #    noun-гейт тут отбрасывал бы корректный тип. Доверяем leaf безусловно;
    #    noun-scan названия (шаг 2) нужен ТОЛЬКО когда leaf нет (иначе он цепляет
    #    спек/фичу — «молния», «карман», «капюшон»).
    if cat_leaf:
        for tok in _TYPE_TOKEN_RE.findall(cat_leaf.lower()):
            # БЕЗ stopword-фильтра: «худи» внесён в _LEADING_STOPWORDS как
            # разговорный тип, но в leaf это и есть искомый тип. Скипаем только
            # короткий мусор (len<3).
            if len(tok) < 3:
                continue
            # _type_lemma (не _lemma): несклоняемые «худи»/«боди» pymorphy
            # лемматизирует в выдуманный глагол «худить»/«бодить», что отравляет
            # тип-гейт (цель «худить» ≠ карточка «худи»). _type_lemma лемматизирует
            # только NOUN-топ-парсы, иначе оставляет surface-форму.
            return _type_lemma(tok)
    # 2. Ведущее существительное названия товара (только при отсутствии leaf).
    for tok in _TYPE_TOKEN_RE.findall(product_name.lower()):
        if len(tok) < 3 or tok in _LEADING_STOPWORDS or _is_spec_or_unit_token(tok):
            continue
        # Латиница/цифры (бренд/артикул) типом не считаем.
        if not re.search(r"[а-яё]", tok):
            break
        if _is_noun_lemma(tok):
            return _lemma(tok)
        # первое русское слово не сущ. (напр. прилагательное) — пропускаем дальше
    return None


def _card_subj_lemmas(card: dict) -> set[str]:
    """Леммы тип-слов карточки из subj_name / subj_root_name (множество).

    subj_name/subj_root_name — это ЯРЛЫК ТИПА карточки (как category leaf:
    «Худи», «Толстовки»), а не свободный текст. Поэтому:
      - НЕ фильтруем по _LEADING_STOPWORDS: «худи» внесён туда как разговорный
        тип (для среза в середине названия), но в subj это и есть искомый тип —
        фильтр выкинул бы его, и карточка-«Худи» давала бы пустой набор лемм, из-за
        чего тип-гейт пропускал бы её мимо проверки (ложный pass);
      - используем _type_lemma (а не _lemma) — симметрично цели: несклоняемое
        «худи» остаётся «худи» с обеих сторон, склоняемое «толстовки»→«толстовка».
    """
    out: set[str] = set()
    for key in ("subj_name", "subj_root_name"):
        val = card.get(key)
        if isinstance(val, str) and val.strip():
            for tok in _TYPE_TOKEN_RE.findall(val.lower()):
                if len(tok) >= 3:
                    out.add(_type_lemma(tok))
    return out


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

# Известный sharding table «vol → basket-NN». На 404 перебираем все NN (01..21).
# Формат: (верхняя_граница_vol_включительно, NN).
_BASKET_THRESHOLDS: list[tuple[int, str]] = [
    (143,  "01"),
    (287,  "02"),
    (431,  "03"),
    (719,  "04"),
    (1007, "05"),
    (1061, "06"),
    (1115, "07"),
    (1169, "08"),
    (1313, "09"),
    (1601, "10"),
    (1655, "11"),
    (1919, "12"),
    (2045, "13"),
    (2189, "14"),
    (2405, "15"),
    (2621, "16"),
    (2837, "17"),
    (3053, "18"),
    (3473, "19"),
    (3793, "20"),
]
# vol >= 3794 → basket-21. Полный список NN для brute-force перебора на 404.
_BASKET_DEFAULT = "21"
_ALL_BASKET_NN: list[str] = [f"{n:02d}" for n in range(1, 22)]  # 01..21

# Regex для извлечения nm_id из ссылки на карточку WB.
# Покрывает .ru/.ge/.am/.by зеркала и относительные ссылки.
_NM_ID_RE = re.compile(
    r"(?:wildberries\.\w+/catalog/|/catalog/)(\d{6,12})/detail\.aspx",
    re.IGNORECASE,
)

# Chrome User-Agent для CDN GET (CDN не банит, но без UA иногда 403).
_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_HTTP_TIMEOUT = 15.0
_SERPER_NUM_RESULTS = 10
_MAX_CANDIDATES = 10  # топ-N уникальных nm_id из Serper (многие дадут 404)

# Ретрай Serper-поиска. Serper флакает при concurrency (троттлинг, пустые
# ответы): один и тот же запрос то даёт 10 nm_id, то 0. Если поиск вернул 0
# organic ИЛИ из них извлеклось 0 nm_id — повторяем с экспоненциальным бэкоффом.
# Покрывает временные пустышки, не залипая на флаке. Бэкофф короткий, попыток
# мало — время растёт максимум в _SERPER_MAX_ATTEMPTS раз только в худшем случае
# (стабильно пустой товар), на успехе ретраев нет.
_SERPER_MAX_ATTEMPTS = 3        # всего попыток (1 основная + 2 ретрая)
_SERPER_BACKOFF_BASE = 1.0      # сек: задержки 1с, 2с (экспонента 2^n)


class _EgPermanentSearchError(Exception):
    """Непреходящая (4xx) ошибка поиска — ретраить бессмысленно.

    Поднимается из `_search_once`, когда бэкенд вернул non-transport 4xx
    (400 «Not enough credits» / 401 / 403): неверный ключ, исчерпан кредит,
    запрещённый запрос. Бэкофф+ретрай таких НЕ чинит — `_search` должен
    немедленно вернуть [] без задержек. 429 (троттлинг) и 5xx — НЕ сюда,
    они транзиентны и ретраятся как раньше.
    """


def _eg_is_permanent_4xx(exc: BaseException) -> bool:
    """True если исключение — non-transport клиентская 4xx (кроме 429).

    Ретраить такие нельзя: 400/401/403 указывают на проблему ключа/кредита/
    запроса. 429 (rate-limit) исключаем — он транзиентный (ретраим с бэкоффом),
    как и 5xx/таймауты/коннект-ошибки.
    """
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status != 429

# Сколько card.json реально скачать прежде чем выбирать лучший. Многие nm_id
# архивные/несуществующие → 404 по всем basket. Перебираем кандидатов по порядку,
# но останавливаемся, набрав _MAX_FETCHED_CARDS успешно скачанных карточек.
_MAX_FETCHED_CARDS = 8

# Карточка считается «богатой» если у неё ≥ _RICH_OPTIONS_THRESHOLD полезных
# options. При сопоставимом матч-скоре богатую предпочитаем бедной.
_RICH_OPTIONS_THRESHOLD = 6
# Разрешённый разрыв в матч-скоре, при котором богатство решает исход. Если
# кандидат с большим числом options отстаёт по скору не более чем на эту
# величину — берём его (богатую карточку), а не пустого лидера по fuzzy.
_SCORE_TIE_BAND = 12.0

# Confidence — параллельно с OzonCardSource.
_CONF_EXACT = 0.93
_CONF_BRAND_LINE = 0.85
# Пониженный confidence для поля «Пол», навеянного гендером brand_line-карточки
# при нейтральном имени товара (гендер-страховка, п.3). Ниже порога, чтобы
# merger/judge не предпочли его более надёжным источникам (vision/llm).
_CONF_GENDER_DOWNWEIGHT = 0.40

_EXACT_THRESHOLD = 78.0
_BRAND_LINE_THRESHOLD = 60.0

# Skip-guard
_SKIP_FILL_RATIO = 0.80

# LRU
_CACHE_MAX = 256

# Brand-line BLACKLIST — model-specific атрибуты.
_BRAND_LINE_BLACKLIST: frozenset[str] = frozenset(name.lower() for name in {
    "Артикул",
    "Артикул WB",
    "Артикул товара",
    "Код производителя",
    "MPN",
    "Партномер",
    "Серийный номер",
    "EAN",
    "GTIN",
    "ASIN",
    "Штрихкод",
    "Дата производства",
    "Модель",
    "Название модели",
    "ID товара",
    "ID карточки",
})


# Attr IDs for «Российский размер» (clothing) and «Российский размер» (footwear).
# Used as fast-path before name-based detection.
_RU_SIZE_ATTR_IDS: frozenset[int] = frozenset({4295, 4298})


def _is_ru_size_target_name(name: str) -> bool:
    """True if target name refers to «Российский размер» (attr 4295/4298).

    Detects by name substring — works for any language variant in the Ozon dict.
    Intentionally conservative: only triggers on the exact phrase «российский размер»
    or «russian size» to avoid accidentally treating other size attributes (EU, INT…)
    as RU size targets.
    """
    low = name.lower()
    return "российский размер" in low or "russian size" in low


def _emit_ru_size_from_card(
    card: dict,
    targets: list["TargetAttribute"],
    results: list["AttributeValue"],
    used_ids: set[int],
    cat_id: Optional[int],
    type_id: Optional[int],
    evidence_short: str,
) -> list["AttributeValue"]:
    """Emit «Российский размер» AttributeValue from card's sizes_table.

    Called ONLY for exact-match cards (not brand_line) since a different model
    of the same brand may have a completely different size run.  Skipped when the
    target was already filled by the normal characteristic pass (used_ids guard).

    Drop-policy (fail-closed):
      - expand_intl_to_ru returns [] → skip silently.
      - resolve_value_id returns None for a candidate → drop that candidate only.
      - If ALL candidates are unresolvable → emit nothing (empty > wrong required).
      - This exactly mirrors the resolved_ids=[r for r in resolved if r is not None] pattern.
    """
    from app.services.enrichment.size_normalizer import extract_wb_sizes  # local re-import OK (already cached)

    # Find size target(s) — typically 4295 for clothing, 4298 for footwear.
    size_targets = [
        t for t in targets
        if t.id in _RU_SIZE_ATTR_IDS or _is_ru_size_target_name(t.name)
    ]
    if not size_targets:
        return results

    # Extract raw size tokens from the card's sizes_table.
    size_tokens = extract_wb_sizes(card)
    if not size_tokens:
        logger.debug("[WbCard] sizes_table absent or empty → no размер emit")
        return results

    logger.info("[WbCard] sizes_table → raw tokens: %s", size_tokens)

    out = list(results)
    for target in size_targets:
        if target.id in used_ids:
            # Already filled by normal characteristic pass — don't overwrite.
            continue

        if not (cat_id and type_id):
            # Can't resolve without category context — skip rather than emit wrong id.
            logger.debug("[WbCard] no cat/type_id → skip размер emit for attr %s", target.id)
            continue

        resolved_ids: list[int] = []
        for token in size_tokens:
            vid = resolve_value_id(cat_id, type_id, target.id, token)
            if vid is not None:
                resolved_ids.append(vid)
        # Dedup while preserving order
        seen_ids: set[int] = set()
        unique_ids: list[int] = []
        for vid in resolved_ids:
            if vid not in seen_ids:
                seen_ids.add(vid)
                unique_ids.append(vid)
        resolved_ids = unique_ids

        if not resolved_ids:
            logger.info(
                "[WbCard] sizes_table tokens %s → 0 resolved value_ids for attr %s "
                "(all unresolvable — leaving empty, not emitting garbage)",
                size_tokens, target.id,
            )
            continue

        logger.info(
            "[WbCard] sizes_table → attr %s: tokens=%s resolved_ids=%s",
            target.id, size_tokens, resolved_ids,
        )
        used_ids.add(target.id)
        out.append(AttributeValue(
            attribute_id=target.id,
            value=size_tokens,         # raw list (merged as collection)
            confidence=_CONF_EXACT,    # exact mode only (see caller gate)
            source=Source.WB_CARD,
            evidence=evidence_short,
            semantic_type=target.semantic_type,
            is_collection=True,
            value_id=None,
            value_ids=resolved_ids,
        ))
    return out


_MULTIVALUE_SPLIT_RE = re.compile(r"[;,]")


def _split_multivalue(raw: str) -> list[str]:
    """Сплит карточной multi-value строки в дедуплицированный список.

    Разделители ";" и ",". Чистит пробелы, регистронезависимый дедуп
    (сохраняя первое вхождение). Если разделителей нет — возвращает [raw]
    (один элемент), чтобы is_collection-атрибут всё равно был списком.
    """
    parts = _MULTIVALUE_SPLIT_RE.split(raw)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        p = p.strip()
        if not p:
            continue
        low = p.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(p)
    return out or [raw.strip()]


def _basket_nn_from_table(nm_id: int) -> str:
    """Возвращает basket-NN из known таблицы по vol (fallback basket-21)."""
    vol = nm_id // 100_000
    for threshold, nn in _BASKET_THRESHOLDS:
        if vol <= threshold:
            return nn
    return _BASKET_DEFAULT


def _build_wb_query(
    full_name: str,
    brand: Optional[str],
    cat_leaf: Optional[str],
    max_tokens: int,
) -> str:
    """Строит Serper-запрос для WB, СОХРАНЯЯ тип-слово товара.

    Проблема: ``_compress_search_query`` (Ozon) срезает категорийный префикс —
    тип-существительное («куртка», «футболка»). Для Ozon SSR это ок (заголовки
    начинаются с типа, fuzzy всё равно матчит). Для WB Serper-поиска фатально:
    по «The North Face Resolve» (без «куртка») Serper отдаёт 2 ссылки, и
    единственная живая — женские шорты. Тип-слово критично сужает выдачу к
    правильному классу товара.

    Решение: компрессим как раньше (чистим спеки/стоп-слова/бренд-дубли), затем
    ПРЕПЕНДИМ тип-слово (cat_leaf-первое-слово или ведущее сущ. названия), если
    его ещё нет в запросе. Тип берём динамически, без хардкода списков.
    """
    base = _compress_search_query(
        full_name, brand, max_tokens=max_tokens, category_name=cat_leaf
    )
    type_word = _wb_query_type_word(full_name, cat_leaf)
    if not type_word:
        return base
    # Уже присутствует (по лемме любого токена запроса) — не дублируем.
    base_lemmas = {_lemma(t) for t in _TYPE_TOKEN_RE.findall(base.lower())}
    if _lemma(type_word) in base_lemmas:
        return base
    return f"{type_word} {base}".strip()


def _wb_query_type_word(product_name: str, cat_leaf: Optional[str]) -> Optional[str]:
    """Тип-слово (в исходной словоформе) для подстановки в Serper-запрос.

    Приоритет — первое значимое слово category leaf (как пишут на WB:
    «Куртки» → «Куртки»), затем ведущее русское существительное названия.
    Возвращаем словоформу как есть (Serper токенизирует, лемма не нужна).
    """
    # Leaf — явный ярлык категории: первое значимое РУССКОЕ слово ВСЕГДА тип,
    # БЕЗ noun-гейта И БЕЗ stopword-фильтра. Несклоняемые («худи»/«пальто»/«боди»)
    # pymorphy парсит как не-сущ.; вдобавок «худи» внесён в _LEADING_STOPWORDS как
    # разговорный тип (для среза в середине названия). Оба фильтра ошибочно
    # выкинули бы тип из leaf, и noun-scan названия подобрал бы спек-слово
    # («молния»/«карман»). В leaf первое значимое слово — это тип по определению,
    # поэтому фильтруем только мусор (len<3 / не-русское). Noun-scan названия —
    # fallback ТОЛЬКО при отсутствии leaf.
    if cat_leaf:
        for tok in cat_leaf.split():
            low = tok.lower()
            if len(low) >= 3 and re.search(r"[а-яё]", low):
                return tok
    for tok in product_name.split():
        low = tok.lower()
        if len(low) < 3 or low in _LEADING_STOPWORDS or _is_spec_or_unit_token(low):
            continue
        if not re.search(r"[а-яё]", low):
            break
        if _is_noun_lemma(low):
            return tok
    return None


def _card_url(nn: str, nm_id: int) -> str:
    vol = nm_id // 100_000
    part = nm_id // 1000
    return (
        f"https://basket-{nn}.wbbasket.ru"
        f"/vol{vol}/part{part}/{nm_id}/info/ru/card.json"
    )


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class WbCardSource(AttributeSource):
    """Копия характеристик с live WB-карточки похожего товара.

    Путь: Serper (поиск nm_id) → CDN card.json (httpx, бесплатно).
    Cost: 1 Serper-запрос/товар. Latency: ~1-3s.
    """

    def __init__(
        self,
        web_search_client: Any = None,
        **kwargs: Any,
    ):
        _ = kwargs  # backward-compat (старые коды передавали scrappey_key и т.п.)

        # Serper-клиент для поиска nm_id. Лениво создаём через factory, если не
        # передан явно. Если SERPER_API_KEY не задан — factory кинет/вернёт None,
        # extract() тогда всегда вернёт [].
        self._search_client = web_search_client
        if self._search_client is None:
            try:
                self._search_client = get_web_search_client()
            except Exception as exc:  # SERPER_API_KEY не задан и т.п.
                logger.warning(
                    "[WbCard] web search client недоступен (%s) — extract() вернёт [].",
                    exc,
                )
                self._search_client = None
        if self._search_client is None:
            logger.warning(
                "[WbCard] SERPER не сконфигурирован (PROVIDER_WEB_SEARCH != 'serper' "
                "или нет ключа) — extract() всегда вернёт []."
            )

        self._judge = WbCardJudge()
        # LRU cache: (brand_lower, model_lower) → list[AttributeValue]
        self._cache: "OrderedDict[tuple[str, str], list[AttributeValue]]" = OrderedDict()

    @property
    def source_type(self) -> Source:
        return Source.WB_CARD

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Применим если product_name достаточный и search-клиент доступен."""
        return bool(
            self._search_client
            and context.product_name
            and len(context.product_name.strip()) >= 5
        )

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list[AttributeValue]] = None,
    ) -> list[AttributeValue]:
        if not targets or not context.product_name or not self._search_client:
            return []

        already_filled = already_filled or []

        # Skip-guard
        filled_ids = {av.attribute_id for av in already_filled if av.confidence >= 0.85}
        if targets and len(filled_ids & {t.id for t in targets}) / len(targets) >= _SKIP_FILL_RATIO:
            logger.debug(
                "[WbCard] skip (≥%.0f%% targets уже filled)",
                _SKIP_FILL_RATIO * 100,
            )
            return []

        effective = filter_already_filled_targets(targets, already_filled)
        if not effective:
            return []

        brand = (context.brand or "").strip()
        model_norm = _normalize_model(context.product_name, brand)
        cache_key = (brand.lower(), model_norm)

        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._filter_for_targets(self._cache[cache_key], effective)

        try:
            all_values = await self._do_extract(context, targets)
        except Exception as exc:
            logger.warning(
                "[WbCard] unexpected error для '%s': %s",
                context.product_name[:60], exc,
            )
            # НЕ кэшируем пустышку: ошибка часто транзиентная (Serper-флак,
            # сетевой сбой). Кэширование [] «залипает» и блокирует ретрай в
            # следующем прогоне. Возвращаем пусто, кэш не трогаем.
            return []

        # Кэшируем только непустой результат. Пустой список — обычно следствие
        # флака Serper (троттлинг/пустой ответ), уже отретраенного в _search;
        # если всё равно пусто, кэшировать [] нельзя — иначе флак-пустышка
        # залипнет в LRU и следующий прогон не сделает повторный поиск. Непустой
        # результат кэшируем как раньше, чтобы не бить Serper повторно.
        if all_values:
            self._cache_put(cache_key, all_values)
        return self._filter_for_targets(all_values, effective)

    def get_judge(self) -> LlmJudge:
        return self._judge

    # ------------------------------------------------------------------
    # Core flow
    # ------------------------------------------------------------------

    async def _do_extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Полный flow: compress → Serper(nm_id) → CDN card.json → pick → map → AVs."""
        full_name = context.product_name.strip()
        cat_leaf = context.category_path[-1] if context.category_path else None
        # Целевой тип товара (лемма) — для query-builder и тип-гейта при выборе.
        target_type = _target_type_lemma(full_name, cat_leaf)
        primary_query = _build_wb_query(
            full_name, context.brand, cat_leaf, max_tokens=5
        )
        fallback_query = _build_wb_query(
            full_name, context.brand, cat_leaf, max_tokens=3
        )

        queries_to_try: list[str] = [primary_query]
        if fallback_query and fallback_query != primary_query:
            queries_to_try.append(fallback_query)

        # ---- SEARCH (Serper → nm_id) ----
        nm_ids: list[int] = []
        used_query: Optional[str] = None
        for q in queries_to_try:
            logger.info("[WbCard] search query: '%s' (was: '%s')", q, full_name[:80])
            candidates = await self._search(q)
            if candidates:
                nm_ids, used_query = candidates, q
                break
            logger.info("[WbCard] no nm_id на query='%s' — пробую fallback", q[:60])

        if not nm_ids or used_query is None:
            logger.info("[WbCard] no nm_id ни для primary ни для fallback")
            return []

        async with httpx.AsyncClient(
            timeout=_HTTP_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": _CHROME_UA},
        ) as client:
            # ---- CARD JSON (CDN) для кандидатов ----
            # Перебираем nm_id по порядку. 404 на одном не прекращает перебор —
            # идём к следующему. Останавливаемся, набрав _MAX_FETCHED_CARDS
            # успешно скачанных карточек (чтобы было из чего выбирать богатую,
            # но не качать все 10 кандидатов).
            cards: list[tuple[int, dict]] = []
            attempted = 0
            for nm_id in nm_ids:
                attempted += 1
                card = await self._fetch_card(client, nm_id)
                if card:
                    cards.append((nm_id, card))
                    if len(cards) >= _MAX_FETCHED_CARDS:
                        break

            if not cards:
                logger.info(
                    "[WbCard] ни один card.json не скачался (%d из %d кандидатов перебрано)",
                    attempted, len(nm_ids),
                )
                return []

            logger.info(
                "[WbCard] скачано %d card.json из %d перебранных кандидатов (всего %d)",
                len(cards), attempted, len(nm_ids),
            )

            # ---- PICK BEST (тип-гейт → match-фильтр → среди релевантных богатую) ----
            best = self._pick_best_card(used_query, cat_leaf, target_type, cards)
            if best is None:
                return []
            nm_id, card, title, score = best
            mode = self._classify_match(score)
            if mode == "skip":
                logger.info("[WbCard] best score=%.1f < %.0f — skip", score, _BRAND_LINE_THRESHOLD)
                return []

            logger.info(
                "[WbCard] match=%s score=%.1f title='%s' nm=%s",
                mode, score, title[:80], nm_id,
            )

            chars = self._extract_options(card)
            if not chars:
                logger.info("[WbCard] no options/characteristics в card.json nm=%s", nm_id)
                return []

            # ---- IMAGES (для downstream VisionSource) ----
            new_image_urls = self._extract_image_urls(card, nm_id)
            if new_image_urls:
                existing = set(context.image_urls or [])
                added = [u for u in new_image_urls if u not in existing]
                if added:
                    context.image_urls = list(context.image_urls or []) + added
                    logger.info(
                        "[WbCard] +%d image URLs для VisionSource (nm=%s)",
                        len(added), nm_id,
                    )

            # ---- MAP & EMIT ----
            return self._map_characteristics(
                chars, targets, context, mode, title, score, card,
            )

    async def _search(self, query: str) -> list[int]:
        """Serper → nm_id с ретраем при пустом результате (троттлинг/флак).

        Serper при concurrency нестабилен: тот же запрос то возвращает 10 nm_id,
        то 0 organic / 0 извлечённых nm_id. Если попытка дала 0 — повторяем с
        экспоненциальным бэкоффом (1с, 2с) до _SERPER_MAX_ATTEMPTS. На успехе
        (≥1 nm_id) выходим сразу. После всех ретраев пусто → graceful [] (выше
        по стеку это fallback, не падение).
        """
        for attempt in range(1, _SERPER_MAX_ATTEMPTS + 1):
            try:
                nm_ids = await self._search_once(query)
            except _EgPermanentSearchError as exc:
                # Непреходящая 4xx (нет кредитов / неверный ключ / запрет):
                # ретрай+бэкофф её не починит → fail-fast, [] без задержек.
                logger.warning(
                    "[WbCard] Serper непреходящая 4xx (%s) — fail-fast без ретрая",
                    exc,
                )
                return []
            if nm_ids:
                if attempt > 1:
                    logger.info(
                        "[WbCard] Serper непустой результат с попытки %d/%d",
                        attempt, _SERPER_MAX_ATTEMPTS,
                    )
                return nm_ids
            if attempt < _SERPER_MAX_ATTEMPTS:
                delay = _SERPER_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.info(
                    "[WbCard] Serper → 0 nm_id (попытка %d/%d), ретрай через %.1fс",
                    attempt, _SERPER_MAX_ATTEMPTS, delay,
                )
                await asyncio.sleep(delay)
        logger.info(
            "[WbCard] Serper → 0 nm_id после %d попыток (флак/нет результатов)",
            _SERPER_MAX_ATTEMPTS,
        )
        return []

    async def _search_once(self, query: str) -> list[int]:
        """Одна Serper-попытка → список уникальных nm_id (top-N).

        Запрос ``site:wildberries.ru <query> detail.aspx`` даёт ~100% прямых
        ссылок на карточки WB. Раньше использовался ``inurl:catalog
        detail.aspx`` — Google игнорит ``inurl:`` НЕДЕТЕРМИНИРОВАННО (~40-50%
        запросов возвращали общую выдачу без WB-ссылок → 0 nm_id). ``site:``
        стабилен. Предпочитаем домен wildberries.ru, зеркала — fallback.
        """
        serper_query = f"site:wildberries.ru {query} detail.aspx".strip()
        try:
            results = await self._search_client.search(
                serper_query, num_results=_SERPER_NUM_RESULTS
            )
        except Exception as exc:
            # Non-transport 4xx (400 «Not enough credits»/401/403) — непреходящая:
            # пробрасываем как _EgPermanentSearchError, чтобы _search не жёг
            # бэкофф-ретраи. Прочие (429/5xx/таймаут/коннект) → транзиентные,
            # деградируем к [] и _search ретраит как раньше.
            if _eg_is_permanent_4xx(exc):
                raise _EgPermanentSearchError(str(exc)) from exc
            # Инструментация: транзиентная ошибка (429/5xx/таймаут/коннект).
            # Логируем HTTP-статус (если есть) для per-attempt-видимости флака.
            _resp = getattr(exc, "response", None)
            _status = getattr(_resp, "status_code", None)
            logger.info(
                "[WbCard] Serper attempt: q='%s' http=%s err=%s (transient)",
                serper_query[:120],
                _status if _status is not None else "n/a",
                exc,
            )
            return []

        organic = getattr(results, "organic_results", None) or []
        # Инструментация (PERMANENT, production-safe): на УСПЕШНОЙ Serper-попытке
        # ответ — HTTP 200 (serper_client.raise_for_status уже прошёл; иначе сюда
        # не дошли бы). Логируем per-attempt query + http=200 + organic-count ДО
        # извлечения nm_id, чтобы zero-флак (200, но 0 organic) был виден в проде
        # отдельно от «200, organic есть, но 0 nm_id». Дёшево, без PII, без prints.
        logger.info(
            "[WbCard] Serper attempt: q='%s' http=200 organic=%d",
            serper_query[:120], len(organic),
        )

        # Собираем (nm_id, prefer_ru) в порядке появления, дедуп.
        ordered_main: list[int] = []   # с wildberries.ru
        ordered_mirror: list[int] = []  # зеркала / относительные
        seen: set[int] = set()
        for item in organic:
            link = getattr(item, "link", "") or ""
            m = _NM_ID_RE.search(link)
            if not m:
                continue
            try:
                nm_id = int(m.group(1))
            except (ValueError, TypeError):
                continue
            if nm_id in seen:
                continue
            seen.add(nm_id)
            if "wildberries.ru" in link.lower():
                ordered_main.append(nm_id)
            else:
                ordered_mirror.append(nm_id)

        nm_ids = (ordered_main + ordered_mirror)[:_MAX_CANDIDATES]
        logger.info("[WbCard] Serper → %d уникальных nm_id: %s", len(nm_ids), nm_ids)
        return nm_ids

    async def _fetch_card(
        self,
        client: httpx.AsyncClient,
        nm_id: int,
    ) -> Optional[dict]:
        """Скачать card.json с CDN. Простой GET; на 404 перебор NN (01..21)."""
        primary_nn = _basket_nn_from_table(nm_id)
        # Сначала known NN, затем остальные (без повтора primary).
        order = [primary_nn] + [nn for nn in _ALL_BASKET_NN if nn != primary_nn]
        for nn in order:
            data = await self._try_basket(client, nn, nm_id)
            if data is not None:
                if nn != primary_nn:
                    logger.info("[WbCard] nm=%s найден на basket-%s (fallback)", nm_id, nn)
                return data
        logger.info("[WbCard] card.json не найден ни на одном basket для nm=%s", nm_id)
        return None

    async def _try_basket(
        self,
        client: httpx.AsyncClient,
        nn: str,
        nm_id: int,
    ) -> Optional[dict]:
        """Один CDN GET. None если non-200 / не JSON / network err."""
        url = _card_url(nn, nm_id)
        try:
            r = await client.get(url)
        except (httpx.TimeoutException, httpx.HTTPError) as exc:
            logger.debug("[WbCard] CDN network err %s: %s", url, exc)
            return None
        if r.status_code != 200:
            return None
        try:
            data = r.json()
        except ValueError:  # включает json.JSONDecodeError
            return None
        except Exception as exc:  # noqa: BLE001 — на всякий случай не падаем
            logger.debug("[WbCard] card.json parse err %s: %s", url, exc)
            return None
        return data if isinstance(data, dict) else None

    # ------------------------------------------------------------------
    # Card JSON parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_options(card: dict) -> list[dict]:
        """Извлечь characteristics из WB card.json.

        Структуры:
          - options: [{"name": "Цвет", "value": "Белый"}, ...]
          - grouped_options: [{"group_name": "...", "options": [...]}]
          - compositions: [{"name": "хлопок", "value": "100"}] ИЛИ ["хлопок 100%"]
            → Состав.

        Возвращает [{name, value, value_ids=[]}], deduped по lowercase name.
        """
        out: list[dict] = []
        seen: set[str] = set()

        def _push(name: Any, value: Any) -> None:
            if not isinstance(name, str):
                return
            name = name.strip()
            if not name:
                return
            name_low = name.lower()
            if name_low in seen:
                return
            if isinstance(value, list):
                texts = [str(v).strip() for v in value if str(v).strip()]
                if not texts:
                    return
                value_str = ", ".join(texts)
            elif isinstance(value, (str, int, float)):
                value_str = str(value).strip()
                if not value_str:
                    return
            else:
                return
            seen.add(name_low)
            out.append({"name": name, "value": value_str, "value_ids": []})

        # 1. Плоские options
        for o in card.get("options") or []:
            if isinstance(o, dict):
                _push(o.get("name"), o.get("value"))

        # 2. grouped_options
        for grp in card.get("grouped_options") or []:
            if isinstance(grp, dict):
                for o in grp.get("options") or []:
                    if isinstance(o, dict):
                        _push(o.get("name"), o.get("value"))

        # 3. compositions → Состав. WB отдаёт либо список {name,value} (доля
        #    материала), либо список строк. Собираем в одну пару «Состав».
        comp_parts: list[str] = []
        for c in card.get("compositions") or []:
            if isinstance(c, dict):
                cname = str(c.get("name") or "").strip()
                cval = c.get("value")
                cval_str = str(cval).strip() if isinstance(cval, (str, int, float)) else ""
                if cname and cval_str:
                    comp_parts.append(f"{cname} {cval_str}")
                elif cname:
                    comp_parts.append(cname)
            elif isinstance(c, str) and c.strip():
                comp_parts.append(c.strip())
        if comp_parts and "состав" not in seen:
            seen.add("состав")
            out.append({"name": "Состав", "value": ", ".join(comp_parts), "value_ids": []})

        return out

    @staticmethod
    def _extract_image_urls(card: dict, nm_id: int, limit: int = 5) -> list[str]:
        """Собрать high-res photo URLs из media.photos[] или media.photo_count."""
        out: list[str] = []
        media = card.get("media") or {}

        for p in media.get("photos") or []:
            if isinstance(p, dict):
                for k in ("url", "big", "src"):
                    url = p.get(k)
                    if isinstance(url, str) and url.startswith("http"):
                        if url not in out:
                            out.append(url)
                        break
            elif isinstance(p, str) and p.startswith("http"):
                if p not in out:
                    out.append(p)
            if len(out) >= limit:
                return out

        photo_count = media.get("photo_count")
        if isinstance(photo_count, int) and photo_count > 0:
            nn = _basket_nn_from_table(nm_id)
            vol = nm_id // 100_000
            part = nm_id // 1000
            base = f"https://basket-{nn}.wbbasket.ru/vol{vol}/part{part}/{nm_id}/images/big"
            for i in range(1, min(photo_count, limit - len(out)) + 1):
                url = f"{base}/{i}.webp"
                if url not in out:
                    out.append(url)
                if len(out) >= limit:
                    break
        return out[:limit]

    # ------------------------------------------------------------------
    # Match scoring
    # ------------------------------------------------------------------

    def _pick_best_card(
        self,
        query: str,
        cat_leaf: Optional[str],
        target_type: Optional[str],
        cards: list[tuple[int, dict]],
    ) -> Optional[tuple[int, dict, str, float]]:
        """Выбрать карточку среди скачанных: тип-гейт → матч → богатство.

        Прежняя стратегия брала ОДИН лучший по fuzzy-скору вслепую — если у
        лидера была бедная карточка (3 options), мы теряли соседнюю богатую с
        чуть меньшим скором. Новая логика:

          0. ЖЁСТКИЙ ТИП-ГЕЙТ: если известен целевой тип (`target_type`, лемма),
             отбрасываем кандидатов, чей `subj_name`/`subj_root_name` несовместим
             с типом (цель «куртка» → карточка «шорты/свитшот/брюки» отвергается,
             а не штрафуется). Совместимость — по совпадению лемм + мягкий гард
             однокоренных форм (футболка↔футболки). Если ВСЕ кандидаты не того
             типа — лучше вернуть None (честный 0), чем подмешать чужой тип.
          1. Скорим каждый кандидат по заголовку (imt_name/subj_name) через
             _pick_best_match (model-token бонус, type-mismatch штраф) и считаем
             число полезных options (_extract_options).
          2. Отсекаем нерелевантные (score < _BRAND_LINE_THRESHOLD) — это
             type/brand-mismatch, такие карточки не нужны даже если богатые.
          3. Среди релевантных тип уже гарантирован гейтом, бренд/модель —
             порогом матча, поэтому БОГАТСТВО (n_opts) становится основным
             ключом выбора: брендовая базовая вещь по fuzzy часто дальше бедной
             на разрыв >_SCORE_TIE_BAND, но именно она правильная. Гард: при
             заданных model-токенах запроса model-совпавшие кандидаты
             приоритетнее (не утащить богатую карточку чужой модели), далее
             n_opts, далее score.

        Возвращает (nm_id, card, title, score) или None.
        """
        # ---- Шаг 0: жёсткий тип-гейт ----
        if target_type:
            gated: list[tuple[int, dict]] = []
            rejected: list[str] = []
            for nm_id, card in cards:
                subj_lemmas = _card_subj_lemmas(card)
                if not subj_lemmas:
                    # subj не задан — не можем судить о типе, пропускаем дальше
                    # (fuzzy/score-фильтр ниже отработает по заголовку).
                    gated.append((nm_id, card))
                    continue
                if self._type_compatible(target_type, subj_lemmas):
                    gated.append((nm_id, card))
                else:
                    rejected.append(f"nm={nm_id}(subj={'/'.join(sorted(subj_lemmas))})")
            if rejected:
                logger.info(
                    "[WbCard] тип-гейт: цель='%s' отбраковал %d/%d карточек: %s",
                    target_type, len(rejected), len(cards), ", ".join(rejected),
                )
            if not gated:
                logger.info(
                    "[WbCard] тип-гейт: НИ ОДНОЙ карточки типа '%s' — честный 0 "
                    "(не подмешиваем чужой тип)", target_type,
                )
                return None
            cards = gated

        # Модель-токены запроса (артикулы 501/M/Resolve2) — если заданы, карточка
        # ОБЯЗАНА их разделять, чтобы считаться model-совпавшей. Это гард против
        # выбора богатой карточки чужой модели только за число опций.
        q_models = _extract_model_tokens(query)

        scored: list[dict] = []
        for nm_id, card in cards:
            title = self._card_title(card)
            _, score = self._pick_best_match(
                query, [{"title": title}], category_leaf=cat_leaf
            )
            n_opts = len(self._extract_options(card))
            # model_match: запрос без model-токенов → нечего различать (True для
            # всех); иначе True только при пересечении model-токенов карточки.
            model_match = (not q_models) or bool(
                q_models & _extract_model_tokens(title)
            )
            scored.append({
                "nm_id": nm_id,
                "card": card,
                "title": title,
                "score": score,
                "n_opts": n_opts,
                "model_match": model_match,
            })

        # Отсечь нерелевантные по матчу (type/brand-mismatch).
        relevant = [c for c in scored if c["score"] >= _BRAND_LINE_THRESHOLD]
        if not relevant:
            # Все кандидаты ниже порога матча — деградируем к старому поведению:
            # вернуть абсолютного лидера по скору, дальше _classify_match → skip.
            top = max(scored, key=lambda c: c["score"], default=None)
            if top is None:
                return None
            return (top["nm_id"], top["card"], top["title"], top["score"])

        # Тип уже гарантирован тип-гейтом, бренд/модель — порогом матча. Среди
        # этих релевантных кандидатов БОГАТСТВО (n_opts) — основной ключ выбора:
        # брендовая базовая вещь («Nike Футболка Nsw Club Tee», 9 опций) часто
        # по fuzzy дальше бедной («Футболка Sportswear Club», 5 опций) на разрыв
        # >_SCORE_TIE_BAND, но именно богатая правильная. Раньше tie-band не давал
        # её выбрать. ГАРД: при заданных model-токенах запроса model-совпавшие
        # кандидаты приоритетнее (чтобы не утащить богатую карточку чужой модели);
        # при равной модельности решает n_opts, далее score.
        best = max(relevant, key=lambda c: (c["model_match"], c["n_opts"], c["score"]))

        if logger.isEnabledFor(logging.INFO):
            ranking = ", ".join(
                f"nm={c['nm_id']}(score={c['score']:.1f},opts={c['n_opts']}"
                f",mm={int(c['model_match'])})"
                for c in sorted(relevant, key=lambda c: (-c["n_opts"], -c["score"]))
            )
            logger.info(
                "[WbCard] pick: %d релевантных → выбран nm=%s (score=%.1f, opts=%d) | %s",
                len(relevant), best["nm_id"], best["score"], best["n_opts"], ranking,
            )

        return (best["nm_id"], best["card"], best["title"], best["score"])

    @staticmethod
    def _type_compatible(target_type: str, subj_lemmas: set[str]) -> bool:
        """Совместим ли тип карточки (subj-леммы) с целевым типом (лемма).

        Жёсткий гейт по типу одежды/товара:
          - точное совпадение лемм («куртка» ∈ {«куртка»}) → True;
          - однокоренные формы через общую основу (футболка↔футболочка,
            куртка↔курточка) → мягко True (стем-affinity ≥ порога);
          - разные корни (куртка vs шорты/свитшот/брюки) → False (реджект).

        Гард от ложного реджекта: учитываем ТОЛЬКО морфологически близкие формы,
        не синонимы разных корней. Это намеренно: «футболка» и «майка» — разные
        леммы, но карточка-«майка» под цель-«футболка» отсекается жёстко, как и
        просили (лучше честный 0, чем чужой тип). Семантические синонимы тут НЕ
        раскрываем — только формы одного корня.
        """
        if target_type in subj_lemmas:
            return True
        # Однокоренные формы: длинная общая основа (≥ 5 симв.) при близкой длине.
        for subj in subj_lemmas:
            common = 0
            for x, y in zip(target_type, subj):
                if x == y:
                    common += 1
                else:
                    break
            shorter = min(len(target_type), len(subj))
            if shorter and common >= 5 and common / shorter >= 0.7:
                return True
        return False

    @staticmethod
    def _card_title(card: dict) -> str:
        """Заголовок карточки для fuzzy-сравнения: brand + imt_name + subj_name."""
        parts: list[str] = []
        for key in ("selling", "imt_name", "subj_name", "subj_root_name"):
            val = card.get(key)
            if key == "selling" and isinstance(val, dict):
                val = val.get("brand_name")
            if isinstance(val, str) and val.strip():
                if val.strip().lower() not in " ".join(parts).lower():
                    parts.append(val.strip())
        return " ".join(parts).strip()

    @staticmethod
    def _pick_best_match(
        query: str,
        tiles: list[dict],
        category_leaf: Optional[str] = None,
    ) -> tuple[Optional[dict], float]:
        """rapidfuzz match с model-token бонусом и type-mismatch штрафом.

        Логика идентична OzonCardSource._pick_best_match: partial_ratio +
        token_sort_ratio, +5 за общий артикул, -30 за несовпадение типа товара
        (только для одежды без артикула).
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return (tiles[0], 100.0) if tiles else (None, 0.0)

        _MODEL_BONUS = 5.0
        _MODEL_BONUS_ALPHA = 3.0  # словесная модель — слабее цифрового артикула
        _TYPE_MISMATCH_PENALTY = 30.0
        _GENDER_MISMATCH_PENALTY = 30.0  # как type-mismatch: карточка чужого пола проигрывает
        q_models = _extract_model_tokens(query)
        q_alpha = _extract_alpha_model_tokens(query)
        # Гендер-сигнал имени товара. Нейтральное имя («Ultraboost 22») → None →
        # штраф не применяется (нельзя утверждать, что карточка неверного пола).
        q_gender = _extract_gender_signal(query)
        best_tile: Optional[dict] = None
        best_score = 0.0
        q = query.lower()
        cat_leaf_low = category_leaf.strip().lower() if category_leaf else None
        for tile in tiles:
            title = (tile.get("title") or "").strip()
            if not title:
                continue
            t = title.lower()
            score = (fuzz.partial_ratio(q, t) + fuzz.token_sort_ratio(q, t)) / 2.0
            if q_models and q_models & _extract_model_tokens(title):
                score += _MODEL_BONUS
            # Алфавитный (словесный) модель-бонус: для словесных моделей без цифр
            # (Resolve, Ultraboost). Меньше цифрового, отдельно от q_models —
            # чтобы НЕ отключить type-mismatch штраф (он гейтится по q_models).
            elif q_alpha and q_alpha & _extract_alpha_model_tokens(title):
                score += _MODEL_BONUS_ALPHA
            if cat_leaf_low and not q_models and cat_leaf_low not in t:
                score -= _TYPE_MISMATCH_PENALTY
            # Гендер-штраф: имя несёт явный пол И заголовок карточки (включает
            # subj_name/subj_root_name через _card_title) несёт ПРОТИВОРЕЧАЩИЙ пол
            # → карточка не того гендера проигрывает. Унисекс/нейтрал — без штрафа.
            if q_gender is not None and _gender_conflict(query, title):
                score -= _GENDER_MISMATCH_PENALTY
            if score > best_score:
                best_score = score
                best_tile = tile
        return best_tile, best_score

    @staticmethod
    def _classify_match(score: float) -> str:
        if score >= _EXACT_THRESHOLD:
            return "exact"
        if score >= _BRAND_LINE_THRESHOLD:
            return "brand_line"
        return "skip"

    # ------------------------------------------------------------------
    # Mapping: char name → target attribute_id → value_id (Ozon dict)
    # ------------------------------------------------------------------

    def _map_characteristics(
        self,
        chars: list[dict],
        targets: list[TargetAttribute],
        context: ExtractionContext,
        mode: str,
        title: str,
        score: float,
        card: Optional[dict] = None,
    ) -> list[AttributeValue]:
        """Сопоставить WB-char names с target.name через Ozon dictionary.

        Маппим WB-характеристики на наш Ozon словарь — финальное API
        публикации у нас Ozon, а WB-карточка лишь источник данных.
        """
        ozon_chars: list[dict] = []
        cat_id: Optional[int] = None
        type_id: Optional[int] = None
        try:
            cat_id = int(context.category_id) if context.category_id else None
            type_id = context.ozon_type_id
            if cat_id and type_id:
                ozon_chars = get_ozon_characteristics_for_type(cat_id, type_id)
        except (ValueError, TypeError):
            cat_id = None
            type_id = None

        attr_id_to_dict_name: dict[int, str] = {}
        for oc in ozon_chars:
            if isinstance(oc, dict) and "id" in oc and "name" in oc:
                attr_id_to_dict_name[int(oc["id"])] = str(oc["name"])

        target_names_low: dict[int, set[str]] = {}
        for t in targets:
            names = {t.name.lower()}
            dn = attr_id_to_dict_name.get(t.id)
            if dn:
                names.add(dn.lower())
            target_names_low[t.id] = names

        name_to_target_id: dict[str, int] = {}
        for tid, names in target_names_low.items():
            for n in names:
                name_to_target_id.setdefault(n, tid)

        try:
            from rapidfuzz import process, fuzz
            all_target_names = list(name_to_target_id.keys())
        except ImportError:
            process = None
            fuzz = None
            all_target_names = []

        target_by_id: dict[int, TargetAttribute] = {t.id: t for t in targets}

        # Verified WB→Ozon field map (eg-importer). Имя WB-характеристики →
        # Ozon attr id напрямую (без fuzzy). Используется как первый шаг гейта:
        # если verified id совпадает с одной из текущих targets — берём его,
        # иначе падаем на старую fuzzy-логику. wb_subject из card.json; если его
        # нет — fallback-merge по (cat_id, type_id) внутри eg_get_field_map.
        wb_subject: Optional[str] = None
        if card:
            for key in ("subj_name", "subj_root_name"):
                val = card.get(key)
                if isinstance(val, str) and val.strip():
                    wb_subject = val.strip()
                    break
        verified_map: dict[str, int] = {}
        try:
            verified_map = eg_get_field_map(wb_subject, cat_id, type_id)
        except Exception as exc:
            logger.debug("[WbCard] eg_get_field_map failed: %s", exc)

        evidence_short = f"wb:{title[:50]} | match={score:.1f}"
        conf = _CONF_EXACT if mode == "exact" else _CONF_BRAND_LINE

        # Гендер-страховка (п.3) для поля «Пол», пришедшего из brand_line-карточки
        # (не exact). Имя товара vs гендер карточки (title включает subj_name).
        name_gender = _extract_gender_signal(context.product_name or "")
        card_gender = _extract_gender_signal(title)

        results: list[AttributeValue] = []
        used_ids: set[int] = set()

        for c in chars:
            char_name = c["name"].strip()
            char_val = c["value"].strip()
            char_name_low = char_name.lower()

            if mode == "brand_line" and char_name_low in _BRAND_LINE_BLACKLIST:
                continue

            target_id: Optional[int] = None

            # 1) Verified path: точный WB name → Ozon attr id из field-map.
            #    Берём только если этот id присутствует среди текущих targets
            #    (иначе verified-id нерелевантен этому запросу публикации).
            verified_id = verified_map.get(char_name_low)
            if verified_id is not None and verified_id in target_by_id:
                target_id = verified_id

            # 2) Fallback: существующая exact/substring/fuzzy логика (без изменений).
            if target_id is None:
                target_id = name_to_target_id.get(char_name_low)
            if target_id is None:
                for tn, tid in name_to_target_id.items():
                    if char_name_low in tn or tn in char_name_low:
                        target_id = tid
                        break
            if target_id is None and process is not None and all_target_names:
                best = process.extractOne(
                    char_name_low, all_target_names, scorer=fuzz.WRatio,
                )
                if best is not None and best[1] >= 88:
                    target_id = name_to_target_id[best[0]]

            if target_id is None or target_id in used_ids:
                continue

            target = target_by_id.get(target_id)
            if target is None:
                continue
            used_ids.add(target_id)

            # Гендер-страховка (п.3): поле «Пол» из brand_line-карточки (не exact).
            # value_gender — пол, который карточка хочет записать. Источники
            # противоречия: (a) имя товара несёт явный пол, конфликтующий со
            # значением → СКИП (как Adidas-кейс, если имя гендерное); (b) имя
            # нейтрально, но карточка-донор сама гендерная (card_gender) и
            # конфликтует со значением → значение навеяно чужим полом донора →
            # понижаем confidence. Exact-режим и нейтральные совпадения не трогаем.
            target_conf = conf
            if mode == "brand_line" and _is_gender_target_name(target.name):
                value_gender = _extract_gender_signal(char_val)
                if value_gender is not None and value_gender != "unisex":
                    if name_gender is not None and value_gender != name_gender \
                            and name_gender != "unisex":
                        logger.info(
                            "[WbCard] гендер-страховка: СКИП '%s'='%s' "
                            "(имя='%s' пол=%s vs значение=%s)",
                            target.name, char_val, context.product_name,
                            name_gender, value_gender,
                        )
                        continue
                    if name_gender is None and card_gender is not None \
                            and card_gender == value_gender:
                        target_conf = min(conf, _CONF_GENDER_DOWNWEIGHT)
                        logger.info(
                            "[WbCard] гендер-страховка: ПОНИЖЕН conf '%s'='%s'→%.2f "
                            "(нейтральное имя, brand_line-карточка пол=%s)",
                            target.name, char_val, target_conf, card_gender,
                        )

            # Коллекционные характеристики WB отдаёт одной строкой с разделителями
            # (";" или ","). Сплитим в список, чтобы значение участвовало в union
            # merge поэлементно и не проигрывало vision/llm целиком.
            if target.is_collection:
                parts = _split_multivalue(char_val)
            else:
                parts = None

            if parts is not None:
                value_out: Union[str, list[str]] = parts
                value_id = None
                value_ids: Optional[list[int]] = None
                if cat_id and type_id:
                    try:
                        resolved = [
                            resolve_value_id(cat_id, type_id, target.id, p) for p in parts
                        ]
                        # value_ids — Optional[list[int]]: None-элементы (часть
                        # parts не легла в словарь) недопустимы внутри списка и
                        # роняют AttributeValue ValidationError → весь extract
                        # падал в except и давал card=N. Оставляем только успешно
                        # резолвнутые int (порядок-агностично: это набор словарных
                        # ID, а не позиционное соответствие parts).
                        resolved_ids = [r for r in resolved if r is not None]
                        if resolved_ids:
                            value_ids = resolved_ids
                    except Exception as exc:
                        logger.debug("[WbCard] resolve_value_id (list) failed: %s", exc)
            else:
                value_out = char_val
                value_ids = None
                value_id = None
                if cat_id and type_id:
                    try:
                        value_id = resolve_value_id(cat_id, type_id, target.id, char_val)
                    except Exception as exc:
                        logger.debug("[WbCard] resolve_value_id failed: %s", exc)

            results.append(AttributeValue(
                attribute_id=target.id,
                value=value_out,
                confidence=target_conf,
                source=Source.WB_CARD,
                evidence=evidence_short,
                semantic_type=target.semantic_type,
                is_collection=target.is_collection,
                value_id=value_id,
                value_ids=value_ids,
            ))

        # ---- LEVER 1: emit «Российский размер» from sizes_table ----
        # sizes_table lives OUTSIDE options/characteristics in card.json.
        # The normal characteristic mapping above cannot reach it.
        #
        # Conservative mode gate: only extract sizes for EXACT-match cards.
        # brand_line cards (different model of same brand) may carry a completely
        # different size run (e.g. a Nike slim-fit tee vs an oversized tee), so
        # emitting their sizes for our product would be wrong. Exact match = same
        # product listing → same sizes.
        if mode == "exact" and card is not None:
            results = _emit_ru_size_from_card(
                card=card,
                targets=targets,
                results=results,
                used_ids=used_ids,
                cat_id=cat_id,
                type_id=type_id,
                evidence_short=evidence_short,
            )

        logger.info(
            "[WbCard] %s mode → %d характеристик скопировано (из %d candidate chars, %d targets)",
            mode, len(results), len(chars), len(targets),
        )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_for_targets(
        values: list[AttributeValue],
        effective: list[TargetAttribute],
    ) -> list[AttributeValue]:
        eff_ids = {t.id for t in effective}
        return [v for v in values if v.attribute_id in eff_ids]

    def _cache_put(self, key: tuple[str, str], value: list[AttributeValue]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > _CACHE_MAX:
            self._cache.popitem(last=False)
