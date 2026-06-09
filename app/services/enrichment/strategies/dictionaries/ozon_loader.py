"""Loader for the Ozon category characteristics dictionary.

The dictionary is built by scripts/build_ozon_dictionary.py and stored at
app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json.

Schema v2 (current):
{
    "schema_version": 2,
    "source": "ozon_seller_api",
    "categories": {
        "<description_category_id>:<type_id>": {
            "description_category_id": int,
            "type_id": int,
            "name": str,
            "path": [str, ...],
            "characteristics": [
                {"id": int, "name": str, "type": str,
                 "is_required": bool, "is_collection": bool, "description": str},
                ...
            ]
        }
    }
}
"""
import gzip
import json
import logging
from pathlib import Path
from typing import Optional
from functools import lru_cache


DATA_DIR = Path(__file__).parent / "data"

logger = logging.getLogger(__name__)

# Module-level singleton for MatcherService (lazy, None until first successful load)
_matcher_instance = None
_matcher_attempted = False


def _get_matcher():
    """Return singleton MatcherService, loading on first call. Returns None on failure."""
    global _matcher_instance, _matcher_attempted
    if _matcher_attempted:
        return _matcher_instance
    _matcher_attempted = True
    try:
        from app.services.matcher import MatcherService
        _matcher_instance = MatcherService(cache_manager=None)
    except Exception as exc:
        logger.warning("MatcherService unavailable (fuzzy/semantic fallback disabled): %s", exc)
        _matcher_instance = None
    return _matcher_instance


@lru_cache(maxsize=1)
def load_ozon_dictionary() -> dict:
    """Load the Ozon dictionary from .json.gz (preferred) or plain .json. Cached."""
    gz_path = DATA_DIR / "ozon_dictionary.json.gz"
    plain_path = DATA_DIR / "ozon_dictionary.json"
    if gz_path.exists():
        with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
    elif plain_path.exists():
        data = json.loads(plain_path.read_text(encoding="utf-8"))
    else:
        return {}
    # Оба формата (legacy flat и schema_version с оберткой) возвращают одинаково
    if "categories" in data:
        return data["categories"]
    return data


def get_ozon_characteristics_for_type(
    description_category_id: int,
    type_id: int,
) -> list[dict]:
    """Вернуть список характеристик для конкретной (category_id, type_id) пары.

    Используется когда ExtractionContext содержит оба поля (основной путь для v2).
    Возвращает [] если пара не найдена в словаре.
    """
    d = load_ozon_dictionary()
    key = f"{description_category_id}:{type_id}"
    entry = d.get(key)
    return entry["characteristics"] if entry else []


def get_ozon_entries_for_category(description_category_id: int) -> list[dict]:
    """Вернуть все type-записи для данного description_category_id.

    Используется как fallback когда type_id неизвестен — перебираем все type-варианты
    категории и возвращаем характеристики первого найденного типа (обычно type_id
    единственный или характеристики совпадают между типами одной категории).
    """
    d = load_ozon_dictionary()
    prefix = f"{description_category_id}:"
    return [v for k, v in d.items() if k.startswith(prefix)]


def get_ozon_characteristics_for_category(description_category_id: int) -> list[dict]:
    """Вернуть характеристики для category_id (без type_id).

    Совместимый API для случаев когда type_id недоступен в контексте.
    При нескольких type_id для одной категории возвращает характеристики первого
    найденного type — достаточно для normalize_target (обогащение метаданными).

    Также поддерживает legacy-формат (plain '<cat_id>' ключи).
    """
    d = load_ozon_dictionary()
    # Сначала пробуем legacy-формат (ключ = строка category_id)
    entry = d.get(str(description_category_id))
    if entry:
        return entry["characteristics"]
    # Затем v2 compound-ключи — берём первый подходящий type
    entries = get_ozon_entries_for_category(description_category_id)
    return entries[0]["characteristics"] if entries else []


def get_ozon_category_name(description_category_id: int) -> Optional[str]:
    """Вернуть human-readable имя категории, или None."""
    d = load_ozon_dictionary()
    # Legacy-формат
    entry = d.get(str(description_category_id))
    if entry:
        return entry["name"]
    # v2 compound-ключи
    entries = get_ozon_entries_for_category(description_category_id)
    return entries[0]["name"] if entries else None


# Homoglyph map: Latin letters that look identical to Cyrillic ones.
# E.g. Latin 'A' vs Cyrillic 'А' are different Unicode codepoints but identical glyphs.
# LLMs and PDFs frequently mix these.
_LATIN_TO_CYRILLIC_HOMOGLYPHS = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
})

# Common technical term aliases: Latin technical token → Cyrillic equivalent.
# Applied BEFORE normalization to align with Ozon dictionary canonical wording.
_TECH_ALIASES = [
    # PSU modularity
    ("fully-modular", "полностью модульный"),
    ("fully modular", "полностью модульный"),
    ("semi-modular", "полумодульный"),
    ("semi modular", "полумодульный"),
    ("non-modular", "немодульный"),
    ("non modular", "немодульный"),
    ("modular", "модульный"),
    # Connector types
    (" pin ", " пин "),
    (" pin", " пин"),
    # Units (raw strings — \b is word boundary, not backspace)
    (r"\bv\b", "в"),   # Volts (e.g. "240 V" → "240 в")
    (r"\bw\b", "вт"),  # Watts
    (r"\bcm\b", "см"),
    (r"\bmm\b", "мм"),
    # Protection
    ("ocp", "защита от перегрузки по току"),
    ("ovp", "защита от перенапряжения"),
    ("uvp", "защита от пониженного напряжения"),
    ("scp", "защита от короткого замыкания"),
    ("opp", "защита от перегрузки по мощности"),
    ("otp", "защита от перегрева"),
]


def _unwrap_array_repr(value: str) -> list[str]:
    """If value looks like a list repr e.g. "['a', 'b']", return ['a', 'b'].
    Otherwise return [value]. Handles both Python list str() and JSON.
    """
    s = value.strip()
    if not (s.startswith("[") and s.endswith("]")):
        return [value]
    inner = s[1:-1]
    parts = []
    for p in inner.split(","):
        p = p.strip().strip("'\"")
        if p:
            parts.append(p)
    return parts if parts else [value]


def _strip_parens(value: str) -> str:
    """Strip trailing parenthetical: 'OCP (защита от перегрузки)' → 'OCP'."""
    import re
    return re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()


def _normalize_token(s: str) -> str:
    """Strong normalization. Order matters:
    1. lower
    2. tech aliases (Latin patterns → Cyrillic, e.g. fully-modular → полностью модульный)
    3. Latin homoglyph → Cyrillic (catches stray Latin lookalikes after aliases)
    4. ё → е
    5. Collapse spaces/dashes, strip punct.
    """
    import re
    s = s.lower()
    for src, dst in _TECH_ALIASES:
        s = re.sub(src, dst, s)
    s = s.translate(_LATIN_TO_CYRILLIC_HOMOGLYPHS)
    s = s.replace("ё", "е")
    s = s.replace("-", " ").replace("+", " + ")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,;:!?\"'()[]")


# ---------------------------------------------------------------------------
# Гендер-синоним-нормализатор (генеральный, без хардкода категорий)
# ---------------------------------------------------------------------------
# Карточка/источники отдают пол как свободные фразы («Мужчинам», «Для мужчин»,
# «men's»), а словарь Ozon хранит канон «Мужской/Женский/Девочки/Мальчики».
# fuzzy≥85 такие фразы НЕ мапит на канон (другая основа), и value_id теряется.
# Здесь — явный map вариант→канон, применяется ДО резолва value_id для целей-пола.
#
# КРИТИЧНО: детские каноны (Девочки/Мальчики) остаются ОТДЕЛЬНЫМИ от взрослых
# (Женский/Мужской) — НИКОГДА не схлопываем детей во взрослых. Порядок проверки:
# сначала детские (длиннее/специфичнее), потом взрослые, чтобы «для девочек» не
# поймался взрослым «жен».
_EG_GENDER_CANON_SYNONYMS: tuple[tuple[str, frozenset[str]], ...] = (
    ("Девочки", frozenset({
        "девочки", "девочка", "для девочек", "girls", "girl",
    })),
    ("Мальчики", frozenset({
        "мальчики", "мальчик", "для мальчиков", "boys", "boy",
    })),
    ("Мужской", frozenset({
        "муж", "мужской", "мужская", "мужское", "мужчинам", "мужчины",
        "для мужчин", "men", "men's", "mens", "male",
    })),
    ("Женский", frozenset({
        "жен", "женский", "женская", "женское", "женщинам", "женщины",
        "для женщин", "women", "women's", "womens", "female",
    })),
)


def normalize_gender_value(s: str) -> Optional[str]:
    """Сопоставить гендер-вариант его канону Ozon (Мужской/Женский/Девочки/Мальчики).

    Генеральный, без хардкода категорий: только явный словарь вариант→канон.
    Вход нормализуется (lower+strip) перед сравнением. Детские каноны держим
    ОТДЕЛЬНО от взрослых — «для девочек» → «Девочки», НЕ «Женский».

    Возвращает канон-строку или None (если вариант неизвестен — не выдумываем).
    """
    if not s:
        return None
    key = str(s).strip().lower()
    if not key:
        return None
    for canon, variants in _EG_GENDER_CANON_SYNONYMS:
        if key in variants:
            return canon
    return None


def _try_match_one_value(value: str, values_list: list) -> Optional[int]:
    """Try multiple match strategies for a single value string against dict values_list."""
    # Gender-synonym pre-pass: вариант пола («Мужчинам»/«men's») → канон Ozon
    # («Мужской»), затем матчим канон по словарю exact-ом. Срабатывает ТОЛЬКО
    # когда (а) value — известный гендер-вариант И (б) канон есть в values_list,
    # т.е. для НЕгендерных атрибутов нейтрально (канона в словаре нет → пропуск).
    gender_canon = normalize_gender_value(value)
    if gender_canon is not None:
        canon_lower = gender_canon.lower()
        for entry in values_list:
            if str(entry.get("value", "")).lower() == canon_lower:
                return entry.get("id")

    # Strategy 0.5: digit-normalized prefix match for pure-numeric codes (e.g. TNVED).
    # TNVED codes come in different granularities: the LLM may produce "6109100010"
    # (EAEU 10-digit subposition) while Ozon stores "6109100000 - Футболки...".
    # Both sides are stripped to digits-only, then we match when one code is a
    # prefix of the other (min 6 digits). Only activates for pure-digit inputs —
    # has zero effect on text attributes.
    import re as _re
    _digits_only = _re.sub(r"\D", "", value)
    if _digits_only == value and len(_digits_only) >= 6:
        # Digit-normalized prefix match for customs codes (TNVED/HS/CN/EAEU).
        #
        # Problem: LLM produces EAEU 10-digit code "6109100010" (national subposition)
        # while Ozon stores "6109100000 - Футболки..." (HS-8 base, last 2 = "00").
        # They share the first 8 digits ("61091000") — the HS-8 subheading — but
        # differ in the last 2 national subposition digits.
        #
        # Strategy: extract digits-only from the option's code portion, then compare
        # the first min(8, len_target, len_option) digits. 8 = HS-8 subheading level
        # (stable, no national divergence). Falls through to prefix check for shorter
        # codes (6-digit HS heading or 4-digit HS chapter lookups).
        #
        # Only activates for pure-digit inputs ≥ 6 digits — zero effect on text attrs.
        _HS8_LEVEL = 8
        _MIN_DIGITS = 6
        for entry in values_list:
            entry_val = str(entry.get("value", ""))
            # Extract the code portion: everything before " - " separator handles both
            # "6109100000 - Футболки..." and spaced "6109 10 000 0 - ..." formats.
            code_part = entry_val.split(" - ")[0] if " - " in entry_val else entry_val
            entry_digits = _re.sub(r"\D", "", code_part)
            if not entry_digits or len(entry_digits) < _MIN_DIGITS:
                continue
            cmp_len = min(len(_digits_only), len(entry_digits), _HS8_LEVEL)
            if cmp_len >= _MIN_DIGITS and _digits_only[:cmp_len] == entry_digits[:cmp_len]:
                return entry.get("id")

    # Strategy 1: exact case-insensitive
    value_lower = value.lower()
    for entry in values_list:
        if str(entry.get("value", "")).lower() == value_lower:
            return entry.get("id")

    # Strategy 2: strong normalization on both sides
    value_norm = _normalize_token(value)
    for entry in values_list:
        if _normalize_token(str(entry.get("value", ""))) == value_norm:
            return entry.get("id")

    # Strategy 3: strip parentheses, retry exact + normalized
    value_stripped = _strip_parens(value)
    if value_stripped != value:
        vs_norm = _normalize_token(value_stripped)
        for entry in values_list:
            entry_val = str(entry.get("value", ""))
            if entry_val.lower() == value_stripped.lower():
                return entry.get("id")
            if _normalize_token(entry_val) == vs_norm:
                return entry.get("id")
            # Also try stripping parens from dict side
            entry_stripped = _strip_parens(entry_val)
            if _normalize_token(entry_stripped) == vs_norm:
                return entry.get("id")

    # Strategy 4: rapidfuzz on normalized tokens
    try:
        from rapidfuzz import fuzz, process
        options = [(str(e.get("value", "")), e.get("id")) for e in values_list]
        norm_options = [_normalize_token(o[0]) for o in options]
        best = process.extractOne(value_norm, norm_options, scorer=fuzz.WRatio)
        if best and best[1] >= 85:  # lowered from 90 (more recall)
            return options[best[2]][1]
    except Exception as exc:
        logger.warning("resolve_value_id rapidfuzz fallback failed: %s", exc)
    return None


def resolve_value_id(
    cat_id: int,
    type_id: int,
    attribute_id: int,
    value: str,
) -> Optional[int]:
    """Найти словарный id для строкового значения характеристики.

    Стратегии (по убыванию строгости):
      1. Exact case-insensitive match.
      2. Strong normalization: Latin homoglyphs → Cyrillic, tech aliases, ё→е.
      3. Strip trailing parentheses content.
      4. Rapidfuzz WRatio ≥ 85.
      5. Если value — list repr ([...]), try each item then aggregate.
      6. Semantic match via MatcherService (fallback).
    """
    chars = get_ozon_characteristics_for_type(cat_id, type_id)
    char = next((c for c in chars if c.get("id") == attribute_id), None)
    if not char:
        return None
    values_list = char.get("values")
    if not values_list:
        return None

    # Try direct match strategies
    vid = _try_match_one_value(value, values_list)
    if vid is not None:
        return vid

    # Try list unwrapping: if value is "['a', 'b']", try matching first/best item
    items = _unwrap_array_repr(value)
    if len(items) > 1 or (len(items) == 1 and items[0] != value):
        for item in items:
            vid = _try_match_one_value(item, values_list)
            if vid is not None:
                return vid

    # Semantic match via MatcherService (если sentence_transformers доступен)
    try:
        matcher = _get_matcher()
        if matcher is None:
            return None
        options = [str(e.get("value", "")) for e in values_list]
        matched = matcher.find_best_match(str(value), options)
        if matched is not None:
            for entry in values_list:
                if str(entry.get("value", "")) == matched:
                    return entry.get("id")
    except Exception as exc:
        logger.warning("resolve_value_id matcher fallback failed: %s", exc)
    return None


def get_attr_value_options(
    cat_id: int,
    type_id: int,
    attribute_id: int,
) -> list[str]:
    """Вернуть список строковых allowed-значений (value-строк) для характеристики.

    Используется LLM-резолвером хвоста: ему нужен полный список разрешённых
    значений из словаря (не truncated target.allowed_values), чтобы выбрать
    ровно одно. Возвращает [] если характеристика/значения не найдены.
    """
    chars = get_ozon_characteristics_for_type(cat_id, type_id)
    char = next((c for c in chars if c.get("id") == attribute_id), None)
    if not char:
        return []
    values_list = char.get("values") or []
    return [str(e.get("value", "")) for e in values_list if e.get("value")]


def get_attr_value_pairs(
    cat_id: int,
    type_id: int,
    attribute_id: int,
) -> dict[str, int]:
    """Вернуть {value-строка: dict-id} для характеристики (тот же источник, что и
    get_attr_value_options, но СОХРАНЯЕТ id).

    Нужен brand-from-name резолверу: он матчит бренд из имени против СТРОК словаря,
    а затем должен привязать его словарный value_id ТОЧНО (без fuzzy) — owner
    чувствителен к неверным id. Возвращает {} если характеристика/значения не
    найдены. Ключи — ровно как в словаре (case/ё сохранены); сравнение exact-ом
    делает вызывающая сторона.
    """
    chars = get_ozon_characteristics_for_type(cat_id, type_id)
    char = next((c for c in chars if c.get("id") == attribute_id), None)
    if not char:
        return {}
    values_list = char.get("values") or []
    pairs: dict[str, int] = {}
    for e in values_list:
        val = str(e.get("value", ""))
        vid = e.get("id")
        if val and vid is not None:
            pairs[val] = vid
    return pairs


def get_attr_value_options_any_type(
    cat_id: int,
    attribute_id: int,
) -> list[str]:
    """Вернуть список строковых allowed-значений для attr без знания type_id.

    Перебирает ВСЕ type-записи категории и возвращает значения из ПЕРВОЙ записи,
    у которой для attribute_id есть непустой values-список. Используется как
    fallback когда ozon_type_id недоступен в контексте — например, для
    brand_value_options при ozon_type_id=None.

    Возвращает [] если ни одна запись не содержит непустых значений.
    """
    entries = get_ozon_entries_for_category(cat_id)
    for entry in entries:
        chars = entry.get("characteristics", [])
        char = next((c for c in chars if c.get("id") == attribute_id), None)
        if not char:
            continue
        values_list = char.get("values") or []
        if values_list:
            return [str(e.get("value", "")) for e in values_list if e.get("value")]
    return []


def get_attr_value_pairs_any_type(
    cat_id: int,
    attribute_id: int,
) -> dict[str, int]:
    """Вернуть {value-строка: dict-id} для attr без знания type_id.

    Перебирает ВСЕ type-записи категории и возвращает пары из ПЕРВОЙ записи,
    у которой для attribute_id есть непустой values-список. Аналог
    get_attr_value_options_any_type, но сохраняет id.

    Используется как fallback в brand_value_id_options при ozon_type_id=None.
    """
    entries = get_ozon_entries_for_category(cat_id)
    for entry in entries:
        chars = entry.get("characteristics", [])
        char = next((c for c in chars if c.get("id") == attribute_id), None)
        if not char:
            continue
        values_list = char.get("values") or []
        if not values_list:
            continue
        pairs: dict[str, int] = {}
        for e in values_list:
            val = str(e.get("value", ""))
            vid = e.get("id")
            if val and vid is not None:
                pairs[val] = vid
        return pairs
    return {}


def is_truncated(cat_id: int, type_id: int, attribute_id: int) -> bool:
    """Return True if the cached values for this attribute were truncated at 5000.

    A characteristic is considered truncated when its dict entry contains
    ``"values_truncated": true``, set by scripts/build_ozon_values.py when
    the Ozon API returned ``has_next=true`` after fetching 5000 values.

    Returns False when:
    - The (cat_id, type_id) pair is not in the dictionary.
    - The attribute_id is not found in characteristics.
    - The characteristic has no ``values_truncated`` flag (i.e. values were not
      yet enriched, or the full set fit within 5000).
    """
    d = load_ozon_dictionary()
    key = f"{cat_id}:{type_id}"
    entry = d.get(key)
    if not entry:
        return False
    for char in entry.get("characteristics", []):
        if char.get("id") == attribute_id:
            return bool(char.get("values_truncated", False))
    return False
