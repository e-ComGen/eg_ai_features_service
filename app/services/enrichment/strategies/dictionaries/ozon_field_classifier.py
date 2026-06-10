"""Classifier for Ozon characteristic fields: platform/manual vs extractable.

Generic structural detection — NOT a hardcoded name list.

Platform fields (return True) are those that:
  - have type == "URL"                         → document/image links
  - have a values dict                         → False (dict field = extractable)
  - otherwise: match a platform-instruction
    pattern in the Ozon API's own description OR a high-precision name pattern.

Real char schema (ozon_seller_api, schema_version 2):
  {"id": int, "name": str, "type": str,
   "is_required": bool, "is_collection": bool,
   "description": str,
   "values": [{"id": int, "value": str}, ...]}  # absent when no dictionary

USAGE — REPORTING ONLY. This heuristic must NEVER be used to drop targets
from the extraction pipeline: a description-text match cannot reliably
distinguish a content field from a real characteristic, and a false positive
would silently lose coverage (it once matched the required field "Название
модели (для объединения в одну карточку)" via the word "объединить"). It is
used solely to compute an honest reporting denominator (optional_honest).

The regex is deliberately HIGH-PRECISION / conservative: it only fires on
phrases that cannot plausibly appear in a real spec characteristic's
description. Ambiguous broad words (видео / оптом / маркетинговый / объединить)
were removed on purpose — better to under-exclude (honest number stays a lower
bound) than to over-exclude (overstate coverage / risk dropping real fields).

is_not_applicable(char, product_context) — MUTUALLY-EXCLUSIVE families
-----------------------------------------------------------------------
Excludes optional attributes that are LOGICALLY IMPOSSIBLE for the specific
product — i.e. they belong to a mutually-exclusive family where the product's
own identity proves a *different* member applies.

Two conservative families are implemented:

1. OS-version fields ("Версия <OS>") — exclude an OS-version attr when the
   product's actual OS is confidently known AND differs from the attr's OS.
   The product OS is derived (in priority order) from:
     a. a filled "Операционная система" attr in product_filled_attrs
     b. brand/category signals from product_name/category_path (Apple → iOS/
        iPadOS/macOS/watchOS; known Android brands + "смартфон" → Android;
        Windows laptop signals → Windows).
   Conservatism: if OS CANNOT be determined confidently → nothing excluded.
   Example: "Версия iOS" on an Android phone → excluded.
            "Версия Android" on the same Android phone → KEPT (matches).

2. Dryer-only fields on washing machines — exclude attrs whose name contains
   "...сушки" / "для сушки" / "загрузка.*сушки" when the product_name or
   category_path strongly indicates a washer WITHOUT a dryer (leaf contains
   "стиральная машина" but NOT "с сушкой").
   Conservatism: combo-machine ("с сушкой") → nothing excluded; ambiguous → kept.

High-precision signals only (description-based):
  https?://           — external URL in the seller instruction
  mp4 | mov           — video file formats (video cover / main video fields)
  json                — JSON-encoded rich-content block
  sku через           — "SKU через запятую" list entry (cross-link fields)
  seller-edu          — Ozon seller-education portal link
  соцсет              — "как в соцсетях" (hashtag field metaphor)
  rich-контент        — explicit rich-content label
  заводск.*упаковок   — factory-packaging count (logistics, not a spec)

High-precision name-based signals (_PLATFORM_NAME_RE):
  озон.видео          — «Озон.Видео: название» (video platform title field)
  объединить в похожие — «Объединить в похожие товары» (grouping field;
                          NOT «Объединить на одной карточке» which is required
                          and already guarded by is_required → False)
  уеи                 — «Количество товара в УЕИ» (unit-of-item logistics)
  нескольких упаковк  — «Планирую доставлять товар в нескольких упаковках»

High-precision name-based signals — fashion service/seller/photo fields
(_PLATFORM_NAME_RE, added for honest optional-denominator on apparel; each
phrase cannot appear in a real extractable spec characteristic):
  название файла       — «Название файла PDF» (document attachment name)
  pdf                  — «Файл PDF / PDF-инструкция» (document attachment)
  документ             — «Документ к товару» (uploaded document field)
  код/артикул продавца — SELLER-internal code/SKU («Код продавца»,
                          «Артикул продавца»), NOT the extractable
                          «Артикул»/model code of the product itself
  рост модели          — model height (per-photo-shoot, not a product spec)
  параметры модели     — model's body parameters on the photo
  размер на модели     — which size the photo model wears
  модель на фото       — «Модель на фото» (per-shoot reference)
  18+ / признак 18     — adult-content flag (18+ age restriction)
  тип ростовки/ростовк — seller size-run type (per-SKU, not fillable)
  размер производителя — manufacturer's own size grid value (per-SKU on tag)
  размер на бирке      — size printed on the product tag (per-SKU)
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional

_PLATFORM_DESC_RE = re.compile(
    r"(https?://"
    r"|\bmp4\b|\bmov\b"
    r"|\bjson\b"
    r"|sku через"
    r"|seller-edu"
    r"|соцсет"
    r"|rich-контент"
    r"|заводск\w*\s+упаковок"
    r")",
    re.IGNORECASE,
)

# High-precision patterns matched against the field *name* (not description).
# Each pattern is chosen to be unambiguous: it cannot appear in a real
# extractable characteristic name. Guard: is_required → False is applied
# BEFORE this regex, so «Объединить на одной карточке» (required) is safe.
_PLATFORM_NAME_RE = re.compile(
    r"(озон\.видео"
    r"|объединить в похожие"
    r"|\bуеи\b"
    r"|нескольких упаковк"
    # --- fashion service / seller / photo fields (honest optional denom) ---
    r"|название файла"
    r"|\bpdf\b"
    r"|\bдокумент"               # «Документ …» (uploaded doc field)
    r"|код продавца"
    r"|артикул продавца"
    r"|рост модели"
    r"|параметры модели"
    r"|размер на модели"
    r"|модел\w*\s+на фото"       # «Модель на фото»
    r"|на фото\s+модел"          # «… на фото модель»
    r"|18\+"
    r"|признак\s+18"
    r"|тип ростовки"
    r"|ростовк"                  # «Ростовка», «тип ростовки»
    r"|размер производителя"
    r"|размер на бирке"
    r")",
    re.IGNORECASE,
)


def is_platform_field(char: dict) -> bool:
    """Return True if *char* is a non-extractable platform/manual field.

    REPORTING ONLY — never use to filter pipeline targets (see module docstring).

    Conservative criteria:
    - is_required               → NEVER platform (hard guard, required is real work)
    - type == "URL"             → platform (link-only entry, nothing to extract)
    - has values list           → has dictionary → LLM can resolve → False
    - name matches _PLATFORM_NAME_RE → high-precision name signal
    - else description matches  → high-precision platform signal in Ozon's text
    """
    if char.get("is_required"):  # required is never a manual-only field
        return False
    if char.get("type") == "URL":
        return True
    if char.get("values"):  # non-empty dictionary → extractable
        return False
    name: str = char.get("name") or ""
    if _PLATFORM_NAME_RE.search(name):
        return True
    desc: str = char.get("description") or ""
    return bool(_PLATFORM_DESC_RE.search(desc))


# ---------------------------------------------------------------------------
# is_not_applicable — MUTUALLY-EXCLUSIVE N/A detection
# ---------------------------------------------------------------------------

@dataclass
class ProductContext:
    """Minimal product context needed for is_not_applicable.

    product_name      — full display name of the product (e.g. "Смартфон Samsung Galaxy A55 5G")
    category_path     — ordered category breadcrumb list (leaf last), e.g. ["Электроника", "Смартфоны"]
    filled_attrs      — dict mapping attr name (lower-cased) → filled value string.
                        Used to read "операционная система" if already resolved by pipeline.
    """
    product_name: str = ""
    category_path: list[str] = field(default_factory=list)
    filled_attrs: dict[str, str] = field(default_factory=dict)


# All known OS names that can appear as "Версия <OS>" fields on Ozon.
# The value is the canonical OS token used in _OS_ALIASES below.
_OS_VERSION_FIELD_RE = re.compile(
    r"^версия\s+(?P<os>ios|macos|mac\s*os|harmonyos|harmony\s*os|windows|android|ipados|watchos)",
    re.IGNORECASE,
)

# Map various spellings/aliases → canonical OS token
_OS_ALIASES: dict[str, str] = {
    "ios": "ios",
    "macos": "macos",
    "mac os": "macos",
    "harmony os": "harmonyos",
    "harmonyos": "harmonyos",
    "windows": "windows",
    "android": "android",
    "ipados": "ipados",
    "watchos": "watchos",
}

# Brand → likely canonical OS token(s).  Conservative: only well-known single-OS brands.
_BRAND_OS_MAP: dict[str, frozenset[str]] = {
    # Apple devices:  brand alone is ambiguous (Mac→macOS, iPhone→iOS, iPad→iPadOS,
    # Watch→watchOS).  We only use brand+category together (see _infer_product_os).
    "apple": frozenset({"ios", "macos", "ipados", "watchos"}),
    # Pure Android phone brands:
    "samsung": frozenset({"android"}),
    "xiaomi": frozenset({"android"}),
    "redmi": frozenset({"android"}),
    "poco": frozenset({"android"}),
    "realme": frozenset({"android"}),
    "oppo": frozenset({"android"}),
    "vivo": frozenset({"android"}),
    "oneplus": frozenset({"android"}),
    "huawei": frozenset({"android", "harmonyos"}),   # Huawei runs either
    "honor": frozenset({"android", "harmonyos"}),    # same
    "infinix": frozenset({"android"}),
    "tecno": frozenset({"android"}),
    "itel": frozenset({"android"}),
    "meizu": frozenset({"android"}),
    "nokia": frozenset({"android"}),
    "motorola": frozenset({"android"}),
}

# Category leaf keywords → Android (phones/tablets)
_ANDROID_CATEGORY_RE = re.compile(r"смартфон|android\s*телефон", re.IGNORECASE)
# Category leaf keywords → Windows (laptops)
_WINDOWS_CATEGORY_RE = re.compile(r"ноутбук|laptop", re.IGNORECASE)
# Product name keywords that strongly imply phone/smartphone (→ Android for non-Apple)
_PHONE_NAME_RE = re.compile(r"\bсмартфон\b", re.IGNORECASE)
# Apple product-name/category keywords → specific OS
_APPLE_IPHONE_RE = re.compile(r"\biphone\b|\bсмартфон apple\b", re.IGNORECASE)
_APPLE_IPAD_RE = re.compile(r"\bipad\b", re.IGNORECASE)
_APPLE_WATCH_RE = re.compile(r"\bapple\s+watch\b|\bwatch\s+series\b|\bwatch\s+ultra\b", re.IGNORECASE)
_APPLE_MAC_RE = re.compile(r"\bmacbook\b|\bimac\b|\bmac\s+mini\b|\bmac\s+pro\b|\bmac\s+studio\b", re.IGNORECASE)


def _infer_product_os(ctx: ProductContext) -> Optional[frozenset[str]]:
    """Derive the set of valid OS tokens for this product.

    Returns a frozenset of canonical OS tokens that ARE valid for this product,
    or None when the OS cannot be determined confidently.

    Conservatism: return None whenever there is ambiguity — never guess.
    """
    # 1. Explicit "Операционная система" attr already filled by pipeline — highest authority
    filled_os_raw = ctx.filled_attrs.get("операционная система", "").strip().lower()
    if filled_os_raw:
        token = _OS_ALIASES.get(filled_os_raw)
        if token:
            return frozenset({token})
        # Partial match inside the value (e.g. "Android 14" → android)
        for alias, canon in _OS_ALIASES.items():
            if alias in filled_os_raw:
                return frozenset({canon})
        # Filled but unrecognised format — can't determine safely
        return None

    name_lower = ctx.product_name.lower()
    leaf = (ctx.category_path[-1] if ctx.category_path else "").lower()
    combined = f"{name_lower} {leaf}"

    # 2. Apple device — determine specific OS from product name/leaf
    if "apple" in combined:
        if _APPLE_WATCH_RE.search(combined):
            return frozenset({"watchos"})
        if _APPLE_IPAD_RE.search(combined):
            return frozenset({"ipados"})
        if _APPLE_IPHONE_RE.search(combined):
            return frozenset({"ios"})
        if _APPLE_MAC_RE.search(combined):
            return frozenset({"macos"})
        # Apple but ambiguous device type — don't exclude anything
        return None

    # 3. Known Android-only phone brand + (smartphone name or category)
    for brand, os_set in _BRAND_OS_MAP.items():
        if brand == "apple":
            continue  # handled above
        if brand in name_lower or brand in leaf:
            if os_set == frozenset({"android"}):
                # Only commit if product is phone/smartphone/tablet context
                if _PHONE_NAME_RE.search(name_lower) or _ANDROID_CATEGORY_RE.search(combined):
                    return frozenset({"android"})
            elif os_set == frozenset({"android", "harmonyos"}):
                # Huawei/Honor — could be either; can't safely exclude either
                return None
            # Other brands: don't infer unless strong signal
            break

    # 4. Windows laptop: "ноутбук" in name OR leaf with no OS indicator
    if _WINDOWS_CATEGORY_RE.search(combined) and "apple" not in combined:
        return frozenset({"windows"})

    return None  # can't determine → keep all


# Dryer-only field names: match "сушки", "сушка", "для сушки", "загрузка белья для сушки"
_DRYER_FIELD_RE = re.compile(r"сушк|сушка\b", re.IGNORECASE)

# Washing machine WITHOUT dryer: leaf or name contains "стиральная машина" but NOT "с сушкой"
_WASHER_RE = re.compile(r"стиральная\s+машин", re.IGNORECASE)
_WASHER_WITH_DRYER_RE = re.compile(r"стиральная\s+машин\w*\s+с\s+сушкой|с\s+функцией\s+сушки", re.IGNORECASE)


def _is_washer_without_dryer(ctx: ProductContext) -> bool:
    """True when the product is provably a washing machine WITHOUT a dryer function."""
    combined = f"{ctx.product_name} {' '.join(ctx.category_path)}"
    if not _WASHER_RE.search(combined):
        return False
    # If combo is mentioned, NOT a pure washer
    return not _WASHER_WITH_DRYER_RE.search(combined)


def is_not_applicable(char: dict, ctx: ProductContext) -> bool:
    """Return True if *char* is a PROVABLY-N/A optional attribute for this product.

    REPORTING ONLY — same restriction as is_platform_field.
    CONSERVATISM: when in doubt, return False (keep counting the attr).

    Rules implemented (general, not per-product hardcode):

    1. OS-version mutual exclusion: "Версия <OS>" attr where the OS does NOT
       match the product's determined OS.  Only fires when product OS can be
       inferred confidently; otherwise False.

    2. Dryer-only fields on pure washing machines: attr name contains dryer
       keywords AND the product is provably a washer-only (no dryer combo).

    Hard guards:
    - is_required → NEVER N/A (required fields are always relevant)
    - Rule only fires on optional attrs (is_required=False)
    """
    if char.get("is_required"):
        return False  # required attrs are always counted

    name: str = (char.get("name") or "").strip()

    # --- Rule 1: OS-version mutual exclusion ---
    m = _OS_VERSION_FIELD_RE.match(name)
    if m:
        field_os_raw = m.group("os").lower().replace(" ", "")
        # Normalise "mac os" → "macos", "harmony os" → "harmonyos"
        field_os = _OS_ALIASES.get(field_os_raw) or _OS_ALIASES.get(
            m.group("os").lower()
        )
        if field_os:
            product_os_set = _infer_product_os(ctx)
            if product_os_set is not None and field_os not in product_os_set:
                return True  # provably impossible OS for this product

    # --- Rule 2: Dryer-only field on a pure washing machine ---
    if _DRYER_FIELD_RE.search(name) and _is_washer_without_dryer(ctx):
        return True

    return False
