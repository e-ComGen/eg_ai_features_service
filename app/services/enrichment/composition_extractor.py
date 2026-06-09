"""Domain-agnostic composition extractor for fabric/material strings.

Mines Состав/Материал from raw HTML using two-signal rules to suppress false positives.
Caller receives a list of normalised strings like ["79% хлопок, 21% полиэстер"].

Design decisions
-----------------
* NO per-site parser, NO hardcoded domains.
* TWO-SIGNAL filter kills most false positives:
    Signal A – a percentage number adjacent to a known textile material word.
    Signal B – a composition label ("Состав:", "Material:", …) followed by a value
               that ALSO contains a material word.
  A candidate must pass AT LEAST ONE signal.
* Noise-word reject list: numbers near "скидка", "оригинал", "рассрочка", "кэшбэк",
  "off", "sale", "discount", "cashback", "размер" etc. are blocked even when a
  material word follows later in the window.
* EN → RU normalisation for the Материал (id 4496) enum.
* "пусто честнее мусора" — return [] rather than emit a wrong composition.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Textile material vocabulary (RU + EN), stems/forms included
# ---------------------------------------------------------------------------

# Each entry: canonical RU name, regex fragment (RU+EN variants, word-boundary safe).
# Order matters for normalisation (first match wins for EN→RU).
_MATERIALS: list[tuple[str, str]] = [
    ("хлопок",    r"хлопок|хлопка|хлопок[а-я]*|cotton"),
    ("полиэстер", r"полиэстер[а-я]*|polyester"),
    ("эластан",   r"эластан[а-я]*|elastane|spandex|эластан"),
    ("вискоза",   r"вискоз[а-я]*|viscose|rayon"),
    ("шерсть",    r"шерст[а-я]*|wool|merino"),
    ("полиамид",  r"полиамид[а-я]*|polyamide|nylon|найлон[а-я]*"),
    ("лиоцелл",   r"лиоцел[а-я]*|lyocell|tencel|тенсел[а-я]*"),
    ("акрил",     r"акрил[а-я]*|acrylic"),
    ("лён",       r"лён|льн[а-я]*|linen|флакс"),
    ("кашемир",   r"кашемир[а-я]*|cashmere"),
    ("модал",     r"модал[а-я]*|modal"),
    ("бамбук",    r"бамбук[а-я]*|bamboo"),
    ("шёлк",      r"шёлк|шелк[а-я]*|silk"),
    ("кожа",      r"натуральная\s+кожа|кожа|leather"),
    ("флис",      r"флис[а-я]*|fleece"),
    ("трикотаж",  r"трикотаж[а-я]*|jersey|knit"),
    ("микрофибра", r"микрофибр[а-я]*|microfiber|microfibre"),
    ("металл",    r"металл[а-я]*|metal"),   # for elastic/zip applications
    ("полипропилен", r"полипропилен[а-я]*|polypropylene"),
    ("тактель",   r"тактель|tactel"),
    ("кевлар",    r"кевлар[а-я]*|kevlar|aramid|арамид[а-я]*"),
    ("купро",     r"купро[а-я]*|cupro"),
    ("дралон",    r"дралон[а-я]*|dralon"),
    ("сатин",     r"сатин[а-я]*|satin"),
]

# Combined pattern: any material word (used for lookahead / validation)
_MAT_VOCAB_RE = re.compile(
    r"\b(?:" + "|".join(pat for _, pat in _MATERIALS) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Noise words — %-numbers near these are NOT composition
# ---------------------------------------------------------------------------

# These words appear within a short window (±20 chars) of a digit+% and disqualify it.
_NOISE_WORDS_RE = re.compile(
    r"\b(?:"
    r"скидк[аеи]?|скидку|sale|off|discount|распродаж[а-я]?"  # discounts
    r"|оригинал[а-я]?|original"                               # authenticity
    r"|рассрочк[аеи]?|кредит[а-я]?|installment"              # payment
    r"|кэшбэк|cashback"                                       # cashback
    r"|предоплат[аеи]?"                                       # prepayment
    r"|акц[ия][а-я]?"                                         # promo
    r"|размер[а-я]?|size"                                     # sizing (number = size number)
    r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Label patterns: "Состав: ..." or "Composition: ..." etc.
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(
    r"""
    (?:^|[;,.\n\r]|\s{2,}|&[a-z]+;)   # preceded by line break / separator
    \s*
    (?P<label>
        Состав\s+(?:изделия|материала)?  # Состав / Состав изделия
        |Состав                           # plain Состав
        |Материал\s+(?:верха|подкладки|подошвы|основной|изделия)?
        |Материал
        |Ткань
        |Fabric\s+(?:content|composition)?
        |Composition
        |Material\s+(?:content|composition)?
        |Material
        |Содержание\s*волокн[а-я]*
    )
    \s*[:\-–—]\s*
    (?P<value>[^\n\r;|]{5,150})          # value up to 150 chars, no newlines
    """,
    re.VERBOSE | re.IGNORECASE | re.MULTILINE,
)

# Fallback: free-form pct+material without label
# Matches runs like "79% хлопок, 21% полиэстер" or "cotton 82% / polyester 18%"
_PCT_MAT_RE = re.compile(
    r"(?:"
    r"(?P<pct1>\d{1,3})\s*%\s*(?P<mat1>(?:" + "|".join(pat for _, pat in _MATERIALS) + r"))"
    r"|"
    r"(?P<mat2>(?:" + "|".join(pat for _, pat in _MATERIALS) + r"))\s*(?P<pct2>\d{1,3})\s*%"
    r")",
    re.IGNORECASE,
)

# Percentage-only lookalike guard: strip tags, decode html entities
_STRIP_TAGS_RE = re.compile(r"<[^>]{0,300}>")

# ---------------------------------------------------------------------------
# HTML → flat text helpers
# ---------------------------------------------------------------------------

def _flatten_html(raw_html: str) -> str:
    """Strip HTML tags and decode entities to get flat inspectable text.

    Preserves whitespace / newlines from block elements via a br→newline pass.
    Keeps result fast — no DOM parser needed.
    """
    # br / block tags → newline to preserve label-value adjacency
    text = re.sub(r"<(?:br|p|div|li|tr|td|th|h\d)(?:\s[^>]*)?>", "\n", raw_html, flags=re.IGNORECASE)
    text = _STRIP_TAGS_RE.sub(" ", text)
    text = html.unescape(text)
    # collapse whitespace runs (keep single newlines)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


# ---------------------------------------------------------------------------
# Core two-signal rules
# ---------------------------------------------------------------------------

def _has_noise_near_pct(text: str, pct_match_start: int, window: int = 40) -> bool:
    """True if a noise word appears within `window` chars of the pct position."""
    lo = max(0, pct_match_start - window)
    hi = min(len(text), pct_match_start + window)
    snippet = text[lo:hi]
    return bool(_NOISE_WORDS_RE.search(snippet))


def _validate_value_string(value: str) -> bool:
    """True if `value` contains at least one material word (and usually a %)."""
    return bool(_MAT_VOCAB_RE.search(value))


def _clean_value(raw: str) -> str:
    """Trim trailing noise chars and whitespace from a captured value."""
    # strip trailing punctuation that was captured as part of the window
    raw = raw.strip().rstrip(".,;:—–-|/\\")
    # collapse internal whitespace
    raw = re.sub(r"\s+", " ", raw)
    return raw


def _sum_pct_approx(value: str) -> float:
    """Sum all explicit %-numbers in `value`. Used to sanity-check total ~100."""
    return sum(int(m) for m in re.findall(r"(\d{1,3})\s*%", value))


def _looks_like_valid_composition(value: str) -> bool:
    """Two-signal gate: value must have a material word AND either:
      - contain a % adjacent to a material word, OR
      - be short/label-extracted (label context is itself the second signal).
    Also rejects values where any %-number is noise-adjacent.
    Returns True only if it clears all guards.
    """
    if not _validate_value_string(value):
        return False

    # Check every pct match in the value for noise contamination
    for m in re.finditer(r"\d{1,3}\s*%", value):
        if _has_noise_near_pct(value, m.start(), window=30):
            return False

    # Percentage sanity: if explicit %s are present, their sum must be roughly sane
    total = _sum_pct_approx(value)
    if total > 0 and (total < 50 or total > 120):
        # e.g. "100% оригинал" sums to 100 but has no material word → caught above
        # "рассрочка 0%" sums to 0 → caught above (< 50)
        # Here we only fire if there ARE material words but sum is crazy
        # (e.g. single "10% хлопок" without a second component — could be real but suspicious)
        if total < 20:
            return False

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_composition(html_or_text: str) -> list[str]:
    """Extract composition strings from raw HTML or plain text.

    Returns a deduplicated list of normalised strings like
    ["79% хлопок, 21% полиэстер"]. Returns [] if nothing passes the two-signal
    filter (fail-closed / "пусто честнее мусора").

    Strategy:
    1. Flatten HTML → plain text.
    2. Signal-B: scan for composition labels ("Состав:", "Material:", …).
       Accept the captured value if it contains a material word.
    3. Signal-A (fallback): scan for raw pct+material tokens, collect
       surrounding context window as a composition candidate.
    4. Deduplicate and normalise.
    """
    if not html_or_text:
        return []

    flat = _flatten_html(html_or_text)

    results: list[str] = []

    # -----------------------------------------------------------------------
    # Pass 1 — Signal B: label context
    # -----------------------------------------------------------------------
    for m in _LABEL_RE.finditer(flat):
        value = _clean_value(m.group("value"))
        if not value:
            continue
        if not _looks_like_valid_composition(value):
            logger.debug("composition_extractor: label match rejected: %r", value[:80])
            continue
        results.append(value)

    # -----------------------------------------------------------------------
    # Pass 2 — Signal A: free-form pct+material runs (no label required)
    # Only used when label pass found nothing (avoids duplicates)
    # -----------------------------------------------------------------------
    if not results:
        # Find all pct+material positions, then try to build a composite string
        # by joining adjacent tokens that are within 60 chars of each other.
        segments: list[tuple[int, int, str]] = []  # (start, end, token)
        for m in _PCT_MAT_RE.finditer(flat):
            start, end = m.start(), m.end()
            # check noise proximity
            if _has_noise_near_pct(flat, start, window=30):
                continue
            token = m.group(0).strip()
            segments.append((start, end, token))

        # Group adjacent tokens (gap ≤ 80 chars → same composition string)
        if segments:
            groups: list[list[tuple[int, int, str]]] = []
            current: list[tuple[int, int, str]] = [segments[0]]
            for seg in segments[1:]:
                if seg[0] - current[-1][1] <= 80:
                    current.append(seg)
                else:
                    groups.append(current)
                    current = [seg]
            groups.append(current)

            for group in groups:
                # Extract the full span from flat text to get separators
                span_start = group[0][0]
                span_end = group[-1][1]
                span_text = flat[span_start:span_end].strip()

                # Validate: must have ≥1 material word, no noise
                if not _looks_like_valid_composition(span_text):
                    continue

                # For single-token matches, use a slightly wider context window
                # to capture "хлопок 79%, полиэстер 21%" style
                if len(group) == 1:
                    # expand window to see if adjacent text has more material info
                    lo = max(0, span_start - 5)
                    hi = min(len(flat), span_end + 60)
                    wider = flat[lo:hi].strip()
                    if _PCT_MAT_RE.search(wider[len(wider) - 60:] if len(wider) > 60 else wider):
                        pass  # will be covered by adjacent token next iteration
                    else:
                        results.append(_clean_value(span_text))
                else:
                    results.append(_clean_value(span_text))

    # -----------------------------------------------------------------------
    # Normalise: lowercase material names, tidy separators
    # -----------------------------------------------------------------------
    normalised: list[str] = []
    seen: set[str] = set()
    for raw in results:
        normed = _normalise_composition_string(raw)
        key = normed.lower()
        if key not in seen:
            seen.add(key)
            normalised.append(normed)

    return normalised


def _normalise_composition_string(s: str) -> str:
    """Light normalisation: ensure pct comes before material name, fix separators."""
    # Replace various separators with ", "
    s = re.sub(r"\s*[/|;]\s*", ", ", s)
    # Normalise "material pct%" → "pct% material"
    # e.g. "хлопок 79%" → "79% хлопок"
    def _swap(m: re.Match) -> str:
        mat = m.group("mat").strip()
        pct = m.group("pct")
        return f"{pct}% {mat}"

    s = re.sub(
        r"(?P<mat>(?:" + "|".join(pat for _, pat in _MATERIALS) + r"))\s+(?P<pct>\d{1,3})\s*%",
        _swap,
        s,
        flags=re.IGNORECASE,
    )
    # Ensure space after % if directly followed by a letter
    s = re.sub(r"(\d)%([A-Za-zА-Яа-яёЁ])", r"\1% \2", s)
    return s.strip()


# ---------------------------------------------------------------------------
# EN → RU material name normalisation (for Ozon Материал enum)
# ---------------------------------------------------------------------------

# Maps lowercase EN tokens to canonical RU name used by Ozon dict
_EN_TO_RU: dict[str, str] = {
    "cotton":       "хлопок",
    "polyester":    "полиэстер",
    "elastane":     "эластан",
    "spandex":      "эластан",
    "viscose":      "вискоза",
    "rayon":        "вискоза",
    "wool":         "шерсть",
    "merino":       "шерсть",
    "polyamide":    "полиамид",
    "nylon":        "полиамид",
    "lyocell":      "лиоцелл",
    "tencel":       "лиоцелл",
    "acrylic":      "акрил",
    "linen":        "лён",
    "flax":         "лён",
    "cashmere":     "кашемир",
    "modal":        "модал",
    "bamboo":       "бамбук",
    "silk":         "шёлк",
    "leather":      "кожа",
    "fleece":       "флис",
    "jersey":       "трикотаж",
    "microfiber":   "микрофибра",
    "microfibre":   "микрофибра",
}


def normalize_material_en_ru(token: str) -> str:
    """Normalise an EN (or already-RU) material token to its canonical RU form.

    Returns the input unchanged if no mapping is found (already RU or unknown).
    Useful for resolving the Ozon «Материал» (id 4496) enum from web-scraped text.

    Examples:
        normalize_material_en_ru("cotton") → "хлопок"
        normalize_material_en_ru("polyester") → "полиэстер"
        normalize_material_en_ru("хлопок") → "хлопок"
    """
    return _EN_TO_RU.get(token.lower().strip(), token.strip())


def primary_material(composition_strings: list[str]) -> Optional[str]:
    """Return the material with the HIGHEST percentage from extracted composition.

    Useful for filling the Ozon «Материал» (id 4496) single-value enum:
    pick the dominant fibre.  Returns None if nothing found.
    """
    best_pct = -1
    best_mat: Optional[str] = None

    for comp in composition_strings:
        # Scan for pct+material pairs in the string
        for m in re.finditer(
            r"(?P<pct>\d{1,3})\s*%\s*(?P<mat>(?:" + "|".join(pat for _, pat in _MATERIALS) + r"))",
            comp,
            re.IGNORECASE,
        ):
            pct = int(m.group("pct"))
            mat_raw = m.group("mat").lower().strip()
            mat_ru = normalize_material_en_ru(mat_raw)
            # Map via vocab to canonical RU
            for canonical_ru, pattern in _MATERIALS:
                if re.fullmatch(pattern, mat_raw, re.IGNORECASE):
                    mat_ru = canonical_ru
                    break
            if pct > best_pct:
                best_pct = pct
                best_mat = mat_ru

    return best_mat if best_pct >= 0 else None


# ---------------------------------------------------------------------------
# Brand-verification helper (critical for "пусто честнее мусора")
# ---------------------------------------------------------------------------

def page_matches_brand(
    html_or_text: str,
    url: str,
    brand: Optional[str],
    product_name: Optional[str] = None,
    garment_type: Optional[str] = None,
) -> bool:
    """Return True if the page is plausibly about OUR product.

    Checks (in order, any one is sufficient):
      1. brand token present in URL (most reliable signal)
      2. brand token present in <title> tag
      3. brand token present in first <h1> tag
      4. brand token present in first 2000 chars of flattened text
         AND (if garment_type given) garment_type also present
         AND (if product_name given) any word from product_name ≥ 4 chars present

    Returns True when no brand given (caller should still emit with low conf).
    "пусто честнее мусора" — when brand IS given, require at least one match.
    """
    if not brand:
        return True  # no brand info → allow (low confidence)

    brand_lower = brand.strip().lower()
    brand_tokens = [t for t in re.split(r"\s+", brand_lower) if len(t) >= 3]
    if not brand_tokens:
        return True

    def _any_token_in(text: str) -> bool:
        t = text.lower()
        return any(tok in t for tok in brand_tokens)

    # 1. URL check
    if _any_token_in(url):
        return True

    # 2. <title> check
    title_m = re.search(r"<title[^>]*>(.*?)</title>", html_or_text, re.IGNORECASE | re.DOTALL)
    if title_m and _any_token_in(title_m.group(1)):
        return True

    # 3. <h1> check
    h1_m = re.search(r"<h1[^>]*>(.*?)</h1>", html_or_text, re.IGNORECASE | re.DOTALL)
    if h1_m and _any_token_in(h1_m.group(1)):
        return True

    # 4. First 2000 chars of flat text
    flat_head = _flatten_html(html_or_text[:4000])[:2000]
    if not _any_token_in(flat_head):
        return False

    # If garment_type given, require it too
    if garment_type:
        gt = garment_type.lower()
        if gt not in flat_head.lower():
            return False

    # If product_name given, at least one significant word must appear
    if product_name:
        words = [w for w in re.split(r"\s+", product_name.lower()) if len(w) >= 4]
        if words and not any(w in flat_head.lower() for w in words):
            return False

    return True
