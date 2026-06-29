"""app/services/enrichment/marketplaces/llm_extractor.py

Generic grounded LLM attribute extractor for marketplace pages without a
dedicated regex parser (e.g. Yandex.Market).

GROUNDING GUARANTEE:
  After LLM extraction, each value is verified to appear as a verbatim
  substring in the source page text. Any value not found in the text is
  discarded. This prevents LLM hallucinations from leaking into fills.

PIPELINE:
  1. Strip HTML → plain text (remove <script>/<style>/tags).
  2. Collapse whitespace; truncate to ~8000 chars around keyword anchors
     when the page is large.
  3. LLM structured call → list of {name, value} pairs.
  4. Grounding check: discard pairs where value is NOT a substring of text.
  5. Map via OzonCardSource._map_characteristics → AttributeValue list.
  6. Patch source tag to the caller-supplied source_tag.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.providers.factory import get_main_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML → text
# ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# Keywords that anchor the context window when truncating large pages.
_ANCHOR_KEYWORDS = [
    "состав", "материал", "характеристик", "бренд", "цвет",
    "размер", "вес", "высота", "ширина", "глубина", "гарантия",
]

_MAX_TEXT_CHARS = 8_000
_CONTEXT_WINDOW = 2_000  # chars around each keyword anchor


def _html_to_text(html: str) -> str:
    """Strip scripts, styles, and HTML tags; collapse whitespace."""
    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _trim_to_relevant(text: str) -> str:
    """Return ≤ _MAX_TEXT_CHARS chars centred around keyword anchors.

    When the page is short enough, return as-is. For larger pages, collect
    windows around keyword matches and return their union up to the limit.
    """
    if len(text) <= _MAX_TEXT_CHARS:
        return text

    text_low = text.lower()
    windows: list[tuple[int, int]] = []
    for kw in _ANCHOR_KEYWORDS:
        pos = text_low.find(kw)
        while pos != -1:
            start = max(0, pos - _CONTEXT_WINDOW // 2)
            end = min(len(text), pos + _CONTEXT_WINDOW // 2)
            windows.append((start, end))
            pos = text_low.find(kw, pos + 1)

    if not windows:
        # No anchors found — take first + last chunk
        half = _MAX_TEXT_CHARS // 2
        return text[:half] + " ... " + text[-half:]

    # Merge overlapping windows and concatenate
    windows.sort()
    merged: list[tuple[int, int]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    parts: list[str] = []
    total = 0
    for start, end in merged:
        chunk = text[start:end]
        if total + len(chunk) > _MAX_TEXT_CHARS:
            chunk = chunk[: _MAX_TEXT_CHARS - total]
            parts.append(chunk)
            break
        parts.append(chunk)
        total += len(chunk)

    return " ... ".join(parts)


# ---------------------------------------------------------------------------
# LLM structured schema
# ---------------------------------------------------------------------------

class _ExtractedPair(BaseModel):
    """Single attribute name-value pair extracted by the LLM."""
    name: str
    value: str


class _ExtractedAttributes(BaseModel):
    """LLM response: list of attribute pairs."""
    attributes: list[_ExtractedPair]


# ---------------------------------------------------------------------------
# Public extractor
# ---------------------------------------------------------------------------

async def llm_extract_attributes(
    html: str,
    context: ExtractionContext,
    targets: list[TargetAttribute],
    source_tag: Source,
    llm_manager=None,
) -> list[AttributeValue]:
    """Extract attributes from a marketplace page via LLM + grounding check.

    Parameters
    ----------
    html:
        Raw HTML of the product page.
    context:
        Extraction context (product name, brand, category, etc.).
    targets:
        Target attributes we want to fill.
    source_tag:
        Source enum to stamp on resulting AttributeValues.
    llm_manager:
        Optional pre-built LLM manager; if None, get_main_manager() is called.

    Returns
    -------
    List of grounded AttributeValues, possibly empty.
    """
    if not html or not targets:
        return []

    # Step 1-2: HTML → trimmed text
    text = _html_to_text(html)
    text = _trim_to_relevant(text)
    if len(text) < 100:
        logger.debug("[llm_extractor] text too short after stripping (%d chars)", len(text))
        return []

    # Step 3: LLM structured extraction
    llm = llm_manager or get_main_manager()
    target_names = [t.name for t in targets[:30]]  # cap to avoid over-long prompt
    system_prompt = (
        "Ты экстрактор характеристик товара. "
        "Извлеки атрибуты товара из текста как список пар {название, значение}. "
        "ТОЛЬКО то, что ДОСЛОВНО есть в тексте — не выдумывай, не перефразируй. "
        "Отвечай строго JSON согласно схеме."
    )
    user_text = (
        f"Товар: «{context.product_name}»\n"
        f"Нужные атрибуты (если есть): {', '.join(target_names)}\n\n"
        f"Текст страницы:\n{text}"
    )

    try:
        parsed, _ = await llm.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=_ExtractedAttributes,
            temperature=0.0,
        )
    except Exception as exc:
        logger.warning("[llm_extractor] LLM call failed: %s", exc)
        return []

    if parsed is None or not parsed.attributes:
        logger.debug("[llm_extractor] LLM returned no attributes")
        return []

    # Step 4: Grounding check — keep only values present verbatim in source text
    text_for_grounding = text.lower()
    grounded_pairs = []
    for pair in parsed.attributes:
        if not pair.name or not pair.value:
            continue
        if pair.value.lower() in text_for_grounding:
            grounded_pairs.append({"name": pair.name, "value": pair.value, "value_ids": []})
        else:
            logger.debug(
                "[llm_extractor] grounding DROP: '%s'='%s' not found verbatim in text",
                pair.name[:40], pair.value[:40],
            )

    if not grounded_pairs:
        logger.debug("[llm_extractor] 0 pairs survived grounding check")
        return []

    logger.info(
        "[llm_extractor] %d/%d pairs passed grounding",
        len(grounded_pairs), len(parsed.attributes),
    )

    # Step 5: Map to AttributeValues via OzonCardSource._map_characteristics
    from app.services.enrichment.sources.ozon_card_source import OzonCardSource
    ozon_helper = OzonCardSource(scrappey_key=None)

    evidence = f"{source_tag}:llm_grounded:{context.product_name[:30]}"
    raw_avs = ozon_helper._map_characteristics(
        grounded_pairs,
        targets,
        context,
        mode="brand_line",
        title=context.product_name[:50],
        score=80.0,
        evidence_override=evidence,
    )

    # Step 6: Patch source tag (OzonCardSource stamps OZON_CARD by default)
    return [av.model_copy(update={"source": source_tag}) for av in raw_avs]
