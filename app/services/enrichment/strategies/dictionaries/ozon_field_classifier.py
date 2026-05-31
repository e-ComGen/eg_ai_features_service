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
"""
from __future__ import annotations
import re

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
