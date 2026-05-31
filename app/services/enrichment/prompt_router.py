"""Prompt router: classifies TargetAttribute to an instruction type.

Used by sources (DescriptionSource, LlmKnowledgeSource, etc.) to build
targets_block with type hints (kind=enum, kind=numeric, etc.) and to
append a meta-guidance block to system_prompt.

Classification rules (in priority order):
  1. model_name  -- name contains article/model keywords
  2. enum        -- non-empty allowed_values present
  3. dimensions  -- numeric type + dimension keywords in name
  4. numeric     -- numeric type or unit in name
  5. text        -- everything else
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from app.services.enrichment.base import TargetAttribute

if TYPE_CHECKING:
    from app.services.enrichment.base import AttributeValue

# Порог уверенности для включения в "уже известно" блок
SKIP_FILLED_CONFIDENCE_THRESHOLD = 0.85

# Pattern: attribute names meaning "model name / article number"
_MODEL_NAME_PATTERN = re.compile(
    r"(название\s+модели"
    r"|артикул"
    r"|model\s*name"
    r"|партномер"
    r"|partnumber|sku"
    r"|\bmodel\b"
    r"|модел[ьяи])",
    re.IGNORECASE,
)

# Pattern: dimension characteristics
_DIMENSIONS_PATTERN = re.compile(
    r"(длина"
    r"|высота"
    r"|ширина"
    r"|глубина"
    r"|размер"
    r"|габарит"
    r"|см\b|мм\b)",
    re.IGNORECASE,
)

# Pattern: unit of measure at end of name (e.g. "Мощность, Вт")
# \w with re.UNICODE covers both latin and cyrillic reliably
_UNIT_PATTERN = re.compile(r",\s*(\w+(?:/\w+)?)\s*$", re.UNICODE)

# Numeric types in Ozon dictionary
_NUMERIC_TYPES = {"Integer", "Decimal", "numeric", "integer", "decimal", "Float", "float"}
_BOOLEAN_TYPES = {"Boolean", "boolean", "bool"}
_URL_TYPES = {"URL", "url", "Url"}


def classify_target(target: TargetAttribute) -> str:
    """Classify attribute -> kind label used for prompt routing.

    Kinds:
      model_name | enum | url | boolean | dimensions | numeric | text

    Rules by descending specificity:
    1. model_name — keywords (артикул/модель) win over typed enums
    2. enum       — non-empty allowed_values present
    3. url        — type=URL (модель не должна фабриковать ссылки)
    4. boolean    — type=Boolean (Да/Нет — особая семантика)
    5. dimensions — numeric type + dimension keyword
    6. numeric    — numeric type or unit
    7. text       — fallback
    """
    name_lower = target.name.lower()

    # 1. model_name checked first -- takes priority over enum
    if _MODEL_NAME_PATTERN.search(name_lower):
        return "model_name"

    # 2. enum -- dictionary values exist
    if target.allowed_values:
        return "enum"

    # 3. url — отдельный kind чтобы инструктировать «не фабрикуй URL»
    if target.type in _URL_TYPES:
        return "url"

    # 4. boolean — Да/Нет вопрос
    if target.type in _BOOLEAN_TYPES:
        return "boolean"

    is_numeric_type = target.type in _NUMERIC_TYPES
    has_unit = bool(_UNIT_PATTERN.search(target.name))

    # 5. dimensions -- numeric type AND dimension keyword
    if (is_numeric_type or has_unit) and _DIMENSIONS_PATTERN.search(name_lower):
        return "dimensions"

    # 6. numeric -- numeric type or unit present
    if is_numeric_type or has_unit:
        return "numeric"

    # 7. text -- fallback
    return "text"


def extract_unit(target_name: str) -> Optional[str]:
    """Extract unit of measure from attribute name.

    Examples:
      "Мощность блока питания, Вт"  -> "Вт"
      "Длина, см"                   -> "см"
      "Цвет товара"                 -> None
      "Скорость вращения, об/мин"   -> "об/мин"
    """
    m = _UNIT_PATTERN.search(target_name)
    return m.group(1) if m else None


def format_target_line(target: TargetAttribute) -> str:
    """Format one targets_block line with type hint and unit.

    Example output:
      - id=4018290, name='Цвет товара', kind=enum, allowed=['Черный','Белый',...]
      - id=4018291, name='Мощность, Вт', kind=numeric, unit=Вт
      - id=4018292, name='Название модели', kind=model_name
      - id=4018293, name='Длина, см', kind=dimensions, unit=см
    """
    kind = classify_target(target)
    unit = extract_unit(target.name)

    parts = [f"id={target.id}", f"name={target.name!r}", f"kind={kind}"]

    if unit:
        parts.append(f"unit={unit}")

    if target.allowed_values:
        # Show first 40 values to keep prompt size manageable
        shown = target.allowed_values[:40]
        parts.append(f"allowed={shown}")

    if target.is_collection:
        parts.append("is_collection=true")

    return "- " + ", ".join(parts)


def build_meta_guidance() -> str:
    """Return instruction block for system_prompt explaining rules by attribute type.

    Appended to all source prompts to fix key quality issues:
    - enum: model picks from allowed list exactly (fixes case/spelling mismatches)
    - model_name: strips brand prefix (fixes low fill rate for model name field)
    - dimensions: correct axis mapping
    - numeric: return number only
    """
    return (
        "\n\nATTRIBUTE TYPE RULES (follow strictly based on 'kind=' hint in target list):\n"
        "• kind=enum      — You MUST pick a value EXACTLY as written in the 'allowed' list. "
        "Case and spelling must match the list entry precisely. "
        "Do not rephrase, translate or correct spelling. "
        "If none fits, skip the attribute.\n"
        "• kind=dimensions — Find dimension values (often LxWxH format). "
        "Map each number to the correct axis using standard product conventions "
        "(e.g. Length = longest side). Return only the numeric value for the requested axis.\n"
        "• kind=numeric   — Return only the numeric value (no unit suffix). "
        "Unit is given in 'unit=' hint for reference only. "
        "If source unit differs from target unit, CONVERT values "
        "(kg→g ×1000, mm→cm ÷10, m→mm ×1000, cm→mm ×10, g→kg ÷1000).\n"
        "• kind=model_name — Extract the model identifier from the product name: "
        "remove category prefix and brand name, keep the model string "
        "(e.g. 'Блок питания Cooler Master MWE Gold 750 V2' -> 'MWE Gold 750 V2').\n"
        "• kind=url       — Return URL ONLY if it appears verbatim in the source text. "
        "NEVER fabricate, guess or construct URLs. If no URL is present, skip the attribute.\n"
        "• kind=boolean   — Return strictly `true` or `false`. "
        "Map natural-language affirmations ('есть', 'имеется', 'поддерживает', 'да') to true, "
        "and negations ('нет', 'отсутствует', 'не поддерживает') to false. "
        "If the source text does not state the property either way, skip the attribute.\n"
        "• kind=text      — Return the relevant text value as found in the source."
    )


def build_already_filled_block(already_filled: "list[AttributeValue]") -> tuple[str, str]:
    """Строит preamble-блок для prompt и системное правило для skip-filled кооперации.

    Возвращает (user_preamble, system_rule):
      user_preamble  — вставляется ПЕРЕД targets_block в user_text
      system_rule    — добавляется в конец system_prompt
    """
    high_conf = [
        av for av in already_filled
        if av.confidence >= SKIP_FILLED_CONFIDENCE_THRESHOLD
    ]
    if not high_conf:
        return "", ""

    lines = [
        f"- attribute_id={av.attribute_id}, name={av.source!r}, value={av.value!r}, confidence={av.confidence:.2f}"
        for av in high_conf
    ]
    user_preamble = (
        "ALREADY RESOLVED (do not re-extract, listed for context only):\n"
        + "\n".join(lines)
        + "\n\n"
    )
    system_rule = (
        "\n\nSkip attributes listed in 'ALREADY RESOLVED' — they are already known. "
        "Focus your extraction on the targets list below."
    )
    return user_preamble, system_rule


def filter_already_filled_targets(
    targets: "list[TargetAttribute]",
    already_filled: "list[AttributeValue]",
) -> "list[TargetAttribute]":
    """Убирает из targets те, что уже заполнены с **is_confident()** (≥source threshold).

    Раньше использовали глобальный threshold 0.85 — это блокировало PDF (threshold
    0.90) перезаписать attrs которые OzonCard заполнил brand_line conf=0.85, хотя
    PDF был бы точнее. is_confident() для каждого source проверяет свой threshold:
    OZON_CARD=0.90, ICECAT=0.90, PDF_DATASHEET=0.90, LLM_KNOWLEDGE=0.92 и т.д.
    Это позволяет более точным sources перебивать менее уверенные.
    """
    filled_ids = {
        av.attribute_id
        for av in already_filled
        if av.is_confident()
    }
    return [t for t in targets if t.id not in filled_ids]
