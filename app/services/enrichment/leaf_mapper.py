"""Deterministic mapper: TargetAttribute → strategy leaf class.

No LLM traversal — uses classify_target() + name keywords.
Every attribute maps to exactly one leaf (fallback = ClassificationLeaf).

Leaf → threshold summary:
  CapacityLeaf            numeric  error_confidence >= 50  (strict)
  PhysicalMagnitudeLeaf   numeric  error_confidence >= 80  (base)
  QuantificationLeaf      numeric  error_confidence >= 80  (base)
  RadialDimensionLeaf     numeric  FAST ACCEPT (num in text)
  AngularDimensionLeaf    numeric  FAST ACCEPT (num in text)
  SurfaceAreaLeaf         numeric  FAST ACCEPT (num in text)
  BiometricDimensionLeaf  numeric  FAST ACCEPT (num in text)
  CompositeAggregationLeaf numeric FAST ACCEPT (num in text)
  MaterialLeaf            text     error_confidence >= 90  (lenient)
  BrandLeaf               text     error_confidence >= 80  (base)
  ConfigurationLeaf       text     error_confidence >= 80  (base)
  ClassificationLeaf      text     error_confidence >= 80  (base, DEFAULT)
  ContextFilterLeaf       text     error_confidence >= 80  (base)
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Type

from app.strategies.definitions.numeric import (
    AngularDimensionLeaf,
    BiometricDimensionLeaf,
    CapacityLeaf,
    CompositeAggregationLeaf,
    NumericBranch,
    PhysicalMagnitudeLeaf,
    QuantificationLeaf,
    RadialDimensionLeaf,
    SurfaceAreaLeaf,
)
from app.strategies.definitions.text import (
    BrandLeaf,
    ClassificationLeaf,
    ConfigurationLeaf,
    ContextFilterLeaf,
    MaterialLeaf,
)
from app.strategies.base import BaseStrategyNode

if TYPE_CHECKING:
    from app.services.enrichment.base import TargetAttribute

# ---------------------------------------------------------------------------
# Keyword patterns per leaf (applied to t.name.lower())
# ---------------------------------------------------------------------------

_CAPACITY_RE = re.compile(
    r"(объём|объем|ёмкость|емкость|capacity"
    r"|памят[ьи]|memory|storage"
    r"|содержим|нетто|наполнени"
    r"|мл\b|литр|litre|liter)",
    re.IGNORECASE,
)

_MATERIAL_RE = re.compile(
    r"(материал|состав|ткань|fabric|material"
    r"|substance|покрытие|наполнитель"
    r"|сырьё|сырье|пропитка|отделка"
    r"|подошва|подкладка|верх изделия"
    r"|внешн|внутрен)",
    re.IGNORECASE,
)

_BRAND_RE = re.compile(
    r"(бренд|brand|производитель|manufacturer"
    r"|торговая марка|марка)",
    re.IGNORECASE,
)

_ANGULAR_RE = re.compile(
    r"(угол|angle|градус|degree|поворот|наклон"
    r"|обзор|fov|field of view)",
    re.IGNORECASE,
)

_SURFACE_RE = re.compile(
    r"(площадь|area|поверхность"
    r"|кв\.?\s*[мм]|м²|кв\s*м)",
    re.IGNORECASE,
)

_BIOMETRIC_RE = re.compile(
    r"(обхват|обхвата|охват|рост|размер\s+одежды"
    r"|размер\s+обуви|голова|талия|бедра|грудь"
    r"|биометр|ergono|wrist|waist|hip|chest|head)",
    re.IGNORECASE,
)

_COMPOSITE_RE = re.compile(
    r"(суммарн|total|совокупн|общий\s+объём|общий\s+объем"
    r"|комплект|набор\s+из|bundle|multipack)",
    re.IGNORECASE,
)

_RADIAL_RE = re.compile(
    r"(диагональ|диаметр|радиус|diagonal"
    r"|diameter|radius|дюйм\b|inch\b|\")",
    re.IGNORECASE,
)

_QUANTIFICATION_RE = re.compile(
    r"(количество|кол-во|count|штук|число"
    r"|шт\b|pcs\b|pieces|комплектация"
    r"|в\s+комплекте|в\s+упаковке|в\s+наборе)",
    re.IGNORECASE,
)

_PHYSICAL_MAGNITUDE_RE = re.compile(
    r"(мощность|power|напряжение|voltage|ток\b|current"
    r"|частота|frequency|давление|pressure"
    r"|температур|luminance|яркость|luminosity"
    r"|сила\s+света|интенсивность|интенс"
    r"|вт\b|ватт|ампер|вольт|герц|дб\b|децибел"
    r"|вт/кг|dbm\b|sar\b)",
    re.IGNORECASE,
)

_CONTEXT_FILTER_RE = re.compile(
    r"(совместим|compatible|подходит\s+для"
    r"|подходящ|applies\s+to)",
    re.IGNORECASE,
)

_CONFIGURATION_RE = re.compile(
    r"(конфигурац|ориентац|интерфейс|разъём|разъем"
    r"|connector|port\b|протокол|стандарт|format\b"
    r"|форм-фактор|form.factor|режим|mode\b"
    r"|ручной|левш|правш)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_to_leaf(target: "TargetAttribute") -> Type[BaseStrategyNode]:
    """Map a TargetAttribute to the most specific strategy leaf class.

    Algorithm (deterministic, no LLM):
    1. classify_target() → kind (numeric/dimensions/text/enum/…)
    2. For numeric kinds: keyword matching on name → specific NumericBranch leaf
    3. For text/enum/boolean: keyword matching → specific TextBranch leaf
    4. Unknown → ClassificationLeaf (abstract default)
    """
    from app.services.enrichment.prompt_router import classify_target

    name = target.name or ""
    kind = classify_target(target)

    # ── Numeric family ──────────────────────────────────────────────────────
    if kind in ("numeric", "dimensions"):
        # Order matters: more specific patterns first.
        if _CAPACITY_RE.search(name):
            return CapacityLeaf
        if _COMPOSITE_RE.search(name):
            return CompositeAggregationLeaf
        if _ANGULAR_RE.search(name):
            return AngularDimensionLeaf
        if _SURFACE_RE.search(name):
            return SurfaceAreaLeaf
        if _BIOMETRIC_RE.search(name):
            return BiometricDimensionLeaf
        if _RADIAL_RE.search(name):
            return RadialDimensionLeaf
        if _QUANTIFICATION_RE.search(name):
            return QuantificationLeaf
        if _PHYSICAL_MAGNITUDE_RE.search(name):
            return PhysicalMagnitudeLeaf
        if kind == "dimensions":
            return RadialDimensionLeaf  # default for dimension kind
        # Generic numeric: NumericBranch (FAST ACCEPT)
        return NumericBranch

    # ── Text / enum / boolean / model_name family ───────────────────────────
    if _MATERIAL_RE.search(name):
        return MaterialLeaf
    if _BRAND_RE.search(name):
        return BrandLeaf
    if _CONTEXT_FILTER_RE.search(name):
        return ContextFilterLeaf
    if _CONFIGURATION_RE.search(name):
        return ConfigurationLeaf

    # ── Default for all remaining (enum, text, boolean, url, model_name) ───
    return ClassificationLeaf


def leaf_threshold(leaf_cls: Type[BaseStrategyNode]) -> int:
    """Return error_confidence threshold for process_judgment of given leaf.

    Mirrors the per-class overrides in numeric.py / text.py:
      CapacityLeaf   → 50
      MaterialLeaf   → 90
      everything else → 80 (base)
    """
    if leaf_cls is CapacityLeaf:
        return 50
    if leaf_cls is MaterialLeaf:
        return 90
    return 80
