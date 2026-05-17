"""Enrichment branches: vision, web-search, and attribute merger."""

from .vision_producer import VisionProducer
from .websearch_producer import WebSearchProducer
from .attribute_merger import AttributeMerger
from .attribute_merger import AttributeValue as _LegacyAttributeValue
from .attribute_merger import Source as _LegacySource

# New canonical base types (cost-aware pipeline, step A)
from .base import (
    Source,
    SOURCE_PRIORITY,
    SOURCE_CONFIDENCE_THRESHOLDS,
    AttributeValue,
    TargetAttribute,
    ExtractionContext,
    LlmJudge,
    AttributeSource,
)

__all__ = [
    # Existing producers / merger
    "VisionProducer",
    "WebSearchProducer",
    "AttributeMerger",
    # New base types
    "Source",
    "SOURCE_PRIORITY",
    "SOURCE_CONFIDENCE_THRESHOLDS",
    "AttributeValue",
    "TargetAttribute",
    "ExtractionContext",
    "LlmJudge",
    "AttributeSource",
]
