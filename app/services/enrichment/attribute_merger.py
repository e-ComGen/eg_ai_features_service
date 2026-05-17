"""
AttributeMerger — merge attribute lists from multiple extraction branches.

Rule: for each attribute_id, keep the variant with the highest confidence.
On tie: source priority  DESCRIPTION > WEB_SEARCH > VISION > LLM_KNOWLEDGE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Source(str, Enum):
    """Origin of an extracted attribute value — defines tie-break priority."""
    DESCRIPTION = "description"   # highest priority
    WEB_SEARCH  = "web_search"
    VISION      = "vision"
    LLM_KNOWLEDGE = "llm_knowledge"  # lowest priority

    # Numeric priority: lower value = higher priority (used for tie-breaking).
    @property
    def priority(self) -> int:
        return {
            Source.DESCRIPTION:   0,
            Source.WEB_SEARCH:    1,
            Source.VISION:        2,
            Source.LLM_KNOWLEDGE: 3,
        }[self]


@dataclass
class AttributeValue:
    """A single extracted attribute value with provenance metadata."""
    attribute_id: str
    value: Any
    confidence: float          # 0.0 – 1.0
    source: Source = Source.DESCRIPTION
    reasoning: Optional[str] = field(default=None, compare=False)


class AttributeMerger:
    """Merge attribute lists from multiple branches.

    For each attribute_id:
      1. Pick the variant with the highest confidence.
      2. On tie, pick by source priority (DESCRIPTION wins over WEB_SEARCH etc.).
    """

    def merge(
        self,
        branches: list[list[AttributeValue]],
    ) -> list[AttributeValue]:
        """Return a deduplicated list — one winner per attribute_id.

        Args:
            branches: Each element is the list of AttributeValue objects produced
                      by one branch (description, vision, web_search, …).
                      Exception objects are NOT expected here — the caller
                      (job_processor) must filter them out first.

        Returns:
            Merged list of AttributeValues, one per attribute_id.
        """
        best: dict[str, AttributeValue] = {}

        for branch in branches:
            if not branch:
                continue
            for attr in branch:
                key = attr.attribute_id
                if key not in best:
                    best[key] = attr
                else:
                    current = best[key]
                    # Higher confidence wins.
                    if attr.confidence > current.confidence:
                        best[key] = attr
                    elif attr.confidence == current.confidence:
                        # Tie: lower priority number = higher source priority.
                        if attr.source.priority < current.source.priority:
                            best[key] = attr

        return list(best.values())
