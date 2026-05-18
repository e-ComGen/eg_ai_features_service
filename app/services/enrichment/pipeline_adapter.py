"""Adapter: CS-Cart payload -> enrichment pipeline -> CS-Cart attributes.

Transforms the existing job_processor input into ExtractionContext / TargetAttribute,
calls PipelineOrchestrator, and converts the result back into the format
expected by job_processor (dict[str, Any]: feature_name -> value).

Design notes
------------
- PipelineOrchestrator uses base.AttributeValue where attribute_id is *int*.
  Legacy job_processor identifies attributes by *name* (str).
  The adapter therefore passes a synthetic int id derived from the position in
  the targets list and records the name<->id mapping to convert back.
- When targets_raw have a numeric "id" / "attribute_id" field that is already
  set, that value is used directly (better for future DB-backed schemas).
- The adapter is a thin layer: it does not cache, does not touch the DB, and
  does not call the LLM directly.  All of that is delegated to the orchestrator.

Spec: docs/architecture/pipeline.md  (step I)
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    TargetAttribute,
)
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.strategies.factory import get_strategy

logger = logging.getLogger(__name__)


class PipelineAdapter:
    """Thin layer: legacy payload <-> new orchestrator types."""

    def __init__(self, orchestrator: Optional[PipelineOrchestrator] = None):
        self._orch = orchestrator or PipelineOrchestrator()

    async def run(
        self,
        product_id: int,
        product_name: str,
        product_description: Optional[str],
        category_id: int,
        category_path: list[str] | None = None,
        brand: Optional[str] = None,
        ean: Optional[str] = None,
        source_urls: list[str] | None = None,
        image_urls: list[str] | None = None,
        targets_raw: list[dict] | None = None,
        max_cost_usd: float = 0.10,
        marketplace: Optional[str] = None,
    ) -> list[AttributeValue]:
        """Build context + targets and run orchestrator.

        Parameters
        ----------
        targets_raw:
            List of dicts as job_processor uses internally.  Expected keys:
            - ``id`` or ``attribute_id`` (int, optional) — CS-Cart attribute ID
            - ``name`` (str) — human-readable attribute name
            - ``type`` (str) — "text" | "numeric" | "enum" | "bool"
            - ``allowed_values`` (list[str], optional)
            - ``semantic_type`` (str, optional)
            - ``description`` (str, optional)

        Returns
        -------
        list[AttributeValue]
            Raw orchestrator output (base.AttributeValue, attribute_id is int).
            Callers that need a name->value dict should use
            :meth:`convert_to_legacy_dict`.
        """
        context = ExtractionContext(
            product_id=product_id,
            product_name=product_name,
            product_description=product_description,
            category_id=category_id,
            category_path=category_path or [],
            brand=brand,
            ean=ean,
            source_urls=source_urls or [],
            image_urls=image_urls or [],
            max_cost_usd=max_cost_usd,
        )

        targets: list[TargetAttribute] = []
        for idx, raw in enumerate(targets_raw or []):
            # Prefer an explicit numeric id; fall back to position index.
            raw_id = raw.get("id") or raw.get("attribute_id")
            try:
                attr_id = int(raw_id) if raw_id is not None else idx
            except (TypeError, ValueError):
                attr_id = idx

            targets.append(
                TargetAttribute(
                    id=attr_id,
                    name=raw.get("name", "unknown"),
                    type=raw.get("type", "text"),
                    allowed_values=raw.get("allowed_values"),
                    semantic_type=raw.get("semantic_type"),
                    description=raw.get("description"),
                )
            )

        # Select orchestrator: if marketplace specified, create one with the appropriate strategy.
        # Otherwise reuse the default orchestrator (avoids unnecessary instantiation).
        strategy = get_strategy(marketplace)
        if marketplace:
            orch = PipelineOrchestrator(strategy=strategy)
        else:
            orch = self._orch
        return await orch.enrich(context, targets)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def convert_to_legacy_dict(
        values: list[AttributeValue],
        targets_raw: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Convert orchestrator output to the {feature_name: value} dict used by job_processor.

        The mapping back from integer attribute_id to feature name relies on
        the same targets_raw list that was passed to :meth:`run`.  When
        targets_raw is not available (e.g. the id was already a real CS-Cart
        int and names are not needed), the int id is stringified as key.

        Parameters
        ----------
        values:
            Return value of :meth:`run`.
        targets_raw:
            Same list that was passed to :meth:`run` so we can map id->name.
        """
        # Build id -> name lookup from targets_raw
        id_to_name: dict[int, str] = {}
        for idx, raw in enumerate(targets_raw or []):
            raw_id = raw.get("id") or raw.get("attribute_id")
            try:
                attr_id = int(raw_id) if raw_id is not None else idx
            except (TypeError, ValueError):
                attr_id = idx
            id_to_name[attr_id] = raw.get("name", str(attr_id))

        result: dict[str, Any] = {}
        for av in values:
            name = id_to_name.get(av.attribute_id, str(av.attribute_id))
            result[name] = av.value
        return result
