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

import asyncio
import logging
import os
from typing import Any, Optional

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    TargetAttribute,
)
from app.services.enrichment.pipeline import PipelineOrchestrator
from app.services.enrichment.strategies.factory import get_strategy

logger = logging.getLogger(__name__)

# Rich card sources (WbCard, IceCat) are OPT-IN on the /process-batch path:
# they make 1 Serper + N card.json fetches per product, so they stay OFF unless
# the deployment opts in. Latency is further bounded by:
#   WB_CARD_MAX_FETCHED (card.json fetch cap), WB_CARD_INJECT_IMAGES=false
#   (suppress the Vision trigger), and RICH_SOURCE_TIMEOUT_S (hard per-source cap).
_RICH_SOURCES_ENABLED = os.getenv("PIPELINE_RICH_SOURCES", "false").strip().lower() in (
    "1", "true", "yes", "on",
)
_RICH_SOURCE_TIMEOUT_S = float(os.getenv("RICH_SOURCE_TIMEOUT_S", "25"))


class _TimeoutSource:
    """Wrap an AttributeSource so extract() can never exceed a hard timeout.

    On timeout OR any error → returns [] (the product keeps every other source's
    fills; one slow card source can never stall or fail the whole batch). Every
    other attribute (is_applicable, source_type, get_judge, …) proxies to inner.
    """

    def __init__(self, inner: Any, timeout_s: float, label: str):
        self._inner = inner
        self._timeout_s = timeout_s
        self._label = label

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not set on the wrapper itself.
        return getattr(self._inner, name)

    async def extract(self, *args: Any, **kwargs: Any) -> list:
        try:
            return await asyncio.wait_for(
                self._inner.extract(*args, **kwargs), timeout=self._timeout_s
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] hard timeout after %.0fs → [] (other sources unaffected)",
                self._label, self._timeout_s,
            )
            return []
        except Exception as exc:  # noqa: BLE001 — never let a card source break the product
            logger.warning("[%s] failed: %s → []", self._label, exc)
            return []


class PipelineAdapter:
    """Thin layer: legacy payload <-> new orchestrator types."""

    def __init__(self, orchestrator: Optional[PipelineOrchestrator] = None):
        # Opt-in rich card sources, each behind a hard timeout. When disabled
        # (default) both stay None → the orchestrator skips those stages exactly
        # as before. WbCardSource self-discovers from product_name (Serper→card.json,
        # free); IceCatSource is brand-verified API. Both no-op without keys.
        self._wb_card = None
        self._icecat = None
        if _RICH_SOURCES_ENABLED:
            from app.services.enrichment.sources.wb_card_source import WbCardSource
            from app.services.enrichment.sources.icecat_source import IceCatSource
            self._wb_card = _TimeoutSource(WbCardSource(), _RICH_SOURCE_TIMEOUT_S, "WbCard")
            self._icecat = _TimeoutSource(IceCatSource(), _RICH_SOURCE_TIMEOUT_S, "IceCat")
            logger.info(
                "[PipelineAdapter] rich sources ENABLED (WbCard+IceCat, timeout=%.0fs)",
                _RICH_SOURCE_TIMEOUT_S,
            )
        self._orch = orchestrator or PipelineOrchestrator(
            wb_card_source=self._wb_card,
            icecat_source=self._icecat,
        )

    async def run(
        self,
        product_id: int,
        product_name: str,
        product_description: Optional[str],
        category_id: int,
        category_path: list[str] | None = None,
        brand: Optional[str] = None,
        ean: Optional[str] = None,
        ozon_type_id: Optional[int] = None,
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
            marketplace=marketplace,
            ozon_type_id=ozon_type_id,  # forwarded by eg-importer → OzonStrategy value_id resolution
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
                    is_required=bool(raw.get("is_required", False)),
                    semantic_type=raw.get("semantic_type"),
                    description=raw.get("description"),
                )
            )

        # Select orchestrator: if marketplace specified, create one with the appropriate strategy.
        # Otherwise reuse the default orchestrator (avoids unnecessary instantiation).
        strategy = get_strategy(marketplace)
        if marketplace:
            # Carry the (timeout-wrapped) rich sources into the marketplace-specific
            # orchestrator too — None when the opt-in flag is off → bare, as before.
            orch = PipelineOrchestrator(
                strategy=strategy,
                wb_card_source=self._wb_card,
                icecat_source=self._icecat,
            )
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
