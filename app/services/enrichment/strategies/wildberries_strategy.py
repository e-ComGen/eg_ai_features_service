"""WildberriesStrategy — marketplace-specific logic for WB Content API.

Validates enum attributes against live WB dictionaries (colors, countries,
seasons, ТН ВЭД) via wb_runtime_lookup. Falls back gracefully (fail-open)
when WB_API_KEY is absent or network is unavailable.
"""
import asyncio
import logging
from typing import Any, Optional

from .base import MarketplaceStrategy, ValidationResult
from app.services.enrichment.base import (
    AttributeValue, TargetAttribute, ExtractionContext,
)
from app.services.enrichment.strategies.dictionaries.wb_runtime_lookup import (
    resolve_value,
    get_directory,
    _COLOR_NAMES,
    _COUNTRY_NAMES,
    _SEASON_NAMES,
    _TNVED_NAMES,
    _GENDER_NAMES,
)

log = logging.getLogger(__name__)

# Attributes that AI must not fill on WB — seller or WB generates them
WB_SKIP_SEMANTIC_TYPES = frozenset({
    "ean", "upc", "gtin", "barcode",  # seller's legal responsibility
    "article", "sku",                  # WB auto-generates nm_id
    "imei", "serial",                  # per-unit identifiers
})

# Phrases banned by WB moderation
WB_BANNED_PHRASES = frozenset({
    "лучший", "№1", "номер один", "лидер рынка",
    "уникальный", "эксклюзив",
})

# Charc names that have WB dictionary backing (lowercased union)
_DICT_BACKED_CHARCS: frozenset[str] = _COLOR_NAMES | _COUNTRY_NAMES | _SEASON_NAMES | _TNVED_NAMES | _GENDER_NAMES


def _is_dict_backed(charc_name: str) -> bool:
    return charc_name.lower().strip() in _DICT_BACKED_CHARCS


def _run_async(coro) -> Any:
    """Run an async coroutine from sync context, reusing existing loop if present."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an async context (e.g. pytest-asyncio) — create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, coro)
                return fut.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


class WildberriesStrategy(MarketplaceStrategy):
    """WB-specific overrides: dictionary validation + banned-phrase guard."""

    @property
    def name(self) -> str:
        return "wb"

    def filter_unsupported_attributes(
        self, targets: list[TargetAttribute],
    ) -> list[TargetAttribute]:
        """Skip attributes that WB-seller fills manually."""
        return [
            t for t in targets
            if not t.semantic_type or t.semantic_type not in WB_SKIP_SEMANTIC_TYPES
        ]

    def validate_value(
        self,
        target: TargetAttribute,
        value: Any,
        context: ExtractionContext,
    ) -> ValidationResult:
        """Validate against WB dictionaries (enum) and banned phrases (text).

        For dictionary-backed charc names (Цвет, Страна производства, Сезон,
        ТН ВЭД): attempts to resolve *value* to a canonical WB string.
        - If resolved: returns is_valid=True with normalized_value set to the
          canonical string.
        - If not resolved: fail-open (returns is_valid=True, original value)
          so pipeline is never blocked by a missing key or network error.

        For text fields: checks WB banned phrases.
        """
        if not isinstance(value, str) or not value.strip():
            return ValidationResult(is_valid=True, normalized_value=value)

        # -- Banned-phrase check for text attributes --
        if target.type == "text":
            v_lower = value.lower()
            for phrase in WB_BANNED_PHRASES:
                if phrase and phrase in v_lower:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Contains banned phrase: '{phrase}'",
                    )

        # -- Dictionary enum resolution --
        if _is_dict_backed(target.name):
            subject_id: Optional[int] = getattr(context, "subject_id", None) or getattr(
                context, "category_id", None
            )
            if subject_id is None:
                # No subject context — skip resolution (fail-open)
                log.debug(
                    "wb_strategy: no subject_id in context for charc=%r; skipping resolve",
                    target.name,
                )
                return ValidationResult(is_valid=True, normalized_value=value)

            try:
                resolved = _run_async(resolve_value(int(subject_id), target.name, value))
            except Exception as exc:
                log.warning(
                    "wb_strategy: resolve_value failed for charc=%r value=%r: %s; fail-open",
                    target.name, value, exc,
                )
                return ValidationResult(is_valid=True, normalized_value=value)

            if resolved:
                canonical = resolved["value"]
                if canonical != value:
                    log.debug(
                        "wb_strategy: normalized charc=%r %r → %r",
                        target.name, value, canonical,
                    )
                return ValidationResult(is_valid=True, normalized_value=canonical)
            else:
                # Not found in WB dictionary — fail-open, keep original
                log.debug(
                    "wb_strategy: charc=%r value=%r not in WB dictionary; keeping as-is (fail-open)",
                    target.name, value,
                )
                return ValidationResult(is_valid=True, normalized_value=value)

        return ValidationResult(is_valid=True, normalized_value=value)

    def normalize_target(self, target: TargetAttribute) -> TargetAttribute:
        """Context-free path — no-op, returns target unchanged."""
        return target

    def normalize_target_with_context(
        self,
        target: TargetAttribute,
        context: ExtractionContext,
    ) -> TargetAttribute:
        """Enrich allowed_values for color targets from WB dictionary.

        If target.allowed_values is already set, returns target unchanged.
        For color targets (semantic_type == 'color' or name contains 'цвет'
        but not 'название') with empty allowed_values, fetches color list
        from WB runtime lookup and populates allowed_values.
        On any failure, returns target unchanged (fail-open).
        """
        # Idempotent: if already populated, return as-is
        if target.allowed_values:
            return target

        # Check if this is a color target
        name_lower = (target.name or "").lower()
        is_color = (
            (target.semantic_type or "").lower() == "color"
            or ("цвет" in name_lower and "название" not in name_lower)
        )
        if not is_color:
            return target

        # Attempt to fetch colors from WB directory
        try:
            items = _run_async(get_directory("colors"))
            names = [item["name"] for item in items if item.get("name")]
            if names:
                return target.model_copy(update={"allowed_values": names})
        except Exception as exc:
            log.warning(
                "wb_strategy: failed to fetch colors for target=%r: %s; fail-open",
                target.name, exc,
            )

        # Fail-open: return target unchanged
        return target

    def resolve_value_ids(
        self,
        attribute_value: AttributeValue,
        context: ExtractionContext,
    ) -> AttributeValue:
        """Keep color-from-name fills alive through the value_id drop-guard.

        WB colours have no numeric value_id (the dictionary is name-only), so a
        value added by ``_apply_color_from_name`` carries value_id=None and would
        be dropped by ``_drop_unresolved_optional_enums``.  It is already
        validated against the WB colour dictionary, so stamp a sentinel
        value_id=0 to survive the drop; a later real WB-API resolve only ever
        overwrites a non-None id.  All other values pass through unchanged.
        """
        # "color_from_name" mirrors pipeline._COLOR_FROM_NAME_EVIDENCE
        # (kept as a literal here to avoid a circular import).
        if (
            attribute_value.value_id is None
            and attribute_value.evidence == "color_from_name"
            and isinstance(attribute_value.value, str)
            and attribute_value.value.strip()
        ):
            return attribute_value.model_copy(update={"value_id": 0})
        return attribute_value

    async def resolve_wb_value(
        self,
        subject_id: int,
        charc_name: str,
        value: str,
    ) -> Optional[dict]:
        """Public async entrypoint for resolving a single WB charc value.

        Returns ``{"value": <canonical>, "id": <int|None>}`` or ``None``.
        """
        return await resolve_value(subject_id, charc_name, value)
