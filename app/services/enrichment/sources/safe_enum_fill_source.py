"""SafeEnumFillSource — gated LLM fill for still-empty short/closed optional enums.

PROBLEM:
  The LLM's self-reported confidence does NOT filter mud.  Levi's jeans →
  Материал="Бязь", Назначение="для дома" at conf=0.95 is a textbook example.
  The earlier force-route to llm_knowledge was REVERTED because of this.

SOLUTION — two-gate architecture:
  Gate A (VERBATIM_EVIDENCE): proposed value (or synonym) appears LITERALLY in
  the fetched web_search/card text for this product. Reuses the composition
  miner's verbatim-check pattern. Cost: zero extra LLM calls.

  Gate B (ADVERSARIAL_VERIFY): a second focused LLM call that is instructed to
  RETRACT the fill if it is a generic guess or not clearly correct for THIS
  specific product. Defaults to retract when uncertain. One batched call per
  product covers ALL candidate fills for that product.

  A fill that passes NEITHER gate → stays EMPTY ("пусто честнее мусора").

SCOPE:
  - Only OPTIONAL (is_required==False) enum targets (have allowed_values).
  - Only SHORT enums: ≤ SAFE_ENUM_MAX_OPTIONS allowed values. Short = the
    domain is closed and small, so LLM constrained extraction is meaningful.
  - Only targets STILL EMPTY after all prior sources.
  - Brand targets are excluded (those are handled by brand-from-name logic).
  - The existing _drop_unresolved_optional_enums guard still applies afterwards
    (value_id=None fills get dropped as usual).

FEATURE FLAG:
  Disabled by default. Set SAFE_LLM_ENUM_FILL_ENABLED=true in env to enable.
  The pipeline imports and calls _run_safe_enum_fill_stage only when the flag
  is True — zero cost or risk when disabled.

COST CONTROL:
  1. Proposal step: one LLM call per product (batched up to CHUNK_SIZE targets).
  2. Adversarial verify: one LLM call per product covering ALL proposed fills
     (not one call per attribute).
  3. Verbatim gate fires first (zero cost), adversarial fires only when verbatim
     cannot confirm (no fetched text OR value not literally in text).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

from pydantic import BaseModel, Field, AliasChoices, model_validator

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    LlmJudge,
    Source,
    TargetAttribute,
)
from app.services.enrichment.prompt_router import (
    format_target_line,
    build_meta_guidance,
    build_already_filled_block,
    filter_already_filled_targets,
)
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.providers.factory import get_main_manager

logger = logging.getLogger(__name__)

# Short-enum threshold: enums with more than this many options are not
# suitable for the "LLM knows this closed domain" assumption.
SAFE_ENUM_MAX_OPTIONS: int = int(os.environ.get("SAFE_ENUM_MAX_OPTIONS", "20"))

# Confidence assigned to fills that pass a gate.
_FILL_CONFIDENCE = 0.82

# Evidence tag prefix so audit can see exactly which gate fired.
_EVIDENCE_PREFIX_VERBATIM = "safe_enum:verbatim_gate"
_EVIDENCE_PREFIX_ADVERSARIAL = "safe_enum:adversarial_gate"

# Adversarial verifier: retract when the answer is "NOT_CONFIRMED".
_ADVERSARIAL_RETRACT_SIGNAL = "NOT_CONFIRMED"

# ──────────────────────────────────────────────────────────────────────────────
# Proposal models
# ──────────────────────────────────────────────────────────────────────────────


class _ProposedFill(BaseModel):
    model_config = {"populate_by_name": True}

    attribute_id: int = Field(
        ..., validation_alias=AliasChoices("attribute_id", "id")
    )
    value: str = Field(
        ...,
        validation_alias=AliasChoices("value", "attribute_value", "extracted_value"),
        description="One of the allowed values, exactly as listed.",
    )
    reasoning: Optional[str] = Field(None, max_length=200)

    @model_validator(mode="before")
    @classmethod
    def _drop_null_value(cls, data):
        if isinstance(data, dict) and data.get("value") is None:
            raise ValueError("null value — skip")
        return data


class _ProposalResponse(BaseModel):
    fills: list[_ProposedFill]

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty(cls, data):
        if isinstance(data, dict):
            data.setdefault("fills", [])
        return data


# ──────────────────────────────────────────────────────────────────────────────
# Adversarial-verifier models
# ──────────────────────────────────────────────────────────────────────────────


class _VerifiedItem(BaseModel):
    attribute_id: int
    verdict: str = Field(
        ...,
        description=(
            "CONFIRMED if this value is clearly correct for THIS specific product "
            "(given title, brand, known specs). NOT_CONFIRMED otherwise."
        ),
    )


class _AdversarialResponse(BaseModel):
    verifications: list[_VerifiedItem]

    @model_validator(mode="before")
    @classmethod
    def _coerce_empty(cls, data):
        if isinstance(data, dict):
            data.setdefault("verifications", [])
        return data


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[а-яёa-z0-9]+", re.IGNORECASE)


def _norm_tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower().replace("ё", "е"))


def _verbatim_check(value: str, source_text: str) -> bool:
    """Return True if `value` (or any token of it) appears literally in source_text.

    Strategy (fail-closed when uncertain):
      1. Normalised whole-phrase match: all tokens of `value` appear as a
         contiguous subsequence in `source_text` tokens.
      2. If value is a single short word (≤6 chars), require phrase match only
         (single-char tokens too risky → always phrase-match).

    This mirrors the composition miner's two-signal rule: we check actual
    fetched HTML/text, not the LLM's memory.
    """
    if not source_text or not value:
        return False

    v_tokens = _norm_tokens(value)
    if not v_tokens:
        return False

    src_tokens = _norm_tokens(source_text)
    n = len(v_tokens)

    # Phrase match (contiguous)
    for i in range(len(src_tokens) - n + 1):
        if src_tokens[i : i + n] == v_tokens:
            return True
    return False


def _is_brand_target(target: TargetAttribute) -> bool:
    """True if this target is a brand/manufacturer attribute."""
    low = target.name.lower()
    return (
        target.id == 31
        or "бренд" in low
        or "brand" in low
        or bool(re.search(r"торгов\w*\s+марк", low))
    )


def _is_short_enum(target: TargetAttribute) -> bool:
    """True if target qualifies for safe-enum fill (optional short closed enum)."""
    return (
        not target.is_required
        and bool(target.allowed_values)
        and len(target.allowed_values) <= SAFE_ENUM_MAX_OPTIONS
        and not _is_brand_target(target)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main source
# ──────────────────────────────────────────────────────────────────────────────


class SafeEnumFillSource(AttributeSource):
    """Gated LLM fill for still-empty short optional enum targets.

    Disabled by default (SAFE_LLM_ENUM_FILL_ENABLED flag). When enabled:
      1. Identifies still-empty short optional enum targets.
      2. Asks LLM to propose one of the allowed values for each.
      3. Gates each proposal: accept only if verbatim evidence in fetched
         text (Gate A) OR adversarial verifier confirms (Gate B).
      4. Returns only the gated fills as AttributeValues.

    The pipeline caller is responsible for:
      - Only calling this source when the flag is enabled.
      - Passing `source_text` (the product's web_search summary or card text)
        as context so Gate A can fire for free.
    """

    def __init__(
        self,
        llm_manager: Optional[StructuredLlmManager] = None,
    ):
        self._llm = llm_manager or get_main_manager()

    @property
    def source_type(self) -> Source:
        return Source.LLM_KNOWLEDGE  # semantically it is LLM knowledge, gated

    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        return bool(context.product_name) and _is_short_enum(target)

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: list[AttributeValue] | None = None,
        source_text: Optional[str] = None,
    ) -> list[AttributeValue]:
        """
        Parameters
        ----------
        targets:
            All short-enum optional targets still empty (filtered by caller).
        already_filled:
            High-confidence AVs from prior stages (skip-fill cooperation).
        source_text:
            Optional fetched text for this product (web_search summary, card
            text, etc.). Used by Gate A. When None Gate A always misses and
            every candidate goes through Gate B.
        """
        if not targets or not context.product_name:
            return []

        # Only operate on short optional enums not yet filled
        effective_targets = [
            t for t in filter_already_filled_targets(targets, already_filled or [])
            if _is_short_enum(t)
        ]
        if not effective_targets:
            return []

        # ── Step 1: Proposal ──────────────────────────────────────────────────
        proposals = await self._propose_fills(context, effective_targets, already_filled or [])
        if not proposals:
            return []

        logger.debug(
            "[SafeEnumFill] %d proposals before gating for product %s",
            len(proposals),
            context.product_id,
        )

        # ── Step 2: Gate A — verbatim evidence ────────────────────────────────
        target_by_id = {t.id: t for t in effective_targets}

        verbatim_passed: list[_ProposedFill] = []
        need_adversarial: list[_ProposedFill] = []

        for prop in proposals:
            if source_text and _verbatim_check(prop.value, source_text):
                verbatim_passed.append(prop)
                logger.info(
                    "[SafeEnumFill] Gate A PASS: attr=%s value=%r (verbatim in source_text)",
                    prop.attribute_id, prop.value,
                )
            else:
                need_adversarial.append(prop)

        # ── Step 3: Gate B — adversarial verify (batched) ────────────────────
        adversarial_passed: list[_ProposedFill] = []
        if need_adversarial:
            confirmed_ids = await self._adversarial_verify(
                context, need_adversarial, target_by_id
            )
            for prop in need_adversarial:
                if prop.attribute_id in confirmed_ids:
                    adversarial_passed.append(prop)
                    logger.info(
                        "[SafeEnumFill] Gate B PASS: attr=%s value=%r (adversarial confirmed)",
                        prop.attribute_id, prop.value,
                    )
                else:
                    logger.info(
                        "[SafeEnumFill] Gate B RETRACT: attr=%s value=%r "
                        "(adversarial retracted — MUD blocked)",
                        prop.attribute_id, prop.value,
                    )

        # ── Step 4: Assemble AttributeValues ──────────────────────────────────
        results: list[AttributeValue] = []

        for prop in verbatim_passed:
            t = target_by_id.get(prop.attribute_id)
            results.append(AttributeValue(
                attribute_id=prop.attribute_id,
                value=prop.value,
                confidence=_FILL_CONFIDENCE,
                source=Source.LLM_KNOWLEDGE,
                evidence=f"{_EVIDENCE_PREFIX_VERBATIM}: {prop.reasoning or ''}".strip(": "),
                semantic_type=t.semantic_type if t else None,
                is_collection=t.is_collection if t else False,
            ))

        for prop in adversarial_passed:
            t = target_by_id.get(prop.attribute_id)
            results.append(AttributeValue(
                attribute_id=prop.attribute_id,
                value=prop.value,
                confidence=_FILL_CONFIDENCE,
                source=Source.LLM_KNOWLEDGE,
                evidence=f"{_EVIDENCE_PREFIX_ADVERSARIAL}: {prop.reasoning or ''}".strip(": "),
                semantic_type=t.semantic_type if t else None,
                is_collection=t.is_collection if t else False,
            ))

        logger.info(
            "[SafeEnumFill] product=%s: %d proposed, %d verbatim, %d adversarial, "
            "%d retracted (mud blocked)",
            context.product_id,
            len(proposals),
            len(verbatim_passed),
            len(adversarial_passed),
            len(need_adversarial) - len(adversarial_passed),
        )
        return results

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    async def _propose_fills(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: list[AttributeValue],
    ) -> list[_ProposedFill]:
        """One LLM call proposes constrained-enum fills for all short-enum targets."""
        already_preamble, already_rule = build_already_filled_block(already_filled)

        system_prompt = (
            "You are a product attribute expert. For each target attribute below, "
            "propose the SINGLE best-matching value from its allowed_values list. "
            "Only propose a value when you are CONFIDENT it is correct for THIS specific "
            "product (given the title, brand, and category). "
            "If you are uncertain → DO NOT include that attribute in your response. "
            "Better to skip than guess. Never invent values not in the allowed_values list. "
            "Return 'fills' list with {attribute_id, value, reasoning}."
            + build_meta_guidance()
            + already_rule
        )

        targets_block_lines = []
        for t in targets:
            opts = ", ".join(f'"{v}"' for v in (t.allowed_values or []))
            line = (
                f"- id={t.id}, name={t.name!r}, "
                f"allowed_values=[{opts}]"
            )
            targets_block_lines.append(line)
        targets_block = "\n".join(targets_block_lines)

        user_text = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n"
            f"Category: {' / '.join(context.category_path) or 'n/a'}\n\n"
            + already_preamble
            + f"Target attributes to fill:\n{targets_block}\n\n"
            "Return only attributes you CONFIDENTLY know. Field: 'fills'."
        )

        try:
            parsed, _ = await self._llm.structured_request(
                system_prompt=system_prompt,
                user_text=user_text,
                response_model=_ProposalResponse,
            )
        except Exception as exc:
            logger.warning("[SafeEnumFill] proposal LLM call failed: %s", exc)
            return []

        context.llm_calls_so_far += 1
        if parsed is None:
            return []

        # Safety: keep only proposals whose value is in the allowed_values list
        target_allowed: dict[int, set[str]] = {
            t.id: {v.lower().replace("ё", "е") for v in (t.allowed_values or [])}
            for t in targets
        }
        valid: list[_ProposedFill] = []
        for f in parsed.fills:
            allowed = target_allowed.get(f.attribute_id, set())
            norm_val = f.value.lower().replace("ё", "е")
            if norm_val in allowed:
                valid.append(f)
            else:
                logger.info(
                    "[SafeEnumFill] proposal dropped (not in allowed_values): "
                    "attr=%s value=%r",
                    f.attribute_id, f.value,
                )
        return valid

    async def _adversarial_verify(
        self,
        context: ExtractionContext,
        proposals: list[_ProposedFill],
        target_by_id: dict[int, TargetAttribute],
    ) -> set[int]:
        """One batched LLM call verifies ALL proposals for this product.

        The verifier is told to RETRACT by default — it must actively confirm.
        Returns the set of attribute_ids whose proposed fill was CONFIRMED.
        """
        if not proposals:
            return set()

        items_block = "\n".join([
            f"- attribute_id={p.attribute_id}, "
            f"attr_name={target_by_id[p.attribute_id].name!r}, "
            f"proposed_value={p.value!r}"
            for p in proposals
            if p.attribute_id in target_by_id
        ])
        if not items_block:
            return set()

        system_prompt = (
            "You are a strict product-data auditor. For each proposed attribute fill, "
            "decide: is this value CLEARLY CORRECT for THIS specific product given its "
            "title and brand? Your default answer is NOT_CONFIRMED — only confirm when "
            "the value is obviously and specifically true for this product.\n\n"
            "Rules:\n"
            "- If the value is a generic/household guess not specific to this product → NOT_CONFIRMED.\n"
            "- If the value could apply to many products of this category but is not "
            "specifically evidenced for this one → NOT_CONFIRMED.\n"
            "- If the value contradicts the product's identity (e.g. a fabric type that "
            "conflicts with what this brand/model is known for) → NOT_CONFIRMED.\n"
            "- Only CONFIRMED when the value is unambiguously characteristic of THIS product.\n"
            "Return 'verifications' list: {attribute_id, verdict: 'CONFIRMED'|'NOT_CONFIRMED'}."
        )

        user_text = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n"
            f"Category: {' / '.join(context.category_path) or 'n/a'}\n\n"
            f"Proposed fills to verify:\n{items_block}\n\n"
            "Audit each fill. Default to NOT_CONFIRMED when unsure."
        )

        try:
            parsed, _ = await self._llm.structured_request(
                system_prompt=system_prompt,
                user_text=user_text,
                response_model=_AdversarialResponse,
            )
        except Exception as exc:
            logger.warning("[SafeEnumFill] adversarial verify LLM call failed: %s", exc)
            return set()

        context.llm_calls_so_far += 1
        if parsed is None:
            return set()

        confirmed: set[int] = set()
        for v in parsed.verifications:
            if v.verdict.upper().strip() == "CONFIRMED":
                confirmed.add(v.attribute_id)
        return confirmed

    def get_judge(self) -> LlmJudge:
        # Reuse the KnowledgeJudge — the adversarial gate already is a specialized
        # judge; the ConfidenceAwareJudgeWrapper will call KnowledgeJudge only for
        # values below SOURCE_CONFIDENCE_THRESHOLDS[LLM_KNOWLEDGE].
        from app.services.enrichment.judges.knowledge_judge import KnowledgeJudge
        return KnowledgeJudge()
