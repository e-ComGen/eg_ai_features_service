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
SAFE_ENUM_MAX_OPTIONS: int = int(os.environ.get("SAFE_ENUM_MAX_OPTIONS", "35"))

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
    reasoning: Optional[str] = Field(None)

    @model_validator(mode="before")
    @classmethod
    def _sanitize(cls, data):
        if isinstance(data, dict):
            if data.get("value") is None:
                raise ValueError("null value — skip")
            if isinstance(data.get("reasoning"), str):
                data["reasoning"] = data["reasoning"][:200]
        return data


class _ProposalResponse(BaseModel):
    fills: list[_ProposedFill]

    @model_validator(mode="before")
    @classmethod
    def _coerce_and_filter(cls, data):
        """Default fills to [] and drop individual items that fail validation.

        Pydantic propagates per-item errors upward and fails the whole response.
        We pre-filter here so a null value or over-long reasoning in ONE item
        does not silently discard ALL valid proposals.
        Only normalises dict items; already-constructed _ProposedFill objects
        (e.g. from tests) are passed through as-is.
        """
        if isinstance(data, dict):
            raw_fills = data.get("fills") or []
            good = []
            for item in raw_fills:
                if not isinstance(item, dict):
                    # Already a model instance (e.g. from tests) — pass through
                    good.append(item)
                    continue
                if item.get("value") is None:
                    continue
                if isinstance(item.get("reasoning"), str):
                    item = {**item, "reasoning": item["reasoning"][:200]}
                good.append(item)
            data["fills"] = good
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
        return Source.SAFE_ENUM_FILL

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
            logger.info(
                "[SafeEnumFill] product=%s: skipped — no targets or no product_name",
                context.product_id,
            )
            return []

        # Only operate on short optional enums not yet filled
        effective_targets = [
            t for t in filter_already_filled_targets(targets, already_filled or [])
            if _is_short_enum(t)
        ]
        logger.info(
            "[SafeEnumFill] product=%s: %d raw targets → %d short-enum optional "
            "targets after filter (source_text=%s)",
            context.product_id,
            len(targets),
            len(effective_targets),
            "present" if source_text else "ABSENT",
        )
        if not effective_targets:
            return []

        # ── Step 1: Proposal ──────────────────────────────────────────────────
        proposals = await self._propose_fills(context, effective_targets, already_filled or [])
        if not proposals:
            logger.info(
                "[SafeEnumFill] SUMMARY product=%s: targets=%d proposed=0 — LLM skipped "
                "all targets (uncertain / targets were already filled in effective filter). "
                "gate_a_pass=0 gate_b_confirmed=0 gate_b_retracted=0 accepted=0",
                context.product_id,
                len(effective_targets),
            )
            return []

        logger.info(
            "[SafeEnumFill] product=%s: %d proposals received from LLM (before gating)",
            context.product_id,
            len(proposals),
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
                reason = "no source_text" if not source_text else "value not literally in text"
                logger.info(
                    "[SafeEnumFill] Gate A MISS: attr=%s value=%r (%s) → going to Gate B",
                    prop.attribute_id, prop.value, reason,
                )
                need_adversarial.append(prop)

        # ── Step 3: Gate B — adversarial verify (batched) ────────────────────
        adversarial_passed: list[_ProposedFill] = []
        if need_adversarial:
            confirmed_ids = await self._adversarial_verify(
                context, need_adversarial, target_by_id,
                resolved_attrs=already_filled or [],
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
                source=Source.SAFE_ENUM_FILL,
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
                source=Source.SAFE_ENUM_FILL,
                evidence=f"{_EVIDENCE_PREFIX_ADVERSARIAL}: {prop.reasoning or ''}".strip(": "),
                semantic_type=t.semantic_type if t else None,
                is_collection=t.is_collection if t else False,
            ))

        logger.info(
            "[SafeEnumFill] SUMMARY product=%s: targets=%d proposed=%d "
            "gate_a_pass=%d gate_b_confirmed=%d gate_b_retracted=%d accepted=%d",
            context.product_id,
            len(effective_targets),
            len(proposals),
            len(verbatim_passed),
            len(adversarial_passed),
            len(need_adversarial) - len(adversarial_passed),
            len(results),
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
            "propose the SINGLE best-matching value from its allowed_values list.\n\n"
            "CORE RULE: Propose whenever the product TYPE, NAME, CATEGORY, or BRAND makes "
            "a value obvious or strongly likely — even if the value is not explicitly stated "
            "word-for-word in the title. Proposing is cheap: the downstream gates will filter "
            "incorrect proposals. Your job is to surface plausible candidates, not to be "
            "overly conservative.\n\n"
            "TYPE-LEVEL DEFAULTS (always propose for these when the product type clearly applies):\n"
            "• Apparel season (Сезон): летнее платье/сарафан → Лето; зимняя куртка/пуховик → Зима; "
            "демисезонная куртка/ветровка → Демисезон; футболка/шорты → Лето or Демисезон.\n"
            "• Apparel style (Стиль): футболка/джинсы → Повседневный; спортивный костюм/кроссовки → "
            "Спортивный; деловой пиджак/блуза → Деловой; платье вечернее → Нарядный.\n"
            "• Apparel purpose (Назначение): футболка/джинсы/платье → Повседневный (не 'для дома'); "
            "спортивная одежда → Спорт; верхняя одежда → Для улицы; пижама/халат → Для дома.\n"
            "• Apparel cut (Покрой): футболка/майка → Прямой; платье A-line → Расклешённый.\n"
            "• Sleeve type (Тип рукава): футболка → Короткий рукав; платье без рукавов/сарафан → "
            "Без рукавов; толстовка/худи → Длинный рукав.\n"
            "• Running shoes (Тип пронации): general running shoe → Нейтральная.\n"
            "• Adult products (Целевая аудитория): explicit adult product → Взрослая.\n\n"
            "SKIP only when the product type gives NO signal at all for the attribute "
            "(e.g. a generic sock → Покрой is truly unknowable). "
            "Do NOT skip just because the value is not explicitly written in the title. "
            "Never invent values outside the allowed_values list. "
            "Never propose a brand value. "
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
            "Return fills for ALL attributes where the product type, name, "
            "or category provides a signal — the gates will verify. Field: 'fills'."
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
        resolved_attrs: list[AttributeValue] | None = None,
    ) -> set[int]:
        """One batched LLM call verifies ALL proposals for this product.

        The verifier is told to RETRACT by default — it must actively confirm.
        Returns the set of attribute_ids whose proposed fill was CONFIRMED.

        Parameters
        ----------
        resolved_attrs:
            Already-merged/resolved AttributeValues from prior pipeline stages.
            Passed to the verifier so it can RETRACT proposals that contradict
            an already-established attribute value (e.g. Windows version proposed
            when OS=«Без ОС» is already resolved; HDD count=1 when HDD=0).
            General rule — no hardcoded attribute names.
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

        # Build a concise already-resolved context block for the verifier so it can
        # detect contradictions (e.g. proposed Версия Windows=«Windows 11» when
        # resolved OS=«Без ОС»). Generic — no hardcoded attribute names/ids.
        resolved_block = ""
        if resolved_attrs:
            lines = [
                f"  {av.attribute_id}: {av.value!r}"
                for av in resolved_attrs
                if av.value is not None
            ]
            if lines:
                resolved_block = (
                    "\nAlready-established attributes for this product "
                    "(do NOT propose contradicting values):\n"
                    + "\n".join(lines[:40])  # cap to keep prompt manageable
                    + "\n"
                )

        system_prompt = (
            "You are a strict product-data auditor. For each proposed attribute fill, "
            "decide: is this value correct for this product? Your default answer is "
            "NOT_CONFIRMED — only confirm when the value is clearly correct or strongly "
            "implied for this product.\n\n"
            "Rules:\n"
            "- CONFIRM when the value is clearly correct for this product, OR is a "
            "well-known default for this product TYPE (e.g. Нейтральная pronation for "
            "a general running shoe, Взрослая audience for an adult product) — "
            "well-known product-type defaults are fine to confirm.\n"
            "- NOT_CONFIRMED when the value is WRONG or contradicted by product knowledge "
            "(e.g. Бязь fabric for denim jeans, для дома purpose for outdoor jeans).\n"
            "- NOT_CONFIRMED when the proposed value CONTRADICTS an already-established "
            "attribute listed in the context (e.g. Windows version when OS=«Без ОС», "
            "HDD count > 0 when HDD capacity=0).\n"
            "- NOT_CONFIRMED when the value is a baseless random guess with no connection "
            "to the product type, brand, or category.\n"
            "- Default to NOT_CONFIRMED only when genuinely uncertain — do not retract "
            "values that are sensible defaults for the product type.\n"
            "Return 'verifications' list: {attribute_id, verdict: 'CONFIRMED'|'NOT_CONFIRMED'}."
        )

        user_text = (
            f"Product: {context.product_name}\n"
            f"Brand: {context.brand or 'unknown'}\n"
            f"Category: {' / '.join(context.category_path) or 'n/a'}\n"
            + resolved_block
            + f"\nProposed fills to verify:\n{items_block}\n\n"
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
        # Reuse the KnowledgeJudge — the adversarial gate already acts as a specialized
        # judge; the ConfidenceAwareJudgeWrapper will call KnowledgeJudge only for
        # values below SOURCE_CONFIDENCE_THRESHOLDS[SAFE_ENUM_FILL].
        from app.services.enrichment.judges.knowledge_judge import KnowledgeJudge
        return KnowledgeJudge()


# ──────────────────────────────────────────────────────────────────────────────
# Public helper: standalone adversarial verify (DRY — reused by pipeline for
# the LLM_KNOWLEDGE post-merge pass in addition to SafeEnumFillSource).
# ──────────────────────────────────────────────────────────────────────────────


async def run_adversarial_verify(
    context: ExtractionContext,
    proposals: list[tuple[int, str, str]],
    resolved_attrs: list[AttributeValue] | None = None,
    llm_manager: Optional[StructuredLlmManager] = None,
) -> set[int]:
    """Standalone adversarial verifier — same Gate B used by SafeEnumFillSource.

    Parameters
    ----------
    context:
        Product extraction context (name, brand, category, etc.)
    proposals:
        List of (attribute_id, attr_name, proposed_value) triples to verify.
    resolved_attrs:
        Already-resolved AttributeValues from the product — used to detect
        contradictions (e.g. Windows version when OS=«Без ОС»).
    llm_manager:
        LLM manager to use. Defaults to get_main_manager().

    Returns
    -------
    Set of attribute_ids whose proposed fill was CONFIRMED by the verifier.
    NOT_CONFIRMED (including on LLM error) → retract (fail-closed).
    """
    if not proposals:
        return set()

    llm = llm_manager or get_main_manager()

    items_block = "\n".join([
        f"- attribute_id={attr_id}, attr_name={attr_name!r}, proposed_value={value!r}"
        for attr_id, attr_name, value in proposals
    ])

    resolved_block = ""
    if resolved_attrs:
        lines = [
            f"  {av.attribute_id}: {av.value!r}"
            for av in resolved_attrs
            if av.value is not None
        ]
        if lines:
            resolved_block = (
                "\nAlready-established attributes for this product "
                "(RETRACT any proposal that contradicts these):\n"
                + "\n".join(lines[:40])
                + "\n"
            )

    system_prompt = (
        "You are a strict product-data auditor. For each proposed attribute fill, "
        "decide: is this value independently-verifiable-correct for THIS specific product? "
        "Your default answer is NOT_CONFIRMED — only confirm when you can independently "
        "establish the exact value for this exact product from your own knowledge.\n\n"
        "CRITICAL RULE — evidence-blindness: You are given ONLY the product identity "
        "(name, brand, category) and the proposed value. Any claim that 'official specs "
        "say X', 'according to specifications', or 'typically X for this type' is NOT "
        "evidence — treat it as if it were absent. Confirm ONLY if YOU can independently "
        "verify this exact value for this exact product/model.\n\n"
        "Rules:\n"
        "- CONFIRM when you can independently verify the value is correct for this EXACT "
        "product model (e.g. OS=Android for Samsung Galaxy S23, storage=256GB for a "
        "specific iPhone model listed by name).\n"
        "- NOT_CONFIRMED when the value is WRONG or contradicted by product knowledge "
        "(e.g. Бязь fabric for denim jeans, Series 3 model for a Series 9 watch listing, "
        "stereo 2.0 for Яндекс Станция Мини which is a MONO speaker).\n"
        "- NOT_CONFIRMED when the proposed value CONTRADICTS an already-established "
        "attribute listed in the context (e.g. True Wireless=Да for an over-ear headphone, "
        "Бязь fabric for a Nike tee when web search showed cotton/polyester blend).\n"
        "- NOT_CONFIRMED when you cannot independently confirm this value for THIS EXACT "
        "product model — even if it sounds plausible for the product category.\n"
        "- Fail closed: when uncertain → NOT_CONFIRMED. Empty is better than wrong.\n"
        "Return 'verifications' list: {attribute_id, verdict: 'CONFIRMED'|'NOT_CONFIRMED'}."
    )

    user_text = (
        f"Product: {context.product_name}\n"
        f"Brand: {context.brand or 'unknown'}\n"
        f"Category: {' / '.join(context.category_path) or 'n/a'}\n"
        + resolved_block
        + f"\nProposed fills to verify:\n{items_block}\n\n"
        "For each fill: is this value strongly implied-correct for THIS specific product? "
        "Default to NOT_CONFIRMED when unsure."
    )

    try:
        parsed, _ = await llm.structured_request(
            system_prompt=system_prompt,
            user_text=user_text,
            response_model=_AdversarialResponse,
        )
    except Exception as exc:
        logger.warning("[AdversarialVerify] LLM call failed: %s", exc)
        return set()  # fail closed — all proposals retracted on error

    context.llm_calls_so_far += 1
    if parsed is None:
        return set()

    confirmed: set[int] = set()
    for v in parsed.verifications:
        if v.verdict.upper().strip() == "CONFIRMED":
            confirmed.add(v.attribute_id)
    return confirmed
