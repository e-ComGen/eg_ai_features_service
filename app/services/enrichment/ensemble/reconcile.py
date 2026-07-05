"""Reconcile -- Phase 2a strict 2-vendor consensus for LlmKnowledgeSource.

Каскад согласия двух независимых LLM (модель A = DeepSeek, модель B = OpenAI
gpt-4o-mini) на остаточных характеристиках, которые донор-маркетплейс не
закрыл: exact-match -> numeric-with-unit match -> fuzzy (rapidfuzz) ->
опциональный vector-cosine (sentence-transformers, best-effort) -> LLM-арбитр
на всё ещё неоднозначное. Значение эмитируется ТОЛЬКО если ОБЕ модели дали
значение для attribute_id И они согласны. Присутствие только у одной модели --
всегда abstain (никогда не эмитить).

Инвариант владельца: "лучше пусто, чем врёт". Confidence эмита фиксирован
~0.9 (config.LLM_ENSEMBLE_CONFIDENCE) -- честно, не раздувается самоотчётом
ни модели A, ни модели B, и остаётся ниже донор-порогов Фазы 1.
"""
from __future__ import annotations

import json as _json
import logging
import math
import re
from pathlib import Path
from typing import Any, Optional, Protocol

from pydantic import BaseModel
from rapidfuzz import fuzz

from app import config
from app.services.enrichment.base import AttributeValue, Source, TargetAttribute
from app.services.providers.structured_adapter import StructuredLlmManager
from app.services.enrichment.ensemble.grounding import ground_value, ground_disagreement

logger = logging.getLogger(__name__)

_EMBEDDER: Any = None

# Phase 3 escalation queue: disagreements (both models answered, arbiter said NO)
# are appended here as JSONL for a future web-search tie-breaker pass.
_DISAGREEMENT_LOG = Path(__file__).resolve().parents[4] / "logs" / "ensemble_disagreements.jsonl"


def _log_disagreement(product: "str | None", attr: str, value_a, value_b) -> None:
    """Append one disagreement record to the Phase 3 escalation queue (best-effort).

    Never raises: any filesystem/serialization error is swallowed with a debug log,
    because failing to log a disagreement must never break the enrichment path.
    """
    try:
        _DISAGREEMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "product": product,
            "attr": attr,
            "value_a": str(value_a),
            "value_b": str(value_b),
        }
        with _DISAGREEMENT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("Failed to log ensemble disagreement", exc_info=True)


class _ParsedAttr(Protocol):
    """Duck-type protocol for a parsed attr object from model A/B (_KnowledgeAttr-compatible)."""
    attribute_id: int
    value: "str | int | float | bool | list"
    confidence: float
    reasoning: Optional[str]


class _AgreementVerdict(BaseModel):
    """LLM arbiter verdict: do two values describe the same real-world fact."""
    same: bool


def _norm_str(v: "str | int | float | bool | list") -> str:
    """Normalize a scalar to a comparable string: strip+lower, collapse whitespace, yo-fold."""
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("ё", "е")  # yo -> ye
    return s


def _vector_cosine_agree(norm1: str, norm2: str) -> Optional[bool]:
    """Optional vector-cosine layer (sentence-transformers), best-effort.

    Lazy model load; any failure (missing package/network/runtime) -> None,
    never raises and never returns False -- no similarity found just means
    "no opinion", not "definitely different".
    """
    global _EMBEDDER
    if not config.LLM_ENSEMBLE_VECTOR_LAYER:
        return None
    try:
        if _EMBEDDER is None:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
        emb1 = _EMBEDDER.encode(norm1)
        emb2 = _EMBEDDER.encode(norm2)
        denom = math.sqrt(float(emb1 @ emb1)) * math.sqrt(float(emb2 @ emb2)) + 1e-12
        sim = float(emb1 @ emb2) / denom
        if sim >= 0.90:
            return True
        return None
    except Exception:
        logger.debug("Vector cosine agreement check failed", exc_info=True)
        return None


def values_agree_fast(
    v1: "str | int | float | bool | list",
    v2: "str | int | float | bool | list",
    target: Optional[TargetAttribute] = None,
) -> Optional[bool]:
    """Fast deterministic cascade: exact -> numeric-with-unit -> fuzzy -> optional vector.

    Returns True ONLY on unambiguous agreement. Anything uncertain (including
    "IPS" vs "VA") -> None, deferring the decision to the LLM arbiter. Anti-
    false-accept invariant: this function must never yield a false True.
    """
    if isinstance(v1, list) and isinstance(v2, list):
        set1 = {_norm_str(x) for x in v1}
        set2 = {_norm_str(x) for x in v2}
        if set1 == set2:
            return True
        return None

    if isinstance(v1, bool) and isinstance(v2, bool):
        return True if v1 == v2 else None

    norm1 = _norm_str(v1)
    norm2 = _norm_str(v2)

    if norm1 == norm2:
        return True

    # Deterministic (non-fuzzy) compact match: strips formatting-only differences
    # (whitespace, hyphens, slashes) -- e.g. "webOS" vs "Web OS" -- WITHOUT touching
    # the fuzzy threshold (which stays conservative to avoid false-accepting genuinely
    # different values that happen to share a long substring, e.g. "IPS" vs "IPS-panel").
    compact1 = re.sub(r"[\s\-_/]+", "", norm1)
    compact2 = re.sub(r"[\s\-_/]+", "", norm2)
    if compact1 and compact1 == compact2:
        return True

    num_match1 = re.match(r"^([\d.,]+)", norm1)
    num_match2 = re.match(r"^([\d.,]+)", norm2)
    if num_match1 and num_match2:
        try:
            num1 = float(num_match1.group(1).replace(",", "."))
            num2 = float(num_match2.group(1).replace(",", "."))
            if math.isclose(num1, num2, rel_tol=1e-6):
                # Gate-2 fix: equal leading numbers alone are not enough -- "5 m" vs
                # "5 mm" must NOT fast-accept. Only accept when at least one side has
                # no unit remainder at all (bare number vs unit-annotated), or both
                # remainders are the same unit text. Different non-empty remainders
                # (potential unit mismatch) fall through to fuzzy/vector/None instead.
                remainder1 = norm1[num_match1.end():].strip()
                remainder2 = norm2[num_match2.end():].strip()
                if not remainder1 or not remainder2 or remainder1 == remainder2:
                    return True
        except (ValueError, TypeError):
            pass

    score = fuzz.WRatio(norm1, norm2)
    if score >= config.LLM_ENSEMBLE_FUZZY_THRESHOLD * 100:
        return True

    if _vector_cosine_agree(norm1, norm2) is True:
        return True

    return None


async def values_agree_llm(
    v1: "str | int | float | bool | list",
    v2: "str | int | float | bool | list",
    attr_name: str,
    llm_manager: StructuredLlmManager,
    cache: Optional[dict[tuple, bool]] = None,
) -> bool:
    """Cheap LLM arbitration call (engine StructuredLlmManager, NOT provider_call.py).

    Fail-closed: any LLM/parse failure -> False (do not agree), never silently
    agree on failure. Symmetric cache keyed by (norm1, norm2, attr_name),
    scoped to one reconcile() call.
    """
    norm1 = _norm_str(v1)
    norm2 = _norm_str(v2)

    if cache is not None:
        key = (norm1, norm2, attr_name)
        key_swapped = (norm2, norm1, attr_name)
        if key in cache:
            return cache[key]
        if key_swapped in cache:
            return cache[key_swapped]

    system_prompt = (
        "You are a strict fact-arbiter. Given two candidate values for the same product attribute "
        "from two different sources, decide if they describe the SAME real-world fact/spec, "
        "allowing for wording/formatting/language differences but NOT allowing genuinely different "
        "facts to be called the same. When in doubt, answer false (same=false) -- a missed match "
        "just means a human/other logic will decide, a false match propagates a wrong spec to a "
        "live product listing."
    )
    user_text = (
        f"Attribute: {attr_name}\n"
        f"Value A: {norm1}\n"
        f"Value B: {norm2}\n"
        "Do these values describe the same real-world fact? Answer true/false."
    )

    verdict, _tokens = await llm_manager.structured_request(
        system_prompt=system_prompt,
        user_text=user_text,
        response_model=_AgreementVerdict,
    )

    result = bool(verdict.same) if verdict is not None else False

    if cache is not None:
        cache[(norm1, norm2, attr_name)] = result

    return result


async def reconcile(
    parsed_a: list,
    parsed_b: list,
    targets: list[TargetAttribute],
    llm_manager: StructuredLlmManager,
    judge: "Any | None" = None,
    context: "Any | None" = None,
    manager_a: "Any | None" = None,
    manager_b: "Any | None" = None,
    grounding_manager: "Any | None" = None,
) -> list[AttributeValue]:
    """Phase 2c — solo-значение судит ПРОТИВОПОЛОЖНЫЙ вендор (cross-vendor), не тот же. Три ветки на attribute_id."""
    by_id_a = {}
    for attr in parsed_a:
        if attr.attribute_id not in by_id_a:
            by_id_a[attr.attribute_id] = attr
    by_id_b = {}
    for attr in parsed_b:
        if attr.attribute_id not in by_id_b:
            by_id_b[attr.attribute_id] = attr

    target_by_id = {t.id: t for t in targets}
    a_ids = set(by_id_a.keys()) & set(target_by_id.keys())
    b_ids = set(by_id_b.keys()) & set(target_by_id.keys())
    both_ids = a_ids & b_ids
    solo_ids = (a_ids | b_ids) - both_ids
    cache: dict = {}
    result: list[AttributeValue] = []
    product_name = getattr(context, "product_name", None) if context is not None else None
    grounding_enabled = config.LLM_ENSEMBLE_GROUNDING_ENABLED

    def _emit(attr_id, value, confidence, evidence, target):
        result.append(AttributeValue(
            attribute_id=attr_id,
            value=value,
            confidence=confidence,
            source=Source.LLM_KNOWLEDGE,
            evidence=(evidence[:297] + "...") if evidence and len(evidence) > 300 else evidence,
            semantic_type=target.semantic_type,
            is_collection=target.is_collection,
        ))

    for attr_id in both_ids:
        attr_a = by_id_a[attr_id]
        attr_b = by_id_b[attr_id]
        target = target_by_id[attr_id]
        fast = values_agree_fast(attr_a.value, attr_b.value, target)
        if fast is True:
            agree = True
        elif fast is None:
            attr_name = str(getattr(target, "name", None) or target.id)
            agree = await values_agree_llm(attr_a.value, attr_b.value, attr_name, llm_manager, cache)
        else:
            agree = False
        if agree:
            evidence_parts = [p for p in (attr_a.reasoning, attr_b.reasoning) if p]
            evidence = "ensemble consensus (A+B): " + " | ".join(evidence_parts)
            evidence = evidence.strip(" |")
            _emit(attr_id, attr_a.value, config.LLM_ENSEMBLE_CONFIDENCE, evidence, target)
        else:
            attr_name_d = str(getattr(target, "name", None) or target.id)
            grounded_val = None
            if grounding_enabled and grounding_manager is not None:
                grounded_val = await ground_disagreement(product_name, attr_name_d, attr_a.value, attr_b.value, grounding_manager)
            if grounded_val is not None:
                _emit(attr_id, grounded_val, config.LLM_ENSEMBLE_CONFIDENCE, f"ensemble disagreement grounded: {grounded_val}", target)
            else:
                _log_disagreement(product_name, attr_name_d, attr_a.value, attr_b.value)

    # Phase 2c: prepare cross-vendor solo judges once. A value produced by model A
    # (DeepSeek) is judged by manager_b (gpt-4o-mini) and vice versa -- the judge never
    # shares a vendor with the producer, so it cannot rubber-stamp its own hallucination.
    solo_judge_mode = config.LLM_ENSEMBLE_SOLO_JUDGE
    cross_judge_for_a = None   # judges A-produced solos -> bound to manager_b (opposite vendor)
    cross_judge_for_b = None   # judges B-produced solos -> bound to manager_a (opposite vendor)
    if solo_judge_mode == "cross" and manager_a is not None and manager_b is not None:
        from app.services.enrichment.judges.knowledge_judge import KnowledgeJudge
        cross_judge_for_a = KnowledgeJudge(llm_manager=manager_b)
        cross_judge_for_b = KnowledgeJudge(llm_manager=manager_a)

    for attr_id in solo_ids:
        target = target_by_id[attr_id]
        from_a = attr_id in by_id_a
        solo = by_id_a[attr_id] if from_a else by_id_b[attr_id]
        if config.LLM_ENSEMBLE_SOLO_POLICY != "judge" or context is None:
            continue
        # Phase 3: enum/categorical solos are where "category-plausible but product-wrong"
        # values hide (e.g. headphone form-factor). Ground ONLY those against an external
        # product-specific source. Non-enum solos keep the cheaper judge path (cost bound).
        is_enum_target = bool(getattr(target, "allowed_values", None))
        if grounding_enabled and grounding_manager is not None and is_enum_target:
            g = await ground_value(product_name, str(getattr(target, "name", None) or target.id), solo.value, grounding_manager)
            if g == "confirm":
                _emit(attr_id, solo.value, 0.85, f"ensemble solo (grounded-confirm): {solo.reasoning or ''}".strip(), target)
                continue
            if g == "refute":
                continue
            # g == "unknown" -> fall through to the existing solo-judge path below.
        # cross-vendor judge (opposite of producer) when wired; else the passed same-vendor judge.
        if solo_judge_mode == "cross" and cross_judge_for_a is not None:
            chosen_judge = cross_judge_for_a if from_a else cross_judge_for_b
        else:
            chosen_judge = judge
        if chosen_judge is None:
            continue
        candidate = AttributeValue(
            attribute_id=attr_id,
            value=solo.value,
            confidence=0.85,
            source=Source.LLM_KNOWLEDGE,
            evidence=solo.reasoning,
            semantic_type=target.semantic_type,
            is_collection=target.is_collection,
        )
        ok = await chosen_judge.validate(candidate, context)
        if ok:
            vendor_note = "cross-vendor" if (solo_judge_mode == "cross" and cross_judge_for_a is not None) else "main"
            evidence = f"ensemble solo (judge-confirmed, {vendor_note}): {solo.reasoning or ''}".strip()
            _emit(attr_id, solo.value, 0.85, evidence, target)

    return result
