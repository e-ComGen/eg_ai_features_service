"""ConfidenceAwareJudgeWrapper — обёртка которая решает звать judge или нет.

Если AttributeValue.is_confident() (т.е. confidence ≥ source threshold) →
trust value без вызова judge, экономия 1 LLM call.
Иначе — вызвать judge, и пометить judge_validated.

Используется PipelineOrchestrator (шаг G) для каждой пары (source, judge).

Spec: docs/architecture/pipeline.md, section "ConfidenceAwareJudgeWrapper".
"""
from typing import Optional
from app.services.enrichment.base import LlmJudge, AttributeValue, ExtractionContext


class ConfidenceAwareJudgeWrapper:
    """Wrapper: skip judge if value.is_confident().

    Цель — экономия LLM calls на judge'ах когда extraction уже уверен.
    """

    def __init__(self, judge: LlmJudge):
        self._judge = judge
        self._skipped_count = 0      # для observability
        self._validated_count = 0
        self._invalidated_count = 0

    @property
    def source(self):
        return self._judge.source

    @property
    def stats(self) -> dict[str, int]:
        return {
            "skipped": self._skipped_count,
            "validated": self._validated_count,
            "invalidated": self._invalidated_count,
        }

    async def maybe_validate(
        self, value: AttributeValue, context: ExtractionContext
    ) -> Optional[AttributeValue]:
        """Returns:
            - AttributeValue (possibly with judge_validated=True) if accepted
            - None if judge rejected the value
        """
        # Confidence shortcut
        if value.is_confident():
            self._skipped_count += 1
            # Mark as trusted (но не judge_validated — это разные понятия)
            return value

        # Low confidence → call judge
        is_valid = await self._judge.validate(value, context)
        context.llm_calls_so_far += 1

        if is_valid:
            self._validated_count += 1
            # Возвращаем копию с judge_validated=True
            return value.model_copy(update={"judge_validated": True})
        else:
            self._invalidated_count += 1
            return None
