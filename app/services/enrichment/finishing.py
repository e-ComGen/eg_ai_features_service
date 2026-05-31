"""FinishingExtractor — финальный проход: повторная попытка для пустых атрибутов.

Запускается после основного pipeline если остались незаполненные targets (required или optional).
Переиспользует существующие AttributeSource-инстансы с усиленной инструкцией в промпте.
Ожидаемый прирост recall: +9-18pp на required fields (WDC-PAVE benchmark).

Лимит: не более MAX_FINISHING_TARGETS за один проход. Required-атрибуты получают приоритет
перед optional — это не меняет стоимость (1 LLM call на source), но гарантирует что при лимите
required-attrs не вытесняются optional.
"""
import logging
from typing import Optional

from app.services.enrichment.base import (
    AttributeSource,
    AttributeValue,
    ExtractionContext,
    TargetAttribute,
)

logger = logging.getLogger(__name__)

# Максимальное число атрибутов за один finishing-проход.
# Required идут первыми, optional добиваются до лимита.
# Один source делает 1 LLM call вне зависимости от числа targets — лимит
# защищает от слишком длинного промпта, а не от числа LLM calls.
_MAX_FINISHING_TARGETS = 15

# Instruction prepended to system_prompt when running in focused recovery mode
_FOCUSED_PREFIX = (
    "FOCUSED RECOVERY MODE: These required attributes are still empty after the main "
    "pipeline. Try harder — re-read the source text carefully, infer the value if it is "
    "explicitly hinted or can be unambiguously derived. Return null (skip the attribute) "
    "only if the value is genuinely absent in the source material. "
    "Do NOT fabricate or guess values that are not supported by evidence.\n\n"
)


class FinishingExtractor:
    """Stage: повторная попытка извлечения для пустых обязательных атрибутов.

    Переиспользует переданные AttributeSource-инстансы (не создаёт новые).
    Каждый source вызывается с focused=True — что добавляет усиленную инструкцию
    к system_prompt. Возвращает новые AttributeValue, которые оркестратор мержит
    в основной список перед финальным _finalize.
    """

    def __init__(self, sources: list[AttributeSource]):
        self._sources = sources

    async def extract_missing(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: list[AttributeValue],
    ) -> list[AttributeValue]:
        """Запустить focused re-extraction для пустых обязательных атрибутов.

        Args:
            context: Текущий контекст извлечения.
            targets: Все нормализованные targets pipeline (включая required).
            already_filled: AttributeValue полученные после основного pipeline (до _finalize).

        Returns:
            Дополнительные AttributeValue от focused pass. Пустой список если нечего делать.
        """
        filled_ids = {av.attribute_id for av in already_filled}
        # Включаем и required, и optional — required получают приоритет при лимите
        missing_all = [t for t in targets if t.id not in filled_ids]
        if not missing_all:
            logger.debug("[Finishing] no missing attributes — skip")
            return []

        # Required первыми, optional в конце; обрезаем до лимита
        missing_required = [t for t in missing_all if t.is_required]
        missing_optional = [t for t in missing_all if not t.is_required]
        missing_targets = (missing_required + missing_optional)[:_MAX_FINISHING_TARGETS]

        logger.debug(
            "[Finishing] %d missing targets (required=%d, optional=%d, limit=%d): %s",
            len(missing_targets),
            len([t for t in missing_targets if t.is_required]),
            len([t for t in missing_targets if not t.is_required]),
            _MAX_FINISHING_TARGETS,
            [t.id for t in missing_targets],
        )

        results: list[AttributeValue] = []
        for source in self._sources:
            applicable = [
                t for t in missing_targets
                if source.is_applicable(context, t)
            ]
            if not applicable:
                continue
            try:
                extracted = await _extract_focused(source, context, applicable)
                results.extend(extracted)
            except Exception as exc:
                logger.warning(
                    "[Finishing] source %s failed: %s",
                    source.source_type.value,
                    exc,
                    exc_info=True,
                )

        logger.debug("[Finishing] recovered %d values", len(results))
        return results


async def _extract_focused(
    source: AttributeSource,
    context: ExtractionContext,
    targets: list[TargetAttribute],
) -> list[AttributeValue]:
    """Вызвать source.extract с патченным system_prompt через focused=True.

    Поскольку AttributeSource.extract не принимает focused параметр, мы патчим
    контекст через временный wrapper-объект который подменяет _llm.structured_request
    чтобы добавить _FOCUSED_PREFIX к system_prompt.
    """
    return await _FocusedSourceProxy(source).extract(context, targets)


class _FocusedSourceProxy:
    """Proxy вокруг AttributeSource: перехватывает structured_request и добавляет focused prefix."""

    def __init__(self, source: AttributeSource):
        self._source = source

    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
    ) -> list[AttributeValue]:
        """Запустить extract с monkey-patched _llm чтобы добавить focused instruction."""
        llm = getattr(self._source, "_llm", None)
        if llm is None:
            # Source не имеет _llm — вызываем напрямую
            return await self._source.extract(context, targets)

        original_fn = llm.structured_request

        async def _focused_request(
            system_prompt: str,
            user_text: str,
            response_model,
            **kwargs,
        ):
            return await original_fn(
                system_prompt=_FOCUSED_PREFIX + system_prompt,
                user_text=user_text,
                response_model=response_model,
                **kwargs,
            )

        llm.structured_request = _focused_request
        try:
            return await self._source.extract(context, targets)
        finally:
            llm.structured_request = original_fn
