"""FinishingExtractor — финальный проход: повторная попытка для обязательных пустых атрибутов.

Запускается после основного pipeline только если остались незаполненные is_required targets.
Переиспользует существующие AttributeSource-инстансы с усиленной инструкцией в промпте.
Ожидаемый прирост recall: +9-18pp на required fields (WDC-PAVE benchmark).
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
        missing_required = [
            t for t in targets
            if t.is_required and t.id not in filled_ids
        ]
        if not missing_required:
            logger.debug("[Finishing] no missing required attributes — skip")
            return []

        logger.debug(
            "[Finishing] %d missing required attributes: %s",
            len(missing_required),
            [t.id for t in missing_required],
        )

        results: list[AttributeValue] = []
        for source in self._sources:
            applicable = [
                t for t in missing_required
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
