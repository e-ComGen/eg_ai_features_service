"""PdfDatasheetJudge — лёгкий судья для PdfDatasheetSource.

Datasheet PDFs от whitelisted доменов — официальные документы производителя.
False positive rate низкий (если URL прошёл хост-фильтр и content-type=application/pdf).
Принимаем все значения с confidence ≥ 0.85 без LLM-вызова.
"""
from app.services.enrichment.base import LlmJudge, AttributeValue, ExtractionContext, Source

_ACCEPT_THRESHOLD = 0.85


class PdfDatasheetJudge(LlmJudge):
    source = Source.PDF_DATASHEET

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        return value.confidence >= _ACCEPT_THRESHOLD
