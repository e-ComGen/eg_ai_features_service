"""CompetitorRagJudge — лёгкий судья для RAG-источника карточек-конкурентов.

CompetitorRagSource использует реальные Ozon-листинги которые прошли модерацию,
поэтому false positive rate низкий. Судья принимает любое значение с confidence ≥ 0.6
без LLM-вызова — экономит токены при высокой точности.

Spec: docs/architecture/pipeline.md (CompetitorRagJudge — lightweight, no LLM).
"""
from app.services.enrichment.base import LlmJudge, AttributeValue, ExtractionContext, Source

# Минимальный порог уверенности для принятия RAG-значения
_ACCEPT_THRESHOLD = 0.6


class CompetitorRagJudge(LlmJudge):
    """Принимает RAG-значения с confidence ≥ 0.6 без LLM-вызова.

    Обоснование: данные из датасета evgmaslov/ozon_ecup — реальные карточки
    которые прошли Ozon-модерацию. Consensus ≥ 2/5 соседей дополнительно
    фильтрует выбросы. LLM-проверка была бы дороже, чем потенциальный false positive.
    """
    source = Source.COMPETITOR_RAG

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        """True если confidence ≥ threshold. Без LLM-вызовов."""
        return value.confidence >= _ACCEPT_THRESHOLD
