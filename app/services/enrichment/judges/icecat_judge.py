"""IceCatJudge — лёгкий судья для IceCat-источника.

IceCat Open API отдаёт brand-verified спецификации от производителя.
Данные проходят верификацию на стороне IceCat, поэтому false positive rate
крайне низкий. Принимаем все значения с confidence ≥ 0.85 без LLM-вызова.

Spec: docs/architecture/pipeline.md (IceCatJudge — lightweight, no LLM).
"""
from app.services.enrichment.base import LlmJudge, AttributeValue, ExtractionContext, Source

# Минимальный порог уверенности для принятия IceCat-значения
_ACCEPT_THRESHOLD = 0.85


class IceCatJudge(LlmJudge):
    """Принимает IceCat-значения с confidence ≥ 0.85 без LLM-вызова.

    Обоснование: IceCat Open — официальные спецификации от бренда (ASUS, HP, Lenovo, etc.).
    Данные верифицированы производителем и прошли проверку на стороне IceCat.
    LLM-проверка здесь была бы дороже, чем потенциальный false positive.
    """
    source = Source.ICECAT

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        """True если confidence ≥ threshold. Без LLM-вызовов."""
        return value.confidence >= _ACCEPT_THRESHOLD
