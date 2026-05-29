"""OzonCardJudge — лёгкий судья для OzonCardSource.

Источник копирует характеристики из живой Ozon-карточки очень похожего/идентичного товара.
Карточки проходят pre-moderation на стороне Ozon (значения уже соответствуют словарю),
поэтому false positive rate низкий. Принимаем все значения с confidence ≥ 0.80 без LLM-вызова.

Spec: см. OzonCardSource — "exact match" (≥90% name similarity, conf=0.93),
"brand_line match" (70-89%, conf=0.85).
"""
from app.services.enrichment.base import AttributeValue, ExtractionContext, LlmJudge, Source

# Минимальный порог уверенности для принятия значения
_ACCEPT_THRESHOLD = 0.80


class OzonCardJudge(LlmJudge):
    """Принимает OzonCard-значения с confidence ≥ 0.80 без LLM-вызова.

    Обоснование: данные приходят из реальной Ozon-карточки идентичного товара
    (после name-similarity ≥ 0.90 либо ≥ 0.70 для brand-line подмножества).
    Карточки прошли модерацию Ozon, поэтому значения соответствуют словарю.
    LLM-проверка обошлась бы дороже потенциального false positive.
    """
    source = Source.OZON_CARD

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        return value.confidence >= _ACCEPT_THRESHOLD
