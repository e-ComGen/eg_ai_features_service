"""WbCardJudge — лёгкий судья для WbCardSource.

Источник копирует характеристики из живой Wildberries-карточки очень похожего/идентичного
товара. Карточки проходят pre-moderation на стороне WB (значения уже соответствуют
словарю WB), поэтому false positive rate низкий. Принимаем все значения с
confidence ≥ 0.80 без LLM-вызова.

Spec: см. WbCardSource — "exact match" (≥ threshold name similarity, conf=0.93),
"brand_line match" (нижний bucket, conf=0.85).
"""
from app.services.enrichment.base import AttributeValue, ExtractionContext, LlmJudge, Source

# Минимальный порог уверенности для принятия значения
_ACCEPT_THRESHOLD = 0.80


class WbCardJudge(LlmJudge):
    """Принимает WbCard-значения с confidence ≥ 0.80 без LLM-вызова.

    Обоснование: данные приходят из реальной WB-карточки идентичного товара
    (после name-similarity ≥ 0.78 либо ≥ 0.60 для brand-line подмножества).
    Карточки прошли модерацию WB, поэтому значения соответствуют словарю.
    LLM-проверка обошлась бы дороже потенциального false positive.
    """
    source = Source.WB_CARD

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        return value.confidence >= _ACCEPT_THRESHOLD
