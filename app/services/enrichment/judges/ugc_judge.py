"""UgcJudge — лёгкий судья для UgcSource (отзывы + Q&A с Ozon/WB).

UGC-источник извлекает характеристики из user-generated content: текст отзывов,
поля pros/cons на WB, вопросы-ответы на Ozon. Это noisy данные — один отзыв
= одно мнение конкретного покупателя, не вердикт производителя. Поэтому
порог принятия выше, чем у card-источников (0.75 vs 0.80), а confidence cap
в источнике ограничен 0.75.

Логика: UgcSource уже агрегирует похожие упоминания (если 2+ отзыва говорят
одно и то же — confidence выше). Значит после source-level фильтрации
данные достаточно надёжные, чтобы пропустить LLM-judge ради экономии.

Spec: см. UgcSource — WB feedbacks JSON (открытый API без auth) +
Ozon reviews/questions HTML через Scrappey.
"""
from app.services.enrichment.base import AttributeValue, ExtractionContext, LlmJudge, Source

# Минимальный порог уверенности для принятия значения.
# UGC noisy, поэтому threshold выше абсолютного floor, но ниже card-источников.
_ACCEPT_THRESHOLD = 0.70


class UgcJudge(LlmJudge):
    """Принимает UGC-значения с confidence ≥ 0.70 без LLM-вызова.

    Обоснование: UgcSource внутренне ограничивает confidence до 0.75 (cap)
    и поднимает выше только при совпадении 2+ отзывов. Если значение
    прошло source-level фильтр (regex по цифрам/единицам + LLM extraction
    с явным указанием "один отзыв — одно мнение"), оно достаточно надёжное.
    LLM-проверка обошлась бы дороже потенциального false positive.

    Critical semantic types (EAN, article, MPN) обрабатываются в
    ConfidenceAwareJudgeWrapper отдельно — для них этот judge никогда
    не вызывается напрямую.
    """
    source = Source.UGC

    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        return value.confidence >= _ACCEPT_THRESHOLD
