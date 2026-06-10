"""Базовые типы и абстрактные интерфейсы для cost-aware pipeline.

Спецификация: docs/architecture/pipeline.md

Каждый "Source" — это источник характеристик (description, vision, web_search, etc).
Каждый "Judge" — валидатор результата source-а (per-source specialized prompts).
PipelineOrchestrator — sequential cost-aware dispatcher.

Этот файл — только типы и интерфейсы. Реализации в sources/, judges/, intelligence/.
"""
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, Optional, Union

from pydantic import BaseModel, Field, field_validator


# Semantic types для которых hallucinations критичны (false positive имеет реальные последствия).
# Для них ConfidenceAwareJudgeWrapper всегда вызывает judge, игнорируя confidence shortcut.
CRITICAL_SEMANTIC_TYPES: frozenset[str] = frozenset({
    "ean",          # European Article Number (штрихкод 13 цифр)
    "upc",          # Universal Product Code (12 цифр, US)
    "gtin",         # Global Trade Item Number
    "barcode",      # generic
    "article",      # артикул производителя
    "sku",          # stock keeping unit
    "serial",       # серийный номер
    "imei",         # IMEI для электроники
    "model_code",   # точный код модели
})


def is_critical_semantic_type(semantic_type: Optional[str]) -> bool:
    """True если semantic_type требует обязательной проверки judge независимо от confidence."""
    if not semantic_type:
        return False
    return semantic_type.lower() in CRITICAL_SEMANTIC_TYPES


class Source(StrEnum):
    """Откуда пришло значение характеристики.

    Используется для:
    - audit trail (видно откуда какое значение)
    - merge tie-break при равной confidence
    - per-source judge dispatch
    """
    DESCRIPTION = "description"        # из текста описания товара
    LLM_KNOWLEDGE = "llm_knowledge"    # из обучающей памяти LLM
    VISION = "vision"                   # с фото товара
    WEB_SEARCH = "web_search"           # из веб-поиска
    COMPETITOR_RAG = "competitor_rag"   # consensus из top-K похожих Ozon-карточек (без LLM)
    ICECAT = "icecat"                   # верифицированные спеки от бренда через IceCat Open API
    PDF_DATASHEET = "pdf_datasheet"     # спеки извлечены из официального datasheet PDF производителя
    OZON_CARD = "ozon_card"             # копия характеристик из live-карточки Ozon (через Apify ozon-scraper-pro)
    WB_CARD = "wb_card"                 # копия характеристик из live-карточки Wildberries (открытое API, без Scrappey)
    UGC = "ugc"                         # отзывы + Q&A с Ozon/WB (user-generated, реальные пользователи)
    SAFE_ENUM_FILL = "safe_enum_fill"   # gated LLM fill: short optional enum, двойной Gate (verbatim + adversarial)


# Source priority при tie-break (если confidence равна).
# Более привязанные к конкретному товару источники имеют выше приоритет.
SOURCE_PRIORITY: dict[Source, int] = {
    Source.DESCRIPTION: 4,         # самый надёжный — описание конкретного товара
    Source.PDF_DATASHEET: 4,       # официальный datasheet производителя — авторитетен как описание
    Source.OZON_CARD: 4,           # копия с pre-modered Ozon-карточки точно такого же товара
    Source.WB_CARD: 4,             # копия с pre-modered WB-карточки точно такого же товара
    Source.VISION: 3,              # тоже про этот товар, но визуально
    Source.ICECAT: 3,              # brand-verified спеки от производителя
    Source.WEB_SEARCH: 2,          # про товар, но внешний источник
    Source.COMPETITOR_RAG: 2,      # реальные Ozon-карточки с модерацией
    Source.UGC: 2,                 # отзывы реальных покупателей — обычно про конкретный товар, но noisy
    Source.LLM_KNOWLEDGE: 1,       # общие знания
    Source.SAFE_ENUM_FILL: 1,      # gated LLM enum fill — уровень знаний, но двойной gate
}


# Per-source confidence thresholds: выше — судью не зовём, доверяем.
SOURCE_CONFIDENCE_THRESHOLDS: dict[Source, float] = {
    Source.DESCRIPTION: 0.95,
    Source.PDF_DATASHEET: 0.90,    # официальный PDF datasheet — высокий порог без judge
    Source.OZON_CARD: 0.90,        # копия с pre-modered Ozon-карточки — высокий порог без LLM-judge
    Source.WB_CARD: 0.90,          # копия с pre-modered WB-карточки — высокий порог без LLM-judge
    Source.ICECAT: 0.90,           # brand-verified: высокий порог без judge
    Source.LLM_KNOWLEDGE: 0.92,
    Source.WEB_SEARCH: 0.88,
    Source.VISION: 0.85,
    Source.COMPETITOR_RAG: 0.80,   # consensus из реальных Ozon-листингов — высокая точность
    Source.UGC: 0.75,              # отзывы/Q&A — шумные, низкий порог, всегда через judge
    Source.SAFE_ENUM_FILL: 0.82,   # gated LLM fill — adversarial guard, но всё равно LLM-инференс
}


_Scalar = Union[str, int, float, bool]


class AttributeValue(BaseModel):
    """Одно извлечённое значение характеристики от какого-то source."""
    attribute_id: int = Field(..., description="ID характеристики в схеме CS-Cart")
    value: Union[_Scalar, list[_Scalar]] = Field(..., description="Извлечённое значение (скаляр или список для is_collection)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность 0-1")
    source: Source = Field(..., description="Откуда пришло значение")
    evidence: Optional[str] = Field(None, description="Цитата/URL/обоснование для audit")
    judge_validated: bool = Field(False, description="Прошёл ли через per-source judge")
    semantic_type: Optional[str] = Field(None, description="copy из TargetAttribute.semantic_type для downstream решений")
    is_collection: bool = Field(False, description="Характеристика-массив (из is_collection словаря)")
    value_id: Optional[int] = Field(None, description="Словарный ID для одиночного значения (Ozon)")
    value_ids: Optional[list[int]] = Field(None, description="Словарные ID для массива значений (Ozon is_collection)")

    @field_validator("evidence")
    @classmethod
    def evidence_max_length(cls, v: Optional[str]) -> Optional[str]:
        """Truncate evidence чтобы не раздувать БД."""
        if v and len(v) > 1000:
            return v[:1000] + "..."
        return v

    def is_confident(self) -> bool:
        """True если confidence ≥ source-specific threshold AND attribute не критичный.

        Critical attrs (EAN, UPC, article) НИКОГДА не считаются is_confident — всегда judge.
        """
        if is_critical_semantic_type(self.semantic_type):
            return False  # критичные всегда требуют judge
        return self.confidence >= SOURCE_CONFIDENCE_THRESHOLDS[self.source]


class TargetAttribute(BaseModel):
    """Описание характеристики которую нужно заполнить."""
    id: int
    name: str
    type: str = Field(..., description="text | numeric | enum | bool")
    allowed_values: Optional[list[str]] = None
    semantic_type: Optional[str] = Field(None, description="color | weight | material_visual | brand | etc — подсказка для classifier")
    description: Optional[str] = None
    is_collection: bool = Field(False, description="Характеристика принимает массив значений")
    is_required: bool = Field(False, description="Обязательная характеристика маркетплейса")


class ExtractionContext(BaseModel):
    """Контекст одного extraction-запроса.

    Передаётся между stages чтобы каждый source знал что уже найдено
    (cost-aware: пропускаем атрибуты которые уже filled с high confidence).
    """
    product_id: int
    product_name: str
    product_description: Optional[str] = None
    category_id: int
    category_path: list[str] = Field(default_factory=list)
    brand: Optional[str] = None
    ean: Optional[str] = None
    mpn: Optional[str] = None              # Manufacturer Part Number (артикул производителя для IceCat / datasheet lookup)
    article: Optional[str] = None          # Артикул производителя (отдельно от MPN — иногда это разные коды)
    source_urls: list[str] = Field(default_factory=list)
    image_urls: list[str] = Field(default_factory=list)

    # Web-search languages (dual-lang). None → source applies its own default
    # (WEBSEARCH_LANGS env, ["ru","en"] by default — searches RU + EN in parallel).
    languages: Optional[list[str]] = None

    # Marketplace-specific fields
    marketplace: Optional[str] = None          # "ozon" | "wb" | None
    ozon_type_id: Optional[int] = None         # Ozon type_id для точного словарного поиска

    # Cost tracking
    cost_so_far_usd: float = 0.0
    max_cost_usd: float = 0.10
    llm_calls_so_far: int = 0


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class LlmJudge(ABC):
    """Per-source валидатор: проверяет что извлечённое значение обосновано.

    Каждый source имеет свой judge с specialized промптом (vision_judge знает
    как распознавать vision-галлюцинации, websearch_judge — источниковую достоверность).
    """
    source: Source  # для какого source предназначен

    @abstractmethod
    async def validate(self, value: AttributeValue, context: ExtractionContext) -> bool:
        """True если значение обосновано, False если выглядит как галлюцинация."""
        ...


class AttributeSource(ABC):
    """Источник характеристик. Реализации: DescriptionSource, LlmKnowledgeSource,
    VisionSource, WebSearchSource (шаги B-E)."""

    @property
    @abstractmethod
    def source_type(self) -> Source:
        """Какой это source."""
        ...

    @abstractmethod
    def is_applicable(self, context: ExtractionContext, target: TargetAttribute) -> bool:
        """Может ли этот source в принципе извлечь этот атрибут?

        Примеры: VisionSource.is_applicable требует image_urls. WebSearchSource
        не имеет смысла для уникальных кастомных товаров.
        """
        ...

    @abstractmethod
    async def extract(
        self,
        context: ExtractionContext,
        targets: list[TargetAttribute],
        already_filled: Optional[list["AttributeValue"]] = None,
    ) -> list[AttributeValue]:
        """Извлечь значения для applicable targets.

        already_filled — список AttributeValue с confidence ≥ 0.85 от предыдущих sources.
        Источник должен исключить их из targets и добавить в prompt как «уже известные».
        Может вернуть пустой список (источник ничего не нашёл).
        Все возвращённые AttributeValue должны иметь source=self.source_type.
        """
        ...

    @abstractmethod
    def get_judge(self) -> LlmJudge:
        """Specialized judge для этого source."""
        ...


# Public API
__all__ = [
    "Source",
    "SOURCE_PRIORITY",
    "SOURCE_CONFIDENCE_THRESHOLDS",
    "CRITICAL_SEMANTIC_TYPES",
    "is_critical_semantic_type",
    "AttributeValue",
    "TargetAttribute",
    "ExtractionContext",
    "LlmJudge",
    "AttributeSource",
]
