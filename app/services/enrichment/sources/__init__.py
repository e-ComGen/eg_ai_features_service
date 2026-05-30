"""Sources — реализации AttributeSource из spec docs/architecture/pipeline.md."""
from .description_source import DescriptionSource
from .llm_knowledge_source import LlmKnowledgeSource
from .vision_source import VisionSource
from .web_search_source import WebSearchSource
from .competitor_rag_source import CompetitorRagSource
from .icecat_source import IceCatSource
from .pdf_datasheet_source import PdfDatasheetSource
from .ozon_card_source import OzonCardSource
from .wb_card_source import WbCardSource
from .ugc_source import UgcSource

__all__ = [
    "DescriptionSource",
    "LlmKnowledgeSource",
    "VisionSource",
    "WebSearchSource",
    "CompetitorRagSource",
    "IceCatSource",
    "PdfDatasheetSource",
    "OzonCardSource",
    "WbCardSource",
    "UgcSource",
]
