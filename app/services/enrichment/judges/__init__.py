"""Per-source judges из spec docs/architecture/pipeline.md."""
from .description_judge import DescriptionJudge
from .knowledge_judge import KnowledgeJudge
from .vision_judge import VisionJudge
from .websearch_judge import WebSearchJudge
from .competitor_rag_judge import CompetitorRagJudge
from .icecat_judge import IceCatJudge
from .ozon_card_judge import OzonCardJudge

__all__ = [
    "DescriptionJudge",
    "KnowledgeJudge",
    "VisionJudge",
    "WebSearchJudge",
    "CompetitorRagJudge",
    "IceCatJudge",
    "OzonCardJudge",
]
