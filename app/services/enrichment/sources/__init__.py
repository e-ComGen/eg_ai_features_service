"""Sources — реализации AttributeSource из spec docs/architecture/pipeline.md."""
from .description_source import DescriptionSource
from .llm_knowledge_source import LlmKnowledgeSource
from .vision_source import VisionSource
from .web_search_source import WebSearchSource

__all__ = [
    "DescriptionSource",
    "LlmKnowledgeSource",
    "VisionSource",
    "WebSearchSource",
]
