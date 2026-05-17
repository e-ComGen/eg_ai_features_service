# CpAiFeatures/app/models.py

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel


class ResearchMode(str, Enum):
    OFF = "off"            # only extract from given description
    FALLBACK = "fallback"  # if description yields nothing, try LLM training knowledge
    DEEP = "deep"          # fallback + (future) web search tool


class FeatureOption(BaseModel):
    type: str
    options: List[Any] = []

    suffix: Optional[str] = ""
    prefix: Optional[str] = ""
    # Operator-configured extraction constraint forwarded from admin_rules.settings.
    # When set, job_processor prepends it to the product text as a priority instruction.
    prompt_hint: Optional[str] = None


class ProductContext(BaseModel):
    existing_features: Union[Dict[str, Any], List[Any]] = {}
    company_id: int = 0


class ProductData(BaseModel):
    id: int
    category_id: int
    id_path: Optional[str] = ""
    name: str
    description: Optional[str] = ""
    price: Union[float, int, str, None] = 0.0
    context: ProductContext
    languages: List[str] = ["en"]


class BatchPayload(BaseModel):
    client_id: int
    products: List[ProductData]
    schemas: Dict[Any, Dict[str, FeatureOption]]
    use_cache: bool = False
    research_mode: ResearchMode = ResearchMode.OFF
