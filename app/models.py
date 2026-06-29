# CpAiFeatures/app/models.py

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, AnyHttpUrl, field_validator


class ResearchMode(str, Enum):
    OFF = "off"            # only extract from given description
    FALLBACK = "fallback"  # if description yields nothing, try LLM training knowledge
    DEEP = "deep"          # fallback + (future) web search tool


class FeatureOption(BaseModel):
    # Real marketplace attribute_id (Ozon). Forwarded by eg-importer so the
    # pipeline's field-map sources (WbCard/IceCat) and value_id resolution land
    # on the right attribute instead of a positional index. None → adapter falls
    # back to the positional index (back-compat for callers that omit it).
    id: Optional[int] = None
    type: str
    options: List[Any] = []

    # Marketplace-required flag. Forwarded by callers (e.g. eg-importer ai_fill)
    # from the real Ozon/WB schema so the pipeline can apply required-only logic
    # (required-enum mud-gate, finishing priority, required gender fallback).
    # Defaults False — back-compat for callers that omit it.
    is_required: bool = False
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
    # Identifiers that unlock the rich enrichment sources. Forwarded by callers
    # (e.g. eg-importer ai_fill): brand → brand-fill + IceCat brand match;
    # ean → IceCat / barcode authoritative specs. process_product already
    # forwards these into the pipeline via getattr.
    brand: Optional[str] = None
    ean: Optional[str] = None
    # Ozon type_id → lets OzonStrategy resolve dictionary value_ids for this exact
    # category/type (the eval passes this; the integration used to drop it).
    ozon_type_id: Optional[int] = None
    context: ProductContext
    languages: List[str] = ["en"]
    source_urls: List[str] = Field(default_factory=list)
    # Vision branch: up to 10 image URLs accepted here; VisionProducer
    # will itself cap at its MAX_IMAGES limit (4) for cost control.
    image_urls: List[str] = Field(default_factory=list)

    @field_validator("source_urls")
    @classmethod
    def validate_source_urls(cls, v: List[str]) -> List[str]:
        if len(v) > 5:
            raise ValueError("source_urls: maximum 5 URLs per product")
        for url in v:
            if not url.startswith("https://"):
                raise ValueError(f"source_urls: only HTTPS URLs allowed, got: {url!r}")
        return v

    @field_validator("image_urls")
    @classmethod
    def validate_image_urls(cls, v: List[str]) -> List[str]:
        if len(v) > 10:
            raise ValueError("image_urls: maximum 10 URLs per product")
        return v


class BatchOptions(BaseModel):
    """Feature-flag options for a batch request."""
    enable_vision: bool = False
    enable_web_search: bool = False
    marketplace: Optional[str] = None  # "default" | "wb" | "ozon"


class BatchPayload(BaseModel):
    client_id: int
    products: List[ProductData]
    schemas: Dict[Any, Dict[str, FeatureOption]]
    use_cache: bool = False
    research_mode: ResearchMode = ResearchMode.OFF
    options: BatchOptions = Field(default_factory=BatchOptions)
