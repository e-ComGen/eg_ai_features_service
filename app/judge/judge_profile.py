from typing import Type, Dict, List, Optional, Any
from pydantic import BaseModel, Field

# 📍 1. СТРУКТУРА ОТВЕТА СУДЬИ (Абсолютно стерильная, никакого переобучения)
class BaseJudgeVerdict(BaseModel):
    analysis: str = Field(
        ...,
        description="Step-by-step audit. You MUST logically evaluate the geometric axioms before setting the violation flags."
    )
    is_supported_by_text: bool = Field(
        description="NLI CHECK: True ONLY if the source text explicitly confirms this exact value belongs to the requested Target Feature. False if it is a hallucination or belongs to a different semantic context."
    )
    violates_rules: bool = Field(
        description="AUDIT CHECK: True if the extracted value violates ANY of the provided Core or Custom Rules."
    )
    error_confidence: int = Field(
        description="0-100. How confident are you that the extraction is INCORRECT or violates rules? 100 = definitively wrong. 0 = perfectly correct."
    )


class JudgeResult(BaseModel):
    is_success: bool
    needs_review: bool
    status_message: str
    raw_verdict: Optional[Any]
    tokens_used: int = 0


class JudgeProfile:
    def __init__(self, response_schema: Type[BaseModel] = BaseJudgeVerdict):
        self.response_schema = response_schema

        # 👇 1. ДИНАМИЧЕСКИЕ НАСТРОЙКИ ПРОМПТА 👇
        self.role: str = "Independent Data Auditor."
        self.goal: str = "Verify if the extracted value is a hallucination, a context mismatch, or a rule violation."
        self.audit_protocol: List[str] = [
            "Read the SOURCE TEXT.",
            "Evaluate the EXTRACTED VALUE against the TARGET FEATURE.",
            "Enforce the provided RULES strictly."
        ]

        # Базовые законы
        self.core_rules: Dict[str, str] = {
            "TRACEABILITY": "The extracted value MUST be explicitly present in the source text. UNLABELLED CONTEXT: Product titles natively contain unlabelled numerical or categorical data. An unlabelled value located in the product title is fully traceable and valid if it logically matches the mathematical or domain constraints of the TARGET FEATURE.",
            "MATH": "NO MATH. Do not accept values derived from calculations.",
            "CONTEXT": "Ensure the value relates EXACTLY to the requested feature without semantic shifts or misattributions."
        }
        self.custom_rules: List[str] = []