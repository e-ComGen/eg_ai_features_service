from abc import ABC, abstractmethod
from typing import List, Type, Any

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List

from app.judge.judge_profile import JudgeProfile, JudgeResult


class WorkerResult(BaseModel):
    analysis: str = Field(
        ...,
        description="STEP 1: Step-by-step logic. You MUST explicitly analyze the source text, map variables, and validate rules before extracting the final value."
    )
    extracted_value: Optional[str] = Field(
        ...,
        description="STEP 2: The exact extracted value based on the analysis above."
    )

    confidence: str

    @field_validator("extracted_value", mode="before")
    @classmethod
    def _coerce_scalar(cls, v: Any) -> Any:
        # DeepSeek and similar providers commonly return numeric extractions
        # as raw int/float (e.g. 60, 2.0) instead of the string the schema
        # asks for. Coerce so a clean numeric answer doesn't get thrown out
        # by Pydantic's strict string check.
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return str(v)
        return v

class DeductionResult(BaseModel):
    context_clues: str = Field(
        ...,
        description="Logical deductions, synonyms, or industry standards found in the text relevant to the target feature."
    )
    confidence_score: int = Field(
        ...,
        description="An integer from 1 to 100. How confident are you that this context directly implies the value for the target feature?"
    )

class BaseStrategyNode(ABC):
    name: str = "BASE"
    description: str = "Root strategy."
    allow_transformation: bool = False

    @classmethod
    def get_judge_profile(cls) -> JudgeProfile:
        """По умолчанию возвращаем базовый профиль."""
        return JudgeProfile()

    @classmethod
    def process_judgment(cls, raw_verdict: BaseModel, tokens: int) -> JudgeResult:
        """
        БАЗОВАЯ ФУНКЦИЯ СУДИМОСТИ.
        Роут сам решает, как трактовать ответ LLM-Судьи.
        """
        if raw_verdict is None:
            return JudgeResult(
                is_success=False, needs_review=False, status_message="Judge LLM Error", tokens_used=tokens
            )

        needs_review = False
        status = "OK"

        if not raw_verdict.is_supported_by_text or raw_verdict.violates_rules:
            if raw_verdict.error_confidence >= 80:
                needs_review = True
                status = f"REJECTED: {getattr(raw_verdict, 'analysis', 'No analysis')}"
            else:
                status = f"PASSED (Doubt {raw_verdict.error_confidence}%): {getattr(raw_verdict, 'analysis', 'No analysis')}"

        return JudgeResult(
            is_success=not needs_review,
            needs_review=needs_review,
            status_message=status,
            raw_verdict=raw_verdict,
            tokens_used=tokens
        )

    @classmethod
    def evaluate_need_for_judgment(cls, extracted_value: Any, text: str, options: list = None) -> str:
        """
        ФИЛЬТР ПРЕ-ВАЛИДАЦИИ:
        Возвращает:
        - "ACCEPT": 100% надежно, берем в базу (Судья не нужен)
        - "REJECT": 100% галлюцинация, сбрасываем (Судья не нужен)
        - "JUDGE": Сомнительно, вызываем LLM-Судью
        """
        if extracted_value is None:
            return "ACCEPT"  # Пустоту не судим

        return "JUDGE"  # По умолчанию всегда зовем Судью

    @classmethod
    def get_response_model(cls, options: list = None) -> Type[BaseModel]:
        """
        Возвращает нужную Pydantic-модель для Structured Outputs.
        ОБЯЗАТЕЛЬНО должен быть переопределен в наследниках (TextBranch, NumericBranch).
        """
        return WorkerResult

    @classmethod
    def get_instruction(cls, unit: str = "", options: list = None, target_languages: list[str] = None) -> str:
        """
        Собирает инструкцию снизу вверх:
        1. Base Rules (System)
        2. Domain Rules (Branch)
        3. Task Logic (Leaf)
        4. Dynamic Constraints (Options & Languages) <-- НОВОЕ
        """
        rules = []

        # 1. BASE LEVEL
        rules.append("--- SYSTEM CONSTRAINTS ---")
        rules.extend([
            "OUTPUT FORMAT: Return ONLY the raw data payload. No prose, no labels.",
            "NULL HANDLING: Return 'None' if the property is physically missing from the text.",
            "SCOPE: Extract properties of the MAIN product only, ignoring accessories.",
            "HONESTY: Do not infer values unless they are implied by standard conventions."
        ])

        # 2. DOMAIN LEVEL
        if hasattr(cls, 'get_domain_rules'):
            rules.append("\n--- DATA TYPE RULES ---")
            rules.extend(cls.get_domain_rules())

        # 3. TASK LEVEL
        if hasattr(cls, 'get_task_logic'):
            rules.append("\n--- EXTRACTION LOGIC ---")
            rules.append(cls.get_task_logic(unit))

        # 👇 4. DYNAMIC LEVEL (Опции словаря и Мультиязычность) 👇
        if hasattr(cls, 'get_dynamic_rules'):
            dynamic_rules = cls.get_dynamic_rules(options, target_languages)
            if dynamic_rules:
                rules.append("\n--- DYNAMIC CONSTRAINTS ---")
                rules.append(dynamic_rules)

        return "\n".join(rules)


    supports_deduction: bool = False

    @classmethod
    def get_deduction_instruction(cls, target_feature: str, unit: str = "") -> str:
        rules = []

        rules.append("--- BASE DEDUCTION RULES ---")
        rules.extend([
            "ROLE: Context Researcher.",
            f"GOAL: Find hidden clues or synonyms for '{target_feature}'.",
            "CONSTRAINT: Base your deduction strictly on the provided text.",
            "OUTPUT: Provide the logical context and rate your confidence (1-100)."
        ])

        if hasattr(cls, 'get_branch_deduction_rules'):
            rules.append("\n--- BRANCH DEDUCTION RULES ---")
            rules.extend(cls.get_branch_deduction_rules())

        if hasattr(cls, 'get_leaf_deduction_logic'):
            rules.append("\n--- LEAF DEDUCTION LOGIC ---")
            rules.append(cls.get_leaf_deduction_logic(target_feature, unit))

        return "\n".join(rules)

    @classmethod
    def is_leaf(cls) -> bool:
        return hasattr(cls, 'get_task_logic')

    @classmethod
    def get_children(cls) -> List[Type['BaseStrategyNode']]:
        return cls.__subclasses__()

    @classmethod
    def get_options_text(cls) -> str:
        # Генерирует меню для Роутера
        options = [f"- {child.name}: {child.description}" for child in cls.get_children() if child.name != "BASE"]
        return "\n".join(options)