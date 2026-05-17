from typing import Optional, List, Type, Any
from pydantic import BaseModel, Field
from .root import RootStrategy
from ...judge.judge_profile import JudgeProfile, JudgeResult


# --- НОВЫЕ СХЕМЫ ОТВЕТОВ ТОЛЬКО ДЛЯ ТЕКСТА ---

class TranslationItem(BaseModel):
    language: str = Field(..., description="Language code (e.g., 'en', 'ru')")
    text: str = Field(..., description="The extracted text, translated to this language.")


class MultiLangWorkerResult(BaseModel):
    analysis: str = Field(
        ...,
        description="STEP 1: Step-by-step logic. Explain why this value was extracted and how you translated it."
    )
    extracted_values: List[TranslationItem] = Field(
        ...,
        description="STEP 2: List of translations for the requested languages. Return an empty list [] if missing."
    )
    confidence: str


class OptionWorkerResult(BaseModel):
    analysis: str = Field(
        ...,
        description="STEP 1: Step-by-step logic. Explain why this specific dictionary option is the best match."
    )
    extracted_value: Optional[str] = Field(
        ...,
        description="STEP 2: The EXACT matched string from the allowed English dictionary."
    )
    confidence: str


# --- ПЕРЕГРУЖЕННЫЙ КЛАСС TEXT_BRANCH ---

class TextBranch(RootStrategy):
    name = "TEXT_BRANCH"
    description = "Qualitative attributes where the expected answer is a TEXT STRING or an ALPHANUMERIC TECHNICAL STANDARD. Strictly excludes pure measurable mathematical scalars."

    @classmethod
    def evaluate_need_for_judgment(cls, extracted_value: Any, text: str, options: list = None) -> str:
        if not extracted_value:
            return "ACCEPT"

        # FAST ACCEPT: Жесткое совпадение со словарем
        if options and extracted_value in options:
            return "ACCEPT"

        return super().evaluate_need_for_judgment(extracted_value, text, options)


    @classmethod
    def get_judge_profile(cls) -> JudgeProfile:
        profile = super().get_judge_profile()
        profile.custom_rules.append(
            "NO_TRANSLATION_ALTERATION: If multilingual extraction is evaluated, strictly verify that any alphanumeric models, specifications, or numbers remain exactly identical across all translated strings."
        )
        return profile

    @classmethod
    def get_domain_rules(cls) -> list[str]:
        return [
            "TYPE: The result must be a text string.",
            "BREVITY: Keep the output concise (1-3 words usually).",
            "CLEANING: Remove trademarks, copyrights, and promotional adjectives."
        ]

    # 👇 ПЕРЕГРУЗКА 1: Выдаем нужную схему в зависимости от наличия опций 👇
    @classmethod
    def get_response_model(cls, options: list = None) -> Type[BaseModel]:
        if options and len(options) > 0:
            return OptionWorkerResult
        return MultiLangWorkerResult

    # 👇 ПЕРЕГРУЗКА 2: Генерируем жесткие правила для промпта 👇
    @classmethod
    def get_dynamic_rules(cls, options: list = None, target_languages: list[str] = None) -> str:
        rules = []

        # Если есть словарь опций (Select/Checkbox)
        if options and len(options) > 0:
            opts_str = ", ".join([str(o) for o in options])
            rules.append(f"ALLOWED DICTIONARY (English): [{opts_str}]")
            rules.append(
                "CRITICAL RULE: You MUST return ONLY a value that perfectly matches an item in this dictionary. If the true value is missing, return None.")
            rules.append(
                "FATAL ERROR PREVENTION: If an ALLOWED DICTIONARY is provided, it is strictly forbidden to return any string not explicitly listed in the dictionary.")

        # Если словаря нет, но есть список языков для перевода
        elif target_languages:
            lang_str = ", ".join(target_languages)
            rules.append(
                f"MULTILINGUAL OUTPUT: You MUST extract the text and translate it into these exact languages: [{lang_str}].")
            rules.append(
                "CRITICAL: Ensure numbers, models, and specifications remain EXACTLY identical across all translations.")

        return "\n".join(rules)


# --- ТВОИ ЛИСТЬЯ (Без изменений) ---

class ConfigurationLeaf(TextBranch):
    name = "CONFIGURATION_EXTRACTOR"
    description = "Standardized operational modes, interface protocols, spatial orientations, or alphanumeric technical formatting standards."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Standards Compliance Engineer.
        GOAL: Extract the specific technical standard, interface configuration, or usage mode.

        ABSTRACT LOGIC:
        SPATIAL LAYOUT: Identify the specific spatial configuration required for user operation.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        # НИКАКИХ УПОМИНАНИЙ ПРАВШЕЙ ИЛИ ГИТАР
        return """
        ABSTRACT DEDUCTION RULES FOR CONFIGURATIONS:
        IMPLICIT DOMINANT STANDARD: If the TARGET FEATURE dictates a spatial operational mode for an asymmetrical physical tool, and the product is described as a standard base model with no lateral constraints explicitly mentioned, deduce the ubiquitous industry-default configuration.
        """


class BrandLeaf(TextBranch):
    # Имя должно совпадать с тем, что выбирает Роутер!
    name = "BRAND_EXTRACTOR"
    description = "Manufacturer name or Brand."

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Entity Recognizer.
        GOAL: Identify the Brand or Manufacturer.
        LOGIC: The Brand is typically the FIRST word of the Title.
        """


class MaterialLeaf(TextBranch):
    name = "MATERIAL_EXTRACTOR"
    description = "The physical substance composing the overarching primary structure, specific sub-components, or formulation texture."
    supports_deduction = True

    @classmethod
    def get_judge_profile(cls) -> JudgeProfile:
        profile = super().get_judge_profile()
        profile.custom_rules.append(
            "MATERIAL_SCOPE: Verify the extracted material belongs strictly to the specific physical component requested by the Target Feature, ignoring overarching body materials if a sub-component is specified."
        )
        return profile

    @classmethod
    def process_judgment(cls, raw_verdict: BaseModel, tokens: int) -> JudgeResult:
        if raw_verdict is None:
            return super().process_judgment(raw_verdict, tokens)

        needs_review = False
        # 👇 ЧИТАЕМ ИЗ ПОЛЯ ANALYSIS 👇
        status = f"OK. Judge Logic: {getattr(raw_verdict, 'analysis', 'No analysis provided')}"

        # СНИСХОДИТЕЛЬНЫЙ СУД: Порог поднят до 90 (допускаем больше сомнений)
        if not raw_verdict.is_supported_by_text or raw_verdict.violates_rules:
            if raw_verdict.error_confidence >= 90:
                needs_review = True
                status = f"REJECTED: {getattr(raw_verdict, 'analysis', 'No analysis')}"
            else:
                status = f"PASSED (Doubt {raw_verdict.error_confidence}% allowed for materials): {getattr(raw_verdict, 'analysis', 'No analysis')}"

        return JudgeResult(
            is_success=not needs_review,
            needs_review=needs_review,
            status_message=status,
            raw_verdict=raw_verdict,
            tokens_used=tokens
        )

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Strict Material Parser.
        GOAL: Extract the exact physical substance.

        CRITICAL COMPONENT ISOLATION ALGORITHM:
        Analyze the TARGET FEATURE. Is it a fractional sub-component or the primary overarching body?
        If it is a sub-component, you are STRICTLY FORBIDDEN from extracting the primary structural material of the main body.
        If the exact substance composing the localized sub-component is not explicitly stated in the text, you MUST return 'None'.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        # НИКАКИХ УПОМИНАНИЙ САПФИРОВ, СТЕКОЛ И ТИТАНА
        return """
        ABSTRACT DEDUCTION RULES FOR MATERIALS:
        FUNCTIONAL EQUIVALENCY: If the TARGET FEATURE represents a generic structural component, extract any explicitly mentioned premium or proprietary substance that serves as the industry-standard functional substitute for that specific physical role.
        """


class ClassificationLeaf(TextBranch):
    name = "CLASSIFICATION_EXTRACTOR"
    description = "Abstract taxonomy, fundamental operating principles, macroscopic physical states, or formal technical certifications."

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Taxonomist.
        GOAL: Identify the specific category, core technology, or style.

        ABSTRACT LOGIC:
        DEFINITION: Extract terms that define 'what the object IS' (Category) or 'how it FUNCTIONS' (Mechanism).
        HIERARCHY: If multiple technologies are present, prioritize the CORE functional mechanism (the primary way it operates) over auxiliary features (connectivity or addons).
        """


class ContextFilterLeaf(TextBranch):
    name = "CONTEXT_VALIDATOR"
    description = "Validation strategy for dependent products."

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Logic Gate.
        GOAL: Validate if the feature applies to this product.
        LOGIC: If product is an Accessory, ensure feature applies to it, not the host device.
        """