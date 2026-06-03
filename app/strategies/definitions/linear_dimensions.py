from typing import Any, Optional, Type

from pydantic import BaseModel, Field, field_validator

from .numeric import NumericBranch
from ...judge.judge_profile import JudgeProfile


class DimensionalWorkerResult(BaseModel):
    analysis: str = Field(
        ...,
        description="STEP 1: Extract all numbers from the text. Sort them mathematically: Maximum = X, Intermediate = Y, Minimum = Z."
    )
    rule_mapping: str = Field(
        ...,
        description="STEP 2: Read the 'EXTRACTION LOGIC' rules for this specific object. Map the sorted numbers to Length, Width, Height exactly as the rules dictate."
    )
    extracted_value: Optional[str] = Field(
        None,
        description="STEP 3: Return ONLY the exact number requested by the TARGET FEATURE based on STEP 2."
    )
    confidence: str

    @field_validator("extracted_value", mode="before")
    @classmethod
    def _coerce_scalar(cls, v: Any) -> Any:
        # Same coercion as WorkerResult — dimensional extractions arrive as
        # raw numbers (60, 2.0, etc.) from providers that ignore the string
        # type hint.
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return str(v)
        return v

class LinearLogicMixin:

    @classmethod
    def get_judge_profile(cls) -> JudgeProfile:
        profile = super().get_judge_profile()

        # 👇 3. ИЗМЕНЯЕМ МОЗГ СУДЬИ ТОЛЬКО ДЛЯ ГАБАРИТОВ 👇
        profile.role = "Strict Geometric Auditor."

        # Добавляем приоритетные протоколы в САМОЕ НАЧАЛО списка протоколов
        priority_protocols = [
            "CUSTOM RULES OVERRIDE BIAS: If specific logic (e.g., geometric axioms, mathematical sorting) is provided in the CUSTOM RULES, it takes ABSOLUTE PRECEDENCE over common formatting habits.",
            "ANTI-SEQUENCE BIAS: You are STRICTLY FORBIDDEN from rejecting a correct value just because it contradicts the typical sequence order (e.g., assuming the first number in 'A x B x C' is always Length). Evaluate purely on CUSTOM RULES."
        ]
        profile.audit_protocol = priority_protocols + profile.audit_protocol

        # Добавляем сами правила (как и было)
        profile.custom_rules.append(
            "SEQUENCE BIAS AUDIT: When evaluating unlabelled dimensions (e.g., 'A x B x C'), STRICTLY verify that the extracted value maps logically to the requested spatial axis based on the physical reality of the object. Do NOT automatically approve the first number as Length or the largest number as Length."
        )

        if hasattr(cls, 'get_task_logic'):
            extractor_logic = cls.get_task_logic(unit="")
            profile.custom_rules.append(
                f"ORIGINAL EXTRACTION RULES (CRITICAL): The extractor was bound by the following logic to determine the correct axis. You MUST use these exact same geometric axioms to verify if the extractor made the correct choice:\n{extractor_logic}"
            )

        return profile
    @classmethod
    def get_response_model(cls, options: list = None) -> Type[BaseModel]:
        return DimensionalWorkerResult

    @classmethod
    def evaluate_need_for_judgment(cls, extracted_value: Any, text: str, options: list = None) -> str:
        # Сначала прогоняем базовую проверку на слепые галлюцинации
        base_status = super().evaluate_need_for_judgment(extracted_value, text, options)

        # Если базовая логика сказала "ACCEPT" (цифра в тексте есть)...
        if base_status == "ACCEPT":
            # ...мы принудительно отправляем на суд!
            # Потому что это габариты, и ИИ мог перепутать Высоту с Длиной.
            return "JUDGE"

        return base_status

    @classmethod
    def get_base_task_logic(cls) -> str:
        return """
        ABSTRACT LOGIC (SHARED FOR LINEAR DIMENSIONS):
        1. EXPLICIT LABELS (CRITICAL): If the text includes dimensional prefixes/acronyms (e.g., 'ШхВхГ', 'W x H x D', 'L/W/H'), you MUST strictly map the numbers to their corresponding labels based on their exact sequence position.
        2. SEQUENCE BIAS OVERRIDE (CRITICAL): For unlabelled sequences (A x B x C), you are STRICTLY FORBIDDEN from assuming the first number is 'Length'. 
        3. ANTI-BIAS REASONING: In your reasoning, you MUST NOT justify your answer by saying "the first value is the length". You MUST justify your extraction based on the Magnitude Sorting Algorithm (Maximum, Intermediate, Minimum) relative to the object's physical shape.
        4. NO DUPLICATION: If the requested axis is not explicitly provided or cannot be logically deduced, return None. Do not duplicate values from other axes.
        5. KINEMATIC STATE: Map numerical values to their corresponding operational state (e.g., folded vs unfolded vs package).
        """

    @classmethod
    def get_base_deduction_logic(cls) -> str:
        return """
        ABSTRACT DEDUCTION RULES FOR LINEAR DIMENSIONS:
        SPATIAL DECODING: Map unlabelled scalars based on standard geometric proportions of the object class, overriding sequence order or relative magnitude biases.
        """

# ---------------------------------------------------------
# ДЕКОМПОЗИЦИЯ: СПЕЦИФИЧНЫЕ ГЕОМЕТРИЧЕСКИЕ ЛИСТЬЯ
# Множественное наследование: (LinearLogicMixin, NumericBranch)
# ---------------------------------------------------------

class SubComponentDimensionLeaf(LinearLogicMixin, NumericBranch):
    name = "SUB_COMPONENT_DIMENSION_EXTRACTOR"
    description = "Dimensions of localized functional sub-parts, modular extensions, internal sleeping/seating areas, or attachments. STRICTLY EXCLUDES overarching main-body dimensions."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return f"""
        ROLE: Spatial Orthogonal Analyst.
        GOAL: Extract the scalar value for the specified linear axis.

        {cls.get_base_task_logic()}

        SPECIFIC LOGIC (LOCALIZED DELTA):
        COMPONENT ISOLATION (CRITICAL): The TARGET FEATURE requests the dimension of a specific SUB-COMPONENT. You are STRICTLY FORBIDDEN from extracting the dimensions of the main object. 
        MAPPING: Extract ONLY the explicit number logically and syntactically linked to that localized sub-component in the text.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        return cls.get_base_deduction_logic()


class UprightStructureDimensionLeaf(LinearLogicMixin, NumericBranch):
    name = "UPRIGHT_STRUCTURE_DIMENSION"
    description = "Dimensions for floor-standing, vertically oriented objects (e.g., large appliances, cabinets, refrigerators, shelving). Characteristic: The vertical axis (Height) is typically the dominant magnitude."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return f"""
        ROLE: Spatial Orthogonal Analyst.
        GOAL: Extract the scalar value for the specified linear axis.

        {cls.get_base_task_logic()}

        SPECIFIC LOGIC (UPRIGHT GEOMETRY):
        MAGNITUDE INFERENCE: For upright structures with unlabelled sequences (A x B x C), the numerically largest value is almost always the vertical Height, regardless of its sequence position. Width and Depth occupy the horizontal plane.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        return cls.get_base_deduction_logic()


class FlatPanelDimensionLeaf(LinearLogicMixin, NumericBranch):
    name = "FLAT_PANEL_DIMENSION"
    description = "Dimensions for planar displays, panels, or reflective surfaces with minimal depth (e.g., screens, monitors, flat boards, rugs)."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return f"""
        ROLE: Spatial Orthogonal Analyst.
        GOAL: Extract the scalar value for the specified linear axis.

        {cls.get_base_task_logic()}

        SPECIFIC LOGIC (PLANAR GEOMETRY):
        MAGNITUDE INFERENCE: For flat panels/surfaces with unlabelled sequences, the smallest value is strictly the Depth/Thickness. 
        CRITICAL RULE FOR LENGTH VS WIDTH: The mathematically LARGEST horizontal value is STRICTLY the Length. The intermediate value is the Width.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        return cls.get_base_deduction_logic()


class LongExtrusionDimensionLeaf(LinearLogicMixin, NumericBranch):
    name = "LONG_EXTRUSION_DIMENSION"
    description = "Dimensions for elongated structural materials, rolled goods, or extrusions (e.g., lumber, profiles, pipes, cables) where one primary axis vastly exceeds the cross-section."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return f"""
        ROLE: Spatial Orthogonal Analyst.
        GOAL: Extract the scalar value for the specified linear axis.

        {cls.get_base_task_logic()}

        SPECIFIC LOGIC (EXTRUSION GEOMETRY):
        MAGNITUDE INFERENCE: For extruded/elongated objects, the overwhelmingly largest numerical value is ALWAYS the primary longitudinal axis (Length). The remaining smaller values define the cross-sectional plane (Thickness/Width/Diameter).
        SEQUENCE OVERRIDE: Industrial formats often write 'Thickness x Width x Length' (T x W x L). Prioritize the magnitude difference over sequence order.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        return cls.get_base_deduction_logic()


class SoftContainerDimensionLeaf(LinearLogicMixin, NumericBranch):
    name = "SOFT_CONTAINER_DIMENSION"
    description = "Dimensions for flexible volumes, wearable carriers, or collapsible gear (e.g., backpacks, bags, strollers, tents). Dimensions may dynamically shift based on load or state."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return f"""
        ROLE: Spatial Orthogonal Analyst.
        GOAL: Extract the scalar value for the specified linear axis.

        {cls.get_base_task_logic()}

        SPECIFIC LOGIC (FLEXIBLE/WHEELED GEOMETRY):
        1. CAPACITY AVOIDANCE: Ensure you extract a spatial linear axis (Length/Width/Height) and NOT volumetric capacity (Liters). 
        2. STATE-DEPENDENT AXIS MAPPING (CRITICAL): These objects dramatically change shape depending on their state.
           - UNFOLDED / DEPLOYED STATE: The vertical axis dominates. You MUST map: MAXIMUM = Height, INTERMEDIATE = Length, MINIMUM = Width.
           - PACKAGED / FOLDED STATE: When packed in a box or carrying bag (e.g., 'Package Length'), the object becomes a standard elongated cuboid or cylinder. For this state, you MUST map: MAXIMUM = Package Length.
        3. ANTI-HALLUCINATION RULE: Analyze the TARGET FEATURE. If it requests a 'Package' (Упаковка) dimension, you are STRICTLY FORBIDDEN from using the Unfolded mapping. You must extract the Maximum scalar of the package dimensions for its Length.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        return cls.get_base_deduction_logic()


class DefaultBoxDimensionLeaf(LinearLogicMixin, NumericBranch):
    name = "DEFAULT_BOX_DIMENSION"
    description = "Standard cuboid dimensions for generic objects, packaging, or parcels without a highly skewed aspect ratio. Use this if the object doesn't fit specific geometric categories."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return f"""
        ROLE: Spatial Orthogonal Analyst.
        GOAL: Extract the scalar value for the specified linear axis.

        {cls.get_base_task_logic()}

        SPECIFIC LOGIC (CUBOID GEOMETRY):
        SEQUENCE FALLBACK: If dimensions are entirely unlabelled and the object lacks a specific skewed topology, default to analyzing the sequence as Length x Width x Height. Length is typically the longest dimension of the horizontal base.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        return cls.get_base_deduction_logic()