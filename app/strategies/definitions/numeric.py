from typing import Any

from pydantic import BaseModel

from .root import RootStrategy
from ..validators.numeric_validator import NumericValidator
from ...judge.judge_profile import JudgeProfile, JudgeResult


class NumericBranch(RootStrategy):
    name = "NUMERIC_BRANCH"
    # FIX: Явно добавили упоминание Area в описание родительского класса
    description = "Quantitative attributes where the expected logical answer is strictly a NUMERICAL VALUE, scalar magnitude, measurable dimension (including 2D Area and 3D Volume), or geometric angle."
    allow_transformation = True

    @classmethod
    def evaluate_need_for_judgment(cls, extracted_value: Any, text: str, options: list = None) -> str:
        if extracted_value is None:
            return "ACCEPT"

        # 1. Защита от слепых галлюцинаций
        if not cls.allow_transformation:
            if not NumericValidator.is_value_in_text(extracted_value, text):
                return "REJECT"  # ⛔ Скрипт убивает мусор бесплатно

        # 2. НОВАЯ ЛОГИКА: Цифра есть в тексте? БЕРЕМ! Экономим токены.
        return "ACCEPT"

    @classmethod
    def get_judge_profile(cls) -> JudgeProfile:
        profile = super().get_judge_profile()
        profile.custom_rules.append(
            "UNIT_INTEGRITY: Verify that the numerical scalar logically matches the source text. If unit conversion was required to match the Target Feature's unit, verify the math is correct."
        )
        return profile

    @classmethod
    def get_domain_rules(cls) -> list[str]:
        return [
            "DATA TYPE: The result must be a numeric value (integer or float).",
            "SANITIZATION: Remove all non-numeric characters (units, currency symbols).",
            "FORMAT: Use a dot '.' for decimal separators.",
            # 👇 3. ДАЛИ ПРЯМУЮ ИНСТРУКЦИЮ ЭКСТРАКТОРУ: Разрешили математику для единиц измерения
            "UNIT ALIGNMENT: If the numerical value in the source text uses a different physical unit than the explicitly requested TARGET UNIT/SUFFIX, you MUST perform standard mathematical conversion to output the final value exactly in the requested TARGET UNIT. If units match, preserve the raw scalar.",
            "SEMANTIC SCOPING (CRITICAL): The extracted scalar must strictly align with any limiting adjectives or modifiers present in the requested feature name. If the feature asks for a specific subset, component, or relative state, you are STRICTLY FORBIDDEN from extracting the overall, total, base, or generic maximum value.",
            "NO CROSS-PROPERTY INFERENCE: Never assume mathematical correlations or physical equivalencies between different domains (e.g., volume to mass). Extract ONLY explicitly stated values."
        ]

class RadialDimensionLeaf(NumericBranch):
    name = "RADIAL_DIMENSION_EXTRACTOR"
    description = "Measurements passing through the center of a circle/sphere or connecting opposite corners of a polygon (e.g., diagonals, diameters, radii)."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Radial Geometry Analyst.
        GOAL: Extract the scalar value representing the cross-sectional, radial, or diagonal distance.

        ABSTRACT LOGIC:
        GEOMETRIC TARGETING: Isolate values defining a span across a circular, cylindrical, or rectangular plane.
        COMPONENT MATCHING: Ensure the dimension applies strictly to the target component (e.g., screen, wheel, lens) and not the overall body.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        return f"""
        ABSTRACT DEDUCTION RULES FOR RADIAL/DIAGONAL DIMENSIONS:
        TYPOGRAPHICAL RECOGNITION: Identify metric abbreviations, punctuation marks, or technical symbols conventionally used in schematics to denote radial or cross-corner spans. Treat these symbols as explicit functional declarations for '{target_feature}', bypassing the need for descriptive nouns.
        """


class AngularDimensionLeaf(NumericBranch):
    name = "ANGULAR_DIMENSION_EXTRACTOR"
    description = "Geometric angles, rotational degrees, sectors, tilt, pan, or field of view (FOV)."
    supports_deduction = True

    @classmethod
    def get_judge_profile(cls) -> JudgeProfile:
        profile = super().get_judge_profile()
        profile.custom_rules.append(
            "DOMAIN_MATCH: Verify the extracted angle corresponds exactly to the physical, optical, or kinematic domain requested by the Target Feature."
        )
        return profile

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Angular Measurement Analyst.
        GOAL: Extract the specific angular, rotational, or sector-based measurement.

        ABSTRACT LOGIC:
        SECTOR IDENTIFICATION: Isolate numeric values representing a geometric angle or field of coverage.
        PHYSICAL CONTEXT MATCHING (CRITICAL): Angles serve vastly different physical purposes (e.g., optical fields of view vs. mechanical limits of rotation/tilt). You MUST analyze the specific modifiers in the TARGET FEATURE and ensure the extracted value corresponds to the exact same physical domain. Never extract an optical angle if a mechanical rotation is requested.
        """
    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        return f"""
        ABSTRACT DEDUCTION RULES FOR ANGLES:
        IMPLICIT KINEMATIC/OPTICAL NOTATION: Isolate numerical scalars paired with standard geometric sector symbols or domain-specific optical acronyms. Treat these entities as direct, implicit validations of '{target_feature}'.
        """


class SurfaceAreaLeaf(NumericBranch):
    name = "SURFACE_AREA_EXTRACTOR"
    # Добавили якорь "coverage area", чтобы роутер выбирал Numeric, а не Text/Configuration
    description = "Two-dimensional spatial measurements, coverage area, footprint, or explicitly declared squared metrics."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Surface Area Analyst.
        GOAL: Extract the numeric value representing a two-dimensional coverage or footprint.

        ABSTRACT LOGIC:
        2D TARGETING: Isolate scalar values strictly associated with squared units.
        NO GEOMETRIC MATH (CRITICAL): You are STRICTLY FORBIDDEN from calculating the area by multiplying length by width. If the total squared area is not explicitly pre-calculated and stated in the text, you MUST return None.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        return f"""
        ABSTRACT DEDUCTION RULES FOR SURFACE AREA:
        PLANAR EXTENT MAPPING: Correlate quantitative values bound to squared dimensional units with '{target_feature}'.
        """

class BiometricDimensionLeaf(NumericBranch):
    name = "BIOMETRIC_DIMENSION_EXTRACTOR"
    description = "Measurements related to biological anatomy, ergonomic sizing constraints, or apparel tailoring dimensions."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Ergonomic & Tailoring Analyst.
        GOAL: Extract the scalar value corresponding to the specific anatomical or tailored dimension.

        ABSTRACT LOGIC:
        AXIAL MAPPING: Correlate the target feature with the specific bodily axis or clothing segment it describes.
        ISOLATION: Strictly differentiate these segmented, ergonomic constraints from the overarching geometric bounding box of the object.
        TAXONOMIC DECODING: Alphanumeric sizing nomenclatures often encapsulate exact numerical dimensions. Extract the raw scalar embedded within the taxonomy without converting its native unit system.
        """

class CompositeAggregationLeaf(NumericBranch):

    @classmethod
    def get_judge_profile(cls) -> JudgeProfile:
        profile = super().get_judge_profile()
        # ⛔ Убираем правило поиска точной цитаты
        profile.core_rules.pop("TRACEABILITY", None)
        # 🟢 Отменяем запрет на математику
        profile.core_rules["MATH"] = "MATH IS ALLOWED. Verify if the extracted value is a correct mathematical aggregation (sum or product) of the explicitly stated individual metrics."
        return profile


    allow_transformation = True
    name = "COMPOSITE_AGGREGATION_EXTRACTOR"
    description = "Quantitative metrics where the TARGET FEATURE explicitly requests the CUMULATIVE TOTAL, overall sum, or aggregated bulk amount of a multi-pack, bundle, GIFT SET, or composite product. If the product consists of multiple items with distinct volumes/weights, this node MUST calculate their sum."

    supports_deduction = True

    @classmethod
    def get_domain_rules(cls) -> list[str]:
        return [
            "DATA TYPE: The result must be a numeric value (integer or float).",
            "SANITIZATION: Remove all non-numeric characters (units, currency symbols).",
            "FORMAT: Use a dot '.' for decimal separators.",
            "UNIT MATCHING (CRITICAL): If the TARGET UNIT/SUFFIX is specified, you MUST ONLY extract the number physically associated with that exact unit.",
            "SEMANTIC SCOPING (CRITICAL): The extracted scalar must strictly align with any limiting adjectives or modifiers present in the requested feature name.",
            "MATH ALLOWED (CRITICAL): You are explicitly ALLOWED to perform arithmetic addition or multiplication ONLY to calculate the requested cumulative total."
        ]

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Inventory Aggregation Auditor.
        GOAL: Calculate the specific cumulative quantitative metric for a composite product.

        ABSTRACT LOGIC:
        CUMULATIVE SCOPE: Mathematically aggregate (add or multiply) the explicitly stated metrics of the individual containers/items to calculate the final total required by the target feature.
        """


class CapacityLeaf(NumericBranch):
    name = "CAPACITY_VOLUME"
    description = "Liquid volume, net content, or internal capacity of a single unit."
    supports_deduction = True

    @classmethod
    def get_judge_profile(cls) -> JudgeProfile:
        profile = super().get_judge_profile()
        profile.custom_rules.append(
            "CAPACITY_SCOPE: Verify the extracted volume represents the exact containment scope requested (e.g., individual unit versus total multipack volume). Do not allow scope mismatches."
        )
        return profile

    @classmethod
    def process_judgment(cls, raw_verdict: BaseModel, tokens: int) -> JudgeResult:
        if raw_verdict is None:
            return super().process_judgment(raw_verdict, tokens)

        needs_review = False
        # 👇 Читаем из analysis 👇
        status = f"OK. Judge Logic: {getattr(raw_verdict, 'analysis', 'No analysis provided')}"

        # Ужесточаем суд для объемов: малейшее сомнение в Scope (уверенность ошибки >= 50) = на ревью
        if not raw_verdict.is_supported_by_text or raw_verdict.violates_rules:
            if raw_verdict.error_confidence >= 50:
                needs_review = True
                # 👇 Читаем из analysis 👇
                status = f"STRICT REJECT (Scope mismatch risk): {getattr(raw_verdict, 'analysis', 'No analysis')}"
            else:
                # 👇 Читаем из analysis 👇
                status = f"PASSED (Doubt {raw_verdict.error_confidence}%): {getattr(raw_verdict, 'analysis', 'No analysis')}"

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
        ROLE: Volumetric Auditor.
        GOAL: Determine the specific net content or internal capacity requested.

        ABSTRACT LOGIC:
        UNIT FILTERING: Focus strictly on volumetric units or mass equivalents representing internal containment.
        SCOPE ISOLATION: Distinguish between base capacity, extension/delta capacity, and total aggregated capacity. Extract ONLY the exact scope requested by the target feature.
        MULTIPACK ISOLATION: If the product is a multi-pack, extract ONLY the explicitly stated volume of one individual item. STRICTLY FORBIDDEN to multiply or aggregate.
        """

class PhysicalMagnitudeLeaf(NumericBranch):

    @classmethod
    def get_judge_profile(cls) -> JudgeProfile:
        profile = super().get_judge_profile()
        profile.custom_rules.append(
            "COMPARATIVE_STATE: If the Target Feature implies a comparative equivalent or benchmark, strictly verify the extracted value represents that comparative state, not the intrinsic base property."
        )
        return profile

    name = "PHYSICAL_MAGNITUDE"
    description = "Intrinsic scalar properties defined by scientific units (e.g., Power, Weight, Voltage)."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        # ДОБАВЛЕНА ЛОГИКА СРАВНЕНИЙ ДЛЯ ЛАМПЫ (Эквиваленты)
        return f"""
        ROLE: Metrologist.
        GOAL: Extract the scalar value corresponding to the requested physical phenomenon.

        ABSTRACT LOGIC:
        PHENOMENON MATCHING: Identify the numeric value associated with the target unit context ('{unit}').
        COMPARATIVE RESOLUTION (CRITICAL): If the context presents multiple scalars representing different states (e.g., an intrinsic baseline vs. a comparative equivalent benchmark), analyze the target feature. If the feature implies a comparative state, extract ONLY the equivalent benchmark value, overriding the base property.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        return """
        ABSTRACT DEDUCTION RULES FOR PHYSICAL MAGNITUDES:
        SHORTHAND DECODING: If multiple scalars sharing the same unit are presented in a shorthand format, correlate the values to the target feature based on standard industry hierarchies.
        """

class QuantificationLeaf(NumericBranch):
    name = "QUANTIFICATION"
    description = "Discrete counts, integer quantities, or cardinality."
    supports_deduction = True

    @classmethod
    def get_task_logic(cls, unit: str) -> str:
        return """
        ROLE: Quantitative Auditor.
        GOAL: Determine the discrete cardinality or total count.

        ABSTRACT LOGIC:
        COUNTING: Identify the integer value representing the quantity of items.
        DISCRETENESS: The output implies a countable number of entities.
        """

    @classmethod
    def get_leaf_deduction_logic(cls, target_feature: str, unit: str = "") -> str:
        # НИКАКИХ УПОМИНАНИЙ GB ИЛИ RAM
        return """
        ABSTRACT DEDUCTION RULES FOR QUANTIFICATION:
        DELIMITED HIERARCHIES: When multiple scalars are presented in a combined, delimited format (e.g., separated by slashes or hyphens), map the TARGET FEATURE to the correct scalar by evaluating their relative magnitudes against the standard functional constraints of the generic object class.
        SUBSET ALLOCATION: If the TARGET FEATURE represents a functional subset of a larger capacity, prioritize the mathematically smaller scalar in the sequence.
        """