from typing import Any
from ..services.llm_manager import OpenAIManager
from .judge_profile import JudgeProfile


class HallucinationJudge:
    def __init__(self, llm_manager: OpenAIManager):
        self.llm = llm_manager

    async def execute_audit(self, text: str, feature_name: str, extracted_value: Any, profile: JudgeProfile) -> tuple[
        Any, int]:

        rules_text = "\n".join([f"- {k}: {v}" for k, v in profile.core_rules.items()])
        if profile.custom_rules:
            rules_text += "\n" + "\n".join([f"- {r}" for r in profile.custom_rules])

        protocol_text = "\n".join([f"{i + 1}. {p}" for i, p in enumerate(profile.audit_protocol)])

        system_prompt = f"""
        ROLE: {profile.role}
        GOAL: {profile.goal}

        AUDIT PROTOCOL:
        {protocol_text}

        RULES TO ENFORCE:
        {rules_text}
        """

        # 👇 ИДЕАЛЬНЫЙ ФИКС СЛОЯ ДАННЫХ (БЕЗ ПЕРЕОБУЧЕНИЯ ПРОМПТОВ) 👇
        # Если пришел словарь с переводами, достаем из него только сами значения
        if isinstance(extracted_value, dict):
            display_value = " / ".join(str(v) for v in extracted_value.values())
        else:
            display_value = extracted_value

        user_prompt = f"""
        <source_text>
        {text}
        </source_text>

        <audit_task>
        TARGET FEATURE: "{feature_name}"
        EXTRACTED VALUE: "{display_value}"
        </audit_task>
        """

        return await self.llm.structured_request(
            system_prompt=system_prompt,
            user_text=user_prompt,
            response_model=profile.response_schema
        )