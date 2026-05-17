import logging
import re
from openai import AsyncOpenAI
import instructor
from pydantic import BaseModel, Field, create_model
from enum import Enum
from typing import Optional


class OllamaManager:
    def __init__(self):
        raw_client = AsyncOpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )
        self.client = instructor.patch(raw_client, mode=instructor.Mode.JSON)
        self.model = "llama3.1"

    def _clean_value(self, value: str) -> str:
        """
        Убивает 'N/A', 'Not specified', 'NULL' и прочий мусор.
        """
        if not value:
            return ""

        # 1. Базовая очистка
        val = str(value).strip().strip("'").strip('"').strip(".").strip()
        val_lower = val.lower()

        # 2. Список стоп-слов (строгое совпадение)
        garbage_exact = [
            "n/a", "null", "none", "unknown", "not specified",
            "not mentioned", "empty", "no info", "not available",
            "is not relevant", "not applicable"
        ]

        if val_lower in garbage_exact:
            return ""

        # 3. Список стоп-фраз (если встречаются внутри)
        garbage_phrases = [
            "does not mention",
            "cannot be determined",
            "no specific",
            "not explicitly"
        ]

        if any(phrase in val_lower for phrase in garbage_phrases):
            return ""

        # 4. Если ответ подозрительно длинный (более 60 символов) - это галлюцинация
        if len(val) > 60:
            return ""

        return val

    async def predict_feature(self, product_info: str, feature_name: str, options: list = None) -> str:
        options = options or []
        is_select = len(options) > 0

        try:
            if is_select:
                # SELECT ЛОГИКА
                # Создаем словарь для Enum, очищая ключи от спецсимволов
                enum_values = {f"OPT_{i}": opt for i, opt in enumerate(options)}
                DynamicEnum = Enum('DynamicOptions', enum_values)

                ResponseModel = create_model(
                    'SelectResponse',
                    value=(Optional[DynamicEnum],
                           Field(..., description=f"Select the exact option for '{feature_name}' from the list."))
                )
                sys_prompt = "You are a classification engine. Map product data to one valid option. If not found, return null."

            else:
                # TEXT ЛОГИКА
                class TextResponse(BaseModel):
                    value: str = Field(...,
                                       description="Extract the technical value (e.g. '120Hz', '4.8L'). Return 'NULL' if missing.")

                ResponseModel = TextResponse
                sys_prompt = "You are a data extractor. Extract values strictly. Do not invent. If missing, return 'NULL'."

            # ЗАПРОС
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Product Data: {product_info}\n\nTask: Extract '{feature_name}'"}
                ],
                response_model=ResponseModel,
                max_retries=2,
                temperature=0.0
            )

            # ОБРАБОТКА
            if resp.value is None:
                return ""

            if is_select:
                raw_result = str(resp.value.value)
            else:
                raw_result = str(resp.value)

            # --- ЗАПУСКАЕМ ДВОРНИКА ---
            return self._clean_value(raw_result)

        except Exception as e:
            logging.error(f"Instructor Error: {e}")
            return ""