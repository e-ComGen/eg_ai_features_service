from typing import Type, TypeVar, Optional, Tuple
from pydantic import BaseModel
from openai import AsyncOpenAI

# Generic Type для Pydantic моделей
T = TypeVar("T", bound=BaseModel)


class OpenAIManager:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        # УБРАЛИ self.router = ... (теперь роутер живет в Pipeline)

    async def structured_request(self,
                                 system_prompt: str,
                                 user_text: str,
                                 response_model: Type[T]) -> Tuple[Optional[T], int]:
        """
        Универсальный метод: отправляет промпт и возвращает Pydantic-объект.
        """
        try:
            completion = await self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                response_format=response_model,
                temperature=0.2,
                seed=42
            )

            usage = completion.usage
            tokens = usage.total_tokens if usage else 0
            parsed = completion.choices[0].message.parsed

            return parsed, tokens

        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return None, 0