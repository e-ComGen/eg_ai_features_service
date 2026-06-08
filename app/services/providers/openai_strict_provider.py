"""OpenAIStrictProvider — structured extraction с token-level enum enforcement.

OpenAI gpt-4.1-mini поддерживает strict json_schema mode через llguidance:
при семплировании токены вне разрешённого enum/Literal просто недостижимы.
Это устраняет retries из-за enum mismatches которые случаются с DeepSeek JSON mode.

Pricing gpt-4.1-mini (май 2026):
    input:  $0.40 / 1M tokens
    output: $1.60 / 1M tokens

Используется ТОЛЬКО для моделей с __has_enum_constraints__=True.
Для остальных случаев дешевле оставаться на DeepSeek.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Цены gpt-4.1-mini (USD / 1M tokens)
_INPUT_PRICE = 0.40
_OUTPUT_PRICE = 1.60


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * _INPUT_PRICE / 1_000_000
        + output_tokens * _OUTPUT_PRICE / 1_000_000
    )


class OpenAIStrictProvider:
    """Вызывает OpenAI gpt-4.1-mini с response_format strict json_schema.

    Метод structured_request() совместим с StructuredLlmManager:
        result, tokens = await provider.structured_request(sys, user, ResponseModel)
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        self._api_key = api_key or config.OPENAI_API_KEY
        self._model = model or config.OPENAI_STRUCTURED_MODEL
        self._client = AsyncOpenAI(api_key=self._api_key)

    async def structured_request(
        self,
        system_prompt: str,
        user_text: str,
        response_model: Type[T],
        timeout: int = 60,
    ) -> Tuple[Optional[T], int]:
        """Strict json_schema запрос к OpenAI с token-level enum enforcement.

        Args:
            timeout: HTTP timeout in seconds for the API call (default 60s).
                     Mirrors the timeout used by DeepSeekProvider / StructuredLlmManager
                     so a stalled API response never hangs the pipeline indefinitely.

        Returns:
            (parsed_instance, total_tokens) — аналогично StructuredLlmManager.
            При ошибке / таймауте — (None, 0).
        """
        schema = response_model.model_json_schema()
        model_name = response_model.__name__

        # Приводим schema к addionalProperties: false для strict режима
        _make_strict_schema(schema)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
                max_tokens=6000,
                timeout=timeout,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": model_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        except Exception as exc:
            logger.error("OpenAIStrictProvider: call failed: %s", exc)
            return None, 0

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = input_tokens + output_tokens
        cost = _estimate_cost(input_tokens, output_tokens)

        # Раздельное логирование для отслеживания расходов на strict-маршрут
        logger.info(
            "OpenAIStrictProvider: model=%s in=%d out=%d cost=$%.5f",
            self._model, input_tokens, output_tokens, cost,
        )

        content = response.choices[0].message.content or ""
        try:
            import json
            data = json.loads(content)
            parsed = response_model.model_validate(data)
            return parsed, total_tokens
        except (ValueError, ValidationError) as exc:
            logger.warning(
                "OpenAIStrictProvider: parse failed for %s: %s\nRaw: %.300s",
                model_name, exc, content,
            )
            return None, total_tokens


def _make_strict_schema(schema: dict) -> None:
    """Рекурсивно добавляет additionalProperties=false и required для strict режима.

    OpenAI strict mode требует:
    - additionalProperties: false на каждом object
    - все properties перечислены в required
    - $defs тоже обработаны рекурсивно
    """
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
        props = schema.get("properties", {})
        if props:
            schema["required"] = list(props.keys())
        for sub in props.values():
            _make_strict_schema(sub)

    # Обрабатываем $defs
    for sub in schema.get("$defs", {}).values():
        _make_strict_schema(sub)

    # anyOf / oneOf / allOf
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []):
            _make_strict_schema(sub)

    # items в array
    if "items" in schema:
        _make_strict_schema(schema["items"])
