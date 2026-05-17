"""StructuredLlmManager — wraps any LlmProvider to expose structured_request().

The existing pipeline (AiPipeline, HallucinationJudge) calls:
    result, tokens = await self.llm.structured_request(sys_prompt, user_text, ResponseModel)

OpenAI's SDK has beta.chat.completions.parse() which returns a Pydantic object
directly.  DeepSeek and OpenRouter don't support that endpoint, so we emulate
it with JSON mode + manual Pydantic parse.

This adapter makes any LlmProvider a drop-in replacement for OpenAIManager
from the perspective of the pipeline callers.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .base import LlmProvider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredLlmManager:
    """Wraps an LlmProvider to provide structured_request() compatible with
    the existing AiPipeline / HallucinationJudge interface.

    Usage::

        manager = StructuredLlmManager(provider=DeepSeekProvider(), model="deepseek-v4-flash")
        result, tokens = await manager.structured_request(sys, user, MyPydanticModel)
    """

    def __init__(self, provider: LlmProvider, model: str) -> None:
        self._provider = provider
        self._model = model

    async def structured_request(
        self,
        system_prompt: str,
        user_text: str,
        response_model: Type[T],
    ) -> Tuple[Optional[T], int]:
        """Send a chat completion and parse the response into *response_model*.

        Returns:
            (parsed_object, total_tokens) — mirrors OpenAIManager.structured_request().
            Returns (None, 0) on any error.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        # Build a JSON schema hint so the model knows what fields to produce.
        schema_hint = self._build_schema_hint(response_model)
        if schema_hint:
            messages[0]["content"] = (
                f"{system_prompt}\n\n"
                f"Respond ONLY with a valid JSON object matching this schema:\n{schema_hint}"
            )

        try:
            llm_resp = await self._provider.complete(
                messages=messages,
                model=self._model,
                temperature=0.2,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.error("StructuredLlmManager: LLM call failed: %s", exc)
            return None, 0

        total_tokens = llm_resp.input_tokens + llm_resp.output_tokens

        try:
            data = json.loads(llm_resp.content)
            parsed = response_model.model_validate(data)
            return parsed, total_tokens
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning(
                "StructuredLlmManager: failed to parse response into %s: %s\nRaw: %.300s",
                response_model.__name__,
                exc,
                llm_resp.content,
            )
            return None, total_tokens

    @staticmethod
    def _build_schema_hint(model: Type[BaseModel]) -> str:
        """Return a compact JSON Schema string for the given Pydantic model."""
        try:
            schema = model.model_json_schema()
            # Keep only fields + required for a compact hint
            fields = schema.get("properties", {})
            required = schema.get("required", [])
            lines = ["{"]
            for field_name, field_info in fields.items():
                req_mark = " (required)" if field_name in required else " (optional)"
                field_type = field_info.get("type") or field_info.get("$ref", "any")
                description = field_info.get("description", "")
                lines.append(f'  "{field_name}": <{field_type}>{req_mark}  // {description}')
            lines.append("}")
            return "\n".join(lines)
        except Exception:
            return ""
