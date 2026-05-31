"""GeminiPdfProvider — Gemini 2.5 Flash с native PDF input + structured output.

OpenRouter принимает контент-парт `{"type": "file", "file": {"filename", "file_data"}}`
где file_data — data-URL `data:application/pdf;base64,...`. Для Gemini models это
маршрутизируется в native PDF understanding (без OCR-прослойки).

Используется только PdfDatasheetSource: short-lived bytes → spec dict → AttributeValue.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Optional, Tuple, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from app import config

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class GeminiPdfProvider:
    """Отправляет PDF-байты в Gemini 2.5 Flash и парсит structured JSON-ответ."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        site_url: str = "https://github.com/e-comgen/cpAiFeatures",
        app_name: str = "e-comgen AI attributes",
    ) -> None:
        self._api_key = api_key or config.OPENROUTER_API_KEY
        if not self._api_key:
            raise ValueError(
                "GeminiPdfProvider: OPEN_ROUTER_API_KEY не задан в .env"
            )
        self._model = model or config.VISION_MODEL  # google/gemini-2.5-flash
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=_OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
        )

    async def extract_from_pdf(
        self,
        pdf_bytes: bytes,
        system_prompt: str,
        user_text: str,
        response_model: Type[T],
        filename: str = "datasheet.pdf",
        timeout: int = 60,
    ) -> Tuple[Optional[T], int]:
        """Послать PDF + промпт в Gemini, вернуть (parsed_instance, total_tokens)."""
        data_url = (
            "data:application/pdf;base64,"
            + base64.b64encode(pdf_bytes).decode("ascii")
        )

        schema_hint = self._build_schema_hint(response_model)
        sys_with_schema = (
            f"{system_prompt}\n\n"
            f"Respond ONLY with a valid JSON object matching this schema:\n{schema_hint}"
        )

        messages = [
            {"role": "system", "content": sys_with_schema},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": data_url,
                        },
                    },
                ],
            },
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
                timeout=timeout,
            )
        except Exception as exc:
            logger.error("GeminiPdfProvider: call failed: %s", exc)
            return None, 0

        usage = response.usage
        total_tokens = (
            (usage.prompt_tokens if usage else 0)
            + (usage.completion_tokens if usage else 0)
        )

        content = response.choices[0].message.content or ""
        try:
            data = json.loads(content)
            parsed = response_model.model_validate(data)
            return parsed, total_tokens
        except (ValueError, ValidationError) as exc:
            logger.warning(
                "GeminiPdfProvider: parse failed for %s: %s\nRaw: %.300s",
                response_model.__name__, exc, content,
            )
            return None, total_tokens

    @staticmethod
    def _build_schema_hint(model: Type[BaseModel]) -> str:
        try:
            schema = model.model_json_schema()
            fields = schema.get("properties", {})
            required = schema.get("required", [])
            lines = ["{"]
            for field_name, field_info in fields.items():
                req_mark = " (required)" if field_name in required else " (optional)"
                field_type = field_info.get("type") or field_info.get("$ref", "any")
                description = field_info.get("description", "")
                lines.append(
                    f'  "{field_name}": <{field_type}>{req_mark}  // {description}'
                )
            lines.append("}")
            return "\n".join(lines)
        except Exception:
            return ""
