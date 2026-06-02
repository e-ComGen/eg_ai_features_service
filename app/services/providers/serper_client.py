"""Serper API client for Google web search.

This is NOT an LlmProvider — it's a dedicated search API client.
Serper returns Google SERP data (organic results, knowledge graph, etc.)
without an LLM call, making it cheap and fast for web research.

Pricing: ~$0.001 per search query (10 000 free queries / month on free tier).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from app import config

logger = logging.getLogger(__name__)

_SERPER_ENDPOINT = "https://google.serper.dev/search"


class OrganicResult(BaseModel):
    title: str
    link: str
    snippet: str = ""  # Serper иногда не возвращает snippet → не падать ValidationError
    position: int


class SerperResults(BaseModel):
    query: str
    organic_results: list[OrganicResult]
    knowledge_graph: Optional[dict[str, Any]] = None
    related_searches: list[str] = []


class SerperClient:
    """Async client for the Serper Google Search API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        gl: str = "ru",   # country code for localised results
        hl: str = "ru",   # interface language
    ) -> None:
        self._api_key = api_key or config.SERPER_API_KEY
        if not self._api_key:
            raise ValueError(
                "Serper API key is not set. "
                "Set SERPER_API_KEY in your .env file."
            )
        self._gl = gl
        self._hl = hl

    async def search(
        self,
        query: str,
        num_results: int = 5,
        timeout: int = 15,
    ) -> SerperResults:
        """Perform a Google search via Serper and return structured results.

        Args:
            query: Search query string.
            num_results: Number of organic results to request (max 100).
            timeout: HTTP timeout in seconds.

        Returns:
            SerperResults with organic_results, knowledge_graph, related_searches.

        Raises:
            httpx.HTTPStatusError: On non-2xx response (e.g. 429 quota exceeded).
            httpx.TimeoutException: On network timeout.
        """
        payload = {
            "q": query,
            "num": num_results,
            "gl": self._gl,
            "hl": self._hl,
        }
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                _SERPER_ENDPOINT,
                json=payload,
                headers=headers,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    logger.error(
                        "Serper quota exceeded (429). Check your plan limits."
                    )
                else:
                    logger.error(
                        "Serper API error %s: %s",
                        exc.response.status_code,
                        exc.response.text[:200],
                    )
                raise

        data = response.json()

        organic: list[OrganicResult] = []
        for item in data.get("organic", []):
            organic.append(
                OrganicResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    position=item.get("position", 0),
                )
            )

        related: list[str] = [
            r.get("query", "")
            for r in data.get("relatedSearches", [])
            if r.get("query")
        ]

        return SerperResults(
            query=query,
            organic_results=organic,
            knowledge_graph=data.get("knowledgeGraph"),
            related_searches=related,
        )
