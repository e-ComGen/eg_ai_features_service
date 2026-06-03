"""Serper API client for Google web search.

This is NOT an LlmProvider — it's a dedicated search API client.
Serper returns Google SERP data (organic results, knowledge graph, etc.)
without an LLM call, making it cheap and fast for web research.

Pricing: ~$0.001 per search query (10 000 free queries / month on free tier).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

import httpx
from pydantic import BaseModel

from app import config

logger = logging.getLogger(__name__)

_SERPER_ENDPOINT = "https://google.serper.dev/search"

# Transport-level retry tuning. Keep the total backoff budget short so it fits
# inside the external asyncio.wait_for(timeout=...) used by websearch_producer.
_MAX_ATTEMPTS = 3          # 1 initial try + 2 retries
_BASE_BACKOFF = 0.5        # seconds; grows exponentially: 0.5, 1.0, 2.0...
_MAX_BACKOFF = 4.0         # cap a single sleep (e.g. honouring Retry-After)

# httpx transport exceptions worth retrying (network flake, not a config issue).
_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.TransportError,  # base class — covers other transport-level flakes
)


def _is_retryable_status(status_code: int) -> bool:
    """Retry on 429 (rate limit) and 5xx (transient server errors) only.

    Other 4xx (401/400/403/404...) are config/key/request problems — a retry
    will not help, so we surface them immediately.
    """
    return status_code == 429 or 500 <= status_code < 600


def _parse_retry_after(response: httpx.Response) -> Optional[float]:
    """Extract a Retry-After delay (in seconds) from a 429 response, if present."""
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        # Serper returns a plain integer seconds value; ignore HTTP-date form.
        return max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        return None


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

        Transport-level failures (connection/read timeouts, dropped sockets,
        HTTP 429, HTTP 5xx) are retried up to ``_MAX_ATTEMPTS`` with exponential
        backoff so a transient network flake on Serper does not wipe out a whole
        web_search pass. Non-transport HTTP errors (401/400/403...) are NOT
        retried — they indicate a key/config/request problem a retry can't fix.

        Raises:
            httpx.HTTPStatusError: On non-2xx response (after retries for 429/5xx).
            httpx.TimeoutException: On network timeout (after retries are exhausted).
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

        data = await self._request_with_retry(payload, headers, timeout)

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

    async def _request_with_retry(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: int,
    ) -> dict[str, Any]:
        """POST to Serper, retrying transport-level failures with backoff.

        ``timeout`` is applied per-attempt (a fresh httpx client per try), so a
        single hung attempt cannot consume the whole retry budget.

        Returns the parsed JSON body on success. Re-raises the last exception
        once all attempts are exhausted, preserving the prior raise behaviour.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(1, _MAX_ATTEMPTS + 1):
            retry_after: Optional[float] = None
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        _SERPER_ENDPOINT,
                        json=payload,
                        headers=headers,
                    )
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        status = exc.response.status_code
                        if status == 429:
                            logger.error(
                                "Serper quota/rate limit (429). "
                                "Check your plan limits."
                            )
                        else:
                            logger.error(
                                "Serper API error %s: %s",
                                status,
                                exc.response.text[:200],
                            )
                        # Non-transport HTTP errors (401/400/403...) are not
                        # worth retrying — surface immediately.
                        if not _is_retryable_status(status):
                            raise
                        last_exc = exc
                        retry_after = _parse_retry_after(exc.response)
                    else:
                        return response.json()
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                last_exc = exc
                logger.warning(
                    "Serper transport error on attempt %d/%d: %s: %s",
                    attempt,
                    _MAX_ATTEMPTS,
                    type(exc).__name__,
                    exc,
                )

            # We only reach here on a retryable failure (transport / 429 / 5xx).
            if attempt >= _MAX_ATTEMPTS:
                break

            backoff = min(_BASE_BACKOFF * (2 ** (attempt - 1)), _MAX_BACKOFF)
            if retry_after is not None:
                backoff = min(max(backoff, retry_after), _MAX_BACKOFF)
            logger.info(
                "Retrying Serper request in %.1fs (attempt %d/%d).",
                backoff,
                attempt + 1,
                _MAX_ATTEMPTS,
            )
            await asyncio.sleep(backoff)

        # All attempts exhausted — re-raise the last failure (graceful for the
        # caller, which already treats exceptions as "source returned None").
        if last_exc is not None:
            raise last_exc
        # Defensive: should be unreachable (loop either returns or sets last_exc).
        raise RuntimeError("Serper request failed without a captured exception.")
