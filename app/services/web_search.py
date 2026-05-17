import asyncio
from dataclasses import dataclass, field
from typing import Optional
from openai import AsyncOpenAI


@dataclass
class Citation:
    url: str
    title: str = ""


@dataclass
class WebSearchResult:
    """Raw web research output. Pipeline turns this into a structured value downstream."""
    answer_text: str = ""
    citations: list[Citation] = field(default_factory=list)
    tokens: int = 0
    web_search_used: bool = False
    error: Optional[str] = None


class WebSearchService:
    """
    Wraps OpenAI Responses API with the `web_search` tool.

    Single responsibility: take a focused question about a product feature,
    return a natural-language answer plus source URLs. Does NOT structure
    the answer — that's the pipeline's structured re-extract step.

    Concurrency is bounded by its own semaphore so heavy DEEP-mode batches
    do not starve the global LLM pool or hit OpenAI rate limits.
    """

    def __init__(self, client: AsyncOpenAI, model: str = "gpt-4o", max_concurrent: int = 10):
        self.client = client
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def search_product_feature(
        self,
        product_text: str,
        feature_name: str,
        suffix: str = "",
    ) -> WebSearchResult:
        snippet = product_text.replace("Title: ", "").replace("Description: ", "").strip()
        snippet = snippet[:400]

        unit_part = f" (expressed in {suffix})" if suffix else ""
        query = (
            f"Find the official '{feature_name}'{unit_part} for this exact product.\n\n"
            f"PRODUCT:\n{snippet}\n\n"
            f"Search the manufacturer's website or trusted retailers. "
            f"Return the exact value with its source URL. "
            f"If you cannot find an authoritative answer, say so explicitly — "
            f"do NOT guess from similar products."
        )

        async with self.semaphore:
            try:
                response = await self.client.responses.create(
                    model=self.model,
                    input=query,
                    tools=[{"type": "web_search"}],
                )
            except Exception as e:
                return WebSearchResult(error=f"WebSearch API error: {e}")

        return self._parse_response(response)

    def _parse_response(self, response) -> WebSearchResult:
        answer_text = (getattr(response, "output_text", "") or "").strip()
        citations: list[Citation] = []
        web_search_used = False

        try:
            for item in (getattr(response, "output", None) or []):
                item_type = str(getattr(item, "type", "") or "")
                if "web_search" in item_type:
                    web_search_used = True

                for block in (getattr(item, "content", None) or []):
                    for ann in (getattr(block, "annotations", None) or []):
                        if getattr(ann, "type", "") == "url_citation":
                            url = getattr(ann, "url", "") or ""
                            if url:
                                citations.append(Citation(
                                    url=url,
                                    title=getattr(ann, "title", "") or ""
                                ))
        except Exception:
            pass

        tokens = 0
        try:
            usage = getattr(response, "usage", None)
            if usage:
                tokens = getattr(usage, "total_tokens", 0) or 0
        except Exception:
            pass

        if answer_text and citations and not web_search_used:
            web_search_used = True

        seen = set()
        unique_citations = []
        for c in citations:
            if c.url not in seen:
                seen.add(c.url)
                unique_citations.append(c)

        return WebSearchResult(
            answer_text=answer_text,
            citations=unique_citations,
            tokens=tokens,
            web_search_used=web_search_used,
            error=None,
        )
