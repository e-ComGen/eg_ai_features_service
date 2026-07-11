"""Concurrent WB card.json fetcher for a batch of nm_ids (audit sampling)."""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from app.services.enrichment.sources.wb_card_cdn import fetch_card


async def fetch_cards(nm_ids: list[int], concurrency: int = 8) -> dict[int, Optional[dict]]:
    """Fetch card.json for each nm_id concurrently (bounded by an asyncio.Semaphore(concurrency)),
    using a single shared httpx.AsyncClient(timeout=15.0). Returns a dict mapping
    nm_id -> card dict (or None if fetch_card returned None for that nm_id, meaning
    the card could not be fetched from any CDN basket).
    """
    if not nm_ids:
        return {}

    async with httpx.AsyncClient(timeout=15.0) as client:
        sem = asyncio.Semaphore(concurrency)

        async def _one(nm_id: int) -> tuple[int, Optional[dict]]:
            async with sem:
                card = await fetch_card(client, nm_id)
                return nm_id, card

        results = await asyncio.gather(*[_one(n) for n in nm_ids])
        return {nm_id: card for nm_id, card in results}
