"""Unit tests for WebSearchProducer.

All tests use mocked LLM — no real API calls.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.enrichment.websearch_producer import WebSearchProducer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_manager(output_text: str = "Product specs: weight 500g, colour red."):
    """Return a mock llm_manager with Responses API stub."""
    response = MagicMock()
    response.output_text = output_text

    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=response)

    manager = MagicMock()
    manager.client = client
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_produce_summary_passes_product_name():
    """Product name should appear in the query sent to the Responses API."""
    manager = _make_llm_manager()
    producer = WebSearchProducer(manager, model="gpt-4o")

    await producer.produce_summary("ACME Hammer 5kg", brand="ACME", ean="1234567890123")

    call_args = manager.client.responses.create.call_args
    query_input = call_args.kwargs.get("input") or call_args.args[0]
    assert "ACME Hammer 5kg" in query_input


@pytest.mark.asyncio
async def test_produce_summary_includes_brand_and_ean():
    """Brand and EAN should both appear in the query when provided."""
    manager = _make_llm_manager()
    producer = WebSearchProducer(manager, model="gpt-4o")

    await producer.produce_summary("Widget Pro", brand="BrandX", ean="9991112223334")

    call_args = manager.client.responses.create.call_args
    query_input = call_args.kwargs.get("input") or call_args.args[0]
    assert "BrandX" in query_input
    assert "9991112223334" in query_input


@pytest.mark.asyncio
async def test_produce_summary_returns_text_on_success():
    """Happy path: should return the model's output_text."""
    expected = "Dimensions: 30x20x10cm, Weight: 1.2kg, Colour: black."
    manager = _make_llm_manager(output_text=expected)
    producer = WebSearchProducer(manager, model="gpt-4o")

    result = await producer.produce_summary("Tool Box")

    assert result == expected


@pytest.mark.asyncio
async def test_produce_summary_empty_output_returns_none():
    """Empty model output should yield None."""
    manager = _make_llm_manager(output_text="")
    producer = WebSearchProducer(manager, model="gpt-4o")

    result = await producer.produce_summary("Unknown Product")

    assert result is None


@pytest.mark.asyncio
async def test_produce_summary_no_product_name_returns_none():
    """Empty product_name should return None without an API call."""
    manager = _make_llm_manager()
    producer = WebSearchProducer(manager, model="gpt-4o")

    result = await producer.produce_summary("")

    assert result is None
    manager.client.responses.create.assert_not_called()


@pytest.mark.asyncio
async def test_produce_summary_timeout_returns_none():
    """TimeoutError should be caught and return None."""
    async def _slow(*args, **kwargs):
        await asyncio.sleep(999)

    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = _slow

    manager = MagicMock()
    manager.client = client
    producer = WebSearchProducer(manager, model="gpt-4o")

    result = await producer.produce_summary(
        "Widget", timeout=0  # immediate timeout
    )

    assert result is None


@pytest.mark.asyncio
async def test_produce_summary_api_error_returns_none():
    """Any exception from the Responses API should be caught and return None."""
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(side_effect=ConnectionError("network down"))

    manager = MagicMock()
    manager.client = client
    producer = WebSearchProducer(manager, model="gpt-4o")

    result = await producer.produce_summary("Widget")

    assert result is None


@pytest.mark.asyncio
async def test_produce_summary_uses_web_search_tool():
    """The Responses API call must include the web_search tool."""
    manager = _make_llm_manager()
    producer = WebSearchProducer(manager, model="gpt-4o")

    await producer.produce_summary("Test Product")

    call_kwargs = manager.client.responses.create.call_args.kwargs
    tools = call_kwargs.get("tools", [])
    assert any(t.get("type") == "web_search" for t in tools)
