"""Unit tests for VisionProducer.

All tests use mocked LLM — no real API calls.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.enrichment.vision_producer import VisionProducer, MAX_IMAGES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_manager(response_text: str = "Red cylindrical product, ~20cm tall."):
    """Return a mock llm_manager whose .client behaves like AsyncOpenAI."""
    choice = MagicMock()
    choice.message.content = response_text

    completion = MagicMock()
    completion.choices = [choice]

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)

    manager = MagicMock()
    manager.client = client
    return manager


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_produce_description_passes_correct_number_of_images():
    """Should cap image list at MAX_IMAGES when more are provided."""
    manager = _make_llm_manager()
    producer = VisionProducer(manager, model="gpt-4o")

    many_urls = [f"https://example.com/img{i}.jpg" for i in range(10)]
    result = await producer.produce_description(many_urls, product_name="Widget")

    assert result is not None
    # Inspect the call to verify only MAX_IMAGES image_url blocks were sent.
    call_args = manager.client.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    user_message = next(m for m in messages if m["role"] == "user")
    image_blocks = [c for c in user_message["content"] if c.get("type") == "image_url"]
    assert len(image_blocks) == MAX_IMAGES


@pytest.mark.asyncio
async def test_produce_description_exact_max_images():
    """When exactly MAX_IMAGES URLs are provided, all should be included."""
    manager = _make_llm_manager()
    producer = VisionProducer(manager, model="gpt-4o")

    urls = [f"https://example.com/img{i}.jpg" for i in range(MAX_IMAGES)]
    result = await producer.produce_description(urls, product_name="Gadget")

    assert result is not None
    call_args = manager.client.chat.completions.create.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    user_message = next(m for m in messages if m["role"] == "user")
    image_blocks = [c for c in user_message["content"] if c.get("type") == "image_url"]
    assert len(image_blocks) == MAX_IMAGES


@pytest.mark.asyncio
async def test_produce_description_empty_list_returns_none():
    """Empty image list should return None without any LLM call."""
    manager = _make_llm_manager()
    producer = VisionProducer(manager, model="gpt-4o")

    result = await producer.produce_description([], product_name="Widget")

    assert result is None
    manager.client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_produce_description_returns_text_on_success():
    """Happy path: should return the LLM's text response."""
    expected = "Blue metallic casing, rectangular shape, approx 15x10cm."
    manager = _make_llm_manager(response_text=expected)
    producer = VisionProducer(manager, model="gpt-4o")

    result = await producer.produce_description(
        ["https://example.com/a.jpg"], product_name="Box"
    )

    assert result == expected


@pytest.mark.asyncio
async def test_produce_description_empty_response_returns_none():
    """When model returns empty string, produce_description should return None."""
    manager = _make_llm_manager(response_text="")
    producer = VisionProducer(manager, model="gpt-4o")

    result = await producer.produce_description(
        ["https://example.com/a.jpg"], product_name="Box"
    )

    assert result is None


@pytest.mark.asyncio
async def test_produce_description_timeout_returns_none():
    """A TimeoutError from asyncio.wait_for should be caught and return None."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    # Simulate timeout by sleeping longer than timeout param.
    async def _slow(*args, **kwargs):
        await asyncio.sleep(999)
    client.chat.completions.create = _slow

    manager = MagicMock()
    manager.client = client
    producer = VisionProducer(manager, model="gpt-4o")

    result = await producer.produce_description(
        ["https://example.com/a.jpg"],
        product_name="Widget",
        timeout=0,  # immediate timeout
    )

    assert result is None


@pytest.mark.asyncio
async def test_produce_description_api_error_returns_none():
    """Any exception from the LLM client should be caught and return None."""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("API down"))

    manager = MagicMock()
    manager.client = client
    producer = VisionProducer(manager, model="gpt-4o")

    result = await producer.produce_description(
        ["https://example.com/a.jpg"], product_name="Widget"
    )

    assert result is None
