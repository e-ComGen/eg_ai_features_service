import pytest
import asyncio
from tests.integration.conftest import skip_unless_live, MAX_COST_PER_TEST
from app.services.providers.deepseek_provider import DeepSeekProvider
from app.services.providers.openrouter_provider import OpenRouterProvider
from app.services.providers.openai_adapter import OpenAIProviderAdapter
from app.services.providers.serper_client import SerperClient
from app.services.providers.fallback_provider import FallbackProvider


@skip_unless_live
@pytest.mark.asyncio
async def test_deepseek_v4_flash_live():
    """DeepSeek V4 flash — реальный вызов, проверка контент + cost."""
    provider = DeepSeekProvider()
    result = await provider.complete(
        messages=[{"role": "user", "content": "Reply with just the word: ok"}],
        model="deepseek-v4-flash",
        max_tokens=10,
        temperature=0,
    )
    assert result.content.strip().lower().startswith("ok")
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert 0 < result.cost_usd < MAX_COST_PER_TEST
    print(f"\n[DeepSeek v4-flash] tokens: {result.input_tokens}+{result.output_tokens}, cost: ${result.cost_usd:.6f}")


@skip_unless_live
@pytest.mark.asyncio
async def test_openrouter_deepseek_via_openrouter_live():
    """DeepSeek через OpenRouter — проверка маршрутизации работает."""
    provider = OpenRouterProvider()
    result = await provider.complete(
        messages=[{"role": "user", "content": "Reply with: hello"}],
        model="deepseek/deepseek-v4-flash",
        max_tokens=10,
        temperature=0,
    )
    assert "hello" in result.content.lower()
    assert 0 < result.cost_usd < MAX_COST_PER_TEST


@skip_unless_live
@pytest.mark.asyncio
async def test_openrouter_gemini_vision_live():
    """Gemini 2.5 Flash vision — реальная картинка."""
    provider = OpenRouterProvider()
    test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    result = await provider.complete(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's on this image? One sentence."},
                {"type": "image_url", "image_url": {"url": test_image_url}}
            ]
        }],
        model="google/gemini-2.5-flash",
        max_tokens=50,
        temperature=0,
    )
    assert len(result.content) > 5
    assert 0 < result.cost_usd < MAX_COST_PER_TEST


@skip_unless_live
@pytest.mark.asyncio
async def test_openai_fallback_live():
    """OpenAI fallback провайдер — реальный вызов gpt-4o-mini."""
    provider = OpenAIProviderAdapter()
    result = await provider.complete(
        messages=[{"role": "user", "content": "Reply with: ok"}],
        model="gpt-4o-mini",
        max_tokens=10,
        temperature=0,
    )
    assert "ok" in result.content.lower()


@skip_unless_live
@pytest.mark.asyncio
async def test_serper_live_search():
    """Serper search — реальный поиск, проверка что есть organic results."""
    client = SerperClient()
    results = await client.search(query="iPhone 15 Pro характеристики", num_results=3)
    assert len(results.organic) > 0
    assert all(r.title and r.link for r in results.organic[:3])
    print(f"\n[Serper] получено {len(results.organic)} результатов")


@skip_unless_live
@pytest.mark.asyncio
async def test_fallback_provider_real_e2e():
    """FallbackProvider — primary работает, fallback не должен вызываться."""
    primary = DeepSeekProvider()
    fallback = OpenAIProviderAdapter()
    fb = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")

    result = await fb.complete(
        messages=[{"role": "user", "content": "Reply: yes"}],
        model="deepseek-v4-flash",
        max_tokens=10,
        temperature=0,
    )
    assert "yes" in result.content.lower()
    assert "[FALLBACK]" not in result.model  # primary succeeded, no fallback


@skip_unless_live
@pytest.mark.asyncio
async def test_structured_output_live():
    """Structured output через StructuredLlmManager — реальный JSON parse."""
    from app.services.providers.structured_adapter import StructuredLlmManager
    from pydantic import BaseModel

    class Color(BaseModel):
        name: str
        hex: str

    manager = StructuredLlmManager(DeepSeekProvider())
    result = await manager.structured_request(
        prompt="Return color red as JSON",
        response_model=Color,
        model="deepseek-v4-flash",
    )
    assert isinstance(result, Color)
    assert result.name.lower() in ("red", "красный")
    assert result.hex.startswith("#") or result.hex.lower().startswith("ff")
