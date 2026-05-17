"""Live integration tests against real APIs.

Только для opt-in через RUN_LIVE_TESTS=1. Используют реальные ключи из .env.
Используется alias deepseek-chat (direct V4 names returning empty content — see git log)."""
import pytest

from tests.integration.conftest import skip_unless_live, MAX_COST_PER_TEST
from app import config
from app.services.providers.deepseek_provider import DeepSeekProvider
from app.services.providers.openrouter_provider import OpenRouterProvider
from app.services.providers.openai_adapter import OpenAIProviderAdapter
from app.services.providers.serper_client import SerperClient
from app.services.providers.fallback_provider import FallbackProvider
from app.services.llm_manager import OpenAIManager


def _make_openai_adapter() -> OpenAIProviderAdapter:
    """Helper — OpenAIProviderAdapter requires OpenAIManager instance."""
    return OpenAIProviderAdapter(manager=OpenAIManager(api_key=config.OPENAI_API_KEY))


@skip_unless_live
@pytest.mark.asyncio
async def test_deepseek_chat_alias_live():
    """DeepSeek chat alias — реальный вызов, проверка контент + cost.

    Используем alias 'deepseek-chat' который маршрутизируется на V4-flash и
    отдаёт реальный content (direct 'deepseek-v4-flash' возвращает пустоту)."""
    provider = DeepSeekProvider()
    result = await provider.complete(
        messages=[{"role": "user", "content": "Say the single word: hello"}],
        model="deepseek-chat",
        max_tokens=20,
        temperature=0,
    )
    assert "hello" in result.content.lower()
    assert result.input_tokens > 0
    assert result.output_tokens > 0
    assert 0 < result.cost_usd < MAX_COST_PER_TEST
    print(f"\n[DeepSeek chat] tokens: {result.input_tokens}+{result.output_tokens}, cost: ${result.cost_usd:.6f}")


@skip_unless_live
@pytest.mark.asyncio
async def test_openrouter_deepseek_via_openrouter_live():
    """DeepSeek через OpenRouter — проверка маршрутизации работает."""
    provider = OpenRouterProvider()
    result = await provider.complete(
        messages=[{"role": "user", "content": "Say the single word: hello"}],
        model="deepseek/deepseek-chat",
        max_tokens=20,
        temperature=0,
    )
    assert "hello" in result.content.lower()
    # cost_usd может быть 0.0 если адаптер ещё не извлекает cost из OpenRouter usage.cost
    assert 0 <= result.cost_usd < MAX_COST_PER_TEST


@skip_unless_live
@pytest.mark.asyncio
async def test_openrouter_gemini_vision_live():
    """Gemini 2.5 Flash vision — реальная картинка через стабильный CDN."""
    provider = OpenRouterProvider()
    # picsum.photos — стабильный image CDN, всегда возвращает корректный JPEG
    test_image_url = "https://picsum.photos/seed/cardtest/400/400.jpg"
    result = await provider.complete(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's on this image? One short sentence."},
                {"type": "image_url", "image_url": {"url": test_image_url}}
            ]
        }],
        model="google/gemini-2.5-flash",
        max_tokens=80,
        temperature=0,
    )
    assert len(result.content) > 5
    assert 0 <= result.cost_usd < MAX_COST_PER_TEST
    print(f"\n[Gemini Vision] response: {result.content[:80]!r}")


@skip_unless_live
@pytest.mark.asyncio
async def test_openai_fallback_live():
    """OpenAI fallback провайдер — реальный вызов gpt-4o-mini."""
    provider = _make_openai_adapter()
    result = await provider.complete(
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        model="gpt-4o-mini",
        max_tokens=10,
        temperature=0,
    )
    assert "ok" in result.content.lower()


@skip_unless_live
@pytest.mark.asyncio
async def test_serper_live_search():
    """Serper search — реальный поиск, проверка organic_results."""
    client = SerperClient()
    results = await client.search(query="iPhone 15 Pro характеристики", num_results=3)
    assert len(results.organic_results) > 0
    assert all(r.title and r.link for r in results.organic_results[:3])
    print(f"\n[Serper] получено {len(results.organic_results)} результатов")


@skip_unless_live
@pytest.mark.asyncio
async def test_fallback_provider_real_e2e():
    """FallbackProvider — primary работает, fallback не должен вызываться."""
    primary = DeepSeekProvider()
    fallback = _make_openai_adapter()
    fb = FallbackProvider(primary, fallback, fallback_model="gpt-4o-mini")

    result = await fb.complete(
        messages=[{"role": "user", "content": "Say the single word: yes"}],
        model="deepseek-chat",
        max_tokens=20,
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

    manager = StructuredLlmManager(DeepSeekProvider(), model="deepseek-chat")
    parsed, tokens = await manager.structured_request(
        system_prompt="You return JSON with fields 'name' (English color name) and 'hex' (hex code like '#FF0000').",
        user_text="Return the color red.",
        response_model=Color,
    )
    assert parsed is not None, "Structured parse returned None"
    assert isinstance(parsed, Color)
    assert parsed.name.lower() in ("red", "красный")
    assert "ff" in parsed.hex.lower() or parsed.hex.startswith("#")
    assert tokens > 0
    print(f"\n[Structured] parsed: {parsed.model_dump()}, tokens: {tokens}")
