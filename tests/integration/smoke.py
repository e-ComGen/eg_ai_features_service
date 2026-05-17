"""Quick smoke test — запусти 'python -m tests.integration.smoke' чтобы проверить все провайдеры за раз."""
import asyncio
import os
from app.services.providers.deepseek_provider import DeepSeekProvider
from app.services.providers.openrouter_provider import OpenRouterProvider
from app.services.providers.serper_client import SerperClient
from app.services.providers.openai_adapter import OpenAIProviderAdapter


async def main():
    print("=== Smoke test: 4 провайдера ===\n")

    # DeepSeek
    try:
        ds = DeepSeekProvider()
        r = await ds.complete(messages=[{"role": "user", "content": "Reply: ok"}], model="deepseek-v4-flash", max_tokens=5)
        print(f"✓ DeepSeek V4 flash: '{r.content.strip()}' (cost ${r.cost_usd:.6f})")
    except Exception as e:
        print(f"✗ DeepSeek: {e}")

    # OpenRouter
    try:
        orp = OpenRouterProvider()
        r = await orp.complete(messages=[{"role": "user", "content": "Reply: ok"}], model="deepseek/deepseek-v4-flash", max_tokens=5)
        print(f"✓ OpenRouter (DeepSeek): '{r.content.strip()}'")
    except Exception as e:
        print(f"✗ OpenRouter: {e}")

    # Serper
    try:
        sc = SerperClient()
        res = await sc.search(query="test query", num_results=1)
        print(f"✓ Serper: {len(res.organic)} результатов")
    except Exception as e:
        print(f"✗ Serper: {e}")

    # OpenAI fallback
    try:
        oa = OpenAIProviderAdapter()
        r = await oa.complete(messages=[{"role": "user", "content": "Reply: ok"}], model="gpt-4o-mini", max_tokens=5)
        print(f"✓ OpenAI gpt-4o-mini: '{r.content.strip()}'")
    except Exception as e:
        print(f"✗ OpenAI: {e}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
