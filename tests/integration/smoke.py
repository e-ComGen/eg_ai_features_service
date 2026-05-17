"""Quick smoke test — запусти 'python -m tests.integration.smoke' чтобы проверить все провайдеры за раз."""
import asyncio
import os
from app.services.providers.deepseek_provider import DeepSeekProvider
from app.services.providers.openrouter_provider import OpenRouterProvider
from app.services.providers.serper_client import SerperClient
from app.services.providers.openai_adapter import OpenAIProviderAdapter
from app.services.llm_manager import OpenAIManager
from app import config


async def main():
    print("=== Smoke test: 4 провайдера ===\n")

    # DeepSeek
    try:
        ds = DeepSeekProvider()
        r = await ds.complete(messages=[{"role": "user", "content": "Say the word: hello"}], model="deepseek-chat", max_tokens=20)
        print(f"[OK] DeepSeek V4 flash: '{r.content.strip()}' (cost ${r.cost_usd:.6f})")
    except Exception as e:
        print(f"[FAIL] DeepSeek: {e}")

    # OpenRouter
    try:
        orp = OpenRouterProvider()
        r = await orp.complete(messages=[{"role": "user", "content": "Say the word: hello"}], model="deepseek/deepseek-chat", max_tokens=20)
        print(f"[OK] OpenRouter (DeepSeek): '{r.content.strip()}'")
    except Exception as e:
        print(f"[FAIL] OpenRouter: {e}")

    # Serper
    try:
        sc = SerperClient()
        res = await sc.search(query="test query", num_results=1)
        print(f"[OK] Serper: {len(res.organic_results)} результатов")
    except Exception as e:
        print(f"[FAIL] Serper: {e}")

    # OpenAI fallback
    try:
        oa = OpenAIProviderAdapter(manager=OpenAIManager(api_key=config.OPENAI_API_KEY))
        r = await oa.complete(messages=[{"role": "user", "content": "Say the word: hello"}], model="gpt-4o-mini", max_tokens=20)
        print(f"[OK] OpenAI gpt-4o-mini: '{r.content.strip()}'")
    except Exception as e:
        print(f"[FAIL] OpenAI: {e}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
