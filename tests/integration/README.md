# Integration tests (live API calls)

⚠️ Эти тесты ходят в реальные API DeepSeek/OpenRouter/Serper/OpenAI и тратят токены.

## Как запустить

### Все pytest integration тесты
```bash
RUN_LIVE_TESTS=1 pytest tests/integration/ -v -s
```

### Только провайдеры (быстрая проверка ключей)
```bash
RUN_LIVE_TESTS=1 pytest tests/integration/test_live_providers.py -v -s
```

### Quick smoke check (без pytest, всё в одном)
```bash
python -m tests.integration.smoke
```

## Стоимость

Один прогон `test_live_providers.py` — около $0.001-0.005.
Один прогон `test_full_pipeline_one_product_e2e` — около $0.01-0.05.

## Что проверяется

1. API ключи валидны
2. Форматы запросов корректны для каждого провайдера
3. Pydantic schema parsing работает на реальных responses
4. Vision принимает image URLs (Gemini через OpenRouter)
5. Fallback chain работает (primary success → no fallback triggered)
