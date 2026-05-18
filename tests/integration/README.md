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

---

## Real-world acceptance test

`test_real_world_pipeline.py` — прогоняет 10 реальных товаров (iPhone, Nike, Sony, Dyson, etc) с known specs.

### Запуск
```bash
RUN_LIVE_TESTS=1 venv/Scripts/python -m pytest tests/integration/test_real_world_pipeline.py -v -s
```

### Cost
Полный прогон 10 товаров: ~$0.10–0.50 (зависит от того сколько stages запустится).

### Метрики
- Coverage — сколько targets заполнено (≥30% required)
- Accuracy — сколько values совпало с ground truth (с tolerance для numeric)
- Source distribution — какие sources реально вызвались (Description / Knowledge / Vision / WebSearch)
- LLM calls per product

### Что проверяет
- Pipeline работает на разнообразных категориях
- Vision branch активируется когда нужен (no-description + есть image)
- WebSearch branch активируется когда нужен (specs requires inet)
- CostPredictor правильно отсекает no-name товары
