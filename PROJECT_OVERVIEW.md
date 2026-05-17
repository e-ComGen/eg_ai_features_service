# CpAiFeatures — AI Worker для извлечения характеристик товаров

## Что это за проект

`CpAiFeatures` — это асинхронный микросервис на **FastAPI**, выступающий в роли AI-воркера. Его задача: получать «сырое» описание товара (название + текстовое описание) и автоматически заполнять структурированные характеристики (Brand, Material, Width, Capacity, Color, и т.д.) с помощью LLM (`OpenAI GPT-4o-mini`).

Сервис устроен как **внутренний worker**, который вызывается оркестратором (по архитектуре — внешний PHP-сервис) через защищённый эндпоинт `POST /process-batch`. Запросы аутентифицируются по секретному заголовку `X-Internal-Secret`.

---

## Ключевые идеи и архитектура

### 1. Decision Tree Router (умная маршрутизация)
Вместо одного «универсального» промпта система строит **дерево стратегий извлечения** (`app/strategies/`). LLM сначала выступает роутером: для каждой характеристики она проходит по дереву от корня (`RootStrategy`) к конкретному «листу» — узкоспециализированному извлекателю.

```
RootStrategy
├── NumericBranch (числа, размеры, объёмы, углы)
│   ├── RadialDimensionLeaf       — диаметры/диагонали
│   ├── AngularDimensionLeaf      — углы, FOV
│   ├── SurfaceAreaLeaf           — площадь
│   ├── BiometricDimensionLeaf    — анатомия/одежда
│   ├── CompositeAggregationLeaf  — сумма по бандлу
│   ├── CapacityLeaf              — объём/ёмкость
│   ├── PhysicalMagnitudeLeaf     — мощность, вес, напряжение
│   ├── QuantificationLeaf        — счётные количества
│   └── Linear Dimensions (Mixin):
│       ├── SubComponentDimensionLeaf
│       ├── UprightStructureDimensionLeaf
│       ├── FlatPanelDimensionLeaf
│       ├── LongExtrusionDimensionLeaf
│       ├── SoftContainerDimensionLeaf
│       └── DefaultBoxDimensionLeaf
└── TextBranch (строки, бренды, материалы)
    ├── ConfigurationLeaf  — режимы, протоколы, стандарты
    ├── BrandLeaf          — бренд/производитель
    ├── MaterialLeaf       — материал
    ├── ClassificationLeaf — таксономия, технология
    └── ContextFilterLeaf  — валидатор зависимости
```

Каждый узел собирает промпт **снизу вверх** — System Constraints + Domain Rules + Task Logic + Dynamic Constraints. Это даёт максимально специализированный промпт под конкретный тип данных.

### 2. Структурированные ответы (Pydantic + Structured Outputs)
LLM вызывается через `client.beta.chat.completions.parse()` с Pydantic-схемой ответа. Каждая ветка дерева возвращает свою модель:
- `WorkerResult` — обычный текст/число (analysis + extracted_value + confidence)
- `MultiLangWorkerResult` — массив переводов на нужные языки
- `OptionWorkerResult` — выбор из словаря допустимых значений
- `DimensionalWorkerResult` — для габаритов (с обязательной сортировкой чисел и mapping rules)

### 3. Smart Deduction (умная дедукция)
Если первое извлечение вернуло `None`, и узел поддерживает `supports_deduction = True`, запускается «исследователь контекста» (`DeductionResult`). Он ищет в тексте косвенные намёки/синонимы/индустриальные стандарты и оценивает уверенность 0–100. При уверенности ≥ 75 — повторный запрос с обогащённым контекстом.

### 4. Hallucination Judge (Судья галлюцинаций)
`app/judge/judge.py` + `app/judge/judge_profile.py`. После извлечения значение проходит трёхступенчатый фильтр:
- **ACCEPT** — 100% надёжно (например, число дословно из текста, или выбор из словаря)
- **REJECT** — 100% мусор (например, числа нет в тексте → `NumericValidator.is_value_in_text`)
- **JUDGE** — сомнительно, вызываем второй LLM-запрос с профилем «независимого аудитора»

У каждого листа свой `JudgeProfile` со специальными правилами (UNIT_INTEGRITY, CAPACITY_SCOPE, MATERIAL_SCOPE и т.д.) и порогами `error_confidence` для отклонения вердикта.

### 5. Многоуровневый кеш
- **DatabaseCacheManager** (`db_cache.py`, SQLite + SQLAlchemy async) — основной кеш. Ключ: `client_id + product_id + feature_name`. Инвалидация по MD5-хешу нормализованного описания (HTML stripped, units canonicalized, words sorted).
- **CacheManager** (`cache_manager.py`) — простой JSON-кеш (legacy/альтернатива).
- **Vector cache** (`vectors.pkl`) — pickled-кеш эмбеддингов в `MatcherService`.

### 6. Fuzzy/Vector Matcher для select-полей
`MatcherService` (`matcher.py`) — для характеристик с фиксированным словарём допустимых значений:
1. Точное сравнение (без пробелов/дефисов)
2. Левенштейн (`rapidfuzz.fuzz.ratio`, порог 85–90%)
3. Семантика (`sentence-transformers all-MiniLM-L6-v2`, cosine ≥ 0.60, GPU если есть)

### 7. Параллельная обработка
- `asyncio.Semaphore(50)` ограничивает глобальную конкурентность LLM-запросов
- `asyncio.gather` запускает все продукты пакета и все характеристики каждого продукта параллельно
- SQLAlchemy async-engine с `pool_size=0`, `max_overflow=-1`, WAL-режимом SQLite — для скорости

---

## Структура проекта

```
CpAiFeatures/
├── run.py                      # Запуск uvicorn (порт 8001)
├── requirements.txt            # FastAPI, OpenAI, sentence-transformers, rapidfuzz...
├── app/
│   ├── main.py                 # FastAPI app + endpoint /process-batch + CSV-логирование
│   ├── config.py               # OPENAI_API_KEY, INTERNAL_SERVICE_SECRET
│   ├── security.py             # verify_internal_token (заголовок X-Internal-Secret)
│   ├── database.py             # SQLAlchemy async engine + RequestLog модель
│   ├── models.py               # Pydantic: BatchPayload, ProductData, FeatureOption
│   ├── prompts.py              # Старые статические промпты (legacy)
│   ├── services/
│   │   ├── ai_pipeline.py      # AiFeaturePipeline — главная оркестрация: Router→Extract→Deduce→Judge
│   │   ├── tree_router.py      # TreeRouter — обход дерева стратегий через LLM
│   │   ├── job_processor.py    # JobProcessor — обработка одного товара (cache, schema, matcher)
│   │   ├── llm_manager.py      # OpenAIManager — обёртка над AsyncOpenAI
│   │   ├── ollama_manager.py   # Альтернативный manager под локальную Ollama (llama3.1)
│   │   ├── db_cache.py         # DatabaseCacheManager — SQLite-кеш с хешированием описания
│   │   ├── cache_manager.py    # Простой JSON-кеш (legacy)
│   │   └── matcher.py          # MatcherService — fuzzy + vector matching по словарю
│   ├── strategies/
│   │   ├── base.py             # BaseStrategyNode + WorkerResult + DeductionResult
│   │   ├── dituction.py        # (битый файл — отдельный DeductionResult без импортов)
│   │   ├── definitions/
│   │   │   ├── root.py
│   │   │   ├── numeric.py      # NumericBranch + 8 числовых листьев
│   │   │   ├── linear_dimensions.py # LinearLogicMixin + 6 геометрических листьев
│   │   │   ├── text.py         # TextBranch + 5 текстовых листьев
│   │   │   └── angular.py      # (пустой)
│   │   └── validators/
│   │       └── numeric_validator.py # NumericValidator — проверка числа в тексте (с дробями)
│   └── judge/
│       ├── judge.py            # HallucinationJudge — выполняет аудит
│       └── judge_profile.py    # JudgeProfile + BaseJudgeVerdict + JudgeResult
├── verify_*.py                 # Тестовые скрипты для проверки кеша/CSV/переименований
├── reproduce_strategy.py       # Дев-скрипт: проверка генерации инструкций
├── features_service.db         # SQLite кеш (с WAL/SHM файлами)
├── saas.db                     # отдельная SQLite база
├── vectors.pkl                 # pickled-кеш эмбеддингов
└── *.csv                       # Логи: ai_service_output, deductions_log, tree_traversal_log, ai_router_logic
```

---

## API

### `POST /process-batch`

**Headers:** `X-Internal-Secret: <secret>`

**Body** (`BatchPayload`):
```json
{
  "client_id": 1,
  "use_cache": true,
  "products": [
    {
      "id": 777,
      "category_id": 5,
      "name": "Samsung 55\" QLED TV",
      "description": "...",
      "price": 999.0,
      "context": {"existing_features": {}, "company_id": 0},
      "languages": ["en", "ru"]
    }
  ],
  "schemas": {
    "5": {
      "Brand":  {"type": "text", "options": []},
      "Screen Size": {"type": "numeric", "suffix": "inch"},
      "Color": {"type": "select", "options": ["Black", "White", "Silver"]}
    }
  }
}
```

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "product_id": 777,
      "filled_features": {"Brand": "Samsung", "Screen Size": "55"},
      "debug_info": { ... }
    }
  ]
}
```

Параллельно каждая обработка пишет в `ai_service_output.csv`: `product_id, feature_name, extracted_value, router_node, router_reasoning, extraction_reasoning, deduced_context, judge_data`.

---

## Полный pipeline одной характеристики

```
ProductData + feature_name
        │
        ▼
[1] DatabaseCacheManager.get_cached_value()    ──► HIT → возврат
        │ MISS
        ▼
[2] TreeRouter.find_instruction()
       LLM-роутер выбирает следующий узел на каждом уровне
       до достижения листа (Leaf)
        │
        ▼
[3] AiFeaturePipeline: динамическая Pydantic-схема + промпт от листа
       OpenAIManager.structured_request() → значение + analysis + confidence
        │
        ▼
[4] Если value=None и leaf.supports_deduction:
       Deduction-LLM → context_clues + score
       если score ≥ 75 → retry с обогащённым контекстом
        │
        ▼
[5] leaf.evaluate_need_for_judgment(value):
       ACCEPT  → принимаем
       REJECT  → выбрасываем (фаст-фильтр)
       JUDGE   → HallucinationJudge.execute_audit() → BaseJudgeVerdict
                  leaf.process_judgment() решает needs_review
        │
        ▼
[6] Для type='select': MatcherService.find_best_match()
       (точное → fuzzy → semantic vector)
        │
        ▼
[7] DatabaseCacheManager.set_cached_value() (JSON-сериализация для словарей)
        │
        ▼
       filled_features + debug_info
```

---

## Особенности реализации

- **Multi-language extraction**: для текстовых характеристик без словаря возвращается массив `{language, text}` со всеми переводами (одним вызовом LLM).
- **Anti-hallucination для чисел**: `NumericValidator.is_value_in_text()` детерминированно проверяет, что число (или его эквивалент в виде дроби `5/4`, `1 1/4`, `1.25`, `1,25`) реально присутствует в тексте.
- **Geometric Auditor**: для линейных размеров судья получает специальный промпт «Strict Geometric Auditor» с защитой от sequence-bias («первое число — это всегда длина»).
- **Extensibility**: новый тип характеристики = новый класс-наследник `BaseStrategyNode` в `app/strategies/definitions/`. `TreeRouter._load_all_strategies()` подгружает их через `pkgutil.iter_modules`.
- **Логирование**:
  - `tree_traversal_log.csv` — каждый шаг роутера
  - `deductions_log.csv` — все запуски дедукции
  - `ai_service_output.csv` — финальные результаты пакета

---

## Стек технологий

| Слой | Инструмент |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Async DB | SQLAlchemy 2 + aiosqlite |
| LLM | OpenAI `gpt-4o-mini` (`client.beta.chat.completions.parse` с `response_format`) |
| Альтернатива | Ollama (`llama3.1`) через `instructor` |
| Структуры | Pydantic v2 |
| Fuzzy matching | rapidfuzz |
| Семантика | sentence-transformers (`all-MiniLM-L6-v2`), torch (CUDA если есть) |
| Кеш эмбеддингов | pickle |
| Хранилище | SQLite (WAL mode) |

---

## Запуск

```bash
pip install -r requirements.txt
python run.py     # uvicorn на 127.0.0.1:8001
```

---

## Замечания по безопасности

- **`app/config.py` содержит реальный `OPENAI_API_KEY` в открытом виде** — критично, рекомендуется ротация ключа и перенос в `.env` (`python-dotenv` уже в requirements).
- `INTERNAL_SERVICE_SECRET` тоже захардкожен — желательно вынести.
- Все `verify_*.py` — это разовые отладочные скрипты, не часть продакшена.
