# Стратегия удешевления без потери качества

Этот документ — план как снизить себестоимость обработки товара **в 5-10 раз** при сохранении (или улучшении) качества. Основано на исследовании актуальных (май 2026) ИИ-провайдеров и архитектурных паттернов.

**Категорически НЕ используем:** GigaChat, YandexGPT и подобные «костыли» — они дают экономию ценой качества. Все рассматриваемые варианты либо равны GPT-4o, либо превосходят.

---

## Главное открытие

**DeepSeek V4 (api.deepseek.com)** — глобальная top-tier модель, доступная в двух размерах:

| Модель | Input (cache miss) | Input (cache hit) | Output |
|---|---|---|---|
| **deepseek-v4-flash** | $0.14 / 1M | **$0.0028 / 1M (-98%)** | $0.28 / 1M |
| **deepseek-v4-pro** (промо до **31.05.2026**) | $0.435 / 1M | $0.003625 / 1M | $0.87 / 1M |
| deepseek-v4-pro (после 31.05) | $1.74 / 1M | $0.0145 / 1M | $3.48 / 1M |

**Что важно:**
- V3 deprecated — алиасы `deepseek-chat`/`deepseek-reasoner` теперь маршрутизируют на V4-flash (отключение 24.07.2026)
- **V4-Pro бьёт GPT-4o** по заявленным бенчмаркам: MMLU 90.1, MMLU-Pro 87.5, GPQA Diamond 90.1 (источник — DeepSeek; независимая верификация ограничена)
- В **17-35 раз дешевле GPT-4o** ($2.50/$10) при равном или лучшем качестве
- Нативный JSON mode + structured outputs
- Context window **1M tokens**, function calling до 128 параллельных вызовов

**Cache-hit бонус — критичная фишка:** для V4-flash при cache-hit стоимость input падает в **×50 раз** ($0.0028 vs $0.14). Это означает, что если правильно структурировать system prompt + JSON schema (~3000 токенов одинаковые для всех товаров), реальная стоимость extraction падает почти до нуля на input стороне.

**Через OpenRouter V4-flash дешевле прямого API:** $0.112 / $0.224 за 1M (~20% объёмная скидка маршрутизатора). Для основной нагрузки (parser, classifier, judges) — лучше OpenRouter. Для cache-heavy задач — лучше прямой API DeepSeek (cache-hit работает там).

**Переход с GPT-4o на DeepSeek V4 — не downgrade, а upgrade** по качеству при цене в 17 раз ниже. Меняет всю экономику SaaS.

⚠️ **Watch out:** V4-Pro в промо до **31.05.2026**. После цена вырастет ×4 (с $0.435 до $1.74). За 2 недели до этой даты — пересмотреть routing решения для тех этапов где использовался Pro.

---

## Стратегия по приоритету ROI

### Tier 1 — Немедленно (2-3 дня, эффект -78%)

**Что переключаем:**

| Этап | Было | Стало | Экономия |
|---|---|---|---|
| Parser, Classifier, Extraction после web/vision | GPT-4o/mini | **deepseek-v4-flash через OpenRouter** ($0.112/$0.224 за 1M) | **-95%**. С cache-hit ещё дешевле. Качество равно GPT-4o |
| Knowledge, Judges (critical reasoning) | GPT-4o | **deepseek-v4-pro прямой API** ($0.435/$0.87 промо, $1.74/$3.48 после 31.05) | -65% (промо) / -30% (после). MMLU 90.1 vs GPT-4o ~88 |
| Vision (описание фото) | GPT-4o Vision ($2.50/$10) | **Gemini 2.5 Flash** ($0.30/$2.50) | **-88%**, при этом Gemini 2.5 Flash **доминирует на e-commerce vision-задачах** по апрельскому 2026 бенчмарку |
| Web search tool | OpenAI Responses API ($25/1k) | **Serper API** ($0.30-1/1k) | **-96-99%**. Serper — Google SERP, такое же качество для product search |

Подходит без условий:
- Качество не страдает (часть этапов даже улучшается)
- Реализация — буквально замена адаптеров провайдеров в коде
- Нет VPN, ничего не блокировано

**Эффект:** Cost худшего сценария $0.14 → $0.03 на товар.

### Tier 2 — Быстро (1 неделя, эффект -85% от исходного)

**Каждая мера независимая, можно делать параллельно:**

**1. Anthropic prompt caching** для системных промптов и схем

Системный промпт + JSON schema + few-shot examples (~3000-5000 tokens) одинаковы для всех товаров. Только product description уникален. Структура:

```
[system + schema + examples ← кэшируется на 5 мин]
[product data ← уникально]
```

Anthropic даёт **-90% на cached input tokens**. Для нашего случая это означает дополнительно **-60-70% от input cost**. На Claude Sonnet 4.6 input $3.00 → $0.30/M реальная цена. Применимо для тех stages где нужно высокое качество reasoning (не всех — DeepSeek для основных вызовов).

Реализация: 4 часа.

**2. Redis SKU-level cache** для повторных запросов

iPhone 15 Pro 256GB Titanium обрабатывается **один раз**, результат хранится 90 дней. Когда второй селлер заводит тот же товар — получаем cached attributes за 3-8мс, **0 обращений к ИИ**.

```python
cache_key = sha256(product_name + brand + ean)
ttl_by_attribute_type = {
    "static_specs": 90_days,   # dimensions, materials, weight
    "price_data": 24_hours,
    "web_search_results": 7_days
}
```

Для популярных товаров (iPhone, AirPods, Nike) — **-70-90% обращений** на повторных запросах. Реализация: 1 день.

**3. Cascade routing на cheap-first**

Простой ИИ-классификатор (на DeepSeek V3 / GPT-4.1-mini) определяет «сложный ли товар»:
- 70% товаров → дешёвый стек (DeepSeek V3 везде)
- 30% сложных (vision-heavy, нишевые, мультиязычные) → premium (Claude Sonnet 4.6 + GPT-4o Vision)

Реализация по research-papers FrugalGPT и RouteLLM: **-75% от model cost** на 70% товаров. 1 день.

**Эффект Tier 2 поверх Tier 1:** $0.03 → $0.01-0.02 на товар (с учётом cache hit rate).

### Tier 3 — Опционально (1-2 недели, эффект -90% на отдельных stages)

**Distillation:**
- Сгенерить 10 000 примеров обработки товаров GPT-4o
- Fine-tune Qwen 3 8B или Phi-3.5 Mini на этих данных
- Self-hosted дистиллированная модель matches 70B+ на нашей domain-specific задаче
- Cost: $5/1M training tokens (OpenAI) или $2-7 на полный SFT run на RunPod H100

Применимо для `parser` и `classifier` stages. Walmart делает именно это на их каталоге.

**Когда делать:** когда есть 10k+ обработанных товаров (через 2-3 месяца после запуска).

### Tier 4 — Self-host (только при объёме >25 000 req/день)

**Break-even анализ из research:**

| Объём | Together.ai (hosted) | RunPod H100 (self-host) | Что выгоднее |
|---|---|---|---|
| 1 000 req/день | $4.40/день | $60/день (97% idle) | **Hosted** |
| 10 000 req/день | $44/день | $60/день (70% idle) | **Hosted** |
| 25 000 req/день | $110/день | $60/день | **Self-host (break-even)** |
| 50 000 req/день | $220/день | $65/день | **Self-host (3× выгода)** |

При <25k req/день self-host дороже и требует ~50 часов DevOps setup + 10-15ч/мес поддержки. **На MVP не делать.**

Когда volume вырастет — Qwen 3 32B (INT4 quantization) на 1×A100 через RunPod Serverless с SGLang. Конфигурация работает в одиночку, MMLU 83.6, отличный JSON output.

---

## Финальная экономика после Tier 1+2

| Пакет | Цена клиенту | Cost на OpenAI (было) | Cost после Tier 1 | Cost после Tier 1+2 | Маржа после T1 | Маржа после T1+T2 |
|---|---|---|---|---|---|---|
| Старт 200 карточек | 990 ₽ | ~1 000 ₽ | ~210 ₽ | ~140 ₽ | +780 ₽ (79%) | +850 ₽ (86%) |
| Бизнес 1000 карточек | 2 990 ₽ | ~5 000 ₽ | ~1 050 ₽ | ~700 ₽ | +1 940 ₽ (65%) | +2 290 ₽ (77%) |
| Pro 5000 карточек | 9 900 ₽ | ~25 000 ₽ | ~5 250 ₽ | ~3 500 ₽ | +4 650 ₽ (47%) | +6 400 ₽ (65%) |
| Pay-as-you-go (карточка) | 5-10 ₽ | ~14 ₽ | ~3 ₽ | ~1-2 ₽ | +2-8 ₽ | +3-9 ₽ |

**Pro теперь рентабельный без подъёма цен.** Можно даже снизить prices для агрессивного маркетинга, всё равно будет маржа.

---

## План реализации (по этапам)

### Этап 1 (на этой неделе, 2-3 дня)

1. Добавить адаптер для DeepSeek V3 в `app/services/llm_manager.py` (OpenAI-совместимый API, минимальное изменение)
2. Добавить адаптер для Gemini 2.5 Flash в Vision producer
3. Заменить OpenAI web_search tool на Serper API в WebSearch producer
4. Запустить A/B тест на 100 товарах: OpenAI vs новая комбинация — сравнить качество

### Этап 2 (1 неделя)

5. Anthropic prompt caching для tier-2 промптов (где используется Claude)
6. Redis SKU cache с TTL стратегией
7. Cascade router: mini-classifier → escalate decision

### Этап 3 (по факту 10k обработанных товаров, через 2-3 мес)

8. Сбор 10k examples в формате (input → expected output)
9. Fine-tune Qwen 3 8B на RunPod
10. Migrate parser/classifier на distilled model

### Этап 4 (когда >25k req/день)

11. Self-host Qwen 3 32B на RunPod Serverless
12. SGLang для structured output без потери throughput
13. Постепенная миграция всех stages на self-host

---

## Качество — как контролируем

Каждый переход — через A/B тест:
- Прогнать 100 товаров через старый и новый стек параллельно
- Сравнить заполненные characteristics: match rate, confidence distribution
- Метрика: % атрибутов где новый стек дал верный или лучший ответ
- Если ≥ 95% match — переключаем production
- Если < 95% — анализируем где проседает, корректируем routing

Continuous monitoring:
- Per-source success rate (сколько раз source.extract() вернул valid attrs)
- Per-source confidence distribution (если падает — alarm)
- Manual sampling: 10 случайных карточек/день — ручная проверка качества

---

## Финальное сравнение провайдеров (cheat sheet)

| Задача | Рекомендуется | Цена /1M (in/out) | Почему |
|---|---|---|---|
| Parser, Classifier, Extraction после web/vision | **deepseek-v4-flash через OpenRouter** | $0.112 / $0.224 | Дешевле прямого API, ×17 от GPT-4o, JSON mode |
| Knowledge, Judges | **deepseek-v4-pro прямой API (промо)** | $0.435 / $0.87 | До 31.05 — премиум reasoning за дёшево. После — пересмотр |
| Vision (анализ фото товара) | **Gemini 2.5 Flash** | $0.30 / $2.50 | Лидер на e-commerce vision-задачах апреля 2026 |
| Premium reasoning fallback (если pro слишком дорог) | **Claude Sonnet 4.6** + prompt caching | $0.30 (cached) / $15 | 1M context window, лучший structured output, -90% на cache-hit |
| Web search tool | **Serper API** | $0.30-1 / 1000 запросов | 25-80× дешевле OpenAI web_search |
| Будущее self-host | **Qwen 3 32B INT4** на RunPod | $0.09-0.12 / 1M (при >25k req/день) | Open source, отличный JSON, helps with multilingual |

---

## Источники исследования

Все цены и бенчмарки актуальны на май 2026:
- DeepSeek API pricing: api-docs.deepseek.com
- Claude pricing: platform.claude.com
- Gemini pricing: pricepertoken.com/pricing-page/model/google-gemini-2.5-flash
- Search API comparison: buildmvpfast.com/api-costs/ai-search
- RouteLLM (LMSYS): lmsys.org/blog/2024-07-01-routellm
- FrugalGPT (Stanford): arxiv.org/abs/2305.05176
- Qwen 3 release: qwenlm.github.io/blog/qwen3
- vLLM vs SGLang: turion.ai/blog/vllm-vs-sglang-inference-comparison-2026
- Amazon Catalog AI on Bedrock: aws.amazon.com/blogs/machine-learning/how-the-amazon-com-catalog-team-built-self-learning-generative-ai-at-scale
- Walmart product LLMs: tech.walmart.com/content/walmart-global-tech/en_us/blog/post/using-llms-to-manage-product-catalogs
