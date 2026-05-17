# Cost breakdown

Сколько LLM-вызовов и денег стоит обработка одного товара в разных сценариях.

---

## Базовые цены (на 2026-05)

| Provider | Model | Input $/1M | Output $/1M |
|---|---|---|---|
| OpenAI | gpt-4o | $2.50 | $10.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| Anthropic | Claude Sonnet 4.6 | $3.00 | $15.00 |
| Yandex | YandexGPT Pro | 1.20 ₽/1k токенов | 1.20 ₽/1k токенов |
| Sber | GigaChat Pro | 1.50 ₽/1k токенов | 1.50 ₽/1k токенов |

**Тестовая модель:** GPT-4o-mini для дешёвых стадий (classifier, judges, knowledge); GPT-4o для vision/web_search.

---

## Cost per LLM call (среднее)

| Тип call | Tokens (in+out avg) | Cost GPT-4o-mini | Cost GPT-4o |
|---|---|---|---|
| Description extraction (один) | 1500 + 500 | $0.000525 | $0.0088 |
| Classifier (routing decisions) | 1000 + 800 | $0.000630 | $0.0105 |
| LLM Knowledge | 800 + 500 | $0.000420 | $0.0070 |
| Vision producer | 1200 + 700 (+image cost) | — (mini не vision) | $0.0100 |
| Vision extraction | 1500 + 500 | $0.000525 | $0.0088 |
| Cost predictor | 600 + 200 | $0.000210 | $0.0035 |
| Web search producer (с web_search tool) | 1500 + 1500 | — | $0.0188 (+ tool) |
| Web search extraction | 2500 + 500 | $0.000675 | $0.0113 |
| Judge (any) | 1000 + 200 | $0.000270 | $0.0045 |

**Тarif tool web_search**: OpenAI берёт **$0.025 за запрос** (фиксированно).

---

## Сценарии

### Сценарий A: Best case (description покрыла всё)

Хороший description от селлера, 5 атрибутов извлекаются сразу.

| Stage | Calls | Cost |
|---|---|---|
| Description (4 stage internal) | 4 | $0.035 (mini) или $0.035 (mini ~) |
| **Total** | **4 calls** | **~$0.035** = ~3.5 ₽ |

### Сценарий B: Typical (3 stages + 1 judge)

Description покрыл часть, classifier → knowledge для брендов → vision для материала.

| Stage | Calls | Cost |
|---|---|---|
| Description (3 calls used) | 3 | $0.026 |
| Classifier | 1 | $0.0105 |
| LLM Knowledge | 1 | $0.0042 |
| Vision producer | 1 | $0.012 (с фото cost) |
| Vision extraction | 1 | $0.0088 |
| Vision judge (low conf) | 1 | $0.0045 |
| **Total** | **8 calls** | **~$0.066** = ~6.6 ₽ |

### Сценарий C: Worst case (полный pipeline + judges)

Description пустое, все стадии нужны, low confidence везде → judges на каждом.

| Stage | Calls | Cost |
|---|---|---|
| Description (failed 4 stages) | 4 | $0.035 |
| Classifier | 1 | $0.0105 |
| Knowledge + judge | 2 | $0.009 |
| Vision producer | 1 | $0.012 |
| Vision extraction | 1 | $0.0088 |
| Vision judge | 1 | $0.0045 |
| Cost predictor | 1 | $0.0035 |
| Web search producer (+tool) | 1 | $0.0188 + $0.025 |
| Web search extraction | 1 | $0.0113 |
| Web search judge | 1 | $0.0045 |
| **Total** | **14 calls** | **~$0.142** = ~14 ₽ |

### Сценарий D: Early skip (только description + classifier видит что web search не нужен)

Нишевый товар, classifier пометил всё как «не найти», early return.

| Stage | Calls | Cost |
|---|---|---|
| Description | 4 | $0.035 |
| Classifier | 1 | $0.0105 |
| Cost predictor → no | 1 | $0.0035 |
| **Total** | **6 calls** | **~$0.049** = ~5 ₽ |

---

## Pricing для клиента (SaaS pay-per-use)

Стандартный SaaS markup × 5–10:

| Pack | Цена клиенту | Cost LavaTop (5%) | Cost LLM (твоё) | Маржа |
|---|---|---|---|---|
| Trial (3 cards) | 0 ₽ | 0 | ~30 ₽ | -30 ₽ (acquisition cost) |
| Starter 200 cards | 990 ₽ | -49 ₽ | -800-1400 ₽ (типичный mix) | +540 ₽ маржа |
| Business 1000 cards | 2990 ₽ | -149 ₽ | -4000-7000 ₽ | низкая/нулевая (зависит от mix) |
| Pro 5000 cards | 9900 ₽ | -495 ₽ | -20000-35000 ₽ | **отрицательная** |

⚠️ **Тарифы выше нужно пересчитать** — Pro с типичным mix Worst-case даёт убыток. Варианты:
- Перейти на GigaChat/YandexGPT для дешёвых стадий (4-10× дешевле OpenAI)
- Cap по `max_cost_credits` per card (например 50 = ~$0.05 max)
- Динамическое pricing per card (basic / advanced tier)

---

## Optimization opportunities

1. **GigaChat/YandexGPT для classifier/judges/knowledge** — нестратегические stages, дешевле в 4-10× для русскоязычного.
2. **Cache** per (product_fingerprint, source) — popular товары не пересчитываем.
3. **Batch extraction** — один LLM call на несколько attrs сразу (но это противоречит "1 stage = 1 LLM call").
4. **Confidence threshold tuning** — поднять до 0.85 для skipping judge → экономия 30-50% judge calls.
5. **Pre-cache barcode lookup** (Stage 5 future) — для FMCG / электроники бесплатные API дают базовые attrs без LLM.

---

## Метрики для мониторинга

В audit_trail хранить per request:
- `total_llm_calls`
- `total_cost_credits`
- `stage_breakdown` (сколько на каждой стадии)
- `early_exit_stage` (где остановились благодаря cost-aware routing)
- `judge_skips` (сколько раз обошли judge благодаря high confidence)
- `cache_hits` (когда будет cache)

Это даст реальные данные для оптимизации тарифов.
