# Cost-aware пайплайн обогащения

## Точка входа и флаги

- `USE_NEW_PIPELINE=true` → `PipelineAdapter` → `PipelineOrchestrator.enrich`
  (`app/services/enrichment/pipeline.py`).
- `PIPELINE_RICH_SOURCES=true` — включает «богатые» источники (card/web/rag).
- `BYPASS_PROMPT_TREE=true` (дефолт, решение владельца 2026-06-16) — пайплайн
  source-merge заполняет атрибуты, а детерминированный `UNIVERSAL_VERIFY`-гейт
  действует как пост-фактум страховка. Промпт-дерево (TreeRouter) — «правильная»
  будущая инвестиция (предотвращать галлюцинации на генерации), пока НЕ вшито.

## Идея

Sequential cost-aware пайплайн. **1 стадия = максимум 1 LLM-вызов**, источники
идут по возрастанию стоимости: сначала бесплатные/дешёвые (описание, card-API),
потом дорогие (LLM-знание, vision, web_search). Каждый следующий источник
заполняет только то, что осталось пустым. Дорогие источники гейтятся
(`CostPredictor`) и флагами.

## Порядок стадий (по `enrich`)

| Стадия | Источник | Тип | Когда срабатывает |
|---|---|---|---|
| 0 | DescriptionSource | verbatim из названия/описания | всегда |
| 0d | `_derive_context_brand` | бэкфилл `context.brand` из имени | если колонка «Бренд» пуста |
| 0c.5 | `_backfill_allowed_values_from_api` | тянет закрытый словарь Ozon | enum-поля без `allowed_values` |
| 0.45 | WbCardSource | WB Content API (донор-карточка) | есть имя ≥5 симв + Serper-ключ |
| 0.5 | OzonCardSource | Scrappey/Serper парс карточки Ozon | имя ≥5 симв + Scrappey-ключ |
| 0.55 | IceCatSource | GTIN/brand→IceCat спеки | есть бренд + имя |
| 0.6 | PdfDatasheetSource | gemini-парс PDF-даташита OEM | передан pdf-источник |
| 0.7 | CompetitorRagSource | Qdrant vector-поиск (конкуренты) | RAG включён |
| 1 | LlmClassifier | роутинг оставшихся таргетов | всегда (1 вызов) |
| 2 | LlmKnowledgeSource | LLM-знание (чанки по 30) | таргеты, отрrouter'енные в LLM |
| 3 | VisionSource | gemini-vision + extraction | есть `image_urls` |
| 4-гейт | CostPredictor | стоит ли web_search | если есть optional-ws таргеты |
| 4 | WebSearchSource | Serper + DeepSeek-summary+extract | одобрено CostPredictor |
| 4.5 | UgcSource | UGC-скрейп | передан ugc-источник |
| 4.7 | TnvedSource | резолв ТН ВЭД (per-category, кэш) | таргет с «ТН ВЭД» в имени |
| 5 | FinishingExtractor | до-извлечение по оставшимся пробелам | если остались пустые |
| 5.5 | `_generate_annotation` | генерация «Аннотация» | таргет «Аннотация» пуст |
| finalize | `_finalize`/`_finalize_async` | гарды + резолв value_id + контракт | всегда |

## Что делает `_finalize` (важный слой)

После сбора всех `AttributeValue` из источников:

1. **placeholder-filter** — дроп «нет/-/n/a/без» из enum-полей.
2. **`_apply_enum_membership_guard`** — дроп значений вне `allowed_values` от
   ЛЮБОГО источника (IceCat/marketplace/donor обходят LLM-гейт). Скаляры — через
   `_map_enum_value` (exact→норм→WRatio≥85); списки (Ⓜ️ многозначные) — через
   `_match_collection_member` (token_set_ratio≥80, дроп вне-словарных членов,
   дедуп). Исключения: ТН ВЭД, бренд.
3. **ТН ВЭД-фикс** — дроп LLM/web-кодов ТН ВЭД кроме тех, что от TnvedSource
   (сентинел `tnved_resolver:`).
4. **color-гарды** — `_apply_color_source_guard` (allowlist = только DESCRIPTION),
   `_drop_multivalue_color_premerge` (дроп палитры донора). Цвет — per-SKU.
5. **mud-гейты** — adversarial Gate B на llm_knowledge, source-corroboration,
   hard-drop `conf=0.0`.
6. **`resolve_value_ids`/`_async`** — резолв строк-ярлыков в `value_id` словаря
   (детерминированный + LLM-tail). Коллекции резолвятся поэлементно.
7. **`_drop_unresolved_optional/required_enums`** — дроп enum-значений, у которых
   после резолва `value_id=None` (фейк-филл, Ozon отверг бы).

## Контракт v2 (выход для eg_importer)

```json
{
  "product_id": 123,
  "filled_features": { "<имя атрибута>": "<значение или [список]>" },
  "debug_info": {
    "<имя>": {
      "attribute_id": 4820, "value_id": 971398593, "value_ids": [..],
      "source": "icecat", "confidence": 0.9, "evidence": "<цитата/сентинел>"
    }
  },
  "skipped": { "<имя>": { "attribute_id": 22232, "reason": "no_data|platform_field" } }
}
```

eg_importer пишет `value_id` как `dictionary_value_id` при импорте. `skipped`
отдаётся в `clean_data` (коммит `fec3284`).

## Модели (провайдеры)

- **Основной:** DeepSeek (`PROVIDER_MAIN=deepseek`). `MAIN_MODEL=deepseek-chat`
  (parser/deduction/judge/extraction), `PREMIUM_MODEL=deepseek-reasoner`.
- **Vision/PDF:** `google/gemini-2.5-flash` через OpenRouter (`VISION_MODEL`).
- **Enum-констрейнт:** `gpt-4.1-mini` (`OPENAI_STRUCTURED_MODEL`) — token-level
  enum enforcement через llguidance, ТОЛЬКО когда у response-модели есть
  enum-констрейнты. $0.40/$1.60 за 1М.
- **web_search:** де-факто DeepSeek (`get_main_manager()` +
  `EXTRACTION_FROM_TEXT_MODEL=deepseek-chat`). `WEB_SEARCH_MODEL=gpt-4o` —
  только legacy-фолбэк через `OpenAIManager`, а `ENABLE_OPENAI_FALLBACK=false`,
  поэтому gpt-4o де-факто НЕ используется.
- **Router:** тот же DeepSeek (`ROUTER_MODEL=deepseek-chat`).
