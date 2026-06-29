# Источники данных, флаги, стоимость

## Источники и их гейты

| Источник | Файл | Платный? | Гейт-флаг / условие | Статус |
|---|---|---|---|---|
| DescriptionSource | sources/ — | нет (LLM) | всегда | ON |
| WbCardSource | sources/wb_card_source.py | Serper (~$0.001) | имя ≥5 симв + SERPER_API_KEY | ON |
| OzonCardSource | sources/ozon_card_source.py | Scrappey (2 кредита) | имя ≥5 + SCRAPPEY_KEY | гейт ключом |
| IceCatSource | sources/icecat_source.py | нет (free API) | бренд + имя ≥5 | ON |
| PdfDatasheetSource | sources/ — | gemini | передан pdf-источник | условно |
| CompetitorRagSource | sources/competitor_rag_source.py | нет (Qdrant) | RAG включён | ON |
| LlmKnowledgeSource | sources/llm_knowledge_source.py | DeepSeek | роутинг классификатора | ON |
| VisionSource | sources/vision_source.py | gemini + DeepSeek | `enable_vision` + image_urls | per-request |
| WebSearchSource | sources/web_search_source.py | Serper + DeepSeek | `enable_web_search` + CostPredictor | per-request |
| TnvedSource | sources/tnved_source.py | DeepSeek (1/категория) | «ТН ВЭД» в имени | ON |
| SafeEnumFillSource | sources/safe_enum_fill_source.py | DeepSeek ×1-2 | `SAFE_LLM_ENUM_FILL_ENABLED` | **ON** |
| LamodaScrapflySource | sources/lamoda_scrapfly_source.py | scrape.do (~30 кред) | `LAMODA_SCRAPFLY_ENABLED` | **OFF** |
| ScrapflyOzonSource | sources/scrapfly_ozon_source.py | Scrapfly | `SCRAPFLY_OZON_FALLBACK_ENABLED` | **OFF** |
| BarcodeSource / BooksSource / Onliner / BestBuy / regard | sources/ | разн. | по наличию идентификатора | условно |

## Дефолтные значения флагов (.env, без секретов)

```
USE_NEW_PIPELINE=true
PIPELINE_RICH_SOURCES=true
BYPASS_PROMPT_TREE=true
SAFE_LLM_ENUM_FILL_ENABLED=true        # Gate A (verbatim) + Gate B (adversarial)
TNVED_SOURCE_FIX_ENABLED=true
UNIT_NORMALIZE_ENABLED=true            # лосслесс конверсия единиц card-значений
LLM_KNOWLEDGE_ADVERSARIAL_ENABLED=1    # mud Gate B на llm_knowledge
ENABLE_OPENAI_FALLBACK=false           # → gpt-4o/gpt-4o-mini де-факто выключены
SCRAPFLY_OZON_FALLBACK_ENABLED=false
LAMODA_SCRAPFLY_ENABLED=false
URL_FETCHER_SCRAPPEY_FALLBACK=0        # Scrappey-фолбэк url_fetcher выключен 06-14
WEB_SEARCH_MODEL=gpt-4o                # только legacy-фолбэк (ENABLE_OPENAI_FALLBACK=false)
```

## Внешние сервисы

- **scrape.do** (`providers/scrapedo_client.py`) — ЕДИНСТВЕННЫЙ платный скрейпер,
  pay-per-credit. Рецепт для RU-антибота (DataDome/SmartCaptcha): `render=true`
  (JS), `super=true` (residential), `geoCode=ru`, `customWait`. Успех = HTTP 200
  И `len(body) ≥ 50 000`. Ретраи на 429/5xx (3 попытки, бэкофф 2/4/8с).
  Замерено: lamoda/sportmaster/zdravcity/exist/4lapy пробиваются; goldapple/
  vseinstrumenti мертвы (DataDome). **Ozon scrape.do НЕ пробивает** (антибот/502).
- **Serper** (Google Search API, ~$0.001/запрос) — card-finding: находит URL
  карточки Ozon/WB в обход флакового внутреннего поиска. Питает WbCardSource
  (nm_id), OzonCardSource (fallback), IceCat MPN-lookup.
- **Scrappey** (≠ scrape.do) — питает `ozon_card_source`, 2 кредита/товар
  (search + features). Выключен в .env с 06-14 (не бьёт RU-ретейл, жёг кредиты;
  раскомментировать чтобы вернуть). Замер 23.06: datacenter-прокси ЛУЧШИЙ для
  Ozon (66%), proxyCountry=Russia ХУЖЕ (33%), browser 0% (ломает SSR-парсер).
- **IceCat** — бесплатный (open/subscription). GTIN/brand→спеки. EN→RU маппинг
  значений (коммит `5b4ffaf`).
- **Ozon Seller API** — бесплатный. `ozon_runtime_lookup.search_value` /
  `list_values` (`/v1/description-category/attribute/values[/search]`), нужны
  `OZON_CLIENT_ID` + `OZON_API_KEY`. Также `/v1/description-category/tree` для
  живого резолва `description_category_id`.
- **Qdrant** (`reference`: сервер romanovka `100.125.33.17` tailnet) — две
  коллекции: `ozon_products` (1.69М точек, конкурентные карточки) и `datasets_rag`
  (OFF/OBF/Amazon/ABO и др.). `QDRANT_URL` переключён на выделенный сервер.

## Стоимость 1 товара (оценка)

Замеренного счётчика токенов в проде НЕТ — счётчики *вызовов* заземлены на код,
размеры промптов оценены по типовому товару (~40 атрибутов). Цены — публичные
тарифы, ₽≈80/$.

| Профиль | LLM-вызовы | Внешние | $/товар | ₽/товар |
|---|---|---|---|---|
| МИНИМУМ (чистый LLM, всё OFF) | 2–3 DeepSeek | — | $0.006–0.012 | ₽0.5–1 |
| ТИПИЧНЫЙ (как настроено: + card + SafeEnumFill) | 5–8 | Serper 1–3 | $0.02–0.04 | ₽1.5–3 |
| МАКСИМУМ (всё ON + vision + web + PDF + Scrappey) | 11–14 | Serper 1–4, Scrappey 2 | $0.05–0.08 | ₽4–6.5 |

Главный качель: модель LLM. Всё на DeepSeek → дёшево. Если включить gpt-4o-роуты
(`ENABLE_OPENAI_FALLBACK=true` + web_search на gpt-4o) — +$0.03/товар, максимум
до ~₽10. Батч-кэш DeepSeek (повтор словарей allowed_values между товарами одной
категории, input по $0.07/М вместо $0.27/М) сдвигает типичный профиль к нижней
границе. Скрейперы по дефолту выключены и стоят копейки даже в максимуме.
