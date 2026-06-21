# API-контракт: cpAiFeatures (enrichment) ↔ eg_importer

**Версия:** v2 (2026-06-21)
**Назначение:** cpAiFeatures обогащает характеристики товара и отдаёт **только то,
что достаётся у него** — значение + авторитетный `value_id` + источник/уверенность/
evidence. **Подсчёт заполнено/не-заполнено, вёрстку Excel и выходной файл строит
eg_importer** (у него есть полный список запрошенных характеристик → diff против
`filled_features`).

---

## Разделение ответственности

| Делает cpAiFeatures (этот сервис) | Делает eg_importer |
|---|---|
| Извлекает значения из источников (карточки МП, IceCat, web, competitor RAG, vision) | Принимает Excel-файл клиента, парсит шаблон МП |
| Резолвит `value_id` по словарю Ozon (авторитетная привязка) | Считает % заполнено / не-заполнено, список пустых |
| Даёт `source`, `confidence`, `evidence` на каждое значение | Верстает выходной Excel + лист-отчёт |
| — | Формирует выходной файл клиенту |

---

## REQUEST — `POST /process_batch` (BatchPayload)

```jsonc
{
  "client_id": 123,
  "use_cache": false,
  "research_mode": "off",                  // "off" | ...
  "options": {
    "enable_vision": false,                // включить VisionSource (image_urls)
    "enable_web_search": false,            // включить web_search/marketplace-пул
    "marketplace": "ozon"                  // "default" | "ozon" | "wb"
  },
  "schemas": {                             // схема характеристик по category_id
    "<category_id>": {
      "<feature_name>": {
        "id": 4567,                        // attribute_id в схеме (СТАБИЛЬНЫЙ ключ)
        "type": "text|enum|number|...",
        "options": [ /* словарь enum, если есть */ ],
        "is_required": true,
        "suffix": "", "prefix": ""
      }
    }
  },
  "products": [
    {
      "id": 1001,                          // product_id (вернётся в ответе)
      "category_id": 17028922,
      "name": "Наушники Sony WH-1000XM5",
      "description": "",
      "brand": "Sony",                     // → brand-fill + IceCat brand-match
      "ean": "4548736141155",              // штрихкод → IceCat GTIN / object-search
      "ozon_type_id": 91565,               // нужен для словарного резолва Ozon
      "languages": ["ru"],
      "source_urls": [],                   // ≤5 HTTPS — донорские страницы
      "image_urls": [],                    // ≤10 — для vision
      "context": { "existing_features": {}, "company_id": 0 }
    }
  ]
}
```

---

## RESPONSE — на каждый товар (ProductResult)

```jsonc
{
  "product_id": 1001,
  "filled_features": {                     // ТОЛЬКО заполненные: feature_name -> value
    "Версия Bluetooth": "5.0",
    "Цвет": ["чёрный"]                     // массив, если is_collection
  },
  "debug_info": {                          // feature_name -> метаданные
    "Версия Bluetooth": {
      "attribute_id": 4567,                // ⬅ NEW: стабильный числовой ключ
      "value_id": null,                    // ⬅ NEW: словарный ID Ozon (одиночное)
      "value_ids": null,                   // ⬅ NEW: словарные ID (коллекция)
      "is_collection": false,              // ⬅ NEW: value — массив или скаляр
      "semantic_type": null,               // ⬅ NEW: color|weight|brand… (подсказка)
      "source": "wb_card",                 // откуда: wb_card|ozon_card|icecat|
                                           //   web_search|llm_knowledge|vision|
                                           //   competitor_rag|lamoda|web_marketplace|…
      "confidence": 0.93,                  // 0.0–1.0
      "evidence": "wb: Sony … | match=…",  // цитата/URL для аудита
      "judge_validated": true,             // прошёл per-source judge
      "router": {}, "extraction_reasoning": "…",
      "deduced_context": null, "source_urls": null
    },
    "Цвет": {
      "attribute_id": 4570,
      "value_id": null,
      "value_ids": [98765, 98766],         // ⬅ для коллекции — список ID
      "is_collection": true,
      "source": "ozon_card", "confidence": 0.9, "evidence": "…",
      "judge_validated": true, "semantic_type": "color",
      "router": {}, "extraction_reasoning": "…",
      "deduced_context": null, "source_urls": null
    }
  },
  "skipped": {                              // ⬅ NEW: причина пустоты per НЕзаполненный target
    "Штрихкод": { "attribute_id": 4571, "reason": "platform_field" },
    "Вес":      { "attribute_id": 4580, "reason": "no_data" }
  },
  "tokens_used": 0,                         // TODO Tier-2 (пока 0)
  "is_cached": false
}
```

### `skipped[feature]` — причина пустоты (что знаем только мы)

| `reason` | Смысл | Как метить в отчёте |
|---|---|---|
| `platform_field` | Учётное/платформенное поле (баркод/НДС/декларации/ИКПУ/артикул) — сервис **намеренно** не заполняет, это данные продавца | 🔒 «ваше поле» — **НЕ** недоработка сервиса |
| `no_data` | Поле извлекаемое, но ни один источник не дал значение | — «нет данных» |

В `skipped` попадают все target из `schema`, которых **нет** в `filled_features`.
eg_importer может либо взять `skipped` напрямую, либо считать пустоту diff-ом —
ценность `skipped` именно в **reason** (его выводит только наш классификатор).

> ⚠️ **Точность `platform_field`:** для **WB** ловится по маркерам надёжно
> (баркод/НДС/ИКПУ/NTIN/сертификаты/декларации). Для **Ozon** классификатор
> намеренно консервативен (не дробит знаменатель) → часть админ-полей вернётся как
> `no_data`, а не `platform_field`. Если нужна агрессивная метка 🔒 для Ozon —
> это отдельная доработка Ozon-классификатора (общий admin-marker список).

### Поля `debug_info[feature]` — что и зачем

| Поле | Тип | Назначение для eg_importer |
|---|---|---|
| `attribute_id` | int | **Ключ для маппинга в колонку шаблона МП** (надёжнее имени) |
| `value_id` | int \| null | **Словарный ID Ozon** — писать в шаблон для enum-полей, иначе значение может не сесть при импорте |
| `value_ids` | list[int] \| null | То же для коллекций (is_collection=true) |
| `is_collection` | bool | Значение — массив; верстать как мультизначение |
| `semantic_type` | str \| null | Тип (color/weight/brand) — опц. для правил отображения |
| `source` | str \| null | Источник → колонка «Источник» в отчёте |
| `confidence` | float | Колонка «Уверенность %»; `< 0.6` → пометка ⚠ «проверить» |
| `evidence` | str \| null | Колонка «Откуда» (цитата/URL) — защита от споров |
| `judge_validated` | bool | Прошёл валидацию судьёй (доп. сигнал доверия) |

### Что eg_importer считает САМ
- **Не-заполненные** = `targets из schema` − `ключи filled_features`.
- **% заполнено** (обязательные/все) — по своей схеме (`is_required` есть у него).
- **Статус поля**: ✔ (в filled) · ⚠ (`confidence < порог`) · — (нет в filled = нет данных).

> ⚠ Платформенные поля (штрихкод/НДС/декларации/ИКПУ) сервис НЕ заполняет
> намеренно — это учётные данные продавца. eg_importer помечает их 🔒, не считая
> «недоработкой» сервиса.

---

## Возможные доп-поля (по запросу eg_importer)
- ✅ **`skipped` (причина пустоты)** — РЕАЛИЗОВАНО (см. выше).
- **`reason: "low_confidence_dropped"`** — отличить отброшенный по low-conf мусор от
  «нет данных»; сейчас неразличимо (кандидаты отсеяны пайплайном раньше). Требует
  проброса dropped-кандидатов из пайплайна — Tier-2.
- **Единица/нормализация** значения — сейчас value уже нормализован (suffix учтён).
- **tokens_used / стоимость** — Tier-2 TODO (для биллинга per-SKU).
