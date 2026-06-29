# Открытые задачи, тесты, eval-бейзлайн

## Открытые задачи (приоритет сверху)

1. **Рестарт воркера :8002 на `b87a171`** (делает eg_importer). До рестарта
   тостер/наушники/многозначные enum-поля текут старым кодом. После — прогнать
   `12_Тостер_synth.xlsx` и файл наушников, подтвердить чистый результат.
2. **Royal Canin «Срок годности» — не нашёлся (НЕ расследовано).** Кейс: «срок
   годности для Royal Canin Medium Adult 4 кг». Гипотеза (НЕ проверена): нет
   verbatim-источника (срок не в описании, нет донора → честный no_data). Проверить
   на трейсе: есть ли срок в описании/донор-карточке; если нет — это by design.
3. **Общий резолв `value_id` для stale-категорий.** Backfill-путь чинит dcid через
   живое дерево, но `resolve_value_ids` использует шаблонный `context.category_id`.
   Для категорий с устаревшим id (тостер 47156221, наушники 17034949) общий резолв
   может писать неверный value_id. Нужно протянуть живой dcid и в основной резолв.
   eg_importer параллельно должен сверить свой маппинг категорий.
4. **eg_importer — форвардить `options` для ВСЕХ словарных полей** (не только
   цвета), включая числовые/многозначные. Список + value_id уже в шаблоне
   (`LookupData.Values`), ноль API-вызовов, сразу верный value_id. Движок и без
   этого справляется через API, но форвардинг дешевле/надёжнее (belt-and-suspenders).

## Текущая рабочая правка (этой сессии)

`b87a171` — enum-membership гард теперь фильтрует члены МНОГОЗНАЧНЫХ словарных
полей (был баг: «Функциональные особенности» тостера писала вне-словарный мусор).
7 тестов в `tests/test_enum_membership_collection.py`. Полная регрессия:
**576 passed, 2 failed** — обе падалки в `tests/test_scrappey_retry.py`
(`_always_hang() takes 2 positional arguments but 3 were given`) — это сломанный
мок, к правке отношения НЕ имеет, чинить отдельно.

## Eval-бейзлайн (БП-eval)

- required ~97–98% (после authoritative ТН ВЭД/Тип через Ozon API, `28c8096`:
  61%→98%).
- optional_honest ~77 ±2% (±2-3пп = fetch-variance, не сигнал; рабочий рычаг =
  надёжность фетча, не filling-логика).
- should-fill ~76.7%.
- Покрытие вышло на ПЛАТО. Рычаги «срок/bool/комплектация/alias» = 0 на масштабе
  (нет verbatim-данных), оставлены, но не двигают. Tested-green-but-zero-at-scale
  trap: тест зелёный, на масштабе 0.
- Fashion-eval шумит ±20пп от одиночного прогона (флак источников). Честный
  знаменатель (исключать provably-N/A optional) — коммит `be113f7`.

## Запуск тестов

```
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 ./venv/Scripts/python.exe -m pytest tests -q
```
Async-тесты используют паттерн `asyncio.run()` (без pytest-asyncio маркера).
Моки Ozon-lookup: `patch(f"{_RUNTIME}.list_values", ...)` где
`_RUNTIME = "app.services.enrichment.strategies.dictionaries.ozon_runtime_lookup"`.

## Связанные тест-файлы по темам

- enum-гард/словари: `test_enum_membership_collection.py`,
  `test_enum_options_backfill.py`, `test_drop_unresolved_enum.py`,
  `test_tree_category_resolve.py`, `test_ozon_runtime_lookup.py`.
- ТН ВЭД: `test_tnved_deprecated_label.py` + регресс-набор ТН ВЭД.
- бренд/цвет: `test_brand_from_name.py`, `test_mud_fixes.py`.
- пайплайн: `test_pipeline_orchestrator.py`, `test_pipeline_ozon_integration.py`,
  `test_ozon_e2e_pipeline.py`, `test_finishing_extractor.py`,
  `test_cost_predictor.py`.
- источники: `test_icecat_source.py`, `test_web_search_source.py`,
  `test_websearch_producer.py`, `test_vision_source.py`, `test_competitor_rag.py`.

## Незакоммиченные изменения в дереве (на момент хэндоффа)

`M`: `app/services/ai_pipeline.py`, `app/services/enrichment/pipeline.py` (мой
коммит b87a171 уже включает enum-фикс; остальные M — рабочая ветка), `prompt_router.py`,
`tree_router.py`, `requirements.txt`. Плюс много `??` временных артефактов проб
(см. 04_operations.md, раздел «незакоммиченный мусор»). Перед мержем — ревизия и
чистка.
