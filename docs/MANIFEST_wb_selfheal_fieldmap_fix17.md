# MANIFEST — FIX-17: WbCardSource live self-heal field-map (реюз eg-importer) (HARDEN)

## Проблема (rooted 2026-07-02, феасибилити-пасс)
WbCardSource (`_map_characteristics`, `wb_card_source.py:1717`) маппит WB-имя-поля → Ozon-атрибут через
ЗАМОРОЖЕННЫЙ статический снапшот `eg_get_field_map` (`eg_wb_ozon_field_map.json`, 6548 ключей). Distinct-key
покрытие ~48% — длинный хвост WB-имён не в кэше → не заполняется. eg-importer достигает ~95% **per-product**
fill НЕ за счёт большего снапшота (его кэш почти идентичен, 6636 ключей, тоже ~48% distinct), а за счёт
**LIVE self-heal**: `field_map_builder._heal_field_map_async` (`eg-importer/mapping/field_map_builder.py:154`)
LLM-мапит (DeepSeek) любое WB-имя, которого нет в кэше, на существующий Ozon-атрибут и доклеивает в растущий
кэш. WbCardSource этого не делает → застревает на 48%.

## Цель
Портировать **только self-heal field-name механизм** в web-fetch, повесить ПЕРЕД/ВОКРУГ статического
`eg_get_field_map`: WB-имя не в кэше → LLM-мапнуть на атрибут из СПИСКА атрибутов этого type → юнион в
растущий per-`(wb_subject,cat_id,type_id)` кэш → персист. Со временем покрытие растёт к per-product ~95%.
`value_normalizer` НЕ реюзаем (родной `resolve_value_id` сильнее). Routing НЕ реюзаем (cat/type уже есть).
Ре-экспорт снапшота НЕ делаем (даёт ~ноль).

## INV-17 (что менять)
### 1. Порт self-heal модуля
- Copy-adapt `field_map_builder.py` self-heal-ядро (`build_field_map_async` + `_heal_field_map_async`) в
  web-fetch (напр. `app/services/enrichment/sources/wb_field_map_selfheal.py`). Использовать **РОДНОЙ**
  web-fetch `app.services.providers.deepseek_provider.DeepSeekProvider` + `app.config.DEEPSEEK_API_KEY`
  (1:1 паритет, подтверждён). `bootstrap.activate()` → заменить на web-fetch env-init (или убрать, если
  config грузится сам).
- Новая кэш-папка `data/field_maps/` в web-fetch (net-new). Файл на `(wb_subject,cat_id,type_id)`, атомарная
  запись (tmp+rename), безопасно при конкуренции.

### 2. Интеграция в `_map_characteristics` (точка `wb_card_source.py:1717`)
- Порядок: (1) статический `eg_get_field_map` (быстро, $0) → (2) для WB-имён, которых нет в результате,
  вызвать self-heal (LLM) → (3) юнион в кэш+персист → (4) downstream БЕЗ ИЗМЕНЕНИЙ (`resolve_value_id`,
  unit-normalize, gender-guard, is_collection, RU-size).
- **Self-heal мапит ТОЛЬКО на атрибут из `get_ozon_characteristics_for_type(cat_id,type_id)`** — НЕ выдумывает
  атрибут, которого нет у type (grounding против галлюна имени-поля). Нет подходящего → верни None (не форсь).
- Флаг `WB_CARD_SELFHEAL_ENABLED` (env, default true). False → только статика (путь до FIX-17).
- **DeepSeek fail/empty/таймаут → graceful fallback на статику-only** (это поле просто не замаплено; НЕ падать,
  НЕ форсить, НЕ кэшировать пустое). Тот же fail-safe принцип, что FIX-16.

## Инварианты (тесты — DeepSeek self-heal МОКается)
- INV-17a: self-heal вызывается ТОЛЬКО для WB-имён, отсутствующих в статическом результате (кэш-hit не зовёт LLM).
- INV-17b: self-heal мапит имя только на существующий атрибут type; мок «нет подходящего» → None, атрибут не создаётся.
- INV-17c: успешный self-heal → имя доклеено в кэш, повторный вызов на то же имя — из кэша, БЕЗ LLM (персист+рост).
- INV-17d: DeepSeek raise/empty → static-only fallback, extract не падает, пустое не кэшируется.
- INV-17e: `WB_CARD_SELFHEAL_ENABLED=false` → LLM не зовётся, поведение байт-в-байт как до FIX-17.
- INV-17f: downstream (`resolve_value_id`/unit-normalize/gender-guard/is_collection) — байт-в-байт прежние;
  diff трогает только новый self-heal-модуль + точку вызова в `_map_characteristics`.
- INV-17g: кэш-запись атомарна (tmp+rename), битый/пустой файл не роняет чтение.

## Oracle (grounding — self-heal мапит ИМЯ, не значение)
| # | WB-имя (не в статике) | type-атрибуты | ожидание |
|---|---|---|---|
| S1 | «Тип разъёма наушников» | [«Разъём для наушников», …] | мапит на «Разъём для наушников», кэш растёт |
| S2 | «Абракадабра-поле-XYZ» | [телефонные атрибуты] | нет подходящего → None, атрибут НЕ создан (без галлюна) |
| S3 | «Диагональ экрана» (уже в статике) | — | self-heal НЕ зовётся (кэш-hit) |
| S4 | любое, DeepSeek лёг | — | static-only, extract цел |

## Acceptance
1. **Unit:** S1–S4 + INV-17a..g (мок DeepSeek).
2. **⚠️ ЖИВОЙ смоук self-heal (ОБЯЗАТЕЛЕН):** реальный DeepSeek на категории с известным длинным хвостом (НЕ
   телефон — там потолок от статики; взять что-то, где статика неполна). Подтвердить: незамапленное WB-имя →
   корректно мапится на реальный Ozon-атрибут → кэш вырос. Неверный маппинг имени = FAIL (усилить промпт grounding).
3. **Gate-1 полный** (ruff+mypy+radon+vulture+bandit+conformance+unit+mutation-self-check).
4. **Gate-2 кросс-семейный** (deepseek+glm на манифест): риск галлюна имени-поля; корректность fallback; не задет downstream.
5. **⚠️ ПРОГОН НА 40 (A/B, отдельная задача после гейта):** РАЗНООБРАЗНЫЕ категории (не только частые). Мерить:
   (a) **per-product fill-rate лифт** static-only vs static+self-heal; (b) **Gate-2D content**: self-healed
   маппинги имён КОРРЕКТНЫ (WB-имя → правильный Ozon-атрибут, не абы какой); (c) сколько LLM-вызовов/латентность/кэш-рост.
   Честное покрытие N/40, INFRA-skip отдельно.

## Вне скоупа
- `value_normalizer` (родной `resolve_value_id` сильнее), routing-таблица (cat/type уже есть), ре-экспорт снапшота
  (даёт ~ноль). WB-card DATA-quality (тот CPU-расхождение) — отдельный трек (FIX-16-стайл identity/content-гард на WB,
  если WB станет авторитетным).
- **Честный потолок:** частые категории (телефон/дрель) уже почти в потолке от статики → лифт от self-heal там
  скромный; главный выигрыш — длинный хвост. 40-набор ДОЛЖЕН быть разнообразным, иначе лифт не увидим.
