# MANIFEST — FIX-9 source cascade: карты всегда, авторитеты раньше маркетплейсов (HARDEN)

## Проблема (rooted 2026-07-01, трассировка живого пути)
Живой путь :8002 /process-batch идёт через `PipelineAdapter()` (main.py:206/296, job_processor.py:100).
Адаптер конструирует `PipelineOrchestrator(...)` в ДВУХ местах (pipeline_adapter.py:100 и 183) и
**НЕ передаёт `ozon_card_source`** → в оркестраторе `self._ozon_card = None` (pipeline.py:4218, без
дефолта) → **Stage 0.5 OzonCard живьём НИКОГДА не запускается.** WbCard+IceCat — за флагом
`PIPELINE_RICH_SOURCES=false` (pipeline_adapter.py:42), тоже OFF. Единственный always-on card-tier
источник — `web_search` (pipeline.py:4187, дефолт `WebSearchSource()`).

**Следствие:** живьём web_search — де-факто ПЕРВЫЙ реальный источник ТТХ. Из-за stop-on-found
(`if not remaining: return`) он заполняет слоты раньше карт, а на сравнит-страницах хватает ТТХ
СОСЕДНЕГО варианта (POCO 2560×1440 вместо 2712×1220, Adreno 610 вместо 710). Карты (WB/Ozon/IceCat),
которые дают ТТХ ИМЕННО этого товара, отключены.

Пользовательская директива: **всегда смотреть карты маркетплейсов; порядок — авторитеты/IceCat →
маркетплейсы (WB/Ozon) → остальное → LLM в конце.**

## Стоимость (честно, one-line surface)
Always-on карты = реальные затраты на КАЖДЫЙ товар: OzonCard ~2 Scrappey-кредита, WbCard 1 Serper +
N fetch, IceCat 1 API. Латентность +10-40s/товар (уже ограничена `_TimeoutSource`
RICH_SOURCE_TIMEOUT_S=25s и WB_CARD_MAX_FETCHED). Пользователь принял ради точности.

---

## FIX-9a — провести карты (карта всегда) [LOW RISK, аддитивно]

### INV-9A-1 (Ozon-карта проведена в обеих сборках адаптера)
Обе `PipelineOrchestrator(...)` в pipeline_adapter.py (стр 100 и 183) получают `ozon_card_source=<timeout-wrapped OzonCardSource>`.

### INV-9A-2 (карты включены по умолчанию, но self-gate на ключах)
`_RICH_SOURCES_ENABLED` дефолт → `true` (pipeline_adapter.py:42: `os.getenv("PIPELINE_RICH_SOURCES", "true")`).
OzonCard конструируется ВСЕГДА (не за rich-флагом — у него свой self-gate на SCRAPPEY_KEY:
`is_applicable`/`extract` возвращают [] без ключа, ozon_card_source.py:960/976/990). WbCard+IceCat —
за rich-флагом (дефолт true теперь). Все три no-op без соответствующих ключей → безопасно на dev.

### Где (точки правки, pipeline_adapter.py)
- `__init__` (после блока rich-sources ~строка 95): сконструировать
  `self._ozon_card = _TimeoutSource(OzonCardSource(), _RICH_SOURCE_TIMEOUT_S, "OzonCard")`
  (импорт `OzonCardSource` внутри, как WbCard). OzonCard проводить БЕЗ rich-флага (always), т.к.
  self-gate на SCRAPPEY_KEY.
- стр 100 конструкция: добавить `ozon_card_source=self._ozon_card`.
- стр 183 конструкция (marketplace-specific): добавить `ozon_card_source=self._ozon_card`.
- стр 42: дефолт "false" → "true".

### Инварианты
- INV-9A-3: без SCRAPPEY_KEY/SERPER_KEY поведение = как сейчас (все карты []-no-op, ни один тест не падает).
- INV-9A-4: `_TimeoutSource` гарантирует, что зависшая карта не валит батч (уже реализовано).
- INV-9A-5: изменения ЛОКАЛЬНЫ в pipeline_adapter.py; сигнатуры оркестратора не менялись (ozon_card_source уже принимается, стр 4164).

### Oracle (input → expected)
| # | условие | expected |
|---|---|---|
| O1 | PipelineAdapter() c SCRAPPEY_KEY в env | `adapter._ozon_card is not None` И оркестратор получил его (Stage 0.5 активна) |
| O2 | PipelineAdapter() БЕЗ ключей | конструируется без ошибок; карты []-no-op; существующие тесты зелены |
| O3 | env PIPELINE_RICH_SOURCES unset | `_RICH_SOURCES_ENABLED is True` (дефолт true) |
| O4 | env PIPELINE_RICH_SOURCES=false | `_RICH_SOURCES_ENABLED is False` (явный override работает) |

### Acceptance
Unit: O1-O4 (мокнуть env/ключи; проверить, что оркестратор получил ozon_card_source — через
инспекцию `adapter._orch._ozon_card is not None`). НЕ-регресс: весь существующий
tests/ (особенно pipeline_adapter / pipeline orchestrator) зелёный. Gate-1 полный.

---

## FIX-9b — IceCat/авторитеты ПЕРЕД маркетплейсами [MED RISK, отдельным раундом ПОСЛЕ 9a]

### Проблема
Под stop-on-found карта маркетплейса (WB/Ozon, стр 4448/4467), заполнив слот, лишает IceCat
(авторитетный brand-API, стр 4546 Stage 0.55) шанса дать ТОЧНОЕ значение этого товара. IceCat
точнее маркетплейс-карты (данные производителя vs «похожий товар»).

### INV-9B-1 (порядок)
IceCat-стадия исполняется ПЕРЕД WbCard(0.45) и OzonCard(0.5). Целевой порядок card-tier:
Description → Barcode → **IceCat** → WbCard → OzonCard → … .

### Зависимость (проверить при имплементации)
IceCat матчит по EAN/бренд+MPN. EAN приходит из BarcodeSource (Stage 0.46, стр 4432, ДО карт) и/или
GTINResolver (Stage 0.54, стр 4518, ПОСЛЕ карт). Если IceCat строго нужен EAN, а Barcode его не дал —
перенести GTINResolver ТОЖЕ перед IceCat. Имплементатор ВЕРИФИЦИРУЕТ: работает ли IceCat на
brand+model из product_name без GTIN-резолвера; если нет — двигать GTIN+IceCat вместе.

### Acceptance
НЕ-регресс существующих тестов + ordering-тест (замокать источники, проверить, что IceCat.extract
вызывается до WbCard.extract при непустом remaining). Живой смоук на 2-3 товарах с известным EAN
(подтверждающий, не блокирующий — как FIX-7: card-путь недетерминирован по доступности).

### ВНЕ СКОУПА 9b
- «LLM в самом конце»: llm_knowledge (Stage 2, стр 4664) идёт до web_search, НО (1) classifier
  роутит каждый target ОДНОМУ основному источнику, (2) merge SOURCE_PRIORITY (FIX-1) уже ранжирует
  WEB_SEARCH(2)>LLM_KNOWLEDGE(1), (3) есть adversarial 4.9. Слепая перестановка routing рискованна.
  Отдельная задача с трассировкой, НЕ в 9a/9b. Директиву «LLM last» на 90% уже обеспечивает
  SOURCE_PRIORITY + то, что карты теперь идут ПЕРВЫМИ (FIX-9a).
