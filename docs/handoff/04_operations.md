# Операционные заметки

## Окружение

- Репо движка: `C:\Users\Venya\PycharmProjects\CpAiFeatures-web-fetch`, ветка
  `feature/web-fetch`.
- venv: `.\venv\Scripts\python.exe`.
- ОС: Windows 10. Шелл: PowerShell (основной) + Bash-инструмент (POSIX).
- Кириллица в консоли крашит cp1251 → ВСЕГДА `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`.
  Вывод дампов писать в UTF-8 файл, а не в stdout.
- **Downloads перенаправлен на `E:\Загрузки`** (НЕ `C:\Users\Venya\Downloads` —
  его нет). xlsx от eg_importer (тостеры/наушники и т.п.) лежат там.

## Запуск воркера движка (:8002)

```
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```
Нужны env: `USE_NEW_PIPELINE=true PIPELINE_RICH_SOURCES=true PYTHONUTF8=1
PYTHONIOENCODING=utf-8`. Старт ~20–30с (грузит embedding-модель, vector-cache).
Нет роута `/health` (вернёт 404). **Рестарт воркера обычно делает eg_importer** —
он управляет жизненным циклом. После правок движка воркер ОБЯЗАТЕЛЬНО
перезапустить на новом коммите, иначе крутится старый код (классический симптом
«фикс не виден»).

## Соседние сервисы

- **ai_orchestrator :8000** — биллинг + `/manage/update/inspect` (анализ файла,
  ждёт POST; GET вернёт 405 — это «живой», не «сломан»).
- Внутренние вызовы к движку идут с заголовком `x-internal-secret:
  INTERNAL_SERVICE_SECRET` (читать из `app.config`, НЕ печатать).

## Как прозондировать движок (probe-паттерн)

`POST http://127.0.0.1:8002/process-batch` с заголовком `x-internal-secret`. Тело:
```json
{
  "client_id": 1, "use_cache": false,
  "options": {"marketplace": "ozon", "enable_vision": false, "enable_web_search": false},
  "schemas": {"<category_id>": {"<имя атрибута>": {"id": 4820, "type": "numeric", "is_required": false}}},
  "products": [{
    "id": 2, "category_id": <cat>, "name": "...", "description": "...",
    "brand": "...", "ozon_type_id": <type_id>,
    "context": {"existing_features": {}, "company_id": 0}
  }]
}
```
Ответ: `filled_features` / `debug_info{value_id,source}` / `skipped{reason}`.
Примеры готовых проб: `scripts/probe_toaster_api_backfill.py`,
`scripts/probe_toaster_slots.py`. Читать xlsx — openpyxl с
`load_workbook(path, data_only=True)`; лист «Шаблон» = данные, лист «validation»
= allowed-значения по индексу колонки.

## Гочи (грабли)

1. **Устаревшие DESCRIPTION_CATEGORY_ID в шаблонах eg_importer.** Лист `configs`
   xlsx содержит `DESCRIPTION_CATEGORY_ID` (тостер 47156221), который values-API
   Ozon ОТВЕРГАЕТ. Движок сам резолвит правильный dcid по `type_id` из живого
   дерева для backfill-пути. НО общий резолв `value_id` всё ещё использует
   `context.category_id` (шаблонный) → для stale-категорий он может писать
   неверный value_id. **Остаток работы** (см. 05_pending).
2. **Кэш `__EMPTY__`-сентинела.** Воркер кэширует no_data в `features_service.db`.
   Старые записи маскируют улучшения. Зондировать с `use_cache=false` / сбрасывать
   кэш.
3. **Делегированные Sonnet-агенты дефолтят в НЕВЕРНЫЙ репо** (`CpAiFeatures` без
   `-web-fetch`). Всегда указывать полный путь репо в промпте агента.
4. **Stale-процессы на портах** — мой временный воркер ловил Errno 10048 (порт
   занят stray PID). Чистить временные воркеры после проб.
5. **scrape.do НЕ пробивает Ozon** (антибот/502). Для Ozon-карточки — Scrappey
   (выключен) или Serper-card-finding.

## Незакоммиченный мусор в рабочем дереве

В корне/`scripts/`/`tests/` лежит много `?? ab_probe_*`, `manual_*_smoke*`,
`*.log`, `sh.exe.stackdump`, `_tmp_*` — это временные артефакты проб, НЕ
закоммичены. Перед мержем ветки их стоит вычистить.
