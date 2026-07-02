# MANIFEST — FIX-13-WB: WB-поиск через scrape.do + актуальный basket-NN (HARDEN)

## Проблема (rooted 2026-07-02, две живые пробы)
WbCardSource выключен (`PIPELINE_WB_CARD_ENABLED=false`), т.к. на живом пути таймаутил и давал 0
заполнений. Корень — ДВА независимых звена (доказаны):
1. **Поиск nm_id через Serper** (`_search_once`, стр 1089): `site:wildberries.ru <q> detail.aspx` в
   Serper флачит при concurrency (троттлинг/пустые ответы) → 3 попытки с бэкоффом → медленно/ненадёжно.
   Проба: **search.wb.ru через scrape.do отдаёт 100 товаров/10 кредитов**, прямой запрос ловит 429
   (нужен scrape.do). Транспорт ТТХ (card.json CDN) — жив и бесплатен, менять НЕ надо.
2. **basket-NN устарел** (`_ALL_BASKET_NN`, стр 255 = range(1,22); fallback `_BASKET_DEFAULT="21"`):
   каталог вырос, новые артикулы шардятся на **basket-37+**. Проба: nm_id 815621985/823775519/823776306
   (vol 8156) все на **basket-37**, старый код (капает на 21) → 404 по всем → 0 ТТХ.

## Цель
Оживить WB: поиск nm_id — через scrape.do+search.wb.ru (выкинуть флакающий Serper из hot-path);
card.json — расширить basket-NN до 40. Парсинг/выбор/маппинг card.json — НЕ трогать (уже работают).

## INV-13 (что менять — ДВЕ точки)
### 1. `_search_once` (стр 1089) — Serper → scrape.do+search.wb.ru
Контракт МЕТОДА не меняется: `async _search_once(query: str) -> list[int]` (top-N уникальных nm_id).
`_search` (retry-обёртка, стр 1049) и весь downstream (`_fetch_card`/`_pick_best_card`/`_map`) — БЕЗ ИЗМЕНЕНИЙ.
- URL: `https://search.wb.ru/exactmatch/ru/common/v5/search?appType=1&curr=rub&dest=-1257786&query={urlencode(query)}&resultset=catalog&sort=popular&spp=30`
- Фетч: `from app.services.providers.scrapedo_client import scrapedo_fetch` (модульный импорт вверху);
  `res = await scrapedo_fetch(url, render=False, super_proxy=True, geo="ru")`.
- Парс: `res.content` может быть чистый JSON ИЛИ обёрнут в HTML — РОБАСТНО извлечь JSON (попытка
  `json.loads`; если не вышло — вырезать от первого `{` до парного `}` / regex на JSON-тело). Взять
  `data.products[]`, из каждого `id` (int, это nm_id), сохранить порядок (search уже сортит по
  popular). Дедуп, `[:_MAX_CANDIDATES]` (=10).
- `query` подаётся ЧИСТЫЙ (как строит query-builder, с тип-словом) — БЕЗ `site:`/`detail.aspx`
  операторов (это был Serper-специфик).
- Ошибка/пусто scrape.do (`not res.success` или 0 products) → graceful `return []` (как раньше;
  выше по стеку это fallback, не падение). Логировать success/status/credits/N-products, как было для Serper.
- Если после свапа `_search_client` (Serper) больше НЕ используется для nm_id — убрать его wiring/импорт,
  чтобы vulture был чист (мёртвый Serper-код не оставлять). НЕ ломать `_search` retry-контракт.

### 2. basket-NN диапазон 21 → 40 (стр 231/255/546)
- `_ALL_BASKET_NN` → `[f"{n:02d}" for n in range(1, 41)]` (01..40).
- `_BASKET_THRESHOLDS` / `_basket_nn_from_table` / `_BASKET_DEFAULT`: обновить так, чтобы высокие vol
  (напр. 8156) резолвились в primary_nn БЛИЖЕ к реальному (basket-37), а не 21 — уменьшить брутфорс.
  Корректность обеспечивает расширенный range (брутфорс всё равно найдёт 37); таблица — оптимизация
  латентности. `_fetch_card`/`_try_basket` (1158-1198) логику НЕ менять, только константы.

## Инварианты (тесты)
- INV-13a: `_search_once` мокнутый scrapedo_fetch (JSON-фикстура search.wb.ru с products[].id) →
  вернул ожидаемые nm_id (top-N, дедуп, порядок). НЕ дёргает Serper.
- INV-13b: JSON внутри HTML-обёртки → тоже распарсен (робастность).
- INV-13c: scrapedo_fetch fail/0 products → `[]` (graceful, без исключения).
- INV-13d: `_fetch_card` находит card.json на basket-37 (мок `_try_basket`: 200 только на "37") —
  доказать, что диапазон покрывает >21.
- INV-13e: `_extract_options`/`_pick_best_card`(тип-гейт)/`_map_characteristics` — БАЙТ-В-БАЙТ прежние
  (diff не трогает парс/выбор/маппинг).
- INV-13f: vulture чист — мёртвого Serper-кода (если `_search_client` осиротел) не осталось.

## Oracle / Acceptance
1. **Unit:** INV-13a..f (мок scrapedo_fetch + мок card.json basket-37).
2. **⚠️ ЖИВОЙ смоук (ОБЯЗАТЕЛЕН, task-class = content-from-source):** `WbCardSource` с реальным
   SCRAPEDO_TOKEN → `extract` на «POCO X6 5G». Подтвердить ЦЕПОЧКУ: search.wb.ru (scrape.do) дал nm_id →
   card.json скачан с **basket-37+** → извлечено **≥5 характеристик** реального телефона (тип-гейт выбрал
   смартфон, не аксессуар). Логировать nm_id, basket-NN, N-char, credits. 0 char / только аксессуар =
   FAIL с диагнозом, НЕ тихий PASS.
3. Gate-1 полный (ruff+mypy+radon+vulture+bandit+conformance+unit+mutation-self-check).

## Вне скоупа
- **×10 unit-баг габаритов** (толщина 8.8→88мм) — в `_map_characteristics` (~1830, cm→mm конверсия),
  card.json отдаёт «9 мм» верно → баг в маппинге. Отдельный **FIX-14** (WB в проде off → не срочно).
- **Re-enable WB в проде** (`PIPELINE_WB_CARD_ENABLED`) — деплой-решение, не код (после FIX-14 + прод-e2e).
- Ozon-карта (FIX-10/11/12, готово). Парс/выбор/маппинг WB card.json — не трогать.
