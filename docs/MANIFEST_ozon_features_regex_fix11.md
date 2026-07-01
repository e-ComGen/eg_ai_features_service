# MANIFEST — FIX-11: Ozon _FEATURES_STATE_RE принимает обе кавычки (HARDEN)

## Проблема (rooted 2026-07-01, живой смоук FIX-10)
scrape.do успешно тянет Ozon-страницы (FIX-10 доказал: 24 плитки, exact-match товар, картинки),
но парсер характеристик даёт 0. Причина — Ozon сменил разметку виджета характеристик:
`data-state='<JSON>'` (одинарная кавычка, raw JSON) → `data-state="<HTML-escaped JSON>"` (двойная
кавычка, внутренние `"` как `&quot;`). Регекс `_FEATURES_STATE_RE`
(`app/services/enrichment/sources/ozon_card_source.py:94`) матчит ТОЛЬКО одинарную кавычку → 0
характеристик. Ломало бы и Scrappey (баг парсера, не транспорта).

## INV-11
Парсер извлекает характеристики из ОБОИХ вариантов разметки:
- `data-state='<raw JSON>'` (старый, одинарная) — как раньше.
- `data-state="<HTML-escaped JSON>"` (новый, двойная) — `html.unescape()` перед `json.loads`.
Никакой другой логики парсинга/маппинга не менять.

## Где (ozon_card_source.py)
1. **Регекс** (стр 94-95): расширить на обе кавычки, ДВЕ группы:
   ```
   _FEATURES_STATE_RE = re.compile(
       r'<div\s+id="state-webCharacteristics-[^"]+"\s+data-state='
       r'(?:\'([^\']+)\'|"([^"]+)")',
   )
   ```
   (группа 1 = одинарная/raw, группа 2 = двойная/escaped).
2. **Разбор** (стр ~1395, цикл `for raw in _FEATURES_STATE_RE.findall(html)`): перейти на `finditer`:
   ```
   import html as _html  # модульный импорт вверху
   for m in _FEATURES_STATE_RE.finditer(html):
       raw = m.group(1) if m.group(1) is not None else _html.unescape(m.group(2))
       ...json.loads(raw)...
   ```
   Логику после `json.loads` (извлечение characteristics/values) НЕ трогать.
3. Обновить docstring-примеры (стр 16, 1384), упоминающие только одинарную кавычку.

## Инварианты (тесты)
- INV-11a: фикстура со СТАРОЙ разметкой (`data-state='{...}'`, raw) → те же характеристики, что и до фикса (регресс).
- INV-11b: фикстура с НОВОЙ разметкой (`data-state="{&quot;...&quot;}"`, escaped) → характеристики извлечены (≥N).
- INV-11c: обе разметки в одном HTML → обе распарсены.
- INV-11d: html.unescape применяется ТОЛЬКО к двойным кавычкам; одинарная ветка не мутируется (raw JSON с `&` в значении не сломается).
- INV-11e: битый JSON в data-state не роняет extract (существующая обработка ошибок сохранена).

## Acceptance
1. **Unit:** INV-11a..e (фикстуры обеих разметок; реальный фрагмент escaped-HTML из живого смоука FIX-10, если сохранён в scratchpad `_ozon_*` — взять оттуда настоящий пример).
2. **⚠️ ЖИВОЙ смоук (ОБЯЗАТЕЛЕН):** OzonCardSource с реальным SCRAPEDO_TOKEN → extract на реальном
   товаре («Bosch GSB 13 RE» и «POCO X6 5G»). Подтвердить: теперь извлечено **≥1 характеристики** с
   реальной Ozon-карточки (то, что FIX-10 не смог из-за регекса). Логировать сколько char + credits.
   0 характеристик = FAIL с диагнозом (какая разметка пришла), НЕ тихий PASS.
3. Gate-1 полный.

## Вне скоупа
- Транспорт (FIX-10, уже сделан). Маппинг характеристик на Ozon-словарь — не трогать.
