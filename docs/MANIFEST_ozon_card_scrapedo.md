# MANIFEST — FIX-10: OzonCardSource fetch через scrape.do (не мёртвый Scrappey) (HARDEN)

## Проблема (rooted 2026-07-01, юзер + код)
`OzonCardSource` (`app/services/enrichment/sources/ozon_card_source.py`) фетчит Ozon-страницы
через **Scrappey**, который **ВЫПИЛЕН 2026-06-14** (`url_fetcher.py:348`: «scrape.do is the PRIMARY
anti-bot tier (Scrappey retired 2026-06-14)»). Поэтому Ozon-карта на живом пути «молчит» — не из-за
отсутствия `SCRAPPEY_KEY`, а потому что провайдер мёртв. Прод-скрейпер = **scrape.do**
(`SCRAPEDO_TOKEN`, есть в `.env`); на него уже переведены yandex/lamoda/generic/YandexMarket.

**Цель:** перевести весь фетч OzonCard на `scrapedo_fetch`, оживив Stage 0.5 (Ozon-карта).

## Клиент (готовый, НЕ трогать)
`app/services/providers/scrapedo_client.py`:
`async scrapedo_fetch(url, *, render=True, super_proxy=True, geo="ru", wait_ms=5000, timeout=120)
-> ScrapflyResult(success: bool, content: Optional[str]=HTML, status_code, credits_used, error)`.
Есть свои ретраи (429/5xx), свой httpx-клиент, self-gate на SCRAPEDO_TOKEN.

## Инвариант INV-10
OzonCard фетчит СТРАНИЦЫ (Ozon search + /features/) ТОЛЬКО через `scrapedo_fetch`. Парсинг HTML
(извлечение характеристик, `state-webCharacteristics` и т.п.) — **НЕ меняется** (тот же парсер, тот
же результат-контракт `list[AttributeValue]`). Меняется ТОЛЬКО транспорт фетча + гейт по токену.

## Где (точки правки, ozon_card_source.py)
1. **Гейт по токену:** `__init__` (стр 960) `self._scrappey_key = ... SCRAPPEY_KEY` →
   `self._scrapedo_token = os.environ.get("SCRAPEDO_TOKEN")` (сохранить backward-compat kwarg, но
   читать SCRAPEDO). `is_applicable` (стр 979) и `extract` (стр 990) — проверять `self._scrapedo_token`
   вместо `_scrappey_key`. Предупреждение (стр 962) — про SCRAPEDO_TOKEN.
2. **Фетч:** заменить реализацию `_scrappey_fetch(client, target_url, session)` (стр 1065-1138) на
   вызов `scrapedo_fetch`:
   ```
   res = await scrapedo_fetch(target_url, render=True, super_proxy=True, geo="ru")
   return res.content if (res.success and res.content) else None
   ```
   Сохранить СИГНАТУРУ метода (принимать `client`, `session`, игнорировать) ИЛИ, если чище,
   ввести `async def _fetch_page(self, target_url, session=None) -> Optional[str]` и переключить
   ВСЕ call-sites (стр 1257, 1299, 1366, 1780). Импорт `from app.services.providers.scrapedo_client
   import scrapedo_fetch` (модульный, вверху). НЕ создавать httpx-клиент под scrape.do (у него свой).
3. **Мёртвый код Scrappey — УДАЛИТЬ** (иначе vulture): `_scrappey_fetch_once` (стр 1140),
   `_build_scrappey_payload` (стр 131), `_SCRAPPEY_ENDPOINT` (стр 71), `_scrappey_key`-атрибут и любые
   Scrappey-only хелперы/константы, ставшие недостижимыми. httpx-`AsyncClient` контексты, которые
   существовали ТОЛЬКО ради Scrappey-фетча (стр ~1427/1465) — упростить/убрать, если после свапа
   не нужны (scrape.do держит свой клиент). НЕ трогать httpx, если он ещё нужен для другого.

## Инварианты (тесты)
- INV-10a: OzonCard.is_applicable/extract возвращают [] БЕЗ SCRAPEDO_TOKEN (self-gate сохранён).
- INV-10b: при наличии токена extract ВЫЗЫВАЕТ scrapedo_fetch (мокнуть scrapedo_fetch → проверить,
  что вызван с ожидаемым Ozon-URL и render/super/geo=ru; парсер получил мок-HTML и вернул значения).
- INV-10c: парсер-контракт не изменился — на фикстуре Ozon-HTML (мок) extract возвращает те же
  характеристики, что и раньше (парсинг не тронут).
- INV-10d: НИ одной ссылки на Scrappey/`publisher.scrappey.com`/`SCRAPPEY_KEY`/`_SCRAPPEY_ENDPOINT`
  не осталось в ozon_card_source.py (grep = 0).
- INV-10e: source_type == Source.OZON_CARD (не менялся).

## Oracle / Acceptance
1. **Unit:** INV-10a..e (мок scrapedo_fetch, мок-HTML фикстура Ozon /features/ с ≥3 характеристиками).
2. **⚠️ ЖИВОЙ смоук (ОБЯЗАТЕЛЬНО, task-class = content-from-source):** сконструировать OzonCardSource с
   реальным SCRAPEDO_TOKEN (load_dotenv), вызвать extract на реальном товаре (напр. «POCO X6 5G» или
   «Bosch GSB 13 RE») → подтвердить, что scrape.do РЕАЛЬНО вернул Ozon-HTML (success=True, content
   непустой) И парсер извлёк ≥1 характеристику. Если scrape.do возвращает пусто/404 на Ozon — это FAIL
   с диагнозом (не тихий PASS). Логировать credits_used. Это доказывает, что карта ОЖИЛА, а не просто
   «код компилируется».
3. Gate-1 полный (ruff+mypy+radon+vulture+bandit+conformance+unit+mutation-self-check).

## Вне скоупа
- WB-карта: она НЕ скрейпер (использует бесплатный WB basket-API + Serper) — «через scrape.do» к ней
  не применяется напрямую; её таймаут — отдельная тема. Сейчас WB выключена (`PIPELINE_WB_CARD_ENABLED
  =false`). Отдельно решить: перестроить WB на scrape.do-скрейп WB-страниц ИЛИ оставить выключенной.
- Парсер Ozon-HTML — НЕ трогать (только транспорт).
- Порядок стадий (FIX-9b, IceCat-first) — отдельно.
