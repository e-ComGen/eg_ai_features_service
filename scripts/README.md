# Marketplace Dictionary Builders

## WB Dictionary

Скачивает категорийные словари Wildberries без регистрации и без токенов.
Использует только публичные endpoints.

### Запуск

```bash
cd C:/Users/Venya/PycharmProjects/CpAiFeatures-web-fetch
venv/Scripts/python -m scripts.build_wb_dictionary --samples 30 --resume
```

### Параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--samples N` | 30 | Количество sample карточек на категорию |
| `--output PATH` | `app/services/enrichment/strategies/dictionaries/data/wb_dictionary.json` | Путь итогового JSON |
| `--resume` | off | Продолжить с `.partial` файла если прерван |

### Длительность

~6–12 часов для всех ~1200 категорий (rate limit 0.3 сек между запросами ≈ 3 req/sec).
Прогресс сохраняется автоматически каждые 100 карточек в `{output}.partial`.

### Результат

`app/services/enrichment/strategies/dictionaries/data/wb_dictionary.json`

Структура:
```json
{
  "12345": {
    "name": "Кроссовки мужские",
    "path": ["Обувь", "Мужская обувь", "Кроссовки"],
    "characteristics": [
      {"id": 14177419, "name": "Цвет"},
      {"id": 14177421, "name": "Материал верха"}
    ]
  }
}
```

### Обновление

Запускать раз в квартал для refresh словаря.

## Cost

0 ₽ — публичные endpoints WB без авторизации.

---

## Ozon Dictionary

Собирает категорийные словари Ozon через Playwright (headless Chrome).
Playwright нужен для обхода Cloudflare — прямые httpx-запросы блокируются.

### Установка (один раз)

```bash
pip install playwright>=1.40.0
playwright install chromium  # ~130 MB
```

### Запуск

```bash
cd C:/Users/Venya/PycharmProjects/CpAiFeatures-web-fetch
venv/Scripts/python scripts/build_ozon_dictionary.py --samples 20 --resume
```

### Параметры

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--samples N` | 20 | Количество sample страниц товаров на категорию |
| `--output PATH` | `app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json` | Путь итогового JSON |
| `--resume` | off | Продолжить с `.partial` файла если прерван |

### Длительность

8–15 часов для всех категорий (Playwright медленнее httpx; rate limit 2 сек между запросами).
Прогресс сохраняется каждые 50 страниц в `{output}.partial`.

### Результат

`app/services/enrichment/strategies/dictionaries/data/ozon_dictionary.json`

Структура:
```json
{
  "502": {
    "name": "Смартфоны",
    "path": ["Электроника", "Смартфоны"],
    "characteristics": [
      {"key": "brand", "name": "Бренд"},
      {"key": "color", "name": "Цвет"}
    ]
  }
}
```

### Seed данные

По умолчанию используется встроенный mini-seed (20 топ-категорий) или данные из
[welel/ozon-scraper](https://github.com/welel/ozon-scraper) (кэшируются в
`scripts/build_ozon_dictionary_lib/_welel_cache.json` после первого fetch).

> **TODO**: Структура welel/ozon-scraper может требовать ручной адаптации
> в `seed_loader._flatten_tree()` — проверить реальные имена полей в репо.

### Cost

0 ₽ — публичные страницы Ozon, без API токенов.
