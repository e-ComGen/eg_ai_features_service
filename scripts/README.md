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
