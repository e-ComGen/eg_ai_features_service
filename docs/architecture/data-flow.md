# Как данные ходят между этапами

Документ показывает что именно передаётся между этапами pipeline. С реальными JSON-примерами и Python-типами — чтобы было понятно «что входит, что выходит».

---

## Что приходит на вход pipeline

Снаружи (от PHP-аддона или другого коннектора) приходит товар плюс список характеристик которые нужно заполнить.

```python
class ProductData(BaseModel):
    """Данные одного товара."""
    id: int                              # внешний id (из CS-Cart product_id)
    name: str                            # "Nike Air Force 1 размер 42"
    description: str | None              # может быть пустым
    category_id: int                     # 42
    category_path: list[str]             # ["Обувь", "Кроссовки", "Мужские"]
    brand: str | None                    # "Nike" — если в CS-Cart хранится отдельно
    ean: str | None                      # штрихкод "0194253401234"
    source_urls: list[str]               # ссылки на товар у поставщика (макс 5)
    image_urls: list[str]                # фото товара (макс 10)

class BatchOptions(BaseModel):
    """Какие этапы вообще запускать. Селлер сам выбирает в настройках."""
    enable_vision: bool = False          # анализ фото (дорогой)
    enable_web_search: bool = False      # поиск в интернете (самое дорогое)
    enable_llm_knowledge: bool = True    # знания ИИ — самое дешёвое, по умолчанию вкл
    max_cost_credits: int = 100          # бюджет на один товар (защита от перерасхода)

class TargetAttribute(BaseModel):
    """Одна характеристика которую нужно заполнить."""
    id: int                              # id из схемы характеристик CS-Cart
    name: str                            # "Материал"
    type: Literal["text", "numeric", "enum"]
    allowed_values: list[str] | None     # для enum — допустимые значения
    semantic_type: str | None            # подсказка для распределителя:
                                          # "color" → лучше через фото
                                          # "weight" → лучше через веб
                                          # "brand" → лучше через знания ИИ
```

JSON пример полного запроса:

```json
{
  "product": {
    "id": 12345,
    "name": "Nike Air Force 1 размер 42",
    "description": "Классические кожаные кроссовки",
    "category_id": 42,
    "category_path": ["Обувь", "Кроссовки"],
    "brand": "Nike",
    "ean": null,
    "source_urls": ["https://nike.com/en/AF1-42"],
    "image_urls": [
      "https://cdn.shop.com/af1_main.jpg",
      "https://cdn.shop.com/af1_side.jpg"
    ]
  },
  "options": {
    "enable_vision": true,
    "enable_web_search": true,
    "max_cost_credits": 50
  },
  "target_attributes": [
    {"id": 101, "name": "Размер", "type": "enum", "allowed_values": ["40","41","42"]},
    {"id": 102, "name": "Цвет", "type": "enum", "semantic_type": "color"},
    {"id": 103, "name": "Материал", "type": "text", "semantic_type": "material_visual"},
    {"id": 104, "name": "Вес", "type": "numeric", "semantic_type": "weight"},
    {"id": 105, "name": "Бренд", "type": "text"}
  ]
}
```

---

## Внутренние типы pipeline

Это типы которые ходят между этапами внутри pipeline.

```python
class Source(StrEnum):
    """Откуда пришла характеристика."""
    DESCRIPTION = "description"          # из описания товара
    LLM_KNOWLEDGE = "llm_knowledge"      # из знаний ИИ
    VISION = "vision"                    # с фото
    WEB_SEARCH = "web_search"            # из веб-поиска

class AttributeValue(BaseModel):
    """Одно значение характеристики которое нашёл какой-то источник."""
    attribute_id: int
    value: str | int | float | bool      # само значение
    confidence: float                     # 0.0 - 1.0, уверенность ИИ
    source: Source                        # откуда пришло
    evidence: str | None                  # обоснование: цитата / URL / описание
    judge_validated: bool = False         # прошёл ли через судью

class ClassifierDecision(BaseModel):
    """Что Распределитель решает для каждой ненайденной характеристики."""
    attribute_id: int
    suggested_sources: list[Source]       # список источников по порядку (сначала дешевле)
    reasoning: str                        # одна строка объяснения почему
```

---

## Поток данных по этапам (на нашем примере)

### Этап 0 — Парсер описания работает над тем что есть

**Что есть на входе:** описание `"Классические кожаные кроссовки"` + 5 характеристик которые нужно заполнить.

**Что Парсер вернул:**
```json
[
  {
    "attribute_id": 101,
    "value": "42",
    "confidence": 0.95,
    "source": "description",
    "evidence": "Размер 42 взят из названия товара"
  },
  {
    "attribute_id": 102,
    "value": "white",
    "confidence": 0.95,
    "source": "description",
    "evidence": "Классические + известный цвет AF1"
  }
]
```

**Проверка заполнения:**
- Нужно было: характеристики `[101, 102, 103, 104, 105]`
- Заполнили: `[101, 102]`
- **Осталось: `[103, 104, 105]`** → передаём Распределителю

### Этап 1 — Распределитель решает где искать остальное

**Распределителю на вход:**
```json
{
  "product": {
    "name": "Nike Air Force 1 размер 42",
    "category_path": ["Обувь", "Кроссовки"],
    "brand": "Nike"
  },
  "unfilled_attributes": [
    {"id": 103, "name": "Материал", "semantic_type": "material_visual"},
    {"id": 104, "name": "Вес", "semantic_type": "weight"},
    {"id": 105, "name": "Бренд", "semantic_type": null}
  ]
}
```

**Распределитель отвечает:**
```json
[
  {
    "attribute_id": 103,
    "suggested_sources": ["vision", "web_search"],
    "reasoning": "Материал хорошо виден на фото; веб как запасной вариант"
  },
  {
    "attribute_id": 104,
    "suggested_sources": ["web_search"],
    "reasoning": "Вес обычно есть в спецификации производителя в интернете"
  },
  {
    "attribute_id": 105,
    "suggested_sources": ["llm_knowledge"],
    "reasoning": "Nike — общеизвестный бренд, ИИ его знает"
  }
]
```

### Этап 2 — Знания ИИ

**Берём только характеристики где `llm_knowledge` идёт первым** → `[105]` (Бренд).

**Промпт к ИИ:**
> «Что ты знаешь про товар: Nike Air Force 1. Заполни характеристики: [Бренд]. Верни JSON с уверенностью по каждому значению.»

**ИИ отвечает:**
```json
[
  {
    "attribute_id": 105,
    "value": "Nike",
    "confidence": 0.95,
    "source": "llm_knowledge"
  }
]
```

`Уверенность 95% ≥ порога 92%` → судью не зовём, доверяем.

### Этап 3 — Фото

**Берём характеристики где `vision` идёт первым** → `[103]` (Материал).

**Промпт Vision-производителя** (с прикреплёнными фото):
> «Опиши товар на изображениях максимально подробно. Сфокусируйся на: цвет, материал, форма, размер, видимые надписи.»

**Текст от Vision:**
> «На изображениях белые кроссовки с гладкой кожаной поверхностью верха, перфорацией на боковых панелях, белой резиновой подошвой...»

**Промпт извлечения характеристик из этого текста:**
> «Из следующего текста извлеки: Материал. Текст: [текст выше]. Верни JSON с уверенностью.»

**Результат извлечения:**
```json
[
  {
    "attribute_id": 103,
    "value": "Кожа",
    "confidence": 0.78,
    "source": "vision",
    "evidence": "гладкой кожаной поверхностью верха"
  }
]
```

`Уверенность 78% < порога 85%` → запускаем `vision_judge`:

**Промпт судьи фото:**
> «Текст от vision: "На изображениях белые кроссовки с гладкой кожаной поверхностью верха...". Значение которое извлекли: Материал = Кожа. Действительно ли этот вывод обоснован тем что видно на фото? Yes/No.»

**Судья отвечает:** `{"valid": true}` → характеристика принята, помечаем `judge_validated: true`.

### Этап 4 — Веб-поиск

Сначала Прогнозист стоимости:

**Прогнозисту на вход:**
```json
{
  "product_name": "Nike Air Force 1",
  "unfilled_attributes": [{"id": 104, "name": "Вес"}]
}
```

**Промпт Прогнозиста:**
> «Товар: Nike Air Force 1. Характеристики которые нужно найти: вес. Есть ли разумная вероятность что мы найдём эти данные в публичном интернете (сайт Nike, отзывы, обзоры)? Yes/No + одна строка причины.»

**Прогнозист отвечает:** `{"worth_it": true, "reason": "Nike публикует характеристики на официальном сайте"}` → запускаем веб-поиск.

**Веб-поиск делает 1 обращение к ИИ с инструментом `web_search`**, запрос: «Nike Air Force 1 weight specifications».

**Текст который вернулся:**
> «По данным Nike.com, AF1 размера 42 весит 380 граммов на одну пару. Согласно отзывам на Wildberries: 400г каждая. Sneaker News: 350-400г.»

**Извлечение характеристик:**
```json
[
  {
    "attribute_id": 104,
    "value": "380",
    "confidence": 0.85,
    "source": "web_search",
    "evidence": "Nike.com официально указывает 380г"
  }
]
```

`Уверенность 85% < порога 88%` → судья:

**Промпт судьи веб-поиска:**
> «Источники: Nike.com (официальный), Wildberries отзывы, Sneaker News. Заявленное значение: 380г. Соответствует ли это наиболее достоверному источнику?»

Судья одобряет → характеристика принята.

### Финальное объединение

Все характеристики собраны:

```json
[
  {"attribute_id": 101, "value": "42", "source": "description", "confidence": 0.95},
  {"attribute_id": 102, "value": "white", "source": "description", "confidence": 0.95},
  {"attribute_id": 103, "value": "Кожа", "source": "vision", "confidence": 0.78, "judge_validated": true},
  {"attribute_id": 104, "value": "380", "source": "web_search", "confidence": 0.85, "judge_validated": true},
  {"attribute_id": 105, "value": "Nike", "source": "llm_knowledge", "confidence": 0.95}
]
```

---

## Что pipeline возвращает наружу

```python
class ExtractionResult(BaseModel):
    """Что вернётся обратно в PHP-аддон или коннектор."""
    request_id: str                            # для трейсинга
    filled_attributes: list[AttributeValue]    # заполненные характеристики
    unfilled_attributes: list[int]             # id характеристик которые не смогли заполнить
    total_llm_calls: int                       # сколько всего обращений к ИИ сделали
    total_cost_credits: int                    # сколько кредитов потратили
    stage_breakdown: dict[str, int]            # сколько на каждой стадии
                                                # для аудита и оптимизации
    audit_trail: list[dict]                    # каждое решение ИИ для отладки
```

JSON пример итога:
```json
{
  "request_id": "req_abc123",
  "filled_attributes": [ /* как выше */ ],
  "unfilled_attributes": [],
  "total_llm_calls": 11,
  "total_cost_credits": 18,
  "stage_breakdown": {
    "description": 3,
    "classifier": 1,
    "knowledge": 1,
    "vision_producer": 1,
    "vision_extract": 1,
    "vision_judge": 1,
    "cost_predictor": 1,
    "websearch_producer": 1,
    "websearch_extract": 1
  },
  "audit_trail": [
    {
      "stage": "description",
      "input_hash": "abc",
      "output_attrs": [101, 102],
      "duration_ms": 1240,
      "model": "gpt-4o-mini"
    }
    /* ... остальные шаги ... */
  ]
}
```

Эта подробная информация в `audit_trail` нужна для:
- Отладки когда что-то пошло не так
- Понимания на каких товарах какие этапы срабатывают чаще
- Оптимизации тарифов (видим где реальный расход)
- Споров с клиентом если он жалуется на качество
