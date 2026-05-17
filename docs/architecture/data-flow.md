# Data flow

Какие данные передаются между стадиями pipeline. С JSON-примерами и Python type definitions.

---

## Input: что приходит в pipeline

```python
class ProductData(BaseModel):
    id: int                              # external (CS-Cart product_id)
    name: str                            # "Nike Air Force 1 размер 42"
    description: str | None              # может быть пустым
    category_id: int                     # 42
    category_path: list[str]             # ["Обувь", "Кроссовки", "Мужские"]
    brand: str | None                    # "Nike" — если CS-Cart хранит отдельно
    ean: str | None                      # "0194253401234"
    source_urls: list[str]               # ссылки на товар у поставщика (max 5)
    image_urls: list[str]                # фото товара (max 10)

class BatchOptions(BaseModel):
    enable_vision: bool = False
    enable_web_search: bool = False
    enable_llm_knowledge: bool = True    # самый дешёвый, default on
    max_cost_credits: int = 100          # бюджет на товар

class TargetAttribute(BaseModel):
    id: int                              # из схемы CS-Cart features
    name: str                            # "Material"
    type: Literal["text", "numeric", "enum"]
    allowed_values: list[str] | None     # для enum
    semantic_type: str | None            # "color", "material_visual", "weight", ...
                                          # подсказка classifier-у
```

JSON example:

```json
{
  "product": {
    "id": 12345,
    "name": "Nike Air Force 1 Размер 42",
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

```python
class Source(StrEnum):
    DESCRIPTION = "description"
    LLM_KNOWLEDGE = "llm_knowledge"
    VISION = "vision"
    WEB_SEARCH = "web_search"

class AttributeValue(BaseModel):
    attribute_id: int
    value: str | int | float | bool      # final value
    confidence: float                     # 0.0-1.0 от extractor
    source: Source                        # откуда пришло
    evidence: str | None                  # цитата / URL / описание основания
    judge_validated: bool = False         # прошёл ли через judge

class ClassifierDecision(BaseModel):
    """Что Classifier возвращает для каждого unfilled attr."""
    attribute_id: int
    suggested_sources: list[Source]       # упорядоченный список (cheapest first)
    reasoning: str                        # одна строка почему
```

---

## Data flow per stage

### Stage 0 → Stage 1 (description → classifier)

**Что Description вернула:**
```json
[
  {"attribute_id": 101, "value": "42", "confidence": 0.95, "source": "description", "evidence": "Размер 42 из названия"},
  {"attribute_id": 102, "value": "white", "confidence": 0.95, "source": "description", "evidence": "Классические + белый известный цвет AF1 — выводи описанием"}
]
```

**Coverage check:**
- target = `[101, 102, 103, 104, 105]`
- filled = `[101, 102]`
- **unfilled = `[103, 104, 105]`** → передаём в Classifier

**Classifier input:**
```json
{
  "product": { "name": "...", "category_path": [...], "brand": "Nike" },
  "unfilled_attributes": [
    {"id": 103, "name": "Материал", "semantic_type": "material_visual"},
    {"id": 104, "name": "Вес", "semantic_type": "weight"},
    {"id": 105, "name": "Бренд", "semantic_type": null}
  ]
}
```

**Classifier output:**
```json
[
  {"attribute_id": 103, "suggested_sources": ["vision", "web_search"], "reasoning": "Материал визуально определяем по фото; web search как fallback"},
  {"attribute_id": 104, "suggested_sources": ["web_search"], "reasoning": "Вес обычно есть в спецификациях производителя в web"},
  {"attribute_id": 105, "suggested_sources": ["llm_knowledge"], "reasoning": "Nike — общеизвестный бренд, LLM знает"}
]
```

### Stage 2 LLM Knowledge

**Input:** только attrs где `llm_knowledge` в suggested_sources → `[105]`.

**LLM prompt:** «What you know about: Nike Air Force 1. Fill these attributes: [Бренд]. Return JSON with confidence per value.»

**Output:**
```json
[
  {"attribute_id": 105, "value": "Nike", "confidence": 0.95, "source": "llm_knowledge"}
]
```

`confidence 0.95 ≥ 0.92` → skip knowledge_judge.

### Stage 3 Vision

**Input:** attrs где `vision` first in suggested_sources → `[103]`.

**Vision producer prompt + image_urls:**
> «Опиши товар на изображениях максимально подробно. Фокус на: цвет, материал, форма, размер...»

**Vision text output:**
> «На изображениях белые кроссовки с гладкой кожаной поверхностью верха, перфорация на боковых панелях, белая резиновая подошва...»

**Extraction LLM call** на этот text:

```json
[
  {"attribute_id": 103, "value": "Кожа", "confidence": 0.78, "source": "vision", "evidence": "гладкой кожаной поверхностью верха"}
]
```

`confidence 0.78 < 0.85` → запускаем `vision_judge`:

**Vision judge prompt:**
> «Текст из vision: "На изображениях белые кроссовки с гладкой кожаной поверхностью верха...". Атрибут: Материал = Кожа. Действительно ли этот вывод обоснован визуально? Yes/No.»

**Judge output:** `{"valid": true}` → attr принят.

### Stage 4 Web Search

**Cost predictor input:**
```json
{
  "product_name": "Nike Air Force 1",
  "unfilled_attributes": [{"id": 104, "name": "Вес"}]
}
```

**Cost predictor LLM prompt:**
> «Товар: Nike Air Force 1. Attrs которые нужно найти: вес. Вероятно ли что мы найдём эти данные в публичном web (Nike.com, отзывы, обзоры)? Yes/No + 1 строка причины.»

**Output:** `{"worth_it": true, "reason": "Nike публикует specs на официальном сайте"}` → запускаем WebSearch.

**Web search producer:** делает 1 LLM call с `web_search` tool, query: «Nike Air Force 1 weight specifications».

**Output text:**
> «По данным Nike.com, AF1 размера 42 весит 380 граммов на одну пару. Согласно отзывам на Wildberries: 400г каждая. Sneaker News: 350-400г.»

**Extraction:** `[{"attribute_id": 104, "value": "380", "confidence": 0.85, "source": "web_search", "evidence": "Nike.com официально"}]`

`0.85 < 0.88` → judge:

**Websearch judge prompt:**
> «Источники: Nike.com (official), Wildberries отзывы, Sneaker News. Заявленное значение: 380г. Соответствует ли наиболее credible источнику?»

Judge approves → attr принят.

### Final merge

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

## Output: что pipeline возвращает

```python
class ExtractionResult(BaseModel):
    request_id: str
    filled_attributes: list[AttributeValue]   # final values, по одному на attr_id
    unfilled_attributes: list[int]             # attr_ids которые не смогли заполнить
    total_llm_calls: int                       # для audit
    total_cost_credits: int                    # для billing
    stage_breakdown: dict[str, int]            # {"description": 3, "classifier": 1, "knowledge": 1, ...}
    audit_trail: list[dict]                    # каждое LLM-decision для debugging
```

JSON:
```json
{
  "request_id": "req_abc123",
  "filled_attributes": [ /* как выше */ ],
  "unfilled_attributes": [],
  "total_llm_calls": 11,
  "total_cost_credits": 18,
  "stage_breakdown": {
    "description": 3, "classifier": 1, "knowledge": 1,
    "vision_producer": 1, "vision_extract": 1, "vision_judge": 1,
    "cost_predictor": 1, "websearch_producer": 1, "websearch_extract": 1
  },
  "audit_trail": [
    {"stage": "description", "input_hash": "abc", "output_attrs": [101, 102], "duration_ms": 1240, "model": "gpt-4o-mini"},
    /* ... */
  ]
}
```
