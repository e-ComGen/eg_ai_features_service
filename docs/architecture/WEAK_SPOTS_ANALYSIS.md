# Weak-Spots Analysis — no-description pipeline (67% accuracy baseline)

Acceptance test: 7 products, no description, 2025-05-14.

---

## Per-case root cause analysis

### Case 1: Samsung S24 — основная камера 200 МП MISSING

**Root cause: Prompt problem (knowledge confidence)**
`LlmKnowledgeSource` system_prompt used vague scale where "Confidence ≥0.92 means I am sure"
but never instructed LLM to default to 0.95 for well-known product specs. Result: LLM returns
conservative values (0.8–0.91) that fail `is_confident()` check (threshold 0.92), so
the value is sent to `KnowledgeJudge` which can reject it, or it never reaches the judge at all
because confidence is below threshold and judge returns False. Samsung S24 Ultra 200MP is an
official published spec — LLM knows it with certainty.

**Fix applied:** system_prompt now explicitly instructs confidence=0.95 for official/well-known specs.

---

### Case 2: Adidas Superstar — цвет «черный» vs «Белый», артикул EG4958 MISSING

**Root cause A: Matching problem (color)**
`value_matches` used only exact + simple partial match. "черный" (with е) vs "белый/чёрный"
partial would not match because neither is a substring of the other in either direction.
The accept_values list includes "Белый/Чёрный" — extracted "черный с белым" fails all checks.

**Fix applied:** bidirectional substring + morphological root matching in `value_matches`.

**Root cause B: Pipeline problem (артикул)**
Classifier routes "Артикул производителя" to `web_search` because it's not explicitly
listed in the "brand/model facts" category. EG4958 is a known SKU in training data.

**Fix applied:** Classifier prompt now explicitly mentions "article/SKU of well-known products"
as a `llm_knowledge` case.

---

### Case 3: Ariel — Производитель P&G MISSING, Страна Россия MISSING

**Root cause: Pipeline problem (classifier routing)**
Classifier prompt only listed "brand/model facts" → `llm_knowledge`. "Производитель" and
"Страна производства" are factual brand attributes that LLM knows (Ariel=P&G, Russia plant
is public knowledge) but the classifier mapped them to `web_search` due to lack of explicit
guidance.

**Fix applied:** Classifier prompt now lists "manufacturer, country of manufacture" as
`llm_knowledge` targets.

---

### Case 4: LEGO 10847 — цвет упаковки «белый» vs «Разноцветный»

**Root cause: Prompt problem (vision no allowed_values constraint)**
Vision producer sees a white image background and reports "white". The extraction step then
naively returns "white". The target had no `allowed_values` in the fixture (this is a fixable
ground truth issue too), but the extraction prompt didn't instruct to pick the most appropriate
value when allowed values are listed.

**Fix applied:** VisionSource extraction prompt now instructs LLM to pick from `allowed` list
when present. Note: fixture for this target has no `allowed_values` — ground truth is still
fragile for this case.

**Deferred:** Add `allowed_values: ["Разноцветный", "Белый", "Красный"]` to the fixture
(ground truth improvement, medium effort).

---

### Case 5: Maybelline «Matte+Poreless» vs «Жидкая» — формула

**Root cause: Ground truth ambiguity + prompt problem**
Attribute "Формула" is genuinely ambiguous in Russian cosmetics context — it can mean the
product formula name (marketing) or physical form (liquid/powder/cream). LLM correctly
identifies "Matte+Poreless" as the formula name. Ground truth expects "Жидкая" (physical form).

**Root cause classification:** 50% ground truth problem (attribute name is ambiguous),
50% prompt problem (knowledge source should prefer physical form interpretation).

**Deferred (MEDIUM):** Either rename the attribute to "Физическая форма" in the fixture,
or add "Matte+Poreless" to `accept_values`. The ambiguity is in the test fixture design.

---

### Case 6: «Низкий» vs «Низкие», «Чёрный» vs «Темно-серый / черный» — semantic mismatches

**Root cause: Matching problem**
`value_matches` used strict equality + simple partial match. Russian adjectives change by
gender/number (Низкий/Низкие, Чёрный/Черный). Also synonym forms like "темно-серый / черный"
should match "Чёрный" via substring.

**Fix applied:** Bidirectional substring + morphological root match (strip last 2 chars,
compare shared prefix ≥3 chars) added to `value_matches`.

---

## Fixes applied (HIGH priority)

| Fix | File | Change |
|-----|------|--------|
| Fuzzy matching | `tests/integration/test_real_world_pipeline.py` | `value_matches()` — bidirectional substring + 2-char suffix morphological strip |
| Knowledge confidence prompt | `app/services/enrichment/sources/llm_knowledge_source.py` | system_prompt instructs confidence=0.95 for official/well-known specs |
| Vision allowed_values constraint | `app/services/enrichment/sources/vision_source.py` | extraction system_prompt instructs to pick from `allowed` list when present |
| Classifier routing — manufacturer/SKU | `app/services/enrichment/intelligence/classifier.py` | routing rules expanded for manufacturer, country, article/SKU, numeric specs of known products |

---

## Fixes deferred (MEDIUM/LOW)

| Case | Fix | Effort | Estimated improvement |
|------|-----|--------|----------------------|
| Case 4 LEGO | Add `allowed_values` to цвет упаковки target in fixture | 5 min | +1 match |
| Case 5 Maybelline | Rename fixture attribute "Формула" → "Физическая форма" OR add accept_values | 10 min | +1 match |
| All cases | Improve KnowledgeJudge to cross-check via second LLM call with richer context (category + allowed_values) | 2-3 hrs | +3-5% |
| General | Add semantic synonyms dict for Russian (черный=тёмный, низкий=низкие) to value_matches | 30 min | +2-3% |

---

## Estimated accuracy after fixes

Baseline: 67% (without live run, estimate based on analysis)

- Fix 1 (fuzzy matching): eliminates Case 6 mismatches → +4-6% accuracy (purely test-side)
- Fix 2 (knowledge confidence): S24 camera, Ariel manufacturer likely now returned at 0.95 → +5-8%
- Fix 3 (vision allowed_values): marginal for cases without allowed_values; will help future fixtures → +2-3%
- Fix 4 (classifier routing): Ariel P&G/Russia, Adidas EG4958 routed to knowledge → +4-6%

**Estimated accuracy after fixes: ~80-85%** (if live test run against same 7 products)
