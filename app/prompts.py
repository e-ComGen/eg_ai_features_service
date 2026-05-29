# app/prompts.py

# --- 1. DIRECT VALUE (Числа с юнитами) ---
# Максимально просто: "Найди число рядом с юнитом".
DIRECT_NUMERIC_PROMPT = """
Task: Extract the numeric value for '{feature_name}'.
Target Unit: '{unit}'
Context: '{context}'

INSTRUCTIONS:
1. Find a number in the text that is explicitly associated with '{unit}' (e.g. "500 {unit}").
2. Ignore numbers associated with other units.
3. Return ONLY the number. If not found, return NULL.
"""

# --- 2. DIRECT TEXT (Текст) ---
# Убрали сложные правила. Просто "Найди это".
DIRECT_TEXT_PROMPT = """
Task: Extract the '{feature_name}' from the product text.
Context: '{context}'

INSTRUCTIONS:
1. Extract the explicit text describing '{feature_name}'.
2. For Brand: It is usually the first word in the Product Name.
3. Return NULL if not mentioned.
"""

# --- 3. BUNDLE (Комплекты) ---
# Простая арифметика.
BUNDLE_PROMPT = """
Task: Calculate total '{feature_name}' in '{unit}'.
Context: '{context}'

INSTRUCTIONS:
1. Identify if this is a single item or a bundle (set of items).
2. If Bundle: Sum up the values (Quantity * Value).
3. If Single: Return the value.
4. Output the final number only.
"""

# --- 4. DIMENSIONS (Габариты) ---
# УБРАНЫ ОПРЕДЕЛЕНИЯ ОСЕЙ. Модель сама знает, что такое Depth.
# Мы просто просим найти размеры и сопоставить их.
DIMENSIONS_PROMPT = """
Task: Extract the '{feature_name}' dimension.
Context: '{context}'

INSTRUCTIONS:
1. Find the dimensions in the text (usually format like LxWxH or similar).
2. Map the found numbers to '{feature_name}' based on standard conventions for this object type.
3. Return ONLY the number. If not found or irrelevant, return NULL.
"""

# --- 5. SELECT (Выбор) ---
# Просим выбрать наиболее подходящий вариант.
SELECT_PROMPT = """
Task: Classify '{feature_name}' into one of the allowed options.
Allowed Options: [{options_list}]
Context: '{context}'

INSTRUCTIONS:
1. Select the option that best matches the product description.
2. Allow for synonyms (e.g. 'Wireless' -> 'Bluetooth').
3. If no option fits, return NULL.
"""

# --- 6. MODEL NAME (Название модели) ---
# Универсальный промпт: убрать бренд-префикс, оставить идентификатор модели.
# Работает для любых категорий (БП, обувь, телефоны, бытовая техника).
MODEL_NAME_PROMPT = """
Task: Extract the model identifier for '{feature_name}' from the product name.
Product Name: '{product_name}'

INSTRUCTIONS:
1. Remove the category prefix (e.g. "Блок питания", "Кроссовки", "Смартфон").
2. Remove the brand name (first recognizable brand word after the category prefix).
3. Return ONLY the model identifier string that remains (e.g. "MWE Gold 750 V2 Full Modular").
4. Do NOT include wattage/capacity/size if it is already encoded in the model name; keep it if it is part of the official model string.
5. If no distinct model identifier exists (e.g. generic no-name), return NULL.
6. Applies across all product categories — use common sense to identify the model substring.
"""