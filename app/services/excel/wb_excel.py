"""WB Excel template parser/writer.

WB Excel шаблон содержит 2 вкладки: 'Товары' и 'Инструкция'.
Колонки в 'Товары': Артикул продавца, Артикул WB, Наименование, Категория продавца,
Бренд, Размер, Цвет, ...характеристики категории...

Спецификация: https://seller.wildberries.ru/instructions/ru/ru/material/A-203
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


WB_PRODUCTS_SHEET = "Товары"
# Имена колонок которые НЕ являются characteristics (фиксированные системные поля)
WB_SYSTEM_COLUMNS = {
    "Артикул продавца", "Артикул WB", "Наименование", "Категория продавца",
    "Группа", "Бренд", "Состав", "Размер", "Цвет", "Описание",
    # WB шаблоны иногда меняются, добавлять по мере встреч
}


class WbExcelReader:
    """Парсит WB Excel шаблон в список products + targets."""

    def __init__(self, file_path: "str | Path"):
        self.path = Path(file_path)

    def read_products(self) -> List[Dict]:
        """Returns list of {row_index, sku, name, brand, description, characteristics: {col_name: value}}.

        Каждая строка = один SKU.
        """
        df = pd.read_excel(self.path, sheet_name=WB_PRODUCTS_SHEET, dtype=str)
        # Drop empty rows
        df = df.dropna(subset=[df.columns[0]])  # article column required

        products = []
        for idx, row in df.iterrows():
            products.append({
                "row_index": idx,
                "sku": row.get("Артикул продавца") or row.get("Артикул WB"),
                "name": row.get("Наименование"),
                "brand": row.get("Бренд"),
                "description": row.get("Описание"),
                "characteristics": {
                    col: row[col]
                    for col in df.columns
                    if col not in WB_SYSTEM_COLUMNS and pd.notna(row[col])
                },
                "_all_columns": list(df.columns),  # for write-back
            })
        return products

    def get_target_attributes(self, products: List[Dict]) -> List[Dict]:
        """Извлекает уникальные target characteristic columns которые нужно заполнить."""
        if not products:
            return []
        all_cols = products[0]["_all_columns"]
        targets = []
        target_id = 1
        for col in all_cols:
            if col in WB_SYSTEM_COLUMNS:
                continue
            targets.append({"id": target_id, "name": col, "type": "text"})
            target_id += 1
        return targets


class WbExcelWriter:
    """Записывает результаты обратно в Excel (сохранение исходной структуры)."""

    def __init__(self, original_path: "str | Path"):
        self.original_path = Path(original_path)

    def write_filled(
        self,
        output_path: "str | Path",
        products_with_filled: List[Dict],  # каждый product с дополнительным полем 'ai_filled': {col: value}
    ) -> Path:
        """Загружает оригинальный файл, заполняет AI-результатами, сохраняет."""
        df = pd.read_excel(self.original_path, sheet_name=WB_PRODUCTS_SHEET, dtype=str)

        for product in products_with_filled:
            idx = product["row_index"]
            ai_filled = product.get("ai_filled", {})
            for col, val in ai_filled.items():
                if col in df.columns and (pd.isna(df.at[idx, col]) or df.at[idx, col] == ""):
                    df.at[idx, col] = val

        output = Path(output_path)
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=WB_PRODUCTS_SHEET, index=False)
            # Copy instruction sheet if exists
            try:
                instr = pd.read_excel(self.original_path, sheet_name="Инструкция", header=None)
                instr.to_excel(writer, sheet_name="Инструкция", index=False, header=False)
            except Exception:
                pass
        return output
