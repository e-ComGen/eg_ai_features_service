"""Converter: (ProductData, list[AttributeValue]) → Ozon /v2/product/import payload."""
from __future__ import annotations

import json
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic import ConfigDict

from app.services.enrichment.base import AttributeValue


# ---------------------------------------------------------------------------
# Ozon API schema models
# ---------------------------------------------------------------------------

class OzonAttributeValue(BaseModel):
    """Одна запись values[] внутри attribute."""
    model_config = ConfigDict(populate_by_name=True)

    dictionary_value_id: Optional[int] = Field(None, alias="dictionary_value_id")
    value: Optional[str] = Field(None)


class OzonAttribute(BaseModel):
    """Один элемент attributes[] в Ozon import item."""
    model_config = ConfigDict(populate_by_name=True)

    complex_id: int = Field(0)
    id: int
    values: list[OzonAttributeValue]


class OzonDimensions(BaseModel):
    """Физические размеры товара (мм/см/in)."""
    model_config = ConfigDict(populate_by_name=True)

    depth: int = Field(..., gt=0)
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)
    dimension_unit: str = Field("mm")

    @field_validator("dimension_unit")
    @classmethod
    def valid_dimension_unit(cls, v: str) -> str:
        allowed = {"mm", "cm", "in"}
        if v not in allowed:
            raise ValueError(f"dimension_unit must be one of {allowed}, got {v!r}")
        return v


class OzonProductInput(BaseModel):
    """Входные данные о товаре от мерчанта."""
    model_config = ConfigDict(populate_by_name=True)

    offer_id: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = Field(None, max_length=4000)
    price: str = Field(...)
    old_price: Optional[str] = Field(None)
    vat: str = Field("0.20")
    barcode: Optional[str] = Field(None)
    currency_code: str = Field("RUB")
    images: list[str] = Field(...)
    weight: int = Field(..., gt=0)
    weight_unit: str = Field("g")
    dimensions: OzonDimensions

    @field_validator("price")
    @classmethod
    def price_positive(cls, v: str) -> str:
        """Цена должна быть строкой с числом > 0."""
        try:
            num = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"price must be numeric string, got {v!r}")
        if num <= 0:
            raise ValueError(f"price must be > 0, got {num}")
        return v

    @field_validator("old_price")
    @classmethod
    def old_price_positive(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        try:
            num = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"old_price must be numeric string, got {v!r}")
        if num <= 0:
            raise ValueError(f"old_price must be > 0, got {num}")
        return v

    @field_validator("images")
    @classmethod
    def images_bounds(cls, v: list[str]) -> list[str]:
        """Ozon принимает от 1 до 15 изображений."""
        if not v:
            raise ValueError("images list must not be empty (Ozon requires at least 1)")
        if len(v) > 15:
            raise ValueError(f"images list must have at most 15 entries, got {len(v)}")
        return v

    @field_validator("vat")
    @classmethod
    def valid_vat(cls, v: str) -> str:
        allowed = {"0", "0.10", "0.20"}
        if v not in allowed:
            raise ValueError(f"vat must be one of {allowed}, got {v!r}")
        return v

    @field_validator("weight_unit")
    @classmethod
    def valid_weight_unit(cls, v: str) -> str:
        allowed = {"g", "kg"}
        if v not in allowed:
            raise ValueError(f"weight_unit must be one of {allowed}, got {v!r}")
        return v


class OzonImportItem(BaseModel):
    """Один товар в items[] для /v2/product/import."""
    model_config = ConfigDict(populate_by_name=True)

    offer_id: str
    name: str
    description_category_id: int
    type_id: int
    barcode: Optional[str] = Field(None)
    price: str
    old_price: Optional[str] = Field(None)
    vat: str
    currency_code: str
    images: list[str]
    primary_image: str
    weight: int
    weight_unit: str
    depth: int
    width: int
    height: int
    dimension_unit: str
    attributes: list[OzonAttribute]


class OzonImportPayload(BaseModel):
    """Тело запроса POST /v2/product/import."""
    model_config = ConfigDict(populate_by_name=True)

    items: list[OzonImportItem]

    def to_api_dict(self) -> dict:
        """Сериализовать в dict для отправки в Ozon API."""
        return self.model_dump(by_alias=True, exclude_none=True)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _build_ozon_attribute(av: AttributeValue) -> OzonAttribute:
    """Преобразовать AttributeValue → OzonAttribute по правилам Ozon.

    Rules:
    - values ALWAYS a list.
    - dictionary_value_id только если value_id / value_ids присутствует.
    - is_collection=True с value_ids → multiple entries with dictionary_value_id.
    - is_collection=True free-text → multiple {value:...} entries.
    - scalar with value_id → one entry {dictionary_value_id, value}.
    - scalar free-text → one entry {value: ...}.
    """
    ozon_values: list[OzonAttributeValue] = []

    if av.is_collection and isinstance(av.value, list):
        values_list = av.value
        ids_list: list[Optional[int]] = []
        if av.value_ids:
            # Выравниваем по длине value_ids: если ids меньше values — хвост без id
            ids_list = list(av.value_ids) + [None] * (len(values_list) - len(av.value_ids))
        else:
            ids_list = [None] * len(values_list)

        for raw_val, vid in zip(values_list, ids_list):
            entry = OzonAttributeValue(value=str(raw_val))
            if vid is not None:
                entry.dictionary_value_id = vid
            ozon_values.append(entry)
    else:
        # Scalar
        raw_val = av.value if not isinstance(av.value, list) else (av.value[0] if av.value else "")
        entry = OzonAttributeValue(value=str(raw_val))
        if av.value_id is not None:
            entry.dictionary_value_id = av.value_id
        ozon_values.append(entry)

    return OzonAttribute(
        complex_id=0,
        id=av.attribute_id,
        values=ozon_values,
    )


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_ozon_import_payload(
    product: OzonProductInput,
    description_category_id: int,
    type_id: int,
    attributes: list[AttributeValue],
    *,
    primary_image: Optional[str] = None,
) -> OzonImportPayload:
    """Собрать OzonImportPayload из входных данных и списка AttributeValue.

    Args:
        product: Данные о товаре от мерчанта.
        description_category_id: Ozon description_category_id.
        type_id: Ozon type_id.
        attributes: Список AttributeValue из pipeline.
        primary_image: Главное изображение; по умолчанию — первое из images.

    Returns:
        Готовый OzonImportPayload для POST /v2/product/import.
    """
    resolved_primary = primary_image or product.images[0]

    ozon_attrs = [_build_ozon_attribute(av) for av in attributes]

    item = OzonImportItem(
        offer_id=product.offer_id,
        name=product.name,
        description_category_id=description_category_id,
        type_id=type_id,
        barcode=product.barcode,
        price=product.price,
        old_price=product.old_price,
        vat=product.vat,
        currency_code=product.currency_code,
        images=product.images,
        primary_image=resolved_primary,
        weight=product.weight,
        weight_unit=product.weight_unit,
        depth=product.dimensions.depth,
        width=product.dimensions.width,
        height=product.dimensions.height,
        dimension_unit=product.dimensions.dimension_unit,
        attributes=ozon_attrs,
    )

    return OzonImportPayload(items=[item])


__all__ = [
    "OzonAttributeValue",
    "OzonAttribute",
    "OzonDimensions",
    "OzonProductInput",
    "OzonImportItem",
    "OzonImportPayload",
    "build_ozon_import_payload",
]
