"""Unit tests for is_not_applicable() in ozon_field_classifier.

Verifies that:
- OS-version fields are excluded only for the provably wrong OS.
- The correct-OS version field is KEPT (not excluded).
- Normal fillable attrs are never excluded.
- Dryer-only fields are excluded on a pure washing machine.
- Ambiguous / unknown-OS products → nothing excluded.
"""
from __future__ import annotations
import pytest

from app.services.enrichment.strategies.dictionaries.ozon_field_classifier import (
    is_not_applicable,
    ProductContext,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _char(name: str, *, required: bool = False) -> dict:
    return {"id": 1, "name": name, "type": "text", "is_required": required}


def _android_phone_ctx(**extra) -> ProductContext:
    return ProductContext(
        product_name="Смартфон Samsung Galaxy A55 5G 8/256GB",
        category_path=["Электроника", "Смартфоны"],
        **extra,
    )


def _washer_ctx() -> ProductContext:
    return ProductContext(
        product_name="Стиральная машина Bosch WGG2540MOE",
        category_path=["Бытовая техника", "Стиральные машины"],
    )


def _washer_with_dryer_ctx() -> ProductContext:
    return ProductContext(
        product_name="Стиральная машина с сушкой LG F4DR510S2W",
        category_path=["Бытовая техника", "Стиральные машины с сушкой"],
    )


# ---------------------------------------------------------------------------
# Rule 1: OS-version mutual exclusion
# ---------------------------------------------------------------------------

class TestOsVersionExclusion:

    def test_ios_version_excluded_on_android_phone(self):
        """'Версия iOS' must be excluded for an Android phone."""
        assert is_not_applicable(_char("Версия iOS"), _android_phone_ctx()) is True

    def test_macos_version_excluded_on_android_phone(self):
        """'Версия MacOS' must be excluded for an Android phone."""
        assert is_not_applicable(_char("Версия MacOS"), _android_phone_ctx()) is True

    def test_harmonyos_version_excluded_on_android_phone(self):
        """'Версия HarmonyOS' must be excluded for Android-only Samsung phone."""
        assert is_not_applicable(_char("Версия HarmonyOS"), _android_phone_ctx()) is True

    def test_android_version_kept_on_android_phone(self):
        """'Версия Android' must NOT be excluded — it applies to this product."""
        assert is_not_applicable(_char("Версия Android"), _android_phone_ctx()) is False

    def test_windows_version_excluded_on_android_phone(self):
        """'Версия Windows' must be excluded on an Android phone."""
        assert is_not_applicable(_char("Версия Windows"), _android_phone_ctx()) is True

    def test_ipados_version_excluded_on_android_phone(self):
        """'Версия iPadOS' must be excluded on an Android phone."""
        assert is_not_applicable(_char("Версия iPadOS"), _android_phone_ctx()) is True

    def test_watchos_version_excluded_on_android_phone(self):
        """'Версия watchOS' must be excluded on an Android phone."""
        assert is_not_applicable(_char("Версия watchOS"), _android_phone_ctx()) is True

    def test_macos_excluded_on_windows_laptop(self):
        """'Версия MacOS' must be excluded on a Windows laptop."""
        ctx = ProductContext(
            product_name="Ноутбук ASUS VivoBook 15 X1504VA Core i5",
            category_path=["Ноутбуки и компьютеры", "Ноутбуки"],
        )
        assert is_not_applicable(_char("Версия MacOS"), ctx) is True

    def test_windows_kept_on_windows_laptop(self):
        """'Версия Windows' must NOT be excluded on a Windows laptop."""
        ctx = ProductContext(
            product_name="Ноутбук ASUS VivoBook 15 X1504VA Core i5",
            category_path=["Ноутбуки и компьютеры", "Ноутбуки"],
        )
        assert is_not_applicable(_char("Версия Windows"), ctx) is False

    def test_ios_excluded_on_apple_watch(self):
        """'Версия iOS' must be excluded on an Apple Watch (watchOS product)."""
        ctx = ProductContext(
            product_name="Смарт-часы Apple Watch Series 9 45mm",
            category_path=["Электроника", "Смарт-часы"],
        )
        assert is_not_applicable(_char("Версия iOS"), ctx) is True

    def test_watchos_kept_on_apple_watch(self):
        """'Версия watchOS' must NOT be excluded on an Apple Watch."""
        ctx = ProductContext(
            product_name="Смарт-часы Apple Watch Series 9 45mm",
            category_path=["Электроника", "Смарт-часы"],
        )
        assert is_not_applicable(_char("Версия watchOS"), ctx) is False

    def test_filled_os_attr_overrides_brand_signal(self):
        """If 'Операционная система' = 'Android' is already filled, iOS is excluded."""
        ctx = _android_phone_ctx(filled_attrs={"операционная система": "Android"})
        assert is_not_applicable(_char("Версия iOS"), ctx) is True

    def test_filled_os_attr_android_keeps_android_version(self):
        """'Версия Android' is NOT excluded when filled OS = Android."""
        ctx = _android_phone_ctx(filled_attrs={"операционная система": "Android"})
        assert is_not_applicable(_char("Версия Android"), ctx) is False

    def test_ambiguous_unknown_os_keeps_all(self):
        """For an unknown-OS product, no OS version attrs are excluded."""
        ctx = ProductContext(
            product_name="Устройство XYZ 3000",
            category_path=["Прочее"],
        )
        assert is_not_applicable(_char("Версия iOS"), ctx) is False
        assert is_not_applicable(_char("Версия Android"), ctx) is False
        assert is_not_applicable(_char("Версия Windows"), ctx) is False

    def test_huawei_ambiguous_not_excluded(self):
        """Huawei (Android/HarmonyOS) is ambiguous — neither OS should be excluded."""
        ctx = ProductContext(
            product_name="Смартфон Huawei Nova 11",
            category_path=["Электроника", "Смартфоны"],
        )
        # Huawei can run either, so neither version field should be excluded
        assert is_not_applicable(_char("Версия Android"), ctx) is False
        assert is_not_applicable(_char("Версия HarmonyOS"), ctx) is False

    def test_required_os_version_field_never_excluded(self):
        """required=True → is_not_applicable must always return False (hard guard)."""
        ctx = _android_phone_ctx()
        assert is_not_applicable(_char("Версия iOS", required=True), ctx) is False

    def test_normal_fillable_attr_kept(self):
        """'Материал корпуса' — a normal spec attr — must never be excluded."""
        ctx = _android_phone_ctx()
        assert is_not_applicable(_char("Материал корпуса"), ctx) is False

    def test_color_attr_kept(self):
        """'Цвет' is a normal extractable attr — must not be excluded."""
        ctx = _android_phone_ctx()
        assert is_not_applicable(_char("Цвет"), ctx) is False


# ---------------------------------------------------------------------------
# Rule 2: Dryer-only fields on washing machines
# ---------------------------------------------------------------------------

class TestDryerFieldExclusion:

    def test_dryer_field_excluded_on_pure_washer(self):
        """'Тип сушки' must be excluded on a pure washing machine."""
        assert is_not_applicable(_char("Тип сушки"), _washer_ctx()) is True

    def test_program_count_dryer_excluded_on_pure_washer(self):
        """'Количество программ сушки' must be excluded on a pure washer."""
        assert is_not_applicable(_char("Количество программ сушки"), _washer_ctx()) is True

    def test_laundry_load_dryer_excluded_on_pure_washer(self):
        """'Загрузка белья для сушки' must be excluded on a pure washer."""
        assert is_not_applicable(_char("Загрузка белья для сушки"), _washer_ctx()) is True

    def test_dryer_field_kept_on_combo_washer_dryer(self):
        """'Тип сушки' must NOT be excluded on a washer-with-dryer combo."""
        assert is_not_applicable(_char("Тип сушки"), _washer_with_dryer_ctx()) is False

    def test_dryer_field_not_excluded_on_non_washer(self):
        """A dryer attr on a non-washer product (e.g. fridge) is NOT excluded."""
        ctx = ProductContext(
            product_name="Холодильник Samsung RT47CG6442S9",
            category_path=["Бытовая техника", "Холодильники"],
        )
        # Not a washer → rule does not fire → keep
        assert is_not_applicable(_char("Тип сушки"), ctx) is False

    def test_wash_load_attr_kept_on_washer(self):
        """'Максимальная загрузка белья' (wash load, NOT dryer) must NOT be excluded."""
        assert is_not_applicable(_char("Максимальная загрузка белья"), _washer_ctx()) is False

    def test_required_dryer_field_never_excluded(self):
        """required dryer field on a pure washer → hard guard returns False."""
        assert is_not_applicable(_char("Тип сушки", required=True), _washer_ctx()) is False
