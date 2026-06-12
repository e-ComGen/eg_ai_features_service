"""Tests for BarcodeSource — verbatim EAN/barcode extractor.

All tests are pure unit tests (no network, no LLM).
Covers:
  - Checksum validation helpers (EAN-13, UPC-A, EAN-8)
  - Text scanning (extract_barcodes_from_text)
  - Target-name detection (is_barcode_target)
  - BarcodeSource.extract: fills barcode attrs, ignores non-barcode attrs,
    skips already-filled, uses context.ean, handles empty text pools.
"""
from __future__ import annotations

import pytest

from app.services.enrichment.base import (
    AttributeValue,
    ExtractionContext,
    Source,
    TargetAttribute,
)
from app.services.enrichment.sources.barcode_source import (
    BarcodeSource,
    BarcodeJudge,
    _ean13_checksum_valid,
    _upc_a_checksum_valid,
    _ean8_checksum_valid,
    extract_barcodes_from_text,
    is_barcode_target,
    validate_barcode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(
    product_id: int = 1,
    product_name: str = "Test Product",
    description: str | None = None,
    ean: str | None = None,
) -> ExtractionContext:
    return ExtractionContext(
        product_id=product_id,
        product_name=product_name,
        product_description=description,
        category_id=100,
        ean=ean,
    )


def _barcode_target(attr_id: int = 10, name: str = "Штрихкод") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type="text")


def _non_barcode_target(attr_id: int = 20, name: str = "Цвет") -> TargetAttribute:
    return TargetAttribute(id=attr_id, name=name, type="enum",
                           allowed_values=["Белый", "Чёрный"])


# ---------------------------------------------------------------------------
# Known-good barcodes from public GS1 test vectors
# ---------------------------------------------------------------------------

# Valid EAN-13: Coca-Cola 5000112637922
_EAN13_VALID = "5000112637922"
# Valid UPC-A: 012345678905
_UPCA_VALID = "012345678905"
# Valid EAN-8: 73513537
_EAN8_VALID = "73513537"

# Corrupted (last digit flipped) — checksum must fail
_EAN13_BAD = "5000112637921"
_UPCA_BAD = "012345678904"
_EAN8_BAD = "73513538"


# ---------------------------------------------------------------------------
# Checksum validators
# ---------------------------------------------------------------------------


class TestEan13ChecksumValid:
    def test_valid_ean13(self):
        assert _ean13_checksum_valid(_EAN13_VALID)

    def test_bad_ean13(self):
        assert not _ean13_checksum_valid(_EAN13_BAD)

    def test_wrong_length(self):
        assert not _ean13_checksum_valid("123456789012")

    def test_non_digits(self):
        assert not _ean13_checksum_valid("500011263792X")


class TestUpcAChecksum:
    def test_valid_upc_a(self):
        assert _upc_a_checksum_valid(_UPCA_VALID)

    def test_bad_upc_a(self):
        assert not _upc_a_checksum_valid(_UPCA_BAD)

    def test_wrong_length(self):
        assert not _upc_a_checksum_valid("0123456789012")


class TestEan8Checksum:
    def test_valid_ean8(self):
        assert _ean8_checksum_valid(_EAN8_VALID)

    def test_bad_ean8(self):
        assert not _ean8_checksum_valid(_EAN8_BAD)

    def test_wrong_length(self):
        assert not _ean8_checksum_valid("7351353")


class TestValidateBarcode:
    def test_dispatches_ean13(self):
        assert validate_barcode(_EAN13_VALID)

    def test_dispatches_upc_a(self):
        assert validate_barcode(_UPCA_VALID)

    def test_dispatches_ean8(self):
        assert validate_barcode(_EAN8_VALID)

    def test_rejects_random_13_digits(self):
        # Random 13 digits are extremely unlikely to pass checksum
        assert not validate_barcode("1234567890123")

    def test_rejects_wrong_length(self):
        assert not validate_barcode("12345")


# ---------------------------------------------------------------------------
# Text scanning
# ---------------------------------------------------------------------------


class TestExtractBarcodesFromText:
    def test_finds_ean13_in_description(self):
        text = f"Товар. Штрихкод: {_EAN13_VALID}. Производство Россия."
        assert extract_barcodes_from_text(text) == [_EAN13_VALID]

    def test_finds_upc_a(self):
        text = f"UPC: {_UPCA_VALID} — official product code."
        assert extract_barcodes_from_text(text) == [_UPCA_VALID]

    def test_finds_ean8(self):
        text = f"EAN-8: {_EAN8_VALID}"
        assert extract_barcodes_from_text(text) == [_EAN8_VALID]

    def test_ignores_random_13_digit_number(self):
        # 9999999999999 — almost certainly invalid checksum
        text = "ID товара: 9999999999999"
        result = extract_barcodes_from_text(text)
        assert "9999999999999" not in result

    def test_deduplicates(self):
        text = f"{_EAN13_VALID} and again {_EAN13_VALID}"
        assert extract_barcodes_from_text(text) == [_EAN13_VALID]

    def test_returns_multiple_distinct_valid(self):
        text = f"EAN-8: {_EAN8_VALID}. EAN-13: {_EAN13_VALID}."
        result = extract_barcodes_from_text(text)
        assert _EAN8_VALID in result
        assert _EAN13_VALID in result

    def test_empty_text(self):
        assert extract_barcodes_from_text("") == []

    def test_no_digits(self):
        assert extract_barcodes_from_text("Нет штрихкода здесь") == []

    def test_digit_sequence_not_isolated(self):
        # Digits embedded in a longer sequence (14 digits) — not matched
        text = "12345678901234"
        assert extract_barcodes_from_text(text) == []


# ---------------------------------------------------------------------------
# Target-name detection
# ---------------------------------------------------------------------------


class TestIsBarcodeTarget:
    @pytest.mark.parametrize("name", [
        "Штрихкод",
        "штрихкод",
        "штрих-код",
        "EAN",
        "ean",
        "barcode",
        "BARCODE",
        "GTIN",
        "гтин",
        "Штрихкод (EAN-13)",
    ])
    def test_barcode_names(self, name: str):
        t = TargetAttribute(id=1, name=name, type="text")
        assert is_barcode_target(t)

    @pytest.mark.parametrize("name", [
        "Цвет",
        "Бренд",
        "Материал",
        "Артикул",
        "Вес",
        "Ширина",
    ])
    def test_non_barcode_names(self, name: str):
        t = TargetAttribute(id=1, name=name, type="text")
        assert not is_barcode_target(t)


# ---------------------------------------------------------------------------
# BarcodeSource.extract
# ---------------------------------------------------------------------------


class TestBarcodeSourceExtract:
    @pytest.mark.asyncio
    async def test_fills_barcode_attr_from_description(self):
        ctx = _ctx(description=f"Штрихкод товара: {_EAN13_VALID}.")
        source = BarcodeSource()
        result = await source.extract(ctx, [_barcode_target()])
        assert len(result) == 1
        assert result[0].value == _EAN13_VALID
        assert result[0].confidence == 0.98
        assert result[0].source == Source.DESCRIPTION
        assert _EAN13_VALID in result[0].evidence

    @pytest.mark.asyncio
    async def test_fills_from_context_ean(self):
        ctx = _ctx(ean=_EAN13_VALID)  # no description
        source = BarcodeSource()
        result = await source.extract(ctx, [_barcode_target()])
        assert len(result) == 1
        assert result[0].value == _EAN13_VALID

    @pytest.mark.asyncio
    async def test_fills_from_source_text_kwarg(self):
        ctx = _ctx()  # no description, no ean
        source = BarcodeSource()
        result = await source.extract(
            ctx, [_barcode_target()],
            source_text=f"EAN: {_UPCA_VALID}",
        )
        assert len(result) == 1
        assert result[0].value == _UPCA_VALID

    @pytest.mark.asyncio
    async def test_context_ean_takes_priority_over_description(self):
        # context.ean is scanned first → its barcode wins
        ctx = _ctx(
            ean=_EAN13_VALID,
            description=f"Also contains UPC: {_UPCA_VALID}",
        )
        source = BarcodeSource()
        result = await source.extract(ctx, [_barcode_target()])
        assert result[0].value == _EAN13_VALID  # not the UPC from description

    @pytest.mark.asyncio
    async def test_no_fill_for_non_barcode_target(self):
        ctx = _ctx(description=f"EAN: {_EAN13_VALID}")
        source = BarcodeSource()
        result = await source.extract(ctx, [_non_barcode_target()])
        assert result == []

    @pytest.mark.asyncio
    async def test_no_fill_when_no_valid_barcode_in_text(self):
        ctx = _ctx(description="Нет штрихкода. ID: 9999999999999.")
        source = BarcodeSource()
        result = await source.extract(ctx, [_barcode_target()])
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_already_filled_target(self):
        ctx = _ctx(description=f"EAN: {_EAN13_VALID}")
        source = BarcodeSource()
        already = [AttributeValue(
            attribute_id=10,
            value="old_barcode",
            confidence=0.99,
            source=Source.DESCRIPTION,
        )]
        result = await source.extract(ctx, [_barcode_target(attr_id=10)], already_filled=already)
        assert result == []

    @pytest.mark.asyncio
    async def test_fills_multiple_barcode_targets_with_same_value(self):
        ctx = _ctx(description=f"EAN: {_EAN13_VALID}")
        source = BarcodeSource()
        t1 = _barcode_target(attr_id=10, name="Штрихкод")
        t2 = _barcode_target(attr_id=11, name="EAN")
        result = await source.extract(ctx, [t1, t2])
        assert len(result) == 2
        assert all(av.value == _EAN13_VALID for av in result)

    @pytest.mark.asyncio
    async def test_empty_context_and_no_text(self):
        ctx = _ctx()
        source = BarcodeSource()
        result = await source.extract(ctx, [_barcode_target()])
        assert result == []


# ---------------------------------------------------------------------------
# BarcodeJudge
# ---------------------------------------------------------------------------


class TestBarcodeJudge:
    @pytest.mark.asyncio
    async def test_accepts_valid_ean13(self):
        judge = BarcodeJudge()
        av = AttributeValue(
            attribute_id=1, value=_EAN13_VALID, confidence=0.98, source=Source.DESCRIPTION,
        )
        assert await judge.validate(av, _ctx())

    @pytest.mark.asyncio
    async def test_rejects_bad_checksum(self):
        judge = BarcodeJudge()
        av = AttributeValue(
            attribute_id=1, value=_EAN13_BAD, confidence=0.98, source=Source.DESCRIPTION,
        )
        assert not await judge.validate(av, _ctx())

    @pytest.mark.asyncio
    async def test_rejects_non_digit_string(self):
        judge = BarcodeJudge()
        av = AttributeValue(
            attribute_id=1, value="not-a-barcode", confidence=0.98, source=Source.DESCRIPTION,
        )
        assert not await judge.validate(av, _ctx())
