"""DefaultStrategy — no-op. Используется когда marketplace не указан."""
from .base import MarketplaceStrategy


class DefaultStrategy(MarketplaceStrategy):
    """Без marketplace-specific логики. Pipeline работает как раньше."""

    # cscart-путь: value_id нет by design (строка→variant_id резолвит PHP),
    # поэтому enum-дроп-guard для нас не применяется — иначе select-поля
    # (Бренд и др.) молча вылетают.
    requires_dictionary_value_ids = False

    @property
    def name(self) -> str:
        return "default"

    # Все методы наследуются от base (no-op)
