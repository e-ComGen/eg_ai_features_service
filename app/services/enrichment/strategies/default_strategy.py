"""DefaultStrategy — no-op. Используется когда marketplace не указан."""
from .base import MarketplaceStrategy


class DefaultStrategy(MarketplaceStrategy):
    """Без marketplace-specific логики. Pipeline работает как раньше."""

    @property
    def name(self) -> str:
        return "default"

    # Все методы наследуются от base (no-op)
