"""Factory для выбора стратегии по имени."""
from typing import Optional
from .base import MarketplaceStrategy
from .default_strategy import DefaultStrategy
from .wildberries_strategy import WildberriesStrategy
from .ozon_strategy import OzonStrategy

_STRATEGIES: dict[str, type[MarketplaceStrategy]] = {
    "default": DefaultStrategy,
    "wb": WildberriesStrategy,
    "wildberries": WildberriesStrategy,  # alias
    "ozon": OzonStrategy,
}


def get_strategy(name: Optional[str] = None) -> MarketplaceStrategy:
    """Returns strategy by name. Falls back to DefaultStrategy if unknown."""
    if not name:
        return DefaultStrategy()
    strategy_class = _STRATEGIES.get(name.lower(), DefaultStrategy)
    return strategy_class()
