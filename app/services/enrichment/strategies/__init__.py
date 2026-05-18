"""Marketplace Strategy pattern — изоляция marketplace-specific логики.

Экспортирует базовый класс, все стратегии и factory function.
"""
from .base import MarketplaceStrategy, ValidationResult
from .default_strategy import DefaultStrategy
from .wildberries_strategy import WildberriesStrategy
from .ozon_strategy import OzonStrategy
from .factory import get_strategy

__all__ = [
    "MarketplaceStrategy",
    "ValidationResult",
    "DefaultStrategy",
    "WildberriesStrategy",
    "OzonStrategy",
    "get_strategy",
]
