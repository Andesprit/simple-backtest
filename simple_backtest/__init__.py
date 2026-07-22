"""A small, transparent framework for long-only strategy backtests."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("simple-backtest")
except PackageNotFoundError:  # Source tree imported without installation.
    __version__ = "0.3.0"

# Core imports
# Commission imports
from simple_backtest.commission import (
    Commission,
    FlatCommission,
    PercentageCommission,
    TieredCommission,
)
from simple_backtest.config.settings import BacktestConfig
from simple_backtest.core.backtest import Backtest
from simple_backtest.core.portfolio import Portfolio
from simple_backtest.core.results import BacktestResults, StrategyResult

# Optimization imports
from simple_backtest.optimization import (
    GridSearchOptimizer,
    Optimizer,
    RandomSearchOptimizer,
    WalkForwardOptimizer,
)
from simple_backtest.strategy.base import Strategy
from simple_backtest.strategy.buy_and_hold import BuyAndHoldStrategy
from simple_backtest.strategy.dca import DCAStrategy
from simple_backtest.strategy.moving_average import MovingAverageStrategy

__all__ = [
    # Core
    "BacktestConfig",
    "Backtest",
    "Portfolio",
    "Strategy",
    "BacktestResults",
    "StrategyResult",
    # Built-in Strategies
    "BuyAndHoldStrategy",
    "DCAStrategy",
    "MovingAverageStrategy",
    # Optimization
    "Optimizer",
    "GridSearchOptimizer",
    "RandomSearchOptimizer",
    "WalkForwardOptimizer",
    # Commission
    "Commission",
    "PercentageCommission",
    "FlatCommission",
    "TieredCommission",
]
