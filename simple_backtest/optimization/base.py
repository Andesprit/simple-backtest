"""Base optimizer class for strategy parameter optimization."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type

import pandas as pd

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.core.backtest import Backtest
from simple_backtest.strategy.base import Strategy


class Optimizer(ABC):
    """Abstract base class for parameter optimization.

    Users can create custom optimizers by inheriting from this class
    and implementing the optimize() method.

    Example:
        class MyOptimizer(Optimizer):
            def optimize(self, data, config, strategy_class, param_space, metric):
                # Your optimization logic here
                return results_df
    """

    def __init__(self, name: str | None = None):
        """Initialize optimizer.

        :param name: Optimizer name (auto-generated if None)
        """
        self._name = name or self.__class__.__name__
        self.failures: List[Dict[str, Any]] = []

    def _record_failure(self, parameters: Dict[str, Any], error: Exception) -> None:
        """Record an expected invalid parameter combination."""
        self.failures.append(
            {
                "parameters": parameters.copy(),
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    @abstractmethod
    def optimize(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        strategy_class: Type[Strategy],
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """Run optimization to find best parameters.

        :param data: OHLCV DataFrame with DatetimeIndex
        :param config: Backtest configuration
        :param strategy_class: Strategy class to optimize
        :param param_space: Dict of param_name: [values] to search
        :param metric: Metric to optimize (e.g., 'sharpe_ratio', 'total_return')
        :return: DataFrame of results sorted by metric (best first)
        """
        pass

    def get_name(self) -> str:
        """Get optimizer name."""
        return self._name

    def _run_backtest(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        strategy: Strategy,
        backtest: Backtest | None = None,
    ) -> Dict[str, Any]:
        """Helper to run a single backtest and return metrics.

        :param data: OHLCV DataFrame
        :param config: Backtest configuration
        :param strategy: Strategy instance to test
        :return: Dict with parameters and metrics
        """
        bt = backtest or Backtest(data, config)
        results = bt.run([strategy])
        strategy_result = results.get_strategy(strategy.get_name())
        return strategy_result.metrics

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self._name}')"
