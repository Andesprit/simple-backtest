"""Base optimizer class for strategy parameter optimization."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Type

import pandas as pd

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.core.backtest import Backtest
from simple_backtest.core.serialization import json_value, write_json
from simple_backtest.metrics.objectives import rank_results
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
        self.summary: Dict[str, Any] = {}

    def _prepare_search(self, param_space: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Validate finite parameter lists and remove duplicate values in insertion order."""
        self.failures = []
        self.summary = {"attempted_evaluations": 0, "unique_candidates": 0}
        cleaned = {}
        for name, values in param_space.items():
            if not isinstance(values, (list, tuple, range)) or not values:
                raise ValueError(f"Parameter '{name}' must have a non-empty sequence of candidates")
            unique: list[Any] = []
            for value in values:
                if value not in unique:
                    unique.append(value)
            cleaned[name] = unique
        # An empty mapping deliberately evaluates the strategy's defaults once.
        return cleaned

    def _finish_search(
        self, rows: list[dict[str, Any]], metric: str, metadata: dict[str, Any]
    ) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        ranked = rank_results(frame, metric)
        self.summary.update(
            successful_evaluations=len(frame),
            failed_evaluations=len(self.failures),
            undefined_objectives=len(frame) - len(ranked),
        )
        ranked.attrs["search"] = json_value(
            {
                **metadata,
                **self.summary,
                "objective": metric,
                "failures": self.failures.copy(),
            }
        )
        return ranked

    def _record_failure(self, parameters: Dict[str, Any], error: Exception) -> None:
        """Record an expected invalid parameter combination."""
        self.failures.append(
            {
                "parameters": parameters.copy(),
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )

    @staticmethod
    def export_json(results: pd.DataFrame, path: str | Path) -> None:
        """Save ranked candidates and the provenance attached to their DataFrame."""
        write_json(path, {"schema_version": 1, "results": results, "metadata": results.attrs})

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
