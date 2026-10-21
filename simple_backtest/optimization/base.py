"""Base optimizer class for strategy parameter optimization."""

import warnings
from abc import ABC, abstractmethod
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Type

import pandas as pd

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.core.backtest import Backtest
from simple_backtest.core.results import StrategyResult, _metric_hint
from simple_backtest.core.serialization import json_value, write_json
from simple_backtest.metrics.objectives import metric_is_maximized, rank_results
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
        # Selection state of the last search: no search, "running" (stopped early if
        # still set), "failed" or "done", plus only the best evaluation so far.
        self._status: str | None = None
        self._objective: str | None = None
        self._param_names: list[str] = []
        self._best: Dict[str, Any] | None = None
        self._last_result: StrategyResult | None = None

    def _prepare_search(
        self, param_space: Dict[str, List[Any]], metric: str | None = None
    ) -> Dict[str, List[Any]]:
        """Validate finite parameter lists and remove duplicate values in insertion order."""
        self.failures = []
        self.summary = {"attempted_evaluations": 0, "unique_candidates": 0}
        self._status, self._objective = "running", metric
        self._param_names = list(param_space)
        self._best = self._last_result = None
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
        self,
        rows: list[dict[str, Any]],
        metric: str,
        metadata: dict[str, Any],
        strategy_class: Type[Strategy],
    ) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        ranked = rank_results(frame, metric)
        self.summary.update(
            successful_evaluations=len(frame),
            failed_evaluations=len(self.failures),
            undefined_objectives=len(frame) - len(ranked),
        )
        if not rows and self.failures:
            valid_parameters = [
                name
                for name, parameter in signature(strategy_class.__init__).parameters.items()
                if name != "self"
                and parameter.kind
                in (
                    parameter.POSITIONAL_OR_KEYWORD,
                    parameter.KEYWORD_ONLY,
                )
            ]
            first = self.failures[0]
            self._status = "failed"
            raise ValueError(
                f"All {len(self.failures)} candidate evaluations failed for "
                f"{strategy_class.__name__}. First failure for {first['parameters']}: "
                f"{first['error_type']}: {first['message']}. "
                f"Valid parameters: {valid_parameters}. "
                "Correct param_space names and values (or lookback_period for history failures) "
                "and retry. Individual failures are available in optimizer.failures."
            )
        self._status = "done"
        if rows and ranked.empty:
            # One bounded completion notice; an empty frame alone looks like success.
            warnings.warn(f"{self.get_name()}: {self._no_winner_reason()}", stacklevel=3)
        ranked.attrs["search"] = json_value(
            {
                **metadata,
                **self.summary,
                "objective": metric,
                "failures": self.failures.copy(),
            }
        )
        return ranked

    def _consider(
        self, parameters: Dict[str, Any], metrics: Dict[str, Any], metric: str, **trial: Any
    ) -> None:
        """Keep only the best evaluated result so far, matching rank_results' stable order."""
        result, self._last_result = self._last_result, None
        if metric not in metrics:
            raise ValueError(f"Metric '{metric}' not found. {_metric_hint(metric, list(metrics))}")
        score = metrics[metric]
        if pd.isna(score):
            return
        best = self._best
        if best is None or (
            score > best["score"] if metric_is_maximized(metric) else score < best["score"]
        ):
            self._best = {
                "score": score,
                "parameters": {name: parameters[name] for name in self._param_names},
                "result": result,
                **trial,
            }

    def _no_winner_reason(self) -> str:
        """Explain why the last search has no selected candidate and how to retry."""
        if self._status is None:
            return "No search has run yet; call optimizer.optimize(...) first."
        if self._status == "running":
            return "The last search stopped with an error before finishing; fix it and rerun."
        if self._status == "failed":
            return (
                f"All {len(self.failures)} candidate evaluations failed; "
                "see optimizer.failures for each error."
            )
        retry = (
            " Retry with metric='total_return', or check"
            if self._objective != "total_return"
            else " Check"
        )
        return (
            f"No candidate had a defined {self._objective} "
            f"({self.summary['successful_evaluations']} evaluated, "
            f"{self.summary['undefined_objectives']} undefined, "
            f"{self.summary['failed_evaluations']} failed), so the ranked results are empty."
            f"{retry} that prices vary and risk_free_rate suits the objective."
        )

    def _selected(self) -> Dict[str, Any]:
        if self._best is None:
            raise ValueError(self._no_winner_reason())
        return self._best

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
        # Avoid one rejection warning per candidate; regular runs still report rejections.
        bt._warn_rejected_orders = False
        results = bt.run([strategy])
        strategy_result = results.get_strategy(strategy.get_name())
        self._last_result = strategy_result
        return strategy_result.metrics

    @property
    def best_parameters(self) -> Dict[str, Any]:
        """Original parameter objects of the last search's top-ranked candidate."""
        return self._selected()["parameters"].copy()

    @property
    def best_result(self) -> StrategyResult:
        """The evaluated StrategyResult of the last search's top-ranked candidate.

        With repeats, this is the single in-sample trial that ranked first, not an average.
        """
        selected = self._selected()
        if selected["result"] is None:
            raise ValueError(
                f"{self.get_name()}._run_backtest returned metrics without keeping the full "
                "result. Run Backtest(data, config).run([strategy_class("
                "**optimizer.best_parameters)]) to inspect it."
            )
        return selected["result"]

    def __str__(self) -> str:
        """Concise outcome of the last search."""
        if self._best is None:
            return f"{self.get_name()}: {self._no_winner_reason()}"
        best, summary = self._best, self.summary
        direction = "higher" if metric_is_maximized(self._objective or "") else "lower"
        parameters = ", ".join(f"{k}={v!r}" for k, v in best["parameters"].items())
        lines = [
            f"{self.get_name()}: best {self._objective} = {best['score']:.4g} "
            f"({direction} is better)",
            f"  parameters: {parameters or 'strategy defaults'}",
            f"  {summary['successful_evaluations']} evaluated, "
            f"{summary['failed_evaluations']} failed, "
            f"{summary['undefined_objectives']} undefined {self._objective}",
        ]
        if "trial" in best:
            lines.append(
                f"  selected trial {best['trial']} (simulation_seed={best['simulation_seed']}): "
                "one in-sample trial, not an average across repeats"
            )
        details = "optimizer.best_result" if best["result"] is not None else "best_parameters"
        failures = ", optimizer.failures" if self.failures else ""
        lines.append(f"  inspect: {details}{failures}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self._name}')"
