"""Results container classes for backtesting with helper methods."""

from typing import Any, Dict, List, Optional

import pandas as pd

from simple_backtest.metrics.objectives import metric_is_maximized


class StrategyResult:
    """Results for a single strategy with convenience methods.

    Attributes:
        name: Strategy name
        metrics: Dict of performance metrics
        portfolio_values: Series of portfolio values over time
        trade_history: List of trade dictionaries
        returns: Series of returns
    """

    def __init__(self, name: str, data: Dict[str, Any]):
        """Initialize strategy result.

        :param name: Strategy name
        :param data: Result data dict with keys: metrics, portfolio_values, trade_history, returns
        """
        self.name = name
        self.metrics = data["metrics"]
        self.portfolio_values = data["portfolio_values"]
        self.trade_history = data["trade_history"]
        self.returns = data["returns"]
        self.errors = data.get("errors", [])

    def summary(self) -> str:
        """Get formatted metrics summary.

        :return: Pretty-formatted string of metrics
        """
        from simple_backtest.metrics.calculator import format_metrics

        return format_metrics(self.metrics)

    def plot_equity_curve(self, show_drawdown: bool = True):
        """Plot equity curve for this strategy.

        :param show_drawdown: Whether to show drawdown subplot
        :return: Plotly figure
        """
        from simple_backtest.visualization.plotter import plot_equity_curve

        return plot_equity_curve({self.name: self._to_dict()}, show_drawdown=show_drawdown)

    def plot_trades(self, price_data):
        """Plot price chart with buy/sell markers for this strategy.

        :param price_data: Original OHLCV DataFrame with DatetimeIndex
        :return: Plotly figure
        """
        from simple_backtest.visualization.plotter import plot_strategy_trades

        figures = plot_strategy_trades(price_data, {self.name: self._to_dict()})
        return figures.get(self.name)

    def export_trades(self, path: str) -> None:
        """Export trade history to CSV.

        :param path: Output file path
        """
        if not self.trade_history:
            raise ValueError("No trades to export")

        df = pd.DataFrame(self.trade_history)
        df.to_csv(path, index=False)

    def export_equity(self, path: str) -> None:
        """Export portfolio values to CSV.

        :param path: Output file path
        """
        self.portfolio_values.to_csv(path)

    def export_metrics(self, path: str) -> None:
        """Export metrics to CSV.

        :param path: Output file path
        """
        df = pd.DataFrame([self.metrics])
        df.insert(0, "strategy", self.name)
        df.to_csv(path, index=False)

    def get_metric(self, metric_name: str) -> float:
        """Get specific metric value.

        :param metric_name: Name of metric (e.g., 'sharpe_ratio')
        :return: Metric value
        :raises KeyError: If metric not found
        """
        if metric_name not in self.metrics:
            available = list(self.metrics.keys())
            raise KeyError(f"Metric '{metric_name}' not found. Available metrics: {available}")
        return self.metrics[metric_name]

    def _to_dict(self) -> Dict[str, Any]:
        """Convert to dict format (for backward compatibility).

        :return: Dict with metrics, portfolio_values, trade_history, returns
        """
        return {
            "metrics": self.metrics,
            "portfolio_values": self.portfolio_values,
            "trade_history": self.trade_history,
            "returns": self.returns,
            "errors": self.errors,
        }

    def __getitem__(self, key: str) -> Any:
        """Dict-style access for backward compatibility.

        :param key: Key to access
        :return: Value
        """
        return getattr(self, key)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"StrategyResult(name='{self.name}', "
            f"sharpe={self.metrics.get('sharpe_ratio', 0):.2f}, "
            f"total_return={self.metrics.get('total_return', 0):.2f}%)"
        )


class BacktestResults:
    """Container for backtest results with comparison methods.

    Provides clean API for accessing individual strategy results,
    comparing strategies, and exporting data.
    """

    def __init__(self, results_dict: Dict[str, Dict[str, Any]]):
        """Initialize results container.

        :param results_dict: Dict mapping strategy names to result dicts
        """
        self._strategies = {}
        self.benchmark = None

        for name, data in results_dict.items():
            result = StrategyResult(name, data)
            if name == "benchmark":
                self.benchmark = result
            else:
                self._strategies[name] = result

    def get_strategy(self, name: str) -> StrategyResult:
        """Get results for a specific strategy.

        :param name: Strategy name
        :return: StrategyResult for that strategy
        :raises KeyError: If strategy not found
        """
        if name not in self._strategies:
            available = list(self._strategies.keys())
            raise KeyError(f"Strategy '{name}' not found. Available strategies: {available}")
        return self._strategies[name]

    def list_strategies(self) -> List[str]:
        """Get list of strategy names (excluding benchmark).

        :return: List of strategy names
        """
        return list(self._strategies.keys())

    def compare(
        self, metrics: Optional[List[str]] = None, include_benchmark: bool = True
    ) -> pd.DataFrame:
        """Compare strategies across metrics in a table.

        :param metrics: List of metric names to compare (default: common metrics)
        :param include_benchmark: Whether to include benchmark in comparison
        :return: DataFrame with strategies as rows and metrics as columns
        """
        if metrics is None:
            metrics = [
                "total_return",
                "cagr",
                "sharpe_ratio",
                "sortino_ratio",
                "max_drawdown",
                "volatility",
                "total_trades",
                "win_rate",
                "profit_factor",
            ]

        comparison = {}
        for name, result in self._strategies.items():
            comparison[name] = {m: result.metrics.get(m, None) for m in metrics}

        if include_benchmark and self.benchmark is not None:
            comparison["benchmark"] = {m: self.benchmark.metrics.get(m, None) for m in metrics}

        df = pd.DataFrame(comparison).T
        # Sort by first metric (descending if it's a positive metric)
        if len(df) > 0 and metrics[0] in df.columns:
            df = df.sort_values(metrics[0], ascending=not metric_is_maximized(metrics[0]))

        return df

    def best_strategy(
        self, metric: str = "sharpe_ratio", maximize: bool | None = None
    ) -> StrategyResult:
        """Get best performing strategy by a metric.

        :param metric: Metric to optimize for (default: sharpe_ratio)
        :return: StrategyResult of best strategy
        :raises ValueError: If no strategies available
        """
        if not self._strategies:
            raise ValueError("No strategies available")

        self._validate_metric(metric)
        should_maximize = metric_is_maximized(metric) if maximize is None else maximize

        selector = max if should_maximize else min
        best_name = selector(
            self._strategies.keys(),
            key=lambda name: self._strategies[name].metrics[metric],
        )
        return self._strategies[best_name]

    def worst_strategy(
        self, metric: str = "sharpe_ratio", maximize: bool | None = None
    ) -> StrategyResult:
        """Get worst performing strategy by a metric.

        :param metric: Metric to optimize for (default: sharpe_ratio)
        :return: StrategyResult of worst strategy
        :raises ValueError: If no strategies available
        """
        if not self._strategies:
            raise ValueError("No strategies available")

        self._validate_metric(metric)
        should_maximize = metric_is_maximized(metric) if maximize is None else maximize

        selector = min if should_maximize else max
        worst_name = selector(
            self._strategies.keys(),
            key=lambda name: self._strategies[name].metrics[metric],
        )
        return self._strategies[worst_name]

    def _validate_metric(self, metric: str) -> None:
        """Require a comparable metric on every strategy."""
        missing = [
            name for name, result in self._strategies.items() if metric not in result.metrics
        ]
        if missing:
            raise KeyError(f"Metric '{metric}' is missing for strategies: {missing}")

    def plot_comparison(self, include_benchmark: bool = True):
        """Plot equity curves for all strategies.

        :param include_benchmark: Whether to include benchmark
        :return: Plotly figure
        """
        from simple_backtest.visualization.plotter import plot_equity_curve

        plot_data = {name: result._to_dict() for name, result in self._strategies.items()}

        if include_benchmark and self.benchmark is not None:
            plot_data["benchmark"] = self.benchmark._to_dict()

        return plot_equity_curve(plot_data)

    def plot_trades(self, price_data) -> Dict[str, Any]:
        """Plot price charts with buy/sell markers for all strategies.

        Creates separate figures for each strategy (excludes benchmark).

        :param price_data: Original OHLCV DataFrame with DatetimeIndex
        :return: Dict mapping strategy names to Plotly figures
        """
        from simple_backtest.visualization.plotter import plot_strategy_trades

        plot_data = {name: result._to_dict() for name, result in self._strategies.items()}
        return plot_strategy_trades(price_data, plot_data)

    def export_all_metrics(self, path: str) -> None:
        """Export metrics for all strategies to CSV.

        :param path: Output file path
        """
        rows = []
        for name, result in self._strategies.items():
            row = {"strategy": name, **result.metrics}
            rows.append(row)

        if self.benchmark is not None:
            row = {"strategy": "benchmark", **self.benchmark.metrics}
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)

    def export_all_trades(self, path: str) -> None:
        """Export trade history for all strategies to CSV.

        :param path: Output file path
        """
        all_trades = []
        for name, result in self._strategies.items():
            for trade in result.trade_history:
                all_trades.append({"strategy": name, **trade})

        if not all_trades:
            raise ValueError("No trades to export")

        df = pd.DataFrame(all_trades)
        df.to_csv(path, index=False)

    def __getitem__(self, key: str) -> Dict[str, Any] | None:
        """Dict-style access for backward compatibility.

        :param key: Strategy name
        :return: Strategy result as dict
        """
        if key == "benchmark":
            return self.benchmark._to_dict() if self.benchmark else None
        return self._strategies[key]._to_dict()

    def __len__(self) -> int:
        """Get number of strategies (excluding benchmark)."""
        return len(self._strategies)

    def __iter__(self):
        """Iterate over strategy names."""
        return iter(self._strategies.keys())

    def __repr__(self) -> str:
        """String representation."""
        return f"BacktestResults({len(self._strategies)} strategies)"

    def items(self):
        """Iterate over (name, result_dict) pairs for backward compatibility.

        Returns:
            Iterator of (strategy_name, result_dict) tuples
        """
        for name, result in self._strategies.items():
            yield name, result._to_dict()
        if self.benchmark:
            yield "benchmark", self.benchmark._to_dict()

    def keys(self):
        """Get strategy names for backward compatibility.

        Returns:
            Iterator of strategy names
        """
        for name in self._strategies.keys():
            yield name
        if self.benchmark:
            yield "benchmark"

    def values(self):
        """Get result dicts for backward compatibility.

        Returns:
            Iterator of result dictionaries
        """
        for result in self._strategies.values():
            yield result._to_dict()
        if self.benchmark:
            yield self.benchmark._to_dict()
