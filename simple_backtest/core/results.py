"""Results container classes for backtesting with helper methods."""

from collections import Counter
from copy import deepcopy
from difflib import get_close_matches
from html import escape
from pathlib import Path
from textwrap import fill, wrap
from typing import Any, Dict, List, Optional

import pandas as pd

from simple_backtest.core.serialization import write_json
from simple_backtest.metrics.objectives import metric_is_maximized


def _metric_hint(metric: str, available: list[str]) -> str:
    matches = get_close_matches(metric, available, n=1) if isinstance(metric, str) else []
    suggestion = f" Did you mean '{matches[0]}'?" if matches else ""
    return f"Available metrics: {available}.{suggestion}"


def _format_value(value: Any, precision: str = ".2f", suffix: str = "", prefix: str = "") -> str:
    return f"{prefix}{value:{precision}}{suffix}" if pd.notna(value) else "n/a"


class StrategyResult:
    """Results for a single strategy with convenience methods.

    Attributes:
        name: Strategy name
        metrics: Dict of performance metrics
        portfolio_values: Closing equity, including a pre-trading cash baseline
        trade_history: List of trade dictionaries
        returns: Trading-bar returns, including the first fill's costs
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
        self.order_outcomes = data.get("order_outcomes", [])
        self._diagnostics_recorded = {key for key in ("errors", "order_outcomes") if key in data}
        self._metadata = deepcopy(data.get("metadata", {}))

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return a detached snapshot of this simulation's provenance."""
        return deepcopy(self._metadata)

    def export_json(self, path: str | Path) -> None:
        """Export equity, trades, diagnostics, metrics, and provenance as portable JSON."""
        write_json(path, {"schema_version": 1, "name": self.name, **self._to_dict()})

    def export_html(self, path: str | Path, price_data: pd.DataFrame | None = None) -> Path:
        """Write one self-contained offline HTML report for this result.

        The report holds metrics, diagnostics, recorded assumptions, an equity and
        drawdown chart and, when ``price_data`` is given, the trade chart.

        :param path: Output .html file path
        :param price_data: Optional price DataFrame you gave to Backtest, for trade context
        :return: The written path
        """
        from simple_backtest.visualization.report import write_report

        results = BacktestResults({})
        results._strategies[self.name] = self
        return write_report(path, results, price_data)

    def explain_error(
        self,
        error_index: int = 0,
        price_data: pd.DataFrame | None = None,
        context_bars: int = 5,
    ) -> str:
        """Explain one recorded strategy error in plain text, without re-running anything.

        Shows the error's time, stage, message, recorded source location, what the
        stage means for timing and fills, and the fills on that bar. With
        ``price_data`` (the DataFrame you gave to Backtest) it also shows up to
        ``context_bars`` bars before the error and, labelled separately, the bar
        where orders execute. Supplied prices are checked against the recorded data
        fingerprint and labelled when they differ.

        :param error_index: Position in ``result.errors`` (negative counts from the end)
        :param price_data: Optional price DataFrame for context
        :param context_bars: Bars of prior context to show, from 1 to 200
        :return: Multi-line explanation; print() it
        """
        from simple_backtest.core.error_context import explain_error

        return explain_error(self, error_index, price_data, context_bars)

    def summary(self) -> str:
        """Get formatted metrics and an explanation of any rejected orders.

        :return: Pretty-formatted string of metrics
        """
        from simple_backtest.metrics.calculator import format_metrics

        summary = format_metrics(self.metrics)
        rejection_summary = self._rejection_summary()
        if rejection_summary:
            summary += f"\n\nOrder Outcomes:\n  {rejection_summary}"
        return summary

    def _rejection_summary(self) -> str | None:
        """Explain rejections once, without treating normal outcomes as errors."""
        reasons = Counter(
            outcome.get("reason") or "unknown"
            for outcome in self.order_outcomes
            if outcome["status"] == "rejected"
        )
        if not reasons:
            return None

        reason, reason_count = reasons.most_common(1)[0]
        hints = {
            "insufficient_cash": (
                "Request fewer shares or use self.buy_budget(self.get_cash()) "
                "to size buys within available cash, including commission."
            ),
            "insufficient_shares": "Sell no more than your held shares (self.get_position()).",
            "zero_size": "Use a positive size or budget when placing an order; use self.hold() to wait.",
            "zero_volume": "Check Volume data and max_volume_participation; this bar permits no fill.",
        }
        hint = hints.get(reason, "Inspect result.order_outcomes for the rejected orders and reasons.")
        final_cash = self.trade_history[-1].get("cash") if self.trade_history else None
        if reason == "zero_size" and final_cash is not None and abs(final_cash) <= 1e-8:
            hint = (
                "Available cash was exhausted; lower investment_amount or raise initial_capital "
                "to keep investing, or use self.hold() while no cash is available."
            )
        count = sum(reasons.values())
        return (
            f"{count} order{'s' if count != 1 else ''} rejected; "
            f"main reason: {reason} ({reason_count}). {hint}"
        )

    def plot_equity_curve(self, show_drawdown: bool = True):
        """Plot equity curve for this strategy.

        :param show_drawdown: Whether to show drawdown subplot
        :return: Plotly figure
        """
        from simple_backtest.visualization.plotter import plot_equity_curve

        return plot_equity_curve({self.name: self._to_dict()}, show_drawdown=show_drawdown)

    def plot_trades(self, price_data):
        """Plot price chart with buy/sell markers for this strategy.

        The chart covers the whole trading period, even without fills, and notes
        below it explain rejected orders, strategy errors or missing prices.

        :param price_data: Original OHLCV DataFrame with DatetimeIndex
        :return: Plotly figure
        """
        from simple_backtest.visualization.plotter import plot_strategy_trades

        figures = plot_strategy_trades(price_data, {self.name: self._plot_dict()})
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
            raise KeyError(f"Metric '{metric_name}' not found. {_metric_hint(metric_name, available)}")
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
            "order_outcomes": self.order_outcomes,
            "metadata": self.metadata,
        }

    def _plot_dict(self) -> Dict[str, Any]:
        """Like _to_dict, but leave out diagnostics a legacy result never recorded."""
        data = self._to_dict()
        for key in {"errors", "order_outcomes"} - self._diagnostics_recorded:
            del data[key]
        return data

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

    def __str__(self) -> str:
        """Show the result's key metrics and diagnostics with explicit units."""
        lines = [fill(self.name, width=120)]
        lines.extend(f"  {label}: {formatted}" for label, formatted in self._metric_rows())
        lines.extend(fill(note, width=120) for note in self._diagnostic_notes())
        return "\n".join(lines)

    def _metric_rows(self) -> list[tuple[str, str]]:
        """Format the key metrics once for printed and HTML reports."""
        rows = []
        closed_trades = any(trade.get("signal") == "sell" for trade in self.trade_history)
        for metric, label, precision, suffix, prefix in (
            ("total_return", "Total Return", ".2f", "%", ""),
            ("cagr", "CAGR", ".2f", "%", ""),
            ("max_drawdown", "Max Drawdown", ".2f", "%", ""),
            ("sharpe_ratio", "Sharpe", ".2f", "", ""),
            ("total_trades", "Trades", ".0f", "", ""),
            ("final_value", "Final Value", ",.2f", "", "$"),
            ("win_rate", "Win Rate", ".2f", "%", ""),
            ("profit_factor", "Profit Factor", ".2f", "", ""),
        ):
            value = self.metrics.get(metric)
            formatted = _format_value(value, precision, suffix, prefix)
            if metric in ("win_rate", "profit_factor") and not closed_trades:
                formatted = "n/a (no closed trades)"
            rows.append((label, formatted))
        return rows

    def _diagnostic_notes(self, max_message: int | None = None) -> list[str]:
        notes = []
        rejected = self._rejection_summary()
        if rejected:
            notes.append(rejected)
        if self.errors:
            count = len(self.errors)
            first = self.errors[0]
            message = str(first.get("message", ""))
            if max_message is not None and len(message) > max_message:
                message = message[: max_message - 3] + "..."
            notes.append(
                f"{count} strategy error{'s' if count != 1 else ''} recorded; "
                f"first: {first.get('error_type', 'error')}: {message}. "
                "Call result.explain_error() to see where and when it failed; raw records are in result.errors."
            )
        return notes


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

    def export_html(self, path: str | Path, price_data: pd.DataFrame | None = None) -> Path:
        """Write one self-contained offline HTML report for all strategies.

        The report holds the comparison, diagnostics, recorded assumptions, an equity
        and drawdown chart and, when ``price_data`` is given, one trade chart per strategy.

        :param path: Output .html file path
        :param price_data: Optional price DataFrame you gave to Backtest, for trade context
        :return: The written path
        """
        from simple_backtest.visualization.report import write_report

        return write_report(path, self, price_data)

    def export_json(self, path: str | Path) -> None:
        """Export all strategy results and the benchmark in one versioned document."""
        write_json(path, {"schema_version": 1, "results": dict(self.items())})

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
        eligible = self._defined_metric_names(metric)
        best_name = selector(
            eligible,
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
        eligible = self._defined_metric_names(metric)
        worst_name = selector(
            eligible,
            key=lambda name: self._strategies[name].metrics[metric],
        )
        return self._strategies[worst_name]

    def _defined_metric_names(self, metric: str) -> list[str]:
        names = [
            name for name, result in self._strategies.items() if pd.notna(result.metrics[metric])
        ]
        if not names:
            raise ValueError(f"Metric '{metric}' is undefined for every strategy")
        return names

    def _validate_metric(self, metric: str) -> None:
        """Require a comparable metric on every strategy."""
        missing = [
            name for name, result in self._strategies.items() if metric not in result.metrics
        ]
        if missing:
            available = sorted({key for result in self._strategies.values() for key in result.metrics})
            raise KeyError(
                f"Metric '{metric}' is missing for strategies: {missing}. "
                f"{_metric_hint(metric, available)}"
            )

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

        plot_data = {name: result._plot_dict() for name, result in self._strategies.items()}
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
        return (
            f"BacktestResults({len(self._strategies)} strategies; "
            "print(results) for the comparison)"
        )

    def __str__(self) -> str:
        """Show returns, risk and trading activity without pandas display truncation."""
        comparison = self._comparison_table()
        if comparison.empty:
            return "No strategies available."

        name_width = min(32, max(len("Strategy"), *(len(str(n)) for n in comparison.index)))
        widths = [max(len(label), comparison[label].map(len).max()) for label in comparison]
        header = f"{'Strategy':<{name_width}}  " + "  ".join(
            f"{label:>{width}}" for label, width in zip(comparison.columns, widths)
        )
        lines = [header, "-" * len(header)]
        for name, metrics in comparison.iterrows():
            name_lines = wrap(str(name), width=name_width) or [""]
            values = [f"{value:>{width}}" for value, width in zip(metrics, widths)]
            lines.append(f"{name_lines[0]:<{name_width}}  " + "  ".join(values))
            lines.extend(name_lines[1:])
        lines.extend(["", fill(self._return_verdict(), width=120)])
        notes = self._diagnostic_notes()
        if notes:
            lines.append("")
            lines.extend(fill(note, width=120) for note in notes)
        return "\n".join(lines)

    def _comparison_table(self) -> pd.DataFrame:
        """Build the same display values for terminal and notebook reports."""
        columns = {
            "total_return": ("Return", "%"),
            "cagr": ("CAGR", "%"),
            "max_drawdown": ("Max Drawdown", "%"),
            "sharpe_ratio": ("Sharpe", ""),
            "total_trades": ("Trades", ""),
        }
        table = self.compare(metrics=list(columns))
        if not table.empty:
            for metric, (_, suffix) in columns.items():
                precision = ".0f" if metric == "total_trades" else ".2f"
                table[metric] = table[metric].map(
                    lambda value: _format_value(value, precision, suffix)
                )
        table.index.name = "Strategy"
        return table.rename(columns={metric: label for metric, (label, _) in columns.items()})

    def _return_verdict(self) -> str:
        eligible = [
            result
            for result in self._strategies.values()
            if pd.notna(result.metrics.get("total_return"))
        ]
        if eligible:
            best = max(eligible, key=lambda result: result.metrics["total_return"])
            best_return = best.metrics["total_return"]
            verdict = f"Best total return: {best.name} ({best_return:.2f}%)."
            benchmark_return = self.benchmark.metrics.get("total_return") if self.benchmark else None
            if pd.notna(benchmark_return):
                difference = best_return - benchmark_return
                if difference == 0:
                    verdict += " It matched benchmark."
                else:
                    direction = "beat" if difference > 0 else "trailed"
                    verdict += (
                        f" It {direction} benchmark by {abs(difference):.2f} percentage points."
                    )
            else:
                verdict += " No benchmark available for comparison."
        else:
            verdict = "No strategy has a defined total return."
        return verdict

    def _diagnostic_notes(self) -> list[str]:
        results = list(self._strategies.values())
        if self.benchmark is not None:
            results.append(self.benchmark)
        return [f"{result.name}: {note}" for result in results for note in result._diagnostic_notes()]

    def _repr_html_(self) -> str:
        """Render the same comparison and diagnostics in a notebook cell."""
        table = self._comparison_table()
        if table.empty:
            return "<p>No strategies available.</p>"
        report = table.to_html(escape=True, border=0, max_rows=None, max_cols=None)
        report += f"<p>{escape(self._return_verdict())}</p>"
        notes = self._diagnostic_notes()
        if notes:
            report += "<ul>" + "".join(f"<li>{escape(note)}</li>" for note in notes) + "</ul>"
        return report

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
