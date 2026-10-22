"""Interactive visualization suite using Plotly."""

from collections.abc import Mapping
from html import escape
from numbers import Real
from textwrap import wrap
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simple_backtest.utils.validation import _normalize_ohlcv_columns

# Color palette for consistent styling
COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]


def _convert_results_to_dict(results) -> Dict[str, Dict[str, Any]]:
    """Convert BacktestResults object to dict format for plotting.

    Args:
        results: BacktestResults object or dict

    Returns:
        Dict mapping strategy names to result dicts
    """
    from simple_backtest.core.results import BacktestResults

    if isinstance(results, BacktestResults):
        results_dict = {}
        for name in results.list_strategies():
            strategy_result = results.get_strategy(name)
            results_dict[name] = {
                "portfolio_values": strategy_result.portfolio_values,
                "trade_history": strategy_result.trade_history,
                "returns": strategy_result.returns,
                "metrics": strategy_result.metrics,
                "errors": strategy_result.errors,
                "order_outcomes": strategy_result.order_outcomes,
                "metadata": strategy_result.metadata,
            }
        if results.benchmark:
            results_dict["benchmark"] = {
                "portfolio_values": results.benchmark.portfolio_values,
                "trade_history": results.benchmark.trade_history,
                "returns": results.benchmark.returns,
                "metrics": results.benchmark.metrics,
                "metadata": results.benchmark.metadata,
            }
        return results_dict

    return results


def plot_equity_curve(results, show_drawdown: bool = False) -> go.Figure:
    """Plot equity curves for all strategies and benchmark.

    Args:
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries

    Returns:
        Plotly Figure with equity curves
    """
    results = _convert_results_to_dict(results)
    fig = (
        make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.7, 0.3],
        )
        if show_drawdown
        else go.Figure()
    )

    for i, (name, result) in enumerate(results.items()):
        portfolio_values = result["portfolio_values"]
        color = COLORS[i % len(COLORS)]

        # Make benchmark dashed and gray
        if name == "benchmark":
            line_style = dict(color="gray", dash="dash", width=2)
        else:
            line_style = dict(color=color, width=2)

        equity_trace = go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            mode="lines",
            name=name,
            line=line_style,
            hovertemplate="<b>%{fullData.name}</b><br>"
            + "Date: %{x}<br>"
            + "Value: $%{y:,.2f}<br>"
            + "<extra></extra>",
        )
        if show_drawdown:
            fig.add_trace(equity_trace, row=1, col=1)
            running_max = portfolio_values.cummax()
            drawdown = (portfolio_values - running_max) / running_max * 100
            fig.add_trace(
                go.Scatter(
                    x=drawdown.index,
                    y=drawdown.values,
                    mode="lines",
                    name=f"{name} drawdown",
                    line=line_style,
                    showlegend=False,
                    hovertemplate="<b>%{fullData.name}</b><br>"
                    + "Date: %{x}<br>"
                    + "Drawdown: %{y:.2f}%<br>"
                    + "<extra></extra>",
                ),
                row=2,
                col=1,
            )
        else:
            fig.add_trace(equity_trace)

    fig.update_layout(
        title="Portfolio Equity Curve",
        hovermode="x unified",
        template="plotly_white",
        height=700 if show_drawdown else 500,
    )
    fig.update_xaxes(
        title_text="Date", row=2 if show_drawdown else None, col=1 if show_drawdown else None
    )
    if show_drawdown:
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    else:
        fig.update_yaxes(title_text="Portfolio Value ($)")

    return fig


def plot_drawdowns(results) -> go.Figure:
    """Plot drawdown chart for all strategies.

    Args:
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries

    Returns:
        Plotly Figure with drawdown curves
    """
    results = _convert_results_to_dict(results)
    fig = go.Figure()

    for i, (name, result) in enumerate(results.items()):
        portfolio_values = result["portfolio_values"]
        color = COLORS[i % len(COLORS)]

        # Calculate drawdown
        running_max = portfolio_values.cummax()
        drawdown = (portfolio_values - running_max) / running_max * 100

        # Make benchmark dashed
        if name == "benchmark":
            line_style = dict(color="gray", dash="dash", width=2)
            fill = None
        else:
            line_style = dict(color=color, width=2)
            fill = "tozeroy"

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name=name,
                line=line_style,
                fill=fill,
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "Date: %{x}<br>"
                + "Drawdown: %{y:.2f}%<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title="Drawdown Over Time",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    return fig


def plot_returns_distribution(results) -> go.Figure:
    """Plot returns distribution histograms.

    Args:
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries

    Returns:
        Plotly Figure with return distributions
    """
    results = _convert_results_to_dict(results)
    fig = go.Figure()

    for i, (name, result) in enumerate(results.items()):
        returns = result["returns"] * 100  # Convert to percentage
        color = COLORS[i % len(COLORS)]

        fig.add_trace(
            go.Histogram(
                x=returns,
                name=name,
                marker_color=color,
                opacity=0.7,
                nbinsx=50,
                bingroup="returns",  # same bin edges for every run, so counts compare
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "Return per bar: %{x:.2f}%<br>"
                + "Count: %{y}<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Return per bar (%)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        barmode="overlay",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def plot_monthly_returns(results) -> Dict[str, go.Figure]:
    """Plot monthly returns heatmap for each strategy.

    Args:
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries

    Returns:
        Dictionary mapping strategy names to their heatmap figures
    """
    results = _convert_results_to_dict(results)
    figures = {}

    for name, result in results.items():
        if name == "benchmark":
            continue  # Skip benchmark

        returns = result["returns"]

        # Resample to monthly returns
        monthly_returns = (1 + returns).resample("ME").prod() - 1
        monthly_returns = monthly_returns * 100  # Convert to percentage

        # Create year-month matrix
        df = pd.DataFrame({"return": monthly_returns})
        df["year"] = df.index.year
        df["month"] = df.index.month

        # Pivot to create heatmap data
        heatmap_data = df.pivot(index="year", columns="month", values="return")

        # Month names
        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=heatmap_data.values,
                x=[month_names[i - 1] for i in heatmap_data.columns],
                y=heatmap_data.index,
                colorscale="RdYlGn",
                zmid=0,
                text=heatmap_data.values,
                texttemplate="%{text:.1f}%",
                textfont={"size": 10},
                hovertemplate="Year: %{y}<br>"
                + "Month: %{x}<br>"
                + "Return: %{z:.2f}%<br>"
                + "<extra></extra>",
            )
        )

        fig.update_layout(
            title=f"Monthly Returns Heatmap - {name}",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
            height=400,
        )

        figures[name] = fig

    return figures


def plot_trades(results) -> Dict[str, go.Figure]:
    """Plot trade scatter for each strategy.

    Args:
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries

    Returns:
        Dictionary mapping strategy names to their trade scatter figures
    """
    results = _convert_results_to_dict(results)
    figures = {}

    for name, result in results.items():
        if name == "benchmark":
            continue

        trade_history = result["trade_history"]
        sell_trades = [t for t in trade_history if t["signal"] == "sell"]

        if not sell_trades:
            continue

        dates = [t["timestamp"] for t in sell_trades]
        pnls = [t["pnl"] for t in sell_trades]
        colors_list = ["green" if pnl > 0 else "red" for pnl in pnls]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=pnls,
                mode="markers",
                marker=dict(
                    size=8,
                    color=colors_list,
                    line=dict(width=1, color="white"),
                ),
                name="Trades",
                hovertemplate="Date: %{x}<br>" + "P&L: $%{y:.2f}<br>" + "<extra></extra>",
            )
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

        fig.update_layout(
            title=f"Trade P&L Analysis - {name}",
            xaxis_title="Date",
            yaxis_title="P&L ($)",
            template="plotly_white",
            height=400,
        )

        figures[name] = fig

    return figures


# Compatible defaults for results that do not record their own risk assumptions.
FALLBACK_PERIODS_PER_YEAR = 252
FALLBACK_RISK_FREE_RATE = 0.0


def _recorded_number(name: str, field: str, value, valid, expected: str):
    """Return a recorded numeric assumption, None when absent, or raise if it is invalid."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or not valid(value)
    ):
        raise ValueError(
            f"{name}: recorded {field}={repr(value)[:40]} is invalid; expected {expected}. "
            f"Correct or remove it in the result metadata."
        )
    return float(value)


def _risk_assumptions(name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve one run's annualization and annual risk-free rate for rolling metrics.

    Each field comes from the run's recorded metadata when present; a missing field
    falls back to 252 periods/year or a 0% target and is marked as assumed.
    """
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    config = metadata.get("config")
    config = config if isinstance(config, Mapping) else {}
    periods = _recorded_number(
        name,
        "periods_per_year",
        metadata.get("periods_per_year"),
        lambda value: value > 0,
        "a finite positive number",
    )
    risk_free = _recorded_number(
        name,
        "risk_free_rate",
        config.get("risk_free_rate"),
        lambda value: value > -1,
        "a finite annual rate above -1",
    )
    return {
        "periods": FALLBACK_PERIODS_PER_YEAR if periods is None else periods,
        "periods_recorded": periods is not None,
        "risk_free": FALLBACK_RISK_FREE_RATE if risk_free is None else risk_free,
        "risk_free_recorded": risk_free is not None,
    }


def _assumption_text(assumptions: Dict[str, Any]) -> str:
    periods = "recorded" if assumptions["periods_recorded"] else "assumed"
    risk_free = "recorded" if assumptions["risk_free_recorded"] else "assumed"
    return (
        f"{assumptions['periods']:g} periods/year ({periods}), "
        f"risk-free {assumptions['risk_free']:.2%}/year ({risk_free})"
    )


def _named(names: list[str]) -> str:
    shown = ", ".join(escape(name) for name in names[:3])
    return shown + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def plot_rolling_metrics(results, window: int = 30) -> go.Figure:
    """Plot rolling annualized Sharpe ratio and volatility over a window of bars.

    Each run uses its own recorded ``periods_per_year`` and annual
    ``config.risk_free_rate`` from result metadata, as the result's metrics do.
    A run without them uses 252 periods/year and a 0% target; the note below the
    chart marks those values as assumed. A window whose returns do not vary has
    a Sharpe ratio of 0. No value is shown before a window has ``window`` returns.

    Args:
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries
        window: Rolling window size in bars (return observations)

    Returns:
        Plotly Figure with rolling metrics
    """
    results = _convert_results_to_dict(results)
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Rolling Sharpe Ratio (annualized)", "Rolling Volatility (annualized %)"),
        vertical_spacing=0.15,
    )

    groups: Dict[str, list[str]] = {}
    too_short = []
    for i, (name, result) in enumerate(results.items()):
        returns = result["returns"]
        color = COLORS[i % len(COLORS)]
        assumptions = _risk_assumptions(name, result)
        periods = assumptions["periods"]
        described = _assumption_text(assumptions)
        groups.setdefault(described, []).append(name)
        if len(returns) < window:
            too_short.append(name)

        # Same definitions as calculate_sharpe_ratio and calculate_volatility per window.
        period_risk_free = (1 + assumptions["risk_free"]) ** (1 / periods) - 1
        excess = (returns - period_risk_free).rolling(window)
        excess_std = excess.std()
        rolling_sharpe = (
            (excess.mean() / excess_std * np.sqrt(periods))
            .where(excess_std != 0, 0.0)
            .where(excess_std.notna())
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(periods) * 100

        hover = (
            f"<b>{escape(name)}</b><br>{{label}}: %{{y:.2f}}<br>"
            f"{window}-bar window, {escape(described)}<extra></extra>"
        )

        # Sharpe subplot
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode="lines",
                name=f"{name} (Sharpe)",
                line=dict(color=color, width=2),
                legendgroup=name,
                hovertemplate=hover.replace("{label}", "Sharpe"),
            ),
            row=1,
            col=1,
        )

        # Volatility subplot
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode="lines",
                name=f"{name} (Vol)",
                line=dict(color=color, width=2),
                legendgroup=name,
                showlegend=False,
                hovertemplate=hover.replace("{label}", "Volatility %"),
            ),
            row=2,
            col=1,
        )

    note_lines = [f"{_named(names)}: {escape(text)}" for text, names in groups.items()][:5]
    if len(groups) > 5:
        note_lines.append(f"{len(groups) - 5} more assumption sets; hover a line to see its own.")
    if too_short:
        note_lines.append(
            f"Fewer than {window} returns, so no rolling values: {_named(too_short)}."
        )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    if note_lines:
        fig.add_annotation(
            text="<br>".join(note_lines),
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            yshift=-50,
            xanchor="left",
            yanchor="top",
            align="left",
            showarrow=False,
            font=dict(size=12),
        )

    fig.update_layout(
        title=f"Rolling Metrics ({window}-bar window)",
        template="plotly_white",
        height=700 + 18 * len(note_lines),
        margin=dict(b=80 + 18 * len(note_lines)),
        hovermode="x unified",
    )

    return fig


def create_comparison_table(results) -> go.Figure:
    """Create comparison table for all strategies.

    Args:
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries

    Returns:
        Plotly Figure with comparison table
    """
    results = _convert_results_to_dict(results)
    # Extract metrics
    metric_names = [
        "total_return",
        "cagr",
        "volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "win_rate",
        "profit_factor",
        "total_trades",
    ]

    metric_labels = [
        "Total Return (%)",
        "CAGR (%)",
        "Volatility (%)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown (%)",
        "Win Rate (%)",
        "Profit Factor",
        "Total Trades",
    ]

    # Build table data
    header = ["Metric"] + list(results.keys())
    cells = []

    for label, metric_key in zip(metric_labels, metric_names):
        row = [label]
        for name in results.keys():
            value = results[name]["metrics"].get(metric_key, 0)
            # Format based on metric type
            if metric_key == "total_trades":
                row.append(f"{int(value)}")
            else:
                row.append(f"{value:.2f}")
        cells.append(row)

    # Transpose for table format
    table_data = list(zip(*cells))

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=header,
                    fill_color="lightblue",
                    align="left",
                    font=dict(size=12, color="black"),
                ),
                cells=dict(
                    values=table_data,
                    fill_color="white",
                    align="left",
                    font=dict(size=11),
                ),
            )
        ]
    )

    fig.update_layout(title="Strategy Comparison", template="plotly_white", height=400)

    return fig


def _trade_window(price_data: pd.DataFrame, result: Dict[str, Any], name: str):
    """Select the prices for the result's own timeline and describe any missing dates."""
    timeline = next(
        (
            series.index
            for series in (result.get("returns"), result.get("portfolio_values"))
            if series is not None and len(series)
        ),
        None,
    )
    if timeline is None:
        return price_data, []

    start, end = timeline[0], timeline[-1]
    plot_data = price_data[(price_data.index >= start) & (price_data.index <= end)]
    if plot_data.empty:
        raise ValueError(
            f"price_data has no prices between {start} and {end}, the dates covered by "
            f"'{name}'. Pass the same price DataFrame you gave to Backtest."
        )
    missing = timeline.difference(price_data.index)
    if missing.empty:
        return plot_data, []
    shown = ", ".join(
        str(date.date()) if date == date.normalize() else str(date) for date in missing[:3]
    )
    more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
    return plot_data, [
        f"Incomplete price data: {len(missing)} of {len(timeline)} backtest dates have no price "
        f"({shown}{more}). This chart does not show the full run; pass the same price "
        "DataFrame you gave to Backtest."
    ]


def _run_notes(name: str, result: Dict[str, Any]) -> list[str]:
    """Explain missing fills with the result's own rejection and error diagnostics."""
    from simple_backtest.core.results import StrategyResult

    has_diagnostics = "errors" in result or "order_outcomes" in result
    notes = (
        # Timelines are optional for plotting; diagnostics never read them.
        StrategyResult(
            name, {"returns": None, "portfolio_values": None, **result}
        )._diagnostic_notes(max_message=120)
        if has_diagnostics
        else []
    )
    if not result["trade_history"]:
        # Only recorded order attempts show that the strategy never placed one.
        if "order_outcomes" in result and not notes:
            notes.insert(0, "No orders were placed: the strategy held on every trading bar.")
        else:
            notes.insert(0, "No fills: no order was executed in this run.")
    return notes


# Pixels reserved below a trade chart's axis for its five-line performance box.
SUMMARY_HEIGHT = 110


def plot_strategy_trades(
    price_data: pd.DataFrame,
    results,
) -> Dict[str, go.Figure]:
    """Plot price chart with buy/sell markers for each strategy.

    Creates a separate figure for each strategy, including strategies without fills, showing:
    - Price data (Close price) over the strategy's whole trading period
    - Buy markers (green up triangles)
    - Sell markers (red down triangles)
    - Profit/Loss summary text
    - Notes below the chart for no fills, rejected orders, strategy errors or missing prices

    Args:
        price_data: Original OHLCV DataFrame with DatetimeIndex
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries

    Returns:
        Dictionary mapping strategy names to their trade plot figures

    Raises:
        ValueError: If price_data has no prices inside a strategy's trading period

    Example:
        >>> figures = plot_strategy_trades(data, results)
        >>> for fig in figures.values():
        ...     fig.show()
    """
    price_data = _normalize_ohlcv_columns(price_data)
    results = _convert_results_to_dict(results)
    figures = {}

    for name, result in results.items():
        # Skip benchmark
        if name == "benchmark":
            continue

        trade_history = result["trade_history"]
        metrics = result["metrics"]
        plot_data, notes = _trade_window(price_data, result, name)
        incomplete = bool(notes)
        notes += _run_notes(name, result)

        fig = go.Figure()

        # Plot close price
        price_trace = go.Scatter(
            x=plot_data.index,
            y=plot_data["Close"],
            mode="lines",
            name="Close Price",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
        )
        fig.add_trace(price_trace)

        # Separate buy and sell trades
        buy_trades = [t for t in trade_history if t["signal"] == "buy"]
        sell_trades = [t for t in trade_history if t["signal"] == "sell"]

        # Plot buy markers
        if buy_trades:
            buy_dates = [t["timestamp"] for t in buy_trades]
            buy_prices = [t["price"] for t in buy_trades]
            buy_shares = [t["shares"] for t in buy_trades]

            buy_marker = go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                name="Buy",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="green",
                    line=dict(color="darkgreen", width=2),
                ),
                hovertemplate="<b>BUY</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>"
                + "Shares: %{customdata:.2f}<extra></extra>",
                customdata=buy_shares,
            )
            fig.add_trace(buy_marker)

        # Plot sell markers
        if sell_trades:
            sell_dates = [t["timestamp"] for t in sell_trades]
            sell_prices = [t["price"] for t in sell_trades]
            sell_shares = [t["shares"] for t in sell_trades]
            sell_pnl = [t.get("pnl", 0) for t in sell_trades]

            sell_marker = go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode="markers",
                name="Sell",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="red",
                    line=dict(color="darkred", width=2),
                ),
                hovertemplate="<b>SELL</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>"
                + "Shares: %{customdata[0]:.2f}<br>"
                + "P&L: $%{customdata[1]:.2f}<extra></extra>",
                customdata=list(zip(sell_shares, sell_pnl)),
            )
            fig.add_trace(sell_marker)

        # Calculate performance metrics for annotation
        total_return = metrics.get("total_return", 0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        win_rate = metrics.get("win_rate", 0)
        total_trades = metrics.get("total_trades", 0)

        # Determine profit/loss color
        pnl_color = "green" if total_return > 0 else "red" if total_return < 0 else "black"

        # Add annotation with strategy performance
        annotation_text = (
            f"<b>Strategy Performance</b><br>"
            f"Total Return: <span style='color:{pnl_color}'>{total_return:+.2f}%</span><br>"
            f"Sharpe Ratio: {sharpe_ratio:.2f}<br>"
            # Win statistics only apply once a trade has closed.
            f"Win Rate: {f'{win_rate:.2f}%' if sell_trades else 'n/a (no closed trades)'}<br>"
            f"Total Trades: {int(total_trades)}"
        )

        # Summary and run notes sit below the axis so they never cover prices or markers.
        fig.add_annotation(
            text=annotation_text,
            xref="paper",
            yref="paper",
            x=0,
            y=0,
            yshift=-60,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            borderpad=10,
            font=dict(size=11),
        )

        note_lines = [escape(line) for note in notes for line in wrap(note, 110)]
        if note_lines:
            fig.add_annotation(
                text="<br>".join(note_lines),
                xref="paper",
                yref="paper",
                x=0,
                y=0,
                yshift=-60 - SUMMARY_HEIGHT,
                xanchor="left",
                yanchor="top",
                align="left",
                showarrow=False,
                font=dict(size=12, color="darkred" if incomplete else "black"),
            )

        title = f"{escape(name)} - Trading Signals"
        if incomplete:
            title += " (incomplete price data)"
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=600 + SUMMARY_HEIGHT + 18 * len(note_lines),
            margin=dict(b=80 + SUMMARY_HEIGHT + 18 * len(note_lines)),
            hovermode="closest",
            showlegend=True,
            # Above the plot area: covers no price or marker and keeps the full width.
            legend=dict(orientation="h", x=0, y=1, xanchor="left", yanchor="bottom"),
        )

        figures[name] = fig

    return figures


def plot_all(results) -> None:
    """Display all plots in sequence.

    Args:
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries
    """
    results = _convert_results_to_dict(results)
    # Equity curve
    fig = plot_equity_curve(results)
    fig.show()

    # Drawdowns
    fig = plot_drawdowns(results)
    fig.show()

    # Returns distribution
    fig = plot_returns_distribution(results)
    fig.show()

    # Monthly returns heatmaps
    monthly_figs = plot_monthly_returns(results)
    for fig in monthly_figs.values():
        fig.show()

    # Trade analysis
    trade_figs = plot_trades(results)
    for fig in trade_figs.values():
        fig.show()

    # Rolling metrics
    fig = plot_rolling_metrics(results)
    fig.show()

    # Comparison table
    fig = create_comparison_table(results)
    fig.show()


def plot_search_trials(
    summary: pd.DataFrame,
    param_names: list[str],
    points: list[Dict[str, Any]],
    selected_score: Any = None,
    title: str = "Search",
) -> go.Figure:
    """Plot every observed objective value per tested combination of a search.

    Each combination has its own position on a categorical axis (no interpolation
    between untested settings). Combinations without plottable values stay visible
    with their counts, and the selected individual trial is drawn as a star.
    """
    info = summary.attrs.get("trial_summary", {})
    objective, unit = info.get("objective", "objective"), info.get("unit", "")
    columns = info.get("parameter_columns", {})
    labels, ticks = [], []
    for row in summary.to_dict("records"):
        label = ", ".join(f"{name}={row[columns.get(name, name)]!r}" for name in param_names)
        label = label or "strategy defaults"
        labels.append(label)
        counts = f"{row['defined']}/{row['attempted']} defined"
        extra = [f"{row[s]} {s}" for s in ("undefined", "failed") if row[s]]
        ticks.append(
            "<br>".join(
                [
                    *(escape(line, quote=False) for line in wrap(label, 30)),
                    ", ".join([counts, *extra]),
                ]
            )
        )
    finite = [p for p in points if np.isfinite(p["objective"])]
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            x=[p["combination"] for p in finite],
            y=[p["objective"] for p in finite],
            boxpoints="all",
            jitter=0.3,
            pointpos=0,
            name="observed trials",
            text=[
                escape(labels[p["combination"]], quote=False)
                + (f"<br>trial {p['trial']}, seed {p['seed']}" if p["trial"] is not None else "")
                for p in finite
            ],
            hovertemplate=f"%{{text}}<br>{objective}=%{{y:.6g}}{unit}<extra></extra>",
        )
    )
    selected = summary.index[summary["contains_selected"]].tolist() if len(summary) else []
    annotations = []
    if selected and selected_score is not None and np.isfinite(selected_score):
        fig.add_trace(
            go.Scatter(
                x=selected,
                y=[selected_score],
                mode="markers",
                marker=dict(symbol="star", size=16, color="gold", line=dict(width=1)),
                name="selected trial (one observation)",
                hovertemplate=f"selected trial<br>{objective}=%{{y:.6g}}{unit}<extra></extra>",
            )
        )
    for position, row in enumerate(summary.to_dict("records")):
        unplotted = sum(
            1 for p in points if p["combination"] == position and not np.isfinite(p["objective"])
        )
        if row["defined"] == 0:
            text = "no defined value<br>" + escape(row["note"].split(";")[0])
        elif unplotted:
            text = f"{unplotted} unbounded value(s) not plotted"
            if position in selected and not np.isfinite(selected_score):
                text += f"<br>selected trial: {objective}={selected_score}"
        else:
            continue
        annotations.append(
            dict(x=position, y=0.5, xref="x", yref="paper", text=text, showarrow=False)
        )
    direction = info.get("direction", "")
    fig.update_layout(
        title=(
            f"{escape(title)}: observed {objective} per tested combination<br>"
            f"<sup>In-sample; {direction}. Each point is one evaluation; boxes show observed "
            "spread, not robustness. The star is the single selected trial.</sup>"
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(len(labels))),
            ticktext=ticks,
            range=[-0.5, max(len(labels), 1) - 0.5],
            title="tested combination",
        ),
        yaxis_title=f"{objective} ({unit})" if unit else objective,
        annotations=annotations,
        template="plotly_white",
    )
    return fig
