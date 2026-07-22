"""Interactive visualization suite using Plotly."""

from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
            }
        if results.benchmark:
            results_dict["benchmark"] = {
                "portfolio_values": results.benchmark.portfolio_values,
                "trade_history": results.benchmark.trade_history,
                "returns": results.benchmark.returns,
                "metrics": results.benchmark.metrics,
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
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
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
                fillcolor=color if name != "benchmark" else None,
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
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
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
                hovertemplate="<b>%{fullData.name}</b><br>"
                + "Return: %{x:.2f}%<br>"
                + "Count: %{y}<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title="Returns Distribution",
        xaxis_title="Daily Return (%)",
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


def plot_rolling_metrics(results, window: int = 30) -> go.Figure:
    """Plot rolling Sharpe and Sortino ratios.

    Args:
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries
        window: Rolling window size

    Returns:
        Plotly Figure with rolling metrics
    """
    results = _convert_results_to_dict(results)
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Rolling Sharpe Ratio", "Rolling Volatility"),
        vertical_spacing=0.15,
    )

    for i, (name, result) in enumerate(results.items()):
        returns = result["returns"]
        color = COLORS[i % len(COLORS)]

        # Rolling Sharpe
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(
            252
        )

        # Rolling Volatility
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100

        # Sharpe subplot
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode="lines",
                name=f"{name} (Sharpe)",
                line=dict(color=color, width=2),
                legendgroup=name,
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
            ),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)

    fig.update_layout(
        title=f"Rolling Metrics ({window}-day window)",
        template="plotly_white",
        height=700,
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


def plot_strategy_trades(
    price_data: pd.DataFrame,
    results,
) -> Dict[str, go.Figure]:
    """Plot price chart with buy/sell markers for each strategy.

    Creates a separate figure for each strategy showing:
    - Price data (Close price)
    - Buy markers (green up triangles)
    - Sell markers (red down triangles)
    - Profit/Loss summary text

    Args:
        price_data: Original OHLCV DataFrame with DatetimeIndex
        results: BacktestResults object or dictionary mapping strategy names to result dictionaries

    Returns:
        Dictionary mapping strategy names to their trade plot figures

    Example:
        >>> figures = plot_strategy_trades(data, results)
        >>> for fig in figures.values():
        ...     fig.show()
    """
    results = _convert_results_to_dict(results)
    figures = {}

    for name, result in results.items():
        # Skip benchmark
        if name == "benchmark":
            continue

        trade_history = result["trade_history"]
        metrics = result["metrics"]

        if not trade_history:
            continue

        fig = go.Figure()

        # Get trading date range
        trade_dates = [t["timestamp"] for t in trade_history]
        if trade_dates:
            start_date = min(trade_dates)
            end_date = max(trade_dates)
            # Add some buffer
            mask = (price_data.index >= start_date) & (price_data.index <= end_date)
            plot_data = price_data[mask]
        else:
            plot_data = price_data

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
        pnl_color = "green" if total_return > 0 else "red"

        # Add annotation with strategy performance
        annotation_text = (
            f"<b>Strategy Performance</b><br>"
            f"Total Return: <span style='color:{pnl_color}'>{total_return:+.2f}%</span><br>"
            f"Sharpe Ratio: {sharpe_ratio:.2f}<br>"
            f"Win Rate: {win_rate:.2f}%<br>"
            f"Total Trades: {int(total_trades)}"
        )

        fig.add_annotation(
            text=annotation_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            borderpad=10,
            font=dict(size=11),
        )

        fig.update_layout(
            title=f"{name} - Trading Signals",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=600,
            hovermode="closest",
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
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
