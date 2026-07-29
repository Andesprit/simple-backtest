"""Smoke and semantic checks for every visualization entry point."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from simple_backtest.visualization.plotter import (
    create_comparison_table,
    plot_drawdowns,
    plot_equity_curve,
    plot_monthly_returns,
    plot_returns_distribution,
    plot_rolling_metrics,
    plot_strategy_trades,
    plot_trades,
)


@pytest.fixture
def visualization_data():
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    equity = pd.Series([1000 + index * 2 for index in range(70)], index=dates)
    returns = equity.pct_change().dropna()
    metrics = {
        "total_return": 13.8,
        "cagr": 20.0,
        "volatility": 5.0,
        "sharpe_ratio": 1.5,
        "sortino_ratio": 2.0,
        "max_drawdown": 1.0,
        "win_rate": 100.0,
        "profit_factor": 2.0,
        "total_trades": 2,
    }
    trades = [
        {
            "timestamp": dates[10],
            "signal": "buy",
            "price": 101.0,
            "shares": 2.0,
            "pnl": None,
        },
        {
            "timestamp": dates[40],
            "signal": "sell",
            "price": 110.0,
            "shares": 2.0,
            "pnl": 18.0,
        },
    ]
    strategy = {
        "portfolio_values": equity,
        "returns": returns,
        "metrics": metrics,
        "trade_history": trades,
    }
    benchmark = {
        "portfolio_values": equity * 0.99,
        "returns": returns,
        "metrics": {**metrics, "total_return": 12.0},
        "trade_history": trades[:1],
    }
    price_data = pd.DataFrame({"Close": equity / 10}, index=dates)
    return {"strategy": strategy, "benchmark": benchmark}, price_data


def test_core_visualizations_return_expected_traces(visualization_data):
    results, _ = visualization_data

    assert len(plot_equity_curve(results).data) == 2
    assert len(plot_equity_curve(results, show_drawdown=True).data) == 4
    assert len(plot_drawdowns(results).data) == 2
    assert len(plot_returns_distribution(results).data) == 2
    assert len(plot_rolling_metrics(results, window=10).data) == 4
    assert isinstance(create_comparison_table(results), go.Figure)


def test_per_strategy_visualizations_skip_benchmark(visualization_data):
    results, price_data = visualization_data

    monthly = plot_monthly_returns(results)
    pnl = plot_trades(results)
    trade_charts = plot_strategy_trades(price_data, results)

    assert list(monthly) == ["strategy"]
    assert list(pnl) == ["strategy"]
    assert list(trade_charts) == ["strategy"]
    assert len(pnl["strategy"].data) == 1
    assert len(trade_charts["strategy"].data) == 3
