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


def _notes(figure):
    return [annotation.text for annotation in figure.layout.annotations[1:]]


def test_trade_chart_spans_result_timeline_not_fill_dates(visualization_data):
    results, price_data = visualization_data

    figure = plot_strategy_trades(price_data, results)["strategy"]

    returns = results["strategy"]["returns"]
    assert list(figure.data[0].x) == list(returns.index)
    assert list(figure.data[1].x) == [price_data.index[10]]
    assert list(figure.data[2].customdata[0]) == [2.0, 18.0]
    assert _notes(figure) == []


def test_portfolio_values_timeline_is_used_without_returns(visualization_data):
    results, price_data = visualization_data
    strategy = {**results["strategy"], "returns": pd.Series(dtype=float)}

    figure = plot_strategy_trades(price_data, {"strategy": strategy})["strategy"]

    assert len(figure.data[0].x) == len(price_data)


def test_legacy_result_without_fills_only_says_no_fills(visualization_data):
    results, price_data = visualization_data
    strategy = {**results["strategy"], "trade_history": []}

    figure = plot_strategy_trades(price_data, {"strategy": strategy})["strategy"]

    assert _notes(figure) == ["No fills: no order was executed in this run."]
    assert "Win Rate: n/a (no closed trades)" in figure.layout.annotations[0].text
    assert len(figure.data) == 1


def test_dictionary_diagnostics_are_escaped_and_bounded(visualization_data):
    results, price_data = visualization_data
    message = "<b>bad</b> " + "x" * 500
    strategy = {
        **results["strategy"],
        "trade_history": [],
        "errors": [{"error_type": "ValueError", "message": message}] * 3,
        "order_outcomes": [],
    }

    note = _notes(plot_strategy_trades(price_data, {"strategy": strategy})["strategy"])[0]

    assert "3 strategy errors recorded" in note
    assert "&lt;b&gt;bad&lt;/b&gt;" in note and "<b>bad" not in note
    assert "x" * 120 not in note and "result.errors" in note
    assert strategy["errors"][0]["message"] == message


def test_partial_prices_are_flagged_and_disjoint_prices_raise(visualization_data):
    results, price_data = visualization_data
    returns = results["strategy"]["returns"]
    partial = price_data.drop([returns.index[0], returns.index[30], returns.index[-1]])

    figure = plot_strategy_trades(partial, results)["strategy"]

    assert "incomplete price data" in figure.layout.title.text
    assert "3 of 69 backtest dates have no price (2024-01-02, 2024-02-01, 2024-03-10)" in "".join(
        _notes(figure)
    ).replace("<br>", " ")
    later = price_data.set_axis(price_data.index + pd.Timedelta(days=365))
    with pytest.raises(ValueError, match="Pass the same price DataFrame"):
        plot_strategy_trades(later, results)


def test_every_run_outcome_gets_an_explained_chart():
    from simple_backtest import Backtest, BacktestConfig, BuyAndHoldStrategy, Strategy

    class NoSignal(Strategy):
        def predict(self, data, trade_history):
            return self.hold()

    class Broken(Strategy):
        def predict(self, data, trade_history):
            raise ValueError("indicator unavailable")

    dates = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    close = pd.Series(range(100, 140), index=dates, dtype=float)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close})
    original = data.copy(deep=True)
    config = BacktestConfig(
        initial_capital=1000,
        lookback_period=5,
        parallel_execution=False,
        error_policy="continue",
        trading_start_date=dates[10],
        trading_end_date=dates[29],
    )
    strategies = [
        BuyAndHoldStrategy(shares=1),
        NoSignal(name="NoSignal"),
        BuyAndHoldStrategy(shares=10000, name="TooBig"),
        Broken(name="Broken"),
    ]
    with pytest.warns(UserWarning):
        results = Backtest(data, config).run(strategies)

    collection = results.plot_trades(data)
    free = plot_strategy_trades(data, results)

    assert list(collection) == list(free) == ["BuyAndHold", "NoSignal", "TooBig", "Broken"]
    expected = {
        "BuyAndHold": [],
        "NoSignal": ["No orders were placed"],
        "TooBig": ["No fills", "insufficient_cash"],
        "Broken": ["No fills", "strategy errors recorded", "indicator unavailable"],
    }
    for name, fragments in expected.items():
        single = results.get_strategy(name).plot_trades(data)
        assert list(single.data[0].x) == list(dates[10:30])
        text = " ".join(_notes(single)).replace("<br>", " ")
        assert all(fragment in text for fragment in fragments)
        assert _notes(single) == _notes(collection[name]) == _notes(free[name])
    pd.testing.assert_frame_equal(data, original)


def test_legends_and_trade_summary_stay_off_plotted_data(visualization_data):
    results, price_data = visualization_data
    trade = plot_strategy_trades(price_data, results)["strategy"]
    drawdowns = plot_drawdowns(results)

    # Plotly's default legend sits outside the plot area, so it hides no line start.
    for figure in (plot_equity_curve(results), plot_equity_curve(results, True), drawdowns):
        assert figure.layout.legend.x is None and figure.layout.legend.y is None
    assert (trade.layout.legend.y, trade.layout.legend.yanchor) == (1, "bottom")  # above it
    summary, *notes = trade.layout.annotations
    assert "Strategy Performance" in summary.text
    assert (summary.yref, summary.y, summary.yanchor) == ("paper", 0, "top")
    assert trade.layout.margin.b >= -summary.yshift + 95  # the box fits below the axis
    # Default fill colors are translucent, so an earlier strategy's line stays visible.
    assert all(trace.fillcolor is None for trace in drawdowns.data)


def test_trade_notes_sit_below_the_performance_summary(visualization_data):
    results, price_data = visualization_data
    strategy = {**results["strategy"], "trade_history": []}

    summary, note = plot_strategy_trades(price_data, {"strategy": strategy})["strategy"].layout[
        "annotations"
    ]

    assert note.y == summary.y == 0 and note.yshift <= summary.yshift - 95


def test_overlaid_return_histograms_share_bins(visualization_data):
    results, _ = visualization_data

    figure = plot_returns_distribution(results)

    assert figure.data[0].bingroup is not None
    assert len({trace.bingroup for trace in figure.data}) == 1


@pytest.mark.parametrize("drop", [("returns",), ("returns", "portfolio_values")])
def test_dictionary_without_timelines_still_plots_fills_and_diagnostics(visualization_data, drop):
    results, price_data = visualization_data
    rejection = {"status": "rejected", "reason": "insufficient_cash"}
    strategy = {key: value for key, value in results["strategy"].items() if key not in drop}
    strategy.update(errors=[], order_outcomes=[rejection])
    original = price_data.copy(deep=True)

    figure = plot_strategy_trades(price_data, {"strategy": strategy})["strategy"]
    empty = plot_strategy_trades(price_data, {"strategy": {**strategy, "trade_history": []}})

    assert len(figure.data[0].x) == len(price_data)
    assert list(figure.data[1].x) == [price_data.index[10]]
    assert list(figure.data[2].x) == [price_data.index[40]]
    assert "insufficient_cash" in " ".join(_notes(figure))
    empty_notes = " ".join(_notes(empty["strategy"]))
    assert "No fills" in empty_notes and "insufficient_cash" in empty_notes
    pd.testing.assert_frame_equal(price_data, original)
