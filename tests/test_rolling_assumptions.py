"""Rolling risk charts use each run's recorded annualization and risk-free rate."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig, BuyAndHoldStrategy
from simple_backtest.metrics.definitions import calculate_sharpe_ratio, calculate_volatility
from simple_backtest.visualization.plotter import (
    plot_returns_distribution,
    plot_rolling_metrics,
)

WINDOW = 30


def _run(freq: str, periods: int, risk_free: float, bars: int = 100):
    rng = np.random.default_rng(91)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0008, 0.01, bars)))
    data = pd.DataFrame(
        {"Open": close, "High": close * 1.02, "Low": close * 0.98, "Close": close},
        index=pd.date_range("2024-01-01", periods=bars, freq=freq, tz="UTC"),
    )
    config = BacktestConfig.zero_commission(
        initial_capital=10000,
        lookback_period=3,
        parallel_execution=False,
        periods_per_year=periods,
        risk_free_rate=risk_free,
    )
    return Backtest(data, config).run([BuyAndHoldStrategy(shares=50)])


@pytest.fixture(scope="module")
def runs():
    return {
        "hourly": (_run("h", 8760, 0.0), 8760, 0.0),
        "daily_target": (_run("D", 252, 0.20), 252, 0.20),
        "monthly": (_run("MS", 12, 0.0), 12, 0.0),
    }


def _canonical(returns, periods, risk_free):
    windows = [returns.iloc[end - WINDOW : end] for end in range(WINDOW, len(returns) + 1)]
    sharpe = [calculate_sharpe_ratio(w, risk_free, periods) for w in windows]
    volatility = [calculate_volatility(w, periods) for w in windows]
    return np.array(sharpe), np.array(volatility)


@pytest.mark.parametrize("case", ["hourly", "daily_target", "monthly"])
@pytest.mark.parametrize("as_dictionary", [False, True])
def test_every_completed_window_matches_canonical_metrics(runs, case, as_dictionary):
    results, periods, risk_free = runs[case]
    strategy = results.get_strategy("BuyAndHold")
    source = {"BuyAndHold": strategy._to_dict()} if as_dictionary else results
    returns_before = strategy.returns.copy()

    fig = plot_rolling_metrics(source, window=WINDOW)

    sharpe, volatility = _canonical(strategy.returns, periods, risk_free)
    assert np.isnan(fig.data[0].y[: WINDOW - 1]).all()
    assert np.isnan(fig.data[1].y[: WINDOW - 1]).all()
    np.testing.assert_allclose(fig.data[0].y[WINDOW - 1 :], sharpe, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fig.data[1].y[WINDOW - 1 :], volatility, rtol=1e-12, atol=1e-12)
    assert f"{periods:g} periods/year (recorded)" in fig.data[0].hovertemplate
    assert f"risk-free {risk_free:.2%}/year (recorded)" in fig.data[1].hovertemplate
    pd.testing.assert_series_equal(strategy.returns, returns_before)


def test_labels_name_bars_not_days(runs):
    results = runs["hourly"][0]

    rolling = plot_rolling_metrics(results, window=WINDOW)
    histogram = plot_returns_distribution(results)

    assert rolling.layout.title.text == "Rolling Metrics (30-bar window)"
    assert "day" not in rolling.layout.title.text.lower()
    assert histogram.layout.xaxis.title.text == "Return per bar (%)"
    assert [trace.name for trace in rolling.data] == [
        "BuyAndHold (Sharpe)",
        "BuyAndHold (Vol)",
        "benchmark (Sharpe)",
        "benchmark (Vol)",
    ]


def test_benchmark_uses_its_recorded_assumptions(runs):
    results, periods, risk_free = runs["daily_target"]

    fig = plot_rolling_metrics(results, window=WINDOW)

    sharpe, _ = _canonical(results.benchmark.returns, periods, risk_free)
    np.testing.assert_allclose(fig.data[2].y[WINDOW - 1 :], sharpe, rtol=1e-12, atol=1e-12)
    assert "risk-free 20.00%/year (recorded)" in fig.data[2].hovertemplate


def test_mixed_runs_keep_their_own_assumptions_and_short_runs_are_named(runs):
    hourly = runs["hourly"][0].get_strategy("BuyAndHold")._to_dict()
    monthly = runs["monthly"][0].get_strategy("BuyAndHold")._to_dict()
    legacy = {key: monthly[key] for key in ("returns", "portfolio_values", "trade_history")}
    short = {**hourly, "returns": hourly["returns"].iloc[:10]}

    fig = plot_rolling_metrics(
        {"hourly": hourly, "monthly": monthly, "legacy": legacy, "short": short}, window=WINDOW
    )

    _, hourly_vol = _canonical(hourly["returns"], 8760, 0.0)
    _, monthly_vol = _canonical(monthly["returns"], 12, 0.0)
    _, legacy_vol = _canonical(monthly["returns"], 252, 0.0)
    np.testing.assert_allclose(fig.data[1].y[WINDOW - 1 :], hourly_vol, rtol=1e-12)
    np.testing.assert_allclose(fig.data[3].y[WINDOW - 1 :], monthly_vol, rtol=1e-12)
    np.testing.assert_allclose(fig.data[5].y[WINDOW - 1 :], legacy_vol, rtol=1e-12)
    assert np.isnan(fig.data[6].y).all()
    note = fig.layout.annotations[-1].text
    assert "hourly, short: 8760 periods/year (recorded)" in note
    assert "monthly: 12 periods/year (recorded)" in note
    assert "legacy: 252 periods/year (assumed), risk-free 0.00%/year (assumed)" in note
    assert "Fewer than 30 returns, so no rolling values: short." in note


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        (
            {"periods_per_year": 8760},
            "8760 periods/year (recorded), risk-free 0.00%/year (assumed)",
        ),
        (
            {"config": {"risk_free_rate": 0.05}},
            "252 periods/year (assumed), risk-free 5.00%/year (recorded)",
        ),
        (None, "252 periods/year (assumed), risk-free 0.00%/year (assumed)"),
        ({"config": None}, "252 periods/year (assumed), risk-free 0.00%/year (assumed)"),
    ],
)
def test_each_missing_field_falls_back_independently(runs, metadata, expected):
    returns = runs["hourly"][0].get_strategy("BuyAndHold").returns

    fig = plot_rolling_metrics({"run": {"returns": returns, "metadata": metadata}})

    assert expected in fig.data[0].hovertemplate
    assert expected in fig.layout.annotations[-1].text


@pytest.mark.parametrize(
    ("metadata", "field"),
    [
        ({"periods_per_year": 0}, "periods_per_year"),
        ({"periods_per_year": float("inf")}, "periods_per_year"),
        ({"periods_per_year": True}, "periods_per_year"),
        ({"periods_per_year": "252"}, "periods_per_year"),
        ({"config": {"risk_free_rate": -1.0}}, "risk_free_rate"),
        ({"config": {"risk_free_rate": float("nan")}}, "risk_free_rate"),
        # An invalid recorded field is reported even when the other one is missing.
        ({"periods_per_year": -5, "config": {}}, "periods_per_year"),
    ],
)
def test_invalid_recorded_assumptions_raise(runs, metadata, field):
    returns = runs["hourly"][0].get_strategy("BuyAndHold").returns

    with pytest.raises(ValueError, match=rf"^Broken: recorded {field}=.* is invalid"):
        plot_rolling_metrics({"Broken": {"returns": returns, "metadata": metadata}})


@pytest.mark.parametrize("risk_free", [0.0, 0.05])
def test_constant_windows_have_no_sharpe_and_volatile_windows_still_match(risk_free):
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    rng = np.random.default_rng(3)
    # Constant growth read from an equity curve carries ~1e-16 rounding noise.
    noisy_flat = pd.Series(10000 * np.cumprod(np.full(46, 1.001))).pct_change().dropna().to_numpy()
    returns = {
        "zero": pd.Series(0.0, index=dates),
        "constant": pd.Series(0.001, index=dates),
        "volatile_then_flat": pd.Series(np.r_[rng.normal(0, 0.01, 45), np.zeros(45)], index=dates),
        # The rolling std can leave ~1e-10 behind here: a Sharpe spike of ~1.8e8 at a 0% target.
        "volatile_then_noisy_flat": pd.Series(
            np.r_[rng.normal(0, 0.01, 45), noisy_flat], index=dates
        ),
    }
    metadata = {"periods_per_year": 252, "config": {"risk_free_rate": risk_free}}

    fig = plot_rolling_metrics(
        {name: {"returns": r, "metadata": metadata} for name, r in returns.items()}
    )

    for trace in (fig.data[0], fig.data[2]):
        assert np.isnan(trace.y).all()
    for sharpe_trace, volatility_trace, name in (
        (fig.data[4], fig.data[5], "volatile_then_flat"),
        (fig.data[6], fig.data[7], "volatile_then_noisy_flat"),
    ):
        sharpe, volatility = _canonical(returns[name], 252, risk_free)
        completed = sharpe_trace.y[WINDOW - 1 :]
        flat = volatility < 1e-9
        assert flat.any() and (~flat).any()
        assert np.isnan(completed[flat]).all() and np.isfinite(completed[~flat]).all()
        np.testing.assert_allclose(completed, sharpe, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(volatility_trace.y[WINDOW - 1 :], volatility, atol=1e-12)


def test_plotting_does_not_rerun_backtest(runs):
    results = runs["monthly"][0]

    with patch.object(Backtest, "run", side_effect=AssertionError("re-ran")):
        plot_rolling_metrics(results)
        plot_returns_distribution(results)
