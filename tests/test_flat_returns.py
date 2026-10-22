"""Flat returns have no defined Sharpe-style ratio, so they cannot win a search."""

import math

import numpy as np
import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig
from simple_backtest.metrics import definitions as defs
from simple_backtest.optimization import GridSearchOptimizer, RandomSearchOptimizer
from simple_backtest.strategy.base import Strategy
from simple_backtest.visualization.report import build_report


def _equity_returns(rate, bars=252):
    """Constant growth measured from an equity curve, so it carries float rounding noise."""
    equity = 10000 * np.cumprod(np.full(bars + 1, 1 + rate))
    return pd.Series(equity).pct_change().dropna()


@pytest.mark.parametrize(
    "returns, risk_free",
    [
        (pd.Series([0.0003] * 252), 0.02),  # About 1.3e17 before the rule.
        (_equity_returns(0.0003), 0.02),  # About 3e13 before, from rounding noise only.
        (pd.Series([0.0] * 252), 0.0),  # Cash only.
        (pd.Series([0.0] * 252), 0.02),
    ],
)
def test_flat_returns_have_undefined_sharpe(returns, risk_free):
    assert math.isnan(defs.calculate_sharpe_ratio(returns, risk_free))


def test_flat_tracking_error_and_noise_only_downside_are_undefined():
    strategy, benchmark = _equity_returns(0.0005), _equity_returns(0.0002)
    assert math.isnan(defs.calculate_information_ratio(strategy, benchmark))
    assert math.isnan(defs.calculate_information_ratio(benchmark, benchmark))
    # Real gains with a rounding-size shortfall gave a Sortino of about 1e15.
    assert math.isnan(defs.calculate_sortino_ratio(pd.Series([0.001, -1e-17] * 126)))


def test_varying_returns_keep_their_exact_ratios():
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0005, 0.01, 252))
    benchmark = pd.Series(rng.normal(0.0003, 0.012, 252))
    tiny_but_real = pd.Series(0.0003 + rng.normal(0, 1e-9, 252))

    assert defs.calculate_sharpe_ratio(returns, 0.02) == -2.224343679220167
    assert defs.calculate_sortino_ratio(returns, 0.02) == -2.792637012508092
    assert defs.calculate_information_ratio(returns, benchmark) == -0.4877313070070956
    assert defs.calculate_sharpe_ratio(tiny_but_real, 0.02) == 3729489.2368414477


class InOut(Strategy):
    """Stays invested (every=0) or sells every `every` bars and buys back the next bar."""

    def __init__(self, every=0, name=None):
        super().__init__(name=name or f"InOut_{every}")
        self.every = every
        self.bar = 0

    def predict(self, data, trade_history):
        self.bar += 1
        if not self.has_position():
            return self.buy_budget(self.get_cash())
        if self.every and self.bar % self.every == 0:
            return self.sell_all()
        return self.hold()

    def reset_state(self):
        super().reset_state()
        self.bar = 0


@pytest.fixture
def steady():
    """Each bar opens at the previous close and gains exactly 0.03%."""
    bars = 120
    price = 100 * 1.0003 ** np.arange(bars + 1)
    data = pd.DataFrame(
        {"Open": price[:-1], "High": price[1:], "Low": price[:-1], "Close": price[1:]},
        index=pd.date_range("2024-01-01", periods=bars, freq="D"),
    )
    config = BacktestConfig.zero_commission(
        initial_capital=10000,
        lookback_period=2,
        parallel_execution=False,
        risk_free_rate=0.02,
        periods_per_year=252,
        execution_price="open",
    )
    return data, config


@pytest.mark.parametrize(
    "optimizer",
    [
        GridSearchOptimizer(verbose=False),
        RandomSearchOptimizer(n_iter=2, random_state=0, verbose=False),
    ],
)
def test_search_does_not_pick_a_flat_candidate(steady, optimizer):
    data, config = steady

    ranked = optimizer.optimize(data, config, InOut, {"every": [0, 2]}, metric="sharpe_ratio")

    assert ranked["every"].tolist() == [2]
    assert optimizer.best_parameters == {"every": 2}
    assert optimizer.summary["undefined_objectives"] == 1


def test_undefined_sharpe_is_shown_as_na(steady):
    data, config = steady
    results = Backtest(data, config).run([InOut(0)])
    result = results.get_strategy("InOut_0")

    assert math.isnan(result.metrics["sharpe_ratio"])
    assert "Sharpe: n/a" in str(result)
    assert "sharpe=n/a" in repr(result)
    assert "Sharpe Ratio: n/a" in result.summary()
    html = build_report(results, data)
    assert "<tr><th>Sharpe</th><td>n/a</td></tr>" in html
    assert "Sharpe Ratio: n\\u002fa" in html  # Trade chart annotation; its JSON escapes "/".
