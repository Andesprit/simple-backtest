"""Deterministic simulation-cost, liquidation, and annualization tests."""

import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig, BuyAndHoldStrategy, DCAStrategy, Strategy
from simple_backtest.utils.validation import DataValidationError


def make_data(prices, *, volume=100.0, index=None):
    if index is None:
        index = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [volume] * len(prices),
        },
        index=index,
    )


class BuyThenSellStrategy(Strategy):
    def __init__(self, shares=1):
        super().__init__()
        self.shares = shares
        self.calls = 0

    def predict(self, data, trade_history):
        actions = [self.buy(self.shares), self.sell(self.shares)]
        action = actions[self.calls] if self.calls < len(actions) else self.hold()
        self.calls += 1
        return action

    def reset_state(self):
        super().reset_state()
        self.calls = 0


def test_spread_and_slippage_move_fill_prices_adversely():
    data = make_data([100.0, 101.0, 102.0, 103.0])
    config = BacktestConfig.zero_commission(
        lookback_period=1,
        parallel_execution=False,
        spread_bps=10,
        slippage_bps=5,
    )

    trades = (
        Backtest(data, config)
        .run([BuyThenSellStrategy()])
        .get_strategy("BuyThenSellStrategy")
        .trade_history
    )

    assert trades[0]["price"] == pytest.approx(101.0 * 1.001)
    assert trades[1]["price"] == pytest.approx(102.0 * 0.999)


def test_volume_participation_clamps_and_records_requested_size():
    data = make_data([100.0, 100.0, 100.0], volume=100)
    config = BacktestConfig.zero_commission(
        lookback_period=1,
        parallel_execution=False,
        max_volume_participation=0.1,
    )

    results = Backtest(data, config).run([BuyAndHoldStrategy(shares=50)])
    trade = results.get_strategy("BuyAndHold").trade_history[0]

    assert trade["shares"] == 10
    assert trade["requested_shares"] == 50
    assert len(results.benchmark.trade_history) == 1
    assert results.benchmark.trade_history[0]["shares"] == 10


def test_volume_participation_requires_volume_at_construction():
    data = make_data([100.0, 100.0, 100.0]).drop(columns="Volume")
    config = BacktestConfig.zero_commission(
        lookback_period=1,
        max_volume_participation=0.1,
    )

    with pytest.raises(DataValidationError, match="Volume"):
        Backtest(data, config)


def test_final_liquidation_policy_applies_to_strategy_and_benchmark():
    data = make_data([100.0, 101.0, 102.0, 103.0])
    config = BacktestConfig(
        initial_capital=1000,
        lookback_period=1,
        commission_type="flat",
        commission_value=1.0,
        parallel_execution=False,
        final_liquidation=True,
    )

    results = Backtest(data, config).run([BuyAndHoldStrategy(shares=1)])

    assert [trade["signal"] for trade in results.get_strategy("BuyAndHold").trade_history] == [
        "buy",
        "sell",
    ]
    assert [trade["signal"] for trade in results.benchmark.trade_history] == ["buy", "sell"]
    assert results.get_strategy("BuyAndHold").metrics["exposure_time"] == pytest.approx(100 * 2 / 3)
    assert results.benchmark.metrics["exposure_time"] == pytest.approx(100 * 2 / 3)


def test_default_policy_marks_open_positions_without_forced_sell():
    data = make_data([100.0, 101.0, 102.0])
    config = BacktestConfig.zero_commission(lookback_period=1, parallel_execution=False)

    result = Backtest(data, config).run([BuyAndHoldStrategy(shares=1)]).get_strategy("BuyAndHold")

    assert [trade["signal"] for trade in result.trade_history] == ["buy"]


def test_dca_schedule_advances_only_after_an_executed_trade():
    data = make_data([10.0, 10.0, 10.0, 10.0])
    config = BacktestConfig(
        initial_capital=5,
        lookback_period=1,
        commission_type="flat",
        commission_value=1.0,
        parallel_execution=False,
    )
    strategy = DCAStrategy(investment_amount=10, interval_days=1)

    result = Backtest(data, config).run([strategy]).get_strategy(strategy.get_name())

    assert result.trade_history == []
    assert strategy.last_trade_date is None


@pytest.mark.parametrize(
    ("index", "expected"),
    [
        (pd.bdate_range("2024-01-01", periods=262), 261),
        (pd.date_range("2024-01-01", periods=366, freq="D"), 365),
        (pd.date_range("2024-01-01", periods=8761, freq="h"), 8760),
        (pd.date_range("2024-01-01", periods=53, freq="7D"), 52),
    ],
)
def test_periods_per_year_is_inferred_from_observed_frequency(index, expected):
    data = make_data([100.0] * len(index), index=index)

    backtest = Backtest(data, BacktestConfig.zero_commission(lookback_period=1))

    assert backtest.periods_per_year == pytest.approx(expected, rel=0.02)
