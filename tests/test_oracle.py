"""Hand-calculated end-to-end accounting and timing oracles."""

import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig, Strategy


class OracleStrategy(Strategy):
    """Place a deterministic sequence of orders from prior-bar signals."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0
        self.observations = []

    def predict(self, data, trade_history):
        self.observations.append(
            {
                "window_end": data.index[-1],
                "execution_time": self._portfolio_state["timestamp"],
                "execution_price": self._portfolio_state["current_price"],
            }
        )
        actions = [self.buy(10), self.hold(), self.buy(5), self.sell(12)]
        action = actions[self.call_count]
        self.call_count += 1
        return action

    def reset_state(self) -> None:
        super().reset_state()
        self.call_count = 0
        self.observations = []


@pytest.fixture
def oracle_data():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [price + 1 for price in prices],
            "Low": [price - 1 for price in prices],
            "Close": prices,
            "Volume": [1000.0] * len(prices),
        },
        index=dates,
    )


@pytest.fixture
def oracle_result(oracle_data):
    config = BacktestConfig(
        initial_capital=10_000,
        lookback_period=2,
        commission_type="percentage",
        commission_value=0.001,
        execution_price="open",
        parallel_execution=False,
        periods_per_year=365,
    )
    strategy = OracleStrategy()
    result = Backtest(oracle_data, config).run([strategy]).get_strategy(strategy.get_name())
    return strategy, result


def test_signal_window_excludes_execution_bar(oracle_result):
    strategy, _ = oracle_result

    assert strategy.observations[0] == {
        "window_end": pd.Timestamp("2020-01-02"),
        "execution_time": pd.Timestamp("2020-01-03"),
        "execution_price": 102.0,
    }


def test_fifo_commission_proration_is_exact(oracle_result):
    _, result = oracle_result
    sell = result.trade_history[-1]

    assert sell["fills"][0]["shares"] == 10
    assert sell["fills"][0]["entry_commission"] == pytest.approx(1.02)
    assert sell["fills"][0]["pnl_before_exit_commission"] == pytest.approx(28.98)
    assert sell["fills"][1]["shares"] == 2
    assert sell["fills"][1]["entry_commission"] == pytest.approx(0.208)
    assert sell["fills"][1]["pnl_before_exit_commission"] == pytest.approx(1.792)
    assert sell["pnl"] == pytest.approx(29.512)
    remaining_position = next(iter(sell["positions"].values()))
    assert remaining_position["shares"] == 3
    assert remaining_position["entry_commission"] == pytest.approx(0.312)


def test_equity_series_and_pnl_identity_are_exact(oracle_result):
    _, result = oracle_result

    assert result.portfolio_values.tolist() == pytest.approx(
        [9998.98, 10008.98, 10018.46, 10032.20]
    )
    sell = result.trade_history[-1]
    remaining_position = next(iter(sell["positions"].values()))
    realized_pnl = sell["pnl"]
    unrealized_pnl = remaining_position["shares"] * (105.0 - remaining_position["entry_price"])
    remaining_entry_commission = remaining_position["entry_commission"]

    assert result.portfolio_values.iloc[-1] - 10_000 == pytest.approx(
        realized_pnl + unrealized_pnl - remaining_entry_commission
    )
