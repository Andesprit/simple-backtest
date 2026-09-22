"""Behavioral regressions for signal timing, equity, and execution boundaries."""

import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig, BuyAndHoldStrategy, Strategy
from simple_backtest.utils.validation import DateRangeError


def candles():
    return pd.DataFrame(
        {
            "Open": [100.0] * 5,
            "High": [130.0] * 5,
            "Low": [90.0] * 5,
            "Close": [100.0, 110.0, 120.0, 115.0, 125.0],
        },
        index=pd.date_range("2024-01-01", periods=5),
    )


def configuration(**kwargs):
    defaults = dict(
        initial_capital=1000,
        lookback_period=1,
        commission_type="flat",
        commission_value=0,
        parallel_execution=False,
        risk_free_rate=0,
    )
    return BacktestConfig(**(defaults | kwargs))


@pytest.mark.parametrize("execution_price", ["open", "close", "typical", "custom"])
def test_prediction_helpers_only_use_prior_close(execution_price):
    class Observe(Strategy):
        def predict(self, data, trade_history):
            prior_close = float(data.Close.iloc[-1])
            assert self.get_portfolio_value() == pytest.approx(
                self.get_cash() + self.get_position() * prior_close
            )
            assert self.buy_cash(100)["size"] == pytest.approx(100 / prior_close)
            assert self.buy_percent(0.1)["size"] == pytest.approx(
                self.get_portfolio_value() * 0.1 / prior_close
            )
            return self.hold() if self.has_position() else self.buy(1)

    Backtest(
        candles(),
        configuration(execution_price=execution_price),
        execution_price_extractor=(lambda row: row.High) if execution_price == "custom" else None,
    ).run([Observe()])


def test_open_fills_are_marked_at_close_for_strategy_and_benchmark():
    results = Backtest(candles(), configuration()).run([BuyAndHoldStrategy(shares=1)])
    strategy = results.get_strategy("BuyAndHold")
    assert strategy.trade_history[0]["price"] == 100
    assert strategy.portfolio_values.iloc[-1] == 1025
    assert results.benchmark.portfolio_values.iloc[-1] == pytest.approx(1250)


def test_initial_fee_is_in_returns_drawdown_and_plotted_equity():
    data = candles().assign(Open=100.0, High=100.0, Low=100.0, Close=100.0)
    results = Backtest(data, configuration(commission_value=10)).run([BuyAndHoldStrategy(shares=1)])
    for result in [results.get_strategy("BuyAndHold"), results.benchmark]:
        assert result.portfolio_values.iloc[0] == 1000
        assert result.returns.tolist() == pytest.approx([-0.01, 0, 0, 0])
        assert result.metrics["max_drawdown"] == pytest.approx(1)
        assert result.metrics["max_drawdown_duration"] == 4
        assert result.metrics["volatility"] > 0
        assert result.metrics["total_return"] == pytest.approx(-1)
        assert result.metrics["exposure_time"] == 100


def test_continue_policy_never_executes_invalid_predictions():
    class InvalidSell(Strategy):
        def predict(self, data, trade_history):
            if not self.has_position():
                return self.buy(1)
            return {"signal": "sell", "size": 1}  # Missing required order_ids.

    result = (
        Backtest(candles(), configuration(error_policy="continue"))
        .run([InvalidSell()])
        .get_strategy("InvalidSell")
    )
    assert [t["signal"] for t in result.trade_history] == ["buy"]
    assert len(result.errors) == 3
    assert all(e["stage"] == "prediction" for e in result.errors)
    assert result.portfolio_values.iloc[-1] == 1025


@pytest.mark.parametrize("timezone", [None, "America/New_York"])
def test_date_boundaries_select_only_bars_inside_requested_interval(timezone):
    data = candles()
    data.index = data.index.tz_localize(timezone)
    start = data.index[1] + pd.Timedelta(hours=1)
    end = data.index[3] + pd.Timedelta(hours=23)
    results = Backtest(data, configuration(trading_start_date=start, trading_end_date=end)).run(
        [BuyAndHoldStrategy(shares=1)]
    )
    for result in [results.get_strategy("BuyAndHold"), results.benchmark]:
        assert result.trade_history[0]["timestamp"] == data.index[2]
        assert result.returns.index.tolist() == [data.index[2], data.index[3]]
        assert all(start <= t["timestamp"] <= end for t in result.trade_history)


def test_date_interval_without_bars_is_rejected():
    with pytest.raises(DateRangeError, match="no trading bars"):
        Backtest(
            candles(),
            configuration(
                trading_start_date="2024-01-02T01:00:00", trading_end_date="2024-01-02T23:00:00"
            ),
        )


@pytest.mark.parametrize("snapshots", [False, True])
def test_incremental_history_preserves_fills_and_isolates_strategy_mutation(snapshots):
    class Accumulate(Strategy):
        def predict(self, data, trade_history):
            assert len(trade_history) == int(self.get_position())
            if trade_history:
                assert trade_history[-1]["total_shares"] == self.get_position()
                trade_history[-1]["price"] = -1
                if snapshots:
                    trade_history[-1]["positions"].clear()
            return self.buy(1)

        def on_trade_executed(self, trade_info):
            trade_info["price"] = -2

    results = Backtest(candles(), configuration(record_position_snapshots=snapshots)).run(
        [Accumulate()]
    )
    trades = results.get_strategy("Accumulate").trade_history
    assert len(trades) == 4
    for i, trade in enumerate(trades, start=1):
        assert trade["price"] == 100
        assert trade["total_shares"] == i
        assert ("positions" in trade) is snapshots
        if snapshots:
            assert len(trade["positions"]) == i


def test_accumulation_does_not_recopy_entire_history_after_each_fill(monkeypatch):
    from simple_backtest.core.portfolio import Portfolio

    original = Portfolio.get_trade_history
    reads = []

    def track_history_reads(self):
        reads.append(len(self.trade_history))
        return original(self)

    monkeypatch.setattr(Portfolio, "get_trade_history", track_history_reads)

    class Accumulate(Strategy):
        def predict(self, data, trade_history):
            return self.buy(1)

    data = pd.concat([candles()] * 20, ignore_index=True)
    data.index = pd.date_range("2024-01-01", periods=len(data))
    Backtest(data, configuration(initial_capital=100000)).run([Accumulate()])
    # Each portfolio is read for metrics and results, independent of fill count.
    assert len(reads) <= 4
