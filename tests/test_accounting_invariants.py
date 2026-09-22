"""Generated simulations checked against an independent cash and FIFO cost ledger."""

from collections import defaultdict, deque

import numpy as np
import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig, Strategy


class MixedOrders(Strategy):
    def predict(self, data, trade_history):
        action = self.rng.integers(4)
        if action == 0:
            return self.buy_budget(float(self.rng.uniform(20, 400)))
        if action == 1:
            return self.buy(float(self.rng.uniform(0.1, 8)))
        if action == 2 and self.has_position():
            return self.sell(self.get_position() * float(self.rng.uniform(0.1, 0.9)))
        return self.sell_all() if self.has_position() else self.hold()


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("fee_model", ["flat", "percentage", "tiered"])
def test_generated_cash_quantity_cost_basis_and_equity_invariants(seed, fee_model):
    rng = np.random.default_rng(seed)
    opens = rng.uniform(20, 150, 60)
    closes = opens * rng.uniform(0.9, 1.1, 60)
    data = pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(opens, closes) + 1,
            "Low": np.minimum(opens, closes) - 1,
            "Close": closes,
            "Volume": rng.integers(0, 30, 60),
        },
        index=pd.date_range("2024-01-01", periods=60),
    )
    fees = {"flat": 0.75, "percentage": 0.003, "tiered": [(100, 0.005), (float("inf"), 0.002)]}
    settings = BacktestConfig(
        initial_capital=1000,
        lookback_period=1,
        commission_type=fee_model,
        commission_value=fees[fee_model],
        slippage_bps=13,
        spread_bps=8,
        max_volume_participation=0.2,
        final_liquidation=bool(seed % 2),
        record_position_snapshots=True,
        periods_per_year=252,
        parallel_execution=False,
        random_seed=seed,
    )
    result = Backtest(data, settings).run([MixedOrders()]).get_strategy("MixedOrders")
    assert not result.errors
    assert any(t["signal"] == "buy" for t in result.trade_history)
    assert any(t["signal"] == "sell" for t in result.trade_history)
    assert any(o["status"] == "partial" for o in result.order_outcomes)
    by_time = defaultdict(list)
    for trade in result.trade_history:
        by_time[trade["timestamp"]].append(trade)

    cash, realized = 1000.0, 0.0
    # Each reference lot stores quantity and total cost per unit, including entry fee.
    lots = deque()
    for timestamp, bar in data.iloc[1:].iterrows():
        for trade in by_time[timestamp]:
            quantity, price, fee = trade["shares"], trade["price"], trade["commission"]
            buy = trade["signal"] == "buy"
            assert quantity > 0
            assert quantity <= bar.Volume * 0.2
            assert price == pytest.approx(bar.Open * (1.0017 if buy else 0.9983))
            value = quantity * price
            expected_fee = (
                0.75
                if fee_model == "flat"
                else value * 0.003
                if fee_model == "percentage"
                else min(value, 100) * 0.005 + max(value - 100, 0) * 0.002
            )
            assert fee == pytest.approx(expected_fee)
            if buy:
                cash -= value + fee
                lots.append([quantity, (value + fee) / quantity])
                if "requested_budget" in trade:
                    assert value + fee <= trade["requested_budget"]
            else:
                cash += value - fee
                remaining, cost = quantity, 0.0
                while remaining > 1e-12:
                    available, unit_cost = lots[0]
                    consumed = min(available, remaining)
                    cost += consumed * unit_cost
                    remaining -= consumed
                    if consumed == available:
                        lots.popleft()
                    else:
                        lots[0][0] -= consumed
                pnl = value - fee - cost
                assert trade["pnl"] == pytest.approx(pnl, abs=1e-9)
                realized += pnl
            held = sum(lot[0] for lot in lots)
            assert cash >= -1e-9
            assert trade["cash"] == pytest.approx(cash, abs=1e-9)
            assert trade["total_shares"] == pytest.approx(held, abs=1e-9)
            assert sum(p["shares"] for p in trade["positions"].values()) == pytest.approx(
                held, abs=1e-9
            )
            assert trade["portfolio_value"] == pytest.approx(cash + held * price)
        held = sum(lot[0] for lot in lots)
        equity = cash + held * bar.Close
        unrealized = sum(qty * (bar.Close - unit_cost) for qty, unit_cost in lots)
        assert result.portfolio_values.loc[timestamp] == pytest.approx(equity)
        assert equity - 1000 == pytest.approx(realized + unrealized, abs=1e-8)
    assert result.metrics["total_return"] == pytest.approx((equity / 1000 - 1) * 100)
