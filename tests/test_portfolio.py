"""Tests for Portfolio class."""

from datetime import datetime

import pytest

from simple_backtest.core.portfolio import Portfolio


def test_portfolio_initialization():
    """Test portfolio initialization."""
    portfolio = Portfolio(initial_capital=1000.0)
    assert portfolio.cash == 1000.0
    assert portfolio.initial_capital == 1000.0
    assert len(portfolio.positions) == 0
    assert len(portfolio.trade_history) == 0


def test_portfolio_invalid_initial_capital():
    """Test portfolio rejects negative capital."""
    with pytest.raises(ValueError, match="must be positive"):
        Portfolio(initial_capital=-100.0)

    with pytest.raises(ValueError, match="must be positive"):
        Portfolio(initial_capital=0.0)


def test_execute_buy():
    """Test buy execution."""
    portfolio = Portfolio(initial_capital=1000.0)
    trade_info = portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
    )

    assert portfolio.cash == 1000.0 - (10 * 50.0) - 5.0  # 495.0
    assert len(portfolio.positions) == 1
    assert trade_info["signal"] == "buy"
    assert trade_info["shares"] == 10
    assert trade_info["price"] == 50.0
    assert trade_info["commission"] == 5.0


def test_execute_buy_insufficient_funds():
    """Test buy fails with insufficient funds."""
    portfolio = Portfolio(initial_capital=100.0)

    with pytest.raises(ValueError, match="Insufficient cash"):
        portfolio.execute_buy(
            shares=10,
            price=50.0,
            commission=5.0,
            timestamp=datetime(2020, 1, 1),
        )


def test_execute_sell():
    """Test sell execution."""
    portfolio = Portfolio(initial_capital=1000.0)

    # First buy
    portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
    )

    # Then sell
    trade_info = portfolio.execute_sell(
        shares=10,
        price=60.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 2),
    )

    # Check P&L includes both entry and exit commissions.
    assert trade_info["pnl"] == pytest.approx(90.0)
    assert len(portfolio.positions) == 0
    assert portfolio.cash == pytest.approx(1000.0 - 505.0 + 600.0 - 5.0)  # 1090.0


def test_execute_sell_insufficient_shares():
    """Test sell fails with insufficient shares."""
    portfolio = Portfolio(initial_capital=1000.0)

    with pytest.raises(ValueError, match="Insufficient shares"):
        portfolio.execute_sell(
            shares=10,
            price=50.0,
            commission=5.0,
            timestamp=datetime(2020, 1, 1),
        )


def test_partial_sell():
    """Test partial position close."""
    portfolio = Portfolio(initial_capital=1000.0)

    portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
    )

    # Sell half
    portfolio.execute_sell(
        shares=5,
        price=60.0,
        commission=2.0,
        timestamp=datetime(2020, 1, 2),
    )

    assert portfolio.get_total_shares() == 5
    assert len(portfolio.positions) == 1


def test_get_portfolio_value():
    """Test portfolio valuation."""
    portfolio = Portfolio(initial_capital=1000.0)

    portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
    )

    # Portfolio value = cash + position value
    # cash = 1000 - 500 - 5 = 495
    # positions = 10 shares * 60 price = 600
    # total = 1095
    value = portfolio.get_portfolio_value(current_price=60.0)
    assert value == pytest.approx(1095.0)


def test_can_afford():
    """Test affordability check."""
    portfolio = Portfolio(initial_capital=1000.0)

    assert portfolio.can_afford(shares=10, price=50.0, commission=5.0)
    assert not portfolio.can_afford(shares=100, price=50.0, commission=5.0)


def test_reset():
    """Test portfolio reset."""
    portfolio = Portfolio(initial_capital=1000.0)

    portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
    )

    portfolio.reset()

    assert portfolio.cash == 1000.0
    assert len(portfolio.positions) == 0
    assert len(portfolio.trade_history) == 0


def test_fifo_selling():
    """Test FIFO order of selling."""
    portfolio = Portfolio(initial_capital=10000.0)

    # Buy at different times
    portfolio.execute_buy(
        shares=10,
        price=50.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 1),
        order_id="ORDER_1",
    )

    portfolio.execute_buy(
        shares=10,
        price=60.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 2),
        order_id="ORDER_2",
    )

    # Sell 10 shares - should sell from ORDER_1 first
    trade_info = portfolio.execute_sell(
        shares=10,
        price=70.0,
        commission=5.0,
        timestamp=datetime(2020, 1, 3),
    )

    # P&L should use the first lot's commission-inclusive cost basis.
    assert trade_info["pnl"] == pytest.approx(190.0)
    assert "ORDER_1" not in portfolio.positions
    assert "ORDER_2" in portfolio.positions


def test_selected_lots_must_cover_entire_sale():
    """Reject a sale that names fewer shares than it requests."""
    portfolio = Portfolio(initial_capital=10000.0)
    timestamp = datetime(2020, 1, 1)
    portfolio.execute_buy(10, 100.0, 0.0, timestamp, order_id="ORDER_1")
    portfolio.execute_buy(10, 100.0, 0.0, timestamp, order_id="ORDER_2")

    before = portfolio.get_state_snapshot()
    with pytest.raises(ValueError, match="Selected order_ids contain only 10 shares"):
        portfolio.execute_sell(
            shares=15,
            price=100.0,
            commission=0.0,
            timestamp=datetime(2020, 1, 2),
            order_ids=["ORDER_1"],
        )

    assert portfolio.get_state_snapshot() == before


def test_selected_lots_reject_duplicate_order_ids():
    """Reject duplicate lot identifiers before mutating the portfolio."""
    portfolio = Portfolio(initial_capital=10000.0)
    portfolio.execute_buy(10, 100.0, 0.0, datetime(2020, 1, 1), order_id="ORDER_1")

    with pytest.raises(ValueError, match="duplicate order_ids"):
        portfolio.execute_sell(
            shares=10,
            price=100.0,
            commission=0.0,
            timestamp=datetime(2020, 1, 2),
            order_ids=["ORDER_1", "ORDER_1"],
        )


def test_buy_rejects_an_existing_order_id_without_deducting_cash():
    """A duplicate lot ID cannot overwrite an existing position."""
    portfolio = Portfolio(initial_capital=10000.0)
    portfolio.execute_buy(10, 100.0, 0.0, datetime(2020, 1, 1), order_id="ORDER_1")
    before = portfolio.get_state_snapshot()

    with pytest.raises(ValueError, match="already exists"):
        portfolio.execute_buy(5, 100.0, 0.0, datetime(2020, 1, 2), order_id="ORDER_1")

    assert portfolio.get_state_snapshot() == before


def test_trade_history_contains_immutable_position_snapshots():
    """Later fills must not rewrite earlier trade records."""
    portfolio = Portfolio(initial_capital=1000.0)
    buy = portfolio.execute_buy(10, 50.0, 5.0, datetime(2020, 1, 1), order_id="ORDER_1")
    portfolio.execute_sell(5, 60.0, 2.0, datetime(2020, 1, 2), ["ORDER_1"])

    assert buy["positions"]["ORDER_1"]["shares"] == 10
    assert portfolio.get_trade_history()[0]["positions"]["ORDER_1"]["shares"] == 10


def test_partial_sale_allocates_entry_commission_proportionally():
    """Realized P&L includes the sold fraction of entry costs."""
    portfolio = Portfolio(initial_capital=1000.0)
    portfolio.execute_buy(10, 50.0, 10.0, datetime(2020, 1, 1), order_id="ORDER_1")

    trade = portfolio.execute_sell(5, 60.0, 2.0, datetime(2020, 1, 2))

    assert trade["pnl"] == pytest.approx(43.0)
    assert portfolio.positions["ORDER_1"]["entry_commission"] == pytest.approx(5.0)


@pytest.mark.parametrize("field,value", [("shares", float("nan")), ("price", float("inf"))])
def test_orders_reject_non_finite_numeric_inputs(field, value):
    """NaN and infinity cannot enter portfolio accounting."""
    portfolio = Portfolio(initial_capital=1000.0)
    values = {"shares": 1.0, "price": 10.0, "commission": 0.0}
    values[field] = value

    with pytest.raises(ValueError, match="finite"):
        portfolio.execute_buy(timestamp=datetime(2020, 1, 1), **values)
