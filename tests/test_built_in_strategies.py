"""Tests for built-in strategy implementations."""

from datetime import datetime

import pandas as pd
import pytest

from simple_backtest.strategy.buy_and_hold import BuyAndHoldStrategy
from simple_backtest.strategy.dca import DCAStrategy
from simple_backtest.strategy.moving_average import MovingAverageStrategy


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    data = pd.DataFrame(
        {
            "Open": [100 + i for i in range(100)],
            "High": [105 + i for i in range(100)],
            "Low": [95 + i for i in range(100)],
            "Close": [100 + i for i in range(100)],
            "Volume": [1000000] * 100,
        },
        index=dates,
    )
    return data


class TestBuyAndHoldStrategy:
    """Tests for BuyAndHoldStrategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = BuyAndHoldStrategy(shares=50)
        assert strategy.get_name() == "BuyAndHold"
        assert strategy.shares == 50
        assert strategy.bought is False

    def test_custom_name(self):
        """Test custom strategy name."""
        strategy = BuyAndHoldStrategy(shares=50, name="MyBuyHold")
        assert strategy.get_name() == "MyBuyHold"

    def test_first_trade_buy_signal(self, sample_data):
        """Test that strategy buys on first prediction."""
        strategy = BuyAndHoldStrategy(shares=50)

        # Mock portfolio state
        strategy._portfolio_state = {
            "cash": 10000,
            "total_shares": 0,
            "portfolio_value": 10000,
            "positions": {},
            "current_price": 100,
            "is_last_day": False,
        }

        prediction = strategy.predict(sample_data.head(10), [])

        assert prediction["signal"] == "buy"
        assert prediction["size"] == 50
        assert prediction["order_ids"] is None
        assert strategy.bought is False

    def test_hold_after_initial_buy(self, sample_data):
        """Test that strategy holds after initial buy."""
        strategy = BuyAndHoldStrategy(shares=50)
        strategy.bought = True

        # Mock portfolio state with position
        strategy._portfolio_state = {
            "cash": 5000,
            "total_shares": 50,
            "portfolio_value": 10000,
            "positions": {"BUY_1": {"shares": 50}},
            "current_price": 100,
            "is_last_day": False,
        }

        prediction = strategy.predict(sample_data.head(10), [])

        assert prediction["signal"] == "hold"
        assert prediction["size"] == 0

    def test_last_day_liquidation_is_owned_by_engine(self, sample_data):
        """Built-ins do not apply a hidden liquidation policy."""
        strategy = BuyAndHoldStrategy(shares=50)
        strategy.bought = True

        # Mock portfolio state on last day
        strategy._portfolio_state = {
            "cash": 5000,
            "total_shares": 50,
            "portfolio_value": 15000,
            "positions": {"BUY_1": {"shares": 50}},
            "current_price": 200,
            "is_last_day": True,
        }

        prediction = strategy.predict(sample_data.tail(10), [])

        assert prediction["signal"] == "hold"
        assert prediction["size"] == 0

    def test_reset_state(self):
        """Test state reset functionality."""
        strategy = BuyAndHoldStrategy(shares=50)
        strategy.bought = True
        strategy._state_initialized = True

        strategy.reset_state()

        assert strategy._state_initialized is False
        # bought flag should persist as it's strategy-specific


class TestDCAStrategy:
    """Tests for Dollar Cost Averaging Strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = DCAStrategy(investment_amount=1000, interval_days=30)
        assert strategy.get_name() == "DCA_30d"  # Default name includes interval
        assert strategy.investment_amount == 1000
        assert strategy.interval_days == 30
        assert strategy.last_trade_date is None

    def test_custom_name(self):
        """Test custom strategy name."""
        strategy = DCAStrategy(investment_amount=500, interval_days=7, name="WeeklyDCA")
        assert strategy.get_name() == "WeeklyDCA"

    def test_first_buy_signal(self, sample_data):
        """Test that strategy buys on first prediction."""
        strategy = DCAStrategy(investment_amount=1000, interval_days=30)

        # Mock portfolio state
        strategy._portfolio_state = {
            "cash": 10000,
            "total_shares": 0,
            "portfolio_value": 10000,
            "positions": {},
            "current_price": 100,
            "is_last_day": False,
        }

        prediction = strategy.predict(sample_data.head(10), [])

        assert prediction["signal"] == "buy"
        assert prediction["size"] == 10  # $1000 / $100 = 10 shares
        assert prediction["order_ids"] is None

    def test_interval_timing(self, sample_data):
        """Test that strategy respects interval timing."""
        strategy = DCAStrategy(investment_amount=1000, interval_days=30)
        strategy.last_trade_date = sample_data.index[0]

        # Mock portfolio state
        strategy._portfolio_state = {
            "cash": 9000,
            "total_shares": 10,
            "portfolio_value": 10000,
            "positions": {"BUY_1": {"shares": 10}},
            "current_price": 100,
            "is_last_day": False,
        }

        # Test before interval - should hold
        data_before_interval = sample_data.head(20)  # 20 days < 30 days
        prediction = strategy.predict(data_before_interval, [])
        assert prediction["signal"] == "hold"

        # Test after interval - should buy
        strategy._portfolio_state["cash"] = 9000
        data_after_interval = sample_data.head(40)  # 40 days > 30 days
        prediction = strategy.predict(data_after_interval, [])
        assert prediction["signal"] == "buy"

    def test_last_day_liquidation_is_owned_by_engine(self, sample_data):
        """DCA follows the same engine-level liquidation policy as other strategies."""
        strategy = DCAStrategy(investment_amount=1000, interval_days=30)
        strategy.last_trade_date = sample_data.index[0]

        # Mock portfolio state on last day
        strategy._portfolio_state = {
            "cash": 5000,
            "total_shares": 50,
            "portfolio_value": 15000,
            "positions": {"BUY_1": {"shares": 50}},
            "current_price": 200,
            "is_last_day": True,
        }

        prediction = strategy.predict(sample_data.tail(10), [])

        assert prediction["signal"] == "buy"
        assert prediction["size"] == 5

    def test_insufficient_cash(self, sample_data):
        """Test behavior when insufficient cash."""
        strategy = DCAStrategy(investment_amount=10000, interval_days=30)

        # Mock portfolio state with insufficient cash
        strategy._portfolio_state = {
            "cash": 500,  # Not enough for $10000 investment
            "total_shares": 0,
            "portfolio_value": 500,
            "positions": {},
            "current_price": 100,
            "is_last_day": False,
        }

        prediction = strategy.predict(sample_data.head(10), [])

        # Should still return buy signal, portfolio will handle affordability
        assert prediction["signal"] == "buy"

    def test_reset_state(self):
        """Test state reset functionality."""
        strategy = DCAStrategy(investment_amount=1000, interval_days=30)
        strategy.last_trade_date = datetime(2020, 1, 1)
        strategy._state_initialized = True

        strategy.reset_state()

        assert strategy._state_initialized is False
        # last_trade_date should be reset
        assert strategy.last_trade_date is None


class TestMovingAverageStrategy:
    """Tests for Moving Average Crossover Strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=50)
        assert strategy.get_name() == "MA_10_30"
        assert strategy.short_window == 10
        assert strategy.long_window == 30
        assert strategy.shares == 50

    def test_custom_name(self):
        """Test custom strategy name."""
        strategy = MovingAverageStrategy(short_window=5, long_window=20, shares=100, name="MyMA")
        assert strategy.get_name() == "MyMA"

    def test_invalid_windows(self):
        """Test that invalid window parameters raise errors."""
        # Short window >= long window
        with pytest.raises(ValueError, match="Short window must be less than long window"):
            MovingAverageStrategy(short_window=30, long_window=10, shares=50)

        # Negative windows
        with pytest.raises(ValueError, match="Moving average windows must be positive"):
            MovingAverageStrategy(short_window=-5, long_window=30, shares=50)

    def test_insufficient_data(self, sample_data):
        """Test behavior with insufficient data."""
        strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=50)

        # Mock portfolio state
        strategy._portfolio_state = {
            "cash": 10000,
            "total_shares": 0,
            "portfolio_value": 10000,
            "positions": {},
            "current_price": 100,
            "is_last_day": False,
        }

        # Only provide 20 data points (< long_window of 30)
        prediction = strategy.predict(sample_data.head(20), [])

        assert prediction["signal"] == "hold"
        assert prediction["size"] == 0

    def test_buy_signal_golden_cross(self, sample_data):
        """Test buy signal on golden cross (short MA > long MA)."""
        strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=50)

        # Create data where short MA > long MA (uptrend)
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        uptrend_data = pd.DataFrame(
            {
                "Open": [100 + i * 2 for i in range(50)],
                "High": [105 + i * 2 for i in range(50)],
                "Low": [95 + i * 2 for i in range(50)],
                "Close": [100 + i * 2 for i in range(50)],
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

        # Mock portfolio state - no position
        strategy._portfolio_state = {
            "cash": 10000,
            "total_shares": 0,
            "portfolio_value": 10000,
            "positions": {},
            "current_price": 200,
            "is_last_day": False,
        }

        prediction = strategy.predict(uptrend_data, [])

        assert prediction["signal"] == "buy"
        assert prediction["size"] == 50

    def test_sell_signal_death_cross(self, sample_data):
        """Test sell signal on death cross (short MA < long MA)."""
        strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=50)

        # Create data where short MA < long MA (downtrend)
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        downtrend_data = pd.DataFrame(
            {
                "Open": [200 - i * 2 for i in range(50)],
                "High": [205 - i * 2 for i in range(50)],
                "Low": [195 - i * 2 for i in range(50)],
                "Close": [200 - i * 2 for i in range(50)],
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

        # Mock portfolio state - has position
        strategy._portfolio_state = {
            "cash": 5000,
            "total_shares": 50,
            "portfolio_value": 10000,
            "positions": {"BUY_1": {"shares": 50}},
            "current_price": 100,
            "is_last_day": False,
        }

        prediction = strategy.predict(downtrend_data, [])

        assert prediction["signal"] == "sell"
        assert prediction["size"] == 50

    def test_hold_when_already_positioned(self, sample_data):
        """Test hold signal when already have position and short MA > long MA."""
        strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=50)

        # Create uptrend data
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        uptrend_data = pd.DataFrame(
            {
                "Open": [100 + i * 2 for i in range(50)],
                "High": [105 + i * 2 for i in range(50)],
                "Low": [95 + i * 2 for i in range(50)],
                "Close": [100 + i * 2 for i in range(50)],
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

        # Mock portfolio state - already has position
        strategy._portfolio_state = {
            "cash": 5000,
            "total_shares": 50,
            "portfolio_value": 15000,
            "positions": {"BUY_1": {"shares": 50}},
            "current_price": 200,
            "is_last_day": False,
        }

        prediction = strategy.predict(uptrend_data, [])

        assert prediction["signal"] == "hold"

    def test_hold_when_no_position_downtrend(self, sample_data):
        """Test hold signal when no position and short MA < long MA."""
        strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=50)

        # Create downtrend data
        dates = pd.date_range(start="2020-01-01", periods=50, freq="D")
        downtrend_data = pd.DataFrame(
            {
                "Open": [200 - i * 2 for i in range(50)],
                "High": [205 - i * 2 for i in range(50)],
                "Low": [195 - i * 2 for i in range(50)],
                "Close": [200 - i * 2 for i in range(50)],
                "Volume": [1000000] * 50,
            },
            index=dates,
        )

        # Mock portfolio state - no position
        strategy._portfolio_state = {
            "cash": 10000,
            "total_shares": 0,
            "portfolio_value": 10000,
            "positions": {},
            "current_price": 100,
            "is_last_day": False,
        }

        prediction = strategy.predict(downtrend_data, [])

        assert prediction["signal"] == "hold"
