"""Tests for Strategy base class."""

import pytest

from simple_backtest.strategy.base import Strategy


class DummyStrategy(Strategy):
    """Dummy strategy implementation for testing."""

    def predict(self, data, trade_history):
        """Return a buy signal."""
        return {"signal": "buy", "size": 10, "order_ids": None}


def test_strategy_initialization():
    """Test strategy initialization."""
    strategy = DummyStrategy()
    assert strategy.get_name() == "DummyStrategy"

    strategy_named = DummyStrategy(name="CustomName")
    assert strategy_named.get_name() == "CustomName"


def test_strategy_must_implement_predict():
    """Test that Strategy is abstract."""
    with pytest.raises(TypeError):
        Strategy()  # Cannot instantiate abstract class


def test_validate_prediction_valid():
    """Test prediction validation with valid input."""
    strategy = DummyStrategy()

    # Valid buy signal
    prediction = {"signal": "buy", "size": 10, "order_ids": None}
    strategy.validate_prediction(prediction)  # Should not raise

    # Valid hold signal
    prediction = {"signal": "hold", "size": 0}
    strategy.validate_prediction(prediction)

    # Valid sell signal
    prediction = {"signal": "sell", "size": 10, "order_ids": None}
    strategy.validate_prediction(prediction)


def test_validate_prediction_missing_keys():
    """Test prediction validation fails with missing keys."""
    strategy = DummyStrategy()

    with pytest.raises(ValueError, match="missing required keys"):
        strategy.validate_prediction({"signal": "buy"})

    with pytest.raises(ValueError, match="missing required keys"):
        strategy.validate_prediction({"size": 10})


def test_validate_prediction_invalid_signal():
    """Test prediction validation fails with invalid signal."""
    strategy = DummyStrategy()

    with pytest.raises(ValueError, match="invalid signal"):
        strategy.validate_prediction({"signal": "invalid", "size": 10})


def test_validate_prediction_invalid_size():
    """Test prediction validation fails with invalid size."""
    strategy = DummyStrategy()

    # Negative size
    with pytest.raises(ValueError, match="invalid size"):
        strategy.validate_prediction({"signal": "buy", "size": -10})

    # Non-numeric size
    with pytest.raises(ValueError, match="invalid size"):
        strategy.validate_prediction({"signal": "buy", "size": "ten"})

    for invalid_size in [float("nan"), float("inf"), True]:
        with pytest.raises(ValueError, match="finite non-negative number"):
            strategy.validate_prediction({"signal": "buy", "size": invalid_size})


def test_validate_prediction_hold_requires_zero_size():
    strategy = DummyStrategy()

    with pytest.raises(ValueError, match="hold.*size 0"):
        strategy.validate_prediction({"signal": "hold", "size": 1})


def test_validate_prediction_sell_without_order_ids():
    """Test sell signal requires order_ids key."""
    strategy = DummyStrategy()

    with pytest.raises(ValueError, match="did not specify 'order_ids'"):
        strategy.validate_prediction({"signal": "sell", "size": 10})


def test_strategy_reset_state():
    """Test strategy state reset."""
    strategy = DummyStrategy()
    strategy._state_initialized = True
    strategy._portfolio_state = {"cash": 100}

    strategy.reset_state()

    assert strategy._state_initialized is False
    assert strategy._portfolio_state is None


def test_on_trade_executed_hook():
    """Test on_trade_executed hook can be overridden."""

    class CustomStrategy(Strategy):
        def __init__(self):
            super().__init__()
            self.trades_executed = []

        def predict(self, data, trade_history):
            return {"signal": "hold", "size": 0}

        def on_trade_executed(self, trade_info):
            self.trades_executed.append(trade_info)

    strategy = CustomStrategy()
    trade_info = {"order_id": "TEST_1", "signal": "buy"}

    strategy.on_trade_executed(trade_info)

    assert len(strategy.trades_executed) == 1
    assert strategy.trades_executed[0]["order_id"] == "TEST_1"


def test_strategy_repr():
    """Test strategy string representation."""
    strategy = DummyStrategy(name="MyStrategy")
    repr_str = repr(strategy)

    assert "DummyStrategy" in repr_str
    assert "MyStrategy" in repr_str
