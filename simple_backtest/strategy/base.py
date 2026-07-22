"""Base Strategy class using Strategy and Template Method design patterns."""

from abc import ABC, abstractmethod
from math import isfinite
from typing import Any, Dict, List, Optional

import pandas as pd


class Strategy(ABC):
    """Abstract base class for trading strategies.

    Strategies trade one long-only, cash-funded instrument. Quantities support
    fractional values, but the engine does not model shorts, leverage, margin,
    derivatives contract multipliers, funding, or currency conversion.

    Implement predict() to define strategy logic. State persists within a backtest
    but is reset between runs.

    Helper methods available in predict():
    - has_position(): Check if holding any units
    - get_position(): Get current units/quantity held
    - get_cash(): Get available cash
    - get_portfolio_value(): Get total portfolio value
    - buy(shares): Return buy signal (shares = units/quantity)
    - sell(shares): Return sell signal (shares = units/quantity)
    - sell_all(): Sell all positions (FIFO)
    - hold(): Return hold signal
    - buy_percent(percent): Buy units worth percent of portfolio
    - buy_cash(amount): Buy units worth specific cash amount
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize strategy.

        :param name: Strategy name (defaults to class name)
        """
        self._name = name or self.__class__.__name__
        self._state_initialized = False
        self._portfolio_state: Dict[str, Any] | None = None  # Injected by backtest engine

    def get_name(self) -> str:
        """Return strategy name."""
        return self._name

    @abstractmethod
    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate trading signal.

        :param data: OHLCV DataFrame with lookback window
        :param trade_history: List of past trades (for backward compatibility)
        :return: Dict with "signal" ("buy"/"hold"/"sell"), "size", "order_ids"

        Note: Use helper methods (has_position(), buy(), sell_all(), etc.) instead of
        parsing trade_history manually.
        """
        raise NotImplementedError("Strategy must implement predict() method")

    # Portfolio state access methods

    def has_position(self) -> bool:
        """Check if currently holding any shares.

        :return: True if holding shares, False otherwise
        """
        if self._portfolio_state is None:
            raise RuntimeError(
                "Portfolio state not available. This method can only be called inside predict()."
            )
        return self._portfolio_state["total_shares"] > 0

    def get_position(self) -> float:
        """Get current total shares held.

        :return: Number of shares currently held
        """
        if self._portfolio_state is None:
            raise RuntimeError(
                "Portfolio state not available. This method can only be called inside predict()."
            )
        return self._portfolio_state["total_shares"]

    def get_cash(self) -> float:
        """Get available cash.

        :return: Current cash balance
        """
        if self._portfolio_state is None:
            raise RuntimeError(
                "Portfolio state not available. This method can only be called inside predict()."
            )
        return self._portfolio_state["cash"]

    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions).

        :return: Total portfolio value
        """
        if self._portfolio_state is None:
            raise RuntimeError(
                "Portfolio state not available. This method can only be called inside predict()."
            )
        return self._portfolio_state["portfolio_value"]

    # Signal helper methods

    def buy(self, shares: float) -> Dict[str, Any]:
        """Helper to return buy signal.

        :param shares: Number of shares to buy
        :return: Buy signal dict
        """
        return {"signal": "buy", "size": shares, "order_ids": None}

    def sell(self, shares: float, order_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Helper to return sell signal.

        :param shares: Number of shares to sell
        :param order_ids: Specific order IDs to sell from (FIFO if None)
        :return: Sell signal dict
        """
        return {"signal": "sell", "size": shares, "order_ids": order_ids}

    def sell_all(self) -> Dict[str, Any]:
        """Helper to sell all positions (FIFO).

        :return: Sell signal dict for all shares
        """
        if self._portfolio_state is None:
            raise RuntimeError(
                "Portfolio state not available. This method can only be called inside predict()."
            )
        return {"signal": "sell", "size": self._portfolio_state["total_shares"], "order_ids": None}

    def hold(self) -> Dict[str, Any]:
        """Helper to return hold signal (no action).

        :return: Hold signal dict
        """
        return {"signal": "hold", "size": 0, "order_ids": None}

    # Position sizing helpers

    def buy_percent(self, percent: float) -> Dict[str, Any]:
        """Buy shares worth percent of current portfolio value.

        :param percent: Percentage as decimal (e.g., 0.1 for 10%)
        :return: Buy signal dict

        Example:
            return self.buy_percent(0.1)  # Invest 10% of portfolio
        """
        if self._portfolio_state is None:
            raise RuntimeError(
                "Portfolio state not available. This method can only be called inside predict()."
            )

        portfolio_value = self._portfolio_state["portfolio_value"]
        current_price = self._portfolio_state["current_price"]

        if current_price <= 0:
            return self.hold()

        shares = (portfolio_value * percent) / current_price
        return self.buy(shares)

    def buy_cash(self, amount: float) -> Dict[str, Any]:
        """Buy shares worth specific cash amount.

        :param amount: Cash amount to invest
        :return: Buy signal dict

        Example:
            return self.buy_cash(5000)  # Invest $5000
        """
        if self._portfolio_state is None:
            raise RuntimeError(
                "Portfolio state not available. This method can only be called inside predict()."
            )

        current_price = self._portfolio_state["current_price"]

        if current_price <= 0:
            return self.hold()

        shares = amount / current_price
        return self.buy(shares)

    def on_trade_executed(self, trade_info: Dict[str, Any]) -> None:
        """Hook called after trade execution.

        :param trade_info: Trade details (order_id, timestamp, signal, shares, price, etc.)
        """
        pass

    def reset_state(self) -> None:
        """Reset internal state before new backtest run."""
        self._state_initialized = False
        self._portfolio_state = None

    def validate_prediction(self, prediction: Dict[str, Any]) -> None:
        """Validate prediction format.

        :param prediction: Dict returned by predict()
        """
        required_keys = {"signal", "size"}
        missing_keys = required_keys - set(prediction.keys())
        if missing_keys:
            raise ValueError(
                f"Strategy {self._name} prediction missing required keys: {missing_keys}"
            )

        signal = prediction.get("signal")
        valid_signals = {"buy", "hold", "sell"}
        if signal not in valid_signals:
            raise ValueError(
                f"Strategy {self._name} returned invalid signal '{signal}'. "
                f"Must be one of: {valid_signals}"
            )

        size = prediction.get("size")
        if (
            not isinstance(size, (int, float))
            or isinstance(size, bool)
            or not isfinite(size)
            or size < 0
        ):
            raise ValueError(
                f"Strategy {self._name} returned invalid size '{size}'. "
                f"Must be a finite non-negative number."
            )

        if signal == "hold" and size != 0:
            raise ValueError(f"Strategy {self._name} returned hold with non-zero size; use size 0")

        if signal == "sell":
            if "order_ids" not in prediction:
                raise ValueError(
                    f"Strategy {self._name} returned 'sell' signal but did not specify 'order_ids'. "
                    f"Must provide list of order_ids to close or None to close oldest positions."
                )

    def __repr__(self) -> str:
        """String representation of strategy."""
        return f"{self.__class__.__name__}(name='{self._name}')"
