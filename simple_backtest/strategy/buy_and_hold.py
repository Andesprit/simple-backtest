"""Buy and hold strategy."""

from typing import Any, Dict, List

import pandas as pd

from simple_backtest.strategy.base import Strategy


class BuyAndHoldStrategy(Strategy):
    """Buy once and hold until end."""

    def __init__(self, shares: float = 100, name: str | None = None):
        """Initialize strategy.

        :param shares: Number of shares to buy
        :param name: Strategy name (defaults to "BuyAndHold")
        """
        super().__init__(name=name or "BuyAndHold")
        self.shares = shares
        self.bought = False

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Buy once, then leave the position open for the engine to mark or liquidate.

        :param data: OHLCV DataFrame (unused)
        :param trade_history: Past trades (unused - for backward compatibility)
        :return: Trading signal dict
        """
        # Buy once at the start
        if not self.bought and not self.has_position():
            return self.buy(self.shares)

        return self.hold()

    def on_trade_executed(self, trade_info: Dict[str, Any]) -> None:
        """Mark the initial purchase only after it actually fills."""
        if trade_info["signal"] == "buy":
            self.bought = True

    def reset_state(self) -> None:
        """Reset state for new backtest."""
        super().reset_state()
        self.bought = False
