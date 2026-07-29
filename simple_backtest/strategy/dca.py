"""Dollar Cost Averaging (DCA) strategy."""

from typing import Any, Dict, List

import pandas as pd

from simple_backtest.strategy.base import Strategy


class DCAStrategy(Strategy):
    """Dollar Cost Averaging - buy fixed amount at regular intervals."""

    def __init__(
        self,
        investment_amount: float = 1000,
        interval_days: int = 7,
        name: str | None = None,
    ):
        """Initialize DCA strategy.

        :param investment_amount: Dollar amount to invest at each interval
        :param interval_days: Number of days between purchases
        :param name: Strategy name (defaults to "DCA_{interval}")
        """
        super().__init__(name=name or f"DCA_{interval_days}d")
        self.investment_amount = investment_amount
        self.interval_days = interval_days
        self.last_trade_date = None

    def predict(self, data: pd.DataFrame, trade_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Buy the configured amount when a fill-based interval is due.

        :param data: OHLCV DataFrame with lookback window
        :param trade_history: Past trades (for backward compatibility)
        :return: Trading signal dict
        """
        current_date = (
            self._portfolio_state.get("timestamp") if self._portfolio_state is not None else None
        )
        if current_date is None and len(data) > 0:
            current_date = data.index[-1]

        # Check if it's time to buy
        if self.last_trade_date is None:
            return self.buy_cash(min(self.investment_amount, self.get_cash()))

        # Calculate days since last buy
        if current_date is not None:
            days_elapsed = (current_date - self.last_trade_date).days

            if days_elapsed >= self.interval_days:
                return self.buy_cash(min(self.investment_amount, self.get_cash()))

        return self.hold()

    def on_trade_executed(self, trade_info: Dict[str, Any]) -> None:
        """Advance the contribution schedule only after a buy fills."""
        if trade_info["signal"] == "buy":
            self.last_trade_date = trade_info["timestamp"]

    def reset_state(self) -> None:
        """Reset state for new backtest."""
        super().reset_state()
        self.last_trade_date = None
