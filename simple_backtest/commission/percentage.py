"""Percentage-based commission."""

from math import isfinite

from simple_backtest.commission.base import Commission


class PercentageCommission(Commission):
    """Commission as a percentage of trade value.

    Most common commission structure used by brokers.

    Example:
        # 0.1% commission
        commission = PercentageCommission(rate=0.001)

        # For a trade of 100 shares @ $50
        cost = commission.calculate(100, 50)  # Returns 5.0 (0.1% of $5000)
    """

    def __init__(self, rate: float, name: str | None = None):
        """Initialize percentage commission.

        :param rate: Commission rate (e.g., 0.001 for 0.1%)
        :param name: Commission name
        """
        super().__init__(name=name or f"Percentage({rate * 100:.3f}%)")

        if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not isfinite(rate):
            raise ValueError(f"Commission rate must be finite, got {rate}")
        if rate < 0:
            raise ValueError(f"Commission rate must be non-negative, got {rate}")

        self.rate = rate

    def calculate(self, shares: float, price: float) -> float:
        """Calculate commission as percentage of trade value.

        :param shares: Number of shares traded
        :param price: Price per share
        :return: Commission amount
        """
        trade_value = shares * price
        return trade_value * self.rate
