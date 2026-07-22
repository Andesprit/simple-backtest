"""Flat fee commission."""

from math import isfinite

from simple_backtest.commission.base import Commission


class FlatCommission(Commission):
    """Fixed fee per trade, regardless of trade size.

    Common for discount brokers with flat-rate pricing.

    Example:
        # $5 per trade
        commission = FlatCommission(fee=5.0)

        # Commission is same regardless of trade size
        cost1 = commission.calculate(10, 50)   # Returns 5.0
        cost2 = commission.calculate(100, 50)  # Returns 5.0
    """

    def __init__(self, fee: float, name: str | None = None):
        """Initialize flat commission.

        :param fee: Flat fee per trade
        :param name: Commission name
        """
        super().__init__(name=name or f"Flat(${fee:.2f})")

        if not isinstance(fee, (int, float)) or isinstance(fee, bool) or not isfinite(fee):
            raise ValueError(f"Commission fee must be finite, got {fee}")
        if fee < 0:
            raise ValueError(f"Commission fee must be non-negative, got {fee}")

        self.fee = fee

    def calculate(self, shares: float, price: float) -> float:
        """Calculate flat commission.

        :param shares: Number of shares traded (unused)
        :param price: Price per share (unused)
        :return: Commission amount
        """
        return self.fee
