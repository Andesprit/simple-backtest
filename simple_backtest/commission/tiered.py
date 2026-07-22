"""Tiered commission based on trade value."""

from math import isfinite
from typing import List, Tuple

from simple_backtest.commission.base import Commission


class TieredCommission(Commission):
    """Commission with different rates based on trade value tiers.

    Volume-based pricing where larger trades get lower rates.

    Example:
        # Define tiers: (threshold, rate)
        tiers = [
            (1000, 0.002),     # $0-1000: 0.2%
            (5000, 0.001),     # $1000-5000: 0.1%
            (float('inf'), 0.0005),  # $5000+: 0.05%
        ]
        commission = TieredCommission(tiers=tiers)

        # Small trade (100 shares @ $5 = $500)
        cost1 = commission.calculate(100, 5)  # 0.2% of $500 = $1.00

        # Large trade (100 shares @ $60 = $6000)
        # $0-1000 at 0.2% + $1000-5000 at 0.1% + $5000-6000 at 0.05%
        cost2 = commission.calculate(100, 60)  # = $2 + $4 + $0.50 = $6.50
    """

    def __init__(self, tiers: List[Tuple[float, float]], name: str | None = None):
        """Initialize tiered commission.

        :param tiers: List of (threshold, rate) tuples, sorted by threshold
        :param name: Commission name
        """
        super().__init__(name=name or "Tiered")

        if not tiers:
            raise ValueError("Tiers list cannot be empty")

        # Validate tiers
        prev_threshold = 0.0
        for i, (threshold, rate) in enumerate(tiers):
            if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
                raise ValueError(f"Tier {i}: threshold must be a finite number or infinity")
            if not isinstance(rate, (int, float)) or isinstance(rate, bool) or not isfinite(rate):
                raise ValueError(f"Tier {i}: rate must be finite")
            if threshold != float("inf") and not isfinite(threshold):
                raise ValueError(f"Tier {i}: threshold must be finite or positive infinity")
            if threshold <= prev_threshold and threshold != float("inf"):
                raise ValueError(
                    f"Tier {i}: thresholds must be in ascending order. "
                    f"Got {threshold} after {prev_threshold}"
                )

            if rate < 0:
                raise ValueError(f"Tier {i}: rate must be non-negative, got {rate}")

            prev_threshold = threshold

        if tiers[-1][0] != float("inf"):
            raise ValueError("Tiers must end with a (float('inf'), rate) tier")

        self.tiers = tiers

    def calculate(self, shares: float, price: float) -> float:
        """Calculate tiered commission based on trade value.

        :param shares: Number of shares traded
        :param price: Price per share
        :return: Commission amount
        """
        trade_value = shares * price
        commission = 0.0
        prev_threshold = 0.0

        for threshold, rate in self.tiers:
            if trade_value <= threshold:
                # Trade value falls in this tier
                commission += (trade_value - prev_threshold) * rate
                break
            else:
                # Trade value exceeds this tier, apply rate to tier range
                commission += (threshold - prev_threshold) * rate
                prev_threshold = threshold

        return commission
