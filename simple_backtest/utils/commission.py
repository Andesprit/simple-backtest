"""Commission calculation functions."""

from math import isfinite
from typing import Callable, List, Tuple

from simple_backtest.config.settings import BacktestConfig


def percentage_commission(shares: float, price: float, rate: float) -> float:
    """Calculate commission as percentage of trade value.

    :param shares: Shares traded
    :param price: Price per share
    :param rate: Commission rate (e.g., 0.001 for 0.1%)
    :return: Commission amount
    """
    return shares * price * rate


def flat_commission(shares: float, price: float, flat_fee: float) -> float:
    """Calculate flat commission per trade.

    :param shares: Unused (for signature compatibility)
    :param price: Unused (for signature compatibility)
    :param flat_fee: Flat commission amount
    :return: Commission amount
    """
    return flat_fee


def tiered_commission(shares: float, price: float, tiers: List[Tuple[float, float]]) -> float:
    """Calculate tiered commission based on trade value.

    :param shares: Shares traded
    :param price: Price per share
    :param tiers: List of (threshold, rate) tuples sorted by threshold
    :return: Commission amount
    """
    if not tiers or tiers[-1][0] != float("inf"):
        raise ValueError("Tiered commissions require a final (float('inf'), rate) tier")

    trade_value = shares * price
    commission = 0.0
    prev_threshold = 0.0

    for threshold, rate in tiers:
        if trade_value <= threshold:
            # Trade value falls in this tier
            commission += (trade_value - prev_threshold) * rate
            break
        else:
            # Trade value exceeds this tier, apply rate to tier range
            commission += (threshold - prev_threshold) * rate
            prev_threshold = threshold

    return commission


def get_commission_calculator(config: BacktestConfig) -> Callable[[float, float], float]:
    """Create commission calculator from config.

    :param config: BacktestConfig with commission settings
    :return: Commission function (shares, price) -> commission
    """
    if config.commission_type == "percentage":
        if isinstance(config.commission_value, list):
            raise ValueError("Percentage commission_value must be numeric")
        rate = float(config.commission_value)
        return lambda shares, price: percentage_commission(shares, price, rate)

    elif config.commission_type == "flat":
        if isinstance(config.commission_value, list):
            raise ValueError("Flat commission_value must be numeric")
        flat_fee = float(config.commission_value)
        return lambda shares, price: flat_commission(shares, price, flat_fee)

    elif config.commission_type == "tiered":
        if not isinstance(config.commission_value, list):
            raise ValueError("Tiered commission_value must be a tier list")
        tiers = config.commission_value
        return lambda shares, price: tiered_commission(shares, price, tiers)

    elif config.commission_type == "custom":
        raise ValueError(
            "commission_calculator must be provided to Backtest when commission_type='custom'"
        )

    else:
        raise ValueError(
            f"Invalid commission_type: {config.commission_type}. "
            f"Must be one of: 'percentage', 'flat', 'tiered', 'custom'"
        )


def create_custom_commission(
    func: Callable[[float, float], float],
) -> Callable[[float, float], float]:
    """Wrap custom commission function with validation.

    :param func: Commission function (shares, price) -> commission
    :return: Validated commission calculator
    """

    def validated_commission(shares: float, price: float) -> float:
        commission = func(shares, price)
        if (
            not isinstance(commission, (int, float))
            or isinstance(commission, bool)
            or not isfinite(commission)
        ):
            raise ValueError(f"Commission must be a finite number, got {commission}")
        if commission < 0:
            raise ValueError(
                f"Commission must be non-negative, got {commission} "
                f"for trade: {shares} shares @ ${price}"
            )
        return commission

    return validated_commission
