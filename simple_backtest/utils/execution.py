"""Execution price extraction from OHLCV data."""

import warnings
from typing import Callable, Literal

import pandas as pd


def get_open_price(row: pd.Series) -> float:
    """Return open price."""
    return float(row["Open"])


def get_close_price(row: pd.Series) -> float:
    """Return close price."""
    return float(row["Close"])


def get_typical_price(row: pd.Series) -> float:
    """Return the OHLC typical price.

    A single OHLC(V) bar does not contain the intrabar observations needed to
    calculate true volume-weighted average price.
    """
    high = float(row["High"])
    low = float(row["Low"])
    close = float(row["Close"])
    return (high + low + close) / 3


def get_vwap(row: pd.Series) -> float:
    """Return typical price through the deprecated ``vwap`` compatibility alias."""
    warnings.warn(
        "get_vwap() is deprecated; use get_typical_price() because a single "
        "OHLCV bar cannot provide true VWAP",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_typical_price(row)


def get_execution_price(
    row: pd.Series,
    method: Literal["open", "close", "typical", "vwap", "custom"] = "open",
    custom_func: Callable[[pd.Series], float] | None = None,
) -> float:
    """Extract execution price using specified method.

    :param row: OHLCV Series
    :param method: Price method ('open', 'close', 'typical', deprecated 'vwap', 'custom')
    :param custom_func: Custom function for 'custom' method
    :return: Execution price
    """
    if method == "open":
        return get_open_price(row)

    elif method == "close":
        return get_close_price(row)

    elif method == "typical":
        return get_typical_price(row)

    elif method == "vwap":
        warnings.warn(
            "execution_price='vwap' is a deprecated alias for 'typical'; "
            "a single OHLCV bar cannot provide true VWAP",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_typical_price(row)

    elif method == "custom":
        if custom_func is None:
            raise ValueError("custom_func must be provided when method='custom'")
        return float(custom_func(row))

    else:
        raise ValueError(
            f"Invalid execution price method: {method}. "
            f"Must be one of: 'open', 'close', 'typical', 'vwap', 'custom'"
        )


def create_execution_price_extractor(
    method: Literal["open", "close", "typical", "vwap", "custom"] = "open",
    custom_func: Callable[[pd.Series], float] | None = None,
) -> Callable[[pd.Series], float]:
    """Create price extractor function.

    :param method: Price method
    :param custom_func: Custom function for 'custom' method
    :return: Extractor function (row) -> price
    """
    if method == "open":
        return get_open_price
    elif method == "close":
        return get_close_price
    elif method == "typical":
        return get_typical_price
    elif method == "vwap":
        warnings.warn(
            "execution_price='vwap' is a deprecated alias for 'typical'; "
            "a single OHLCV bar cannot provide true VWAP",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_typical_price
    elif method == "custom":
        if custom_func is None:
            raise ValueError("custom_func must be provided when method='custom'")
        return custom_func
    else:
        raise ValueError(
            f"Invalid execution price method: {method}. "
            f"Must be one of: 'open', 'close', 'typical', 'vwap', 'custom'"
        )


def validate_ohlcv_row(row: pd.Series, require_volume: bool = False) -> None:
    """Validate OHLC(V) row format.

    :param row: Series to validate
    :param require_volume: If True, Volume column is required
    """
    required_columns = ["Open", "High", "Low", "Close"]
    if require_volume:
        required_columns.append("Volume")

    missing_columns = [col for col in required_columns if col not in row.index]

    if missing_columns:
        raise KeyError(f"Missing required OHLC columns: {missing_columns}")

    # Validate price relationships
    open_price = row["Open"]
    high = row["High"]
    low = row["Low"]
    close = row["Close"]

    # Check price values
    price_values = [open_price, high, low, close]
    if any(val < 0 for val in price_values):
        raise ValueError(f"OHLC values must be non-negative: {row.to_dict()}")

    # Check volume if present
    if "Volume" in row.index:
        volume = row["Volume"]
        if volume < 0:
            raise ValueError(f"Volume must be non-negative: {volume}")

    if high < low:
        raise ValueError(f"High ({high}) must be >= Low ({low})")

    if not (low <= open_price <= high):
        raise ValueError(f"Open ({open_price}) must be between Low ({low}) and High ({high})")

    if not (low <= close <= high):
        raise ValueError(f"Close ({close}) must be between Low ({low}) and High ({high})")
