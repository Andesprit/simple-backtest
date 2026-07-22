"""Input validation with custom exceptions."""

import warnings
from datetime import datetime
from typing import List

import pandas as pd

from simple_backtest.strategy.base import Strategy


class BacktestError(Exception):
    """Base exception for backtesting."""

    pass


class DataValidationError(BacktestError):
    """Data validation error."""

    pass


class DateRangeError(BacktestError):
    """Date range validation error."""

    pass


class StrategyError(BacktestError):
    """Strategy validation error."""

    pass


class StrategyExecutionError(StrategyError, RuntimeError):
    """A strategy callback or order failed during a backtest."""

    def __init__(
        self,
        strategy_name: str,
        timestamp: datetime,
        stage: str,
        cause: Exception,
    ) -> None:
        self.strategy_name = strategy_name
        self.timestamp = timestamp
        self.stage = stage
        self.cause = cause
        super().__init__(f"Strategy '{strategy_name}' {stage} failed at {timestamp}: {cause}")

    def __reduce__(self):
        """Preserve contextual constructor arguments across worker processes."""
        serialized_cause = RuntimeError(f"{type(self.cause).__name__}: {self.cause}")
        return (
            type(self),
            (self.strategy_name, self.timestamp, self.stage, serialized_cause),
        )


def validate_dataframe(
    data: pd.DataFrame, strict: bool = True, require_volume: bool = False
) -> None:
    """Validate DataFrame structure for backtesting with actionable error messages.

    The engine accepts one OHLC(V) series at a time. Its accounting model is
    long-only and cash-funded; asset-specific mechanics are not inferred here.

    :param data: DataFrame to validate (OHLCV format)
    :param strict: If True, raise errors; if False, warn only
    :param require_volume: If True, Volume column is required; if False, it's optional
    """
    # Check if DataFrame
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError(
            f"Expected pandas DataFrame, got {type(data).__name__}.\n"
            f"\n"
            f"→ Fix: Load your data into a DataFrame:\n"
            f"  import pandas as pd\n"
            f"  df = pd.read_csv('data.csv', index_col=0, parse_dates=True)\n"
            f"\n"
            f"  Or if you have lists/arrays:\n"
            f"  df = pd.DataFrame({{'Open': [...], 'High': [...], ...}})"
        )

    if data.empty:
        raise DataValidationError(
            "DataFrame is empty (0 rows).\n"
            "\n"
            "→ Fix: Ensure you loaded data correctly:\n"
            "  print(df.head())  # Check if data loaded\n"
            "  print(len(df))    # Check number of rows"
        )

    # Handle MultiIndex columns (common with yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        # Get unique values from all levels except the first (which should be OHLCV names)
        if data.columns.nlevels == 2:
            # Typical yfinance structure: (OHLCV, Ticker)
            level_1_values = data.columns.get_level_values(1).unique()

            if len(level_1_values) == 1:
                # Single ticker - flatten by dropping the second level
                data.columns = data.columns.droplevel(1)
            else:
                # Multiple tickers - user needs to select one
                raise DataValidationError(
                    f"DataFrame has MultiIndex columns with multiple tickers: {list(level_1_values)}\n"
                    f"\n"
                    f"The framework works with single-asset data at a time.\n"
                    f"\n"
                    f"→ Fix: Select a single ticker:\n"
                    f"  # Option 1: Select by ticker name\n"
                    f"  data = data.xs('{level_1_values[0]}', level=1, axis=1)\n"
                    f"\n"
                    f"  # Option 2: If you want a specific ticker\n"
                    f"  ticker = 'AAPL'  # Replace with your ticker\n"
                    f"  data = data.xs(ticker, level=1, axis=1)\n"
                    f"\n"
                    f"  # Option 3: Download single ticker from yfinance\n"
                    f"  data = yf.download('AAPL', start='2020-01-01', end='2023-12-31')"
                )
        else:
            # Complex MultiIndex - provide generic error
            raise DataValidationError(
                f"DataFrame has complex MultiIndex columns with {data.columns.nlevels} levels.\n"
                f"\n"
                f"Column structure: {data.columns.tolist()[:5]}...\n"
                f"\n"
                f"→ Fix: Flatten to single-level columns with standard OHLCV names:\n"
                f"  # Check current structure\n"
                f"  print(data.columns)\n"
                f"  \n"
                f"  # Then flatten appropriately based on your data structure"
            )

    # Check required columns
    required_columns = ["Open", "High", "Low", "Close"]
    if require_volume:
        required_columns.append("Volume")

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        col_examples = "      'open': 'Open',\n      'high': 'High',\n      'low': 'Low',\n      'close': 'Close'"
        if require_volume or "Volume" in missing_columns:
            col_examples += ",\n      'volume': 'Volume'"

        raise DataValidationError(
            f"Missing required columns: {missing_columns}\n"
            f"\n"
            f"Required columns: {required_columns}\n"
            f"Your columns: {list(data.columns)}\n"
            f"\n"
            f"→ Fix option 1: Rename columns to match (case-sensitive!):\n"
            f"  df.rename(columns={{\n"
            f"{col_examples}\n"
            f"  }}, inplace=True)\n"
            f"\n"
            f"→ Fix option 2: Rename all columns at once:\n"
            f"  df.columns = {required_columns}"
        )

    # Check for duplicate columns
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise DataValidationError(f"DataFrame has duplicate column names: {duplicates}")

    # Validate data types
    cols_to_validate = ["Open", "High", "Low", "Close"]
    if "Volume" in data.columns:
        cols_to_validate.append("Volume")

    for col in cols_to_validate:
        col_data = data[col]

        if not pd.api.types.is_numeric_dtype(col_data):
            dtype = col_data.dtype if hasattr(col_data, "dtype") else type(col_data)
            # Get sample values safely (handle both Series and DataFrame)
            try:
                sample_values = col_data.head(3).tolist()
            except AttributeError:
                sample_values = str(col_data.head(3))

            raise DataValidationError(
                f"Column '{col}' must be numeric, got {dtype}.\n"
                f"\n"
                f"Current type: {dtype}\n"
                f"Sample values: {sample_values}\n"
                f"\n"
                f"→ Fix: Convert to numeric:\n"
                f"  df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')\n"
                f"  df.dropna(inplace=True)  # Remove rows that couldn't convert"
            )

    # Check for missing values
    missing_counts = data[required_columns].isnull().sum()
    if missing_counts.any():
        nan_columns = missing_counts[missing_counts > 0].index.tolist()
        nan_count = {col: missing_counts[col] for col in nan_columns}

        if strict:
            raise DataValidationError(
                f"Found NaN values in columns: {nan_columns}\n"
                f"\n"
                f"NaN counts: {nan_count}\n"
                f"Total rows: {len(data)}\n"
                f"\n"
                f"→ Fix option 1: Forward-fill missing values:\n"
                f"  df.fillna(method='ffill', inplace=True)\n"
                f"\n"
                f"→ Fix option 2: Drop rows with NaN:\n"
                f"  df.dropna(inplace=True)\n"
                f"\n"
                f"→ Fix option 3: Fill with specific value:\n"
                f"  df.fillna(0, inplace=True)"
            )
        else:
            warnings.warn(
                f"DataFrame contains missing values:\n{missing_counts[missing_counts > 0]}",
                UserWarning,
            )

    # Check for infinite values
    inf_counts = data[cols_to_validate].isin([float("inf"), float("-inf")]).sum()
    if inf_counts.any():
        raise DataValidationError(
            f"DataFrame contains infinite values:\n{inf_counts[inf_counts > 0]}"
        )

    # Verify date index
    if not isinstance(data.index, pd.DatetimeIndex):
        if strict:
            raise DataValidationError(
                f"DataFrame index must be DatetimeIndex, got {type(data.index).__name__}.\n"
                f"\n"
                f"Current index type: {type(data.index).__name__}\n"
                f"First few index values: {data.index[:3].tolist()}\n"
                f"\n"
                f"→ Fix option 1: Convert existing index to datetime:\n"
                f"  df.index = pd.to_datetime(df.index)\n"
                f"\n"
                f"→ Fix option 2: Parse dates when loading CSV:\n"
                f"  df = pd.read_csv('data.csv', index_col=0, parse_dates=True)\n"
                f"\n"
                f"→ Fix option 3: Set a datetime column as index:\n"
                f"  df['Date'] = pd.to_datetime(df['Date'])\n"
                f"  df.set_index('Date', inplace=True)"
            )
        else:
            warnings.warn(
                f"DataFrame index should be DatetimeIndex, got {type(data.index)}. "
                "Date-based operations may not work correctly.",
                UserWarning,
            )

    # Check if index is sorted
    if isinstance(data.index, pd.DatetimeIndex):
        if not data.index.is_monotonic_increasing:
            if strict:
                raise DataValidationError(
                    f"DataFrame index must be sorted in ascending order.\n"
                    f"\n"
                    f"First few dates: {data.index[:3].tolist()}\n"
                    f"Last few dates: {data.index[-3:].tolist()}\n"
                    f"\n"
                    f"→ Fix: Sort your data by date:\n"
                    f"  df.sort_index(inplace=True)"
                )
            else:
                warnings.warn(
                    "DataFrame index (dates) is not sorted in ascending order. "
                    "This may cause unexpected behavior.",
                    UserWarning,
                )

    # Check for duplicate indices
    if data.index.duplicated().any():
        duplicate_count = data.index.duplicated().sum()
        if strict:
            raise DataValidationError(f"DataFrame has {duplicate_count} duplicate index values")
        else:
            warnings.warn(
                f"DataFrame has {duplicate_count} duplicate index values. "
                "This may cause unexpected behavior.",
                UserWarning,
            )

    # Validate OHLC relationships
    invalid_ohlc = (
        (data["High"] < data["Low"])
        | (data["Open"] < data["Low"])
        | (data["Open"] > data["High"])
        | (data["Close"] < data["Low"])
        | (data["Close"] > data["High"])
    )

    if invalid_ohlc.any():
        invalid_count = invalid_ohlc.sum()
        if strict:
            raise DataValidationError(
                f"DataFrame has {invalid_count} rows with invalid OHLC relationships. "
                f"First invalid row:\n{data[invalid_ohlc].head(1)}"
            )
        else:
            warnings.warn(
                f"DataFrame has {invalid_count} rows with invalid OHLC relationships.",
                UserWarning,
            )

    price_columns = ["Open", "High", "Low", "Close"]
    nonpositive_prices = (data[price_columns] <= 0).any()
    if nonpositive_prices.any():
        raise DataValidationError(
            "Price columns must be strictly positive. Invalid columns: "
            f"{nonpositive_prices[nonpositive_prices].index.tolist()}"
        )

    # Prices and volume cannot be negative; zero volume is valid.
    negative_values = (data[cols_to_validate] < 0).any()
    if negative_values.any():
        raise DataValidationError(
            f"DataFrame contains negative values in columns: "
            f"{negative_values[negative_values].index.tolist()}"
        )

    # Check for suspicious gaps in dates (if DatetimeIndex)
    if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
        date_diffs = data.index.to_series().diff()
        median_diff = date_diffs.median()
        large_gaps = date_diffs > median_diff * 5  # Gaps 5x larger than median

        if large_gaps.any():
            gap_count = large_gaps.sum()
            warnings.warn(
                f"Found {gap_count} large gaps in date index (>5x median gap). "
                f"This may indicate missing data.",
                UserWarning,
            )


def validate_date_range(
    data: pd.DataFrame,
    trading_start_date: datetime | None,
    trading_end_date: datetime | None,
    lookback_period: int,
) -> None:
    """Validate trading date range compatibility with data.

    :param data: DataFrame with DatetimeIndex
    :param trading_start_date: Trading start (None = auto)
    :param trading_end_date: Trading end (None = auto)
    :param lookback_period: Rows needed before trading
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise DateRangeError("DataFrame must have DatetimeIndex for date range validation")

    if data.empty:
        raise DateRangeError("Cannot validate date range on empty DataFrame")

    data_start = data.index[0]
    data_end = data.index[-1]
    total_rows = len(data)

    # Validate lookback period
    if lookback_period >= total_rows:
        raise DateRangeError(
            f"lookback_period ({lookback_period}) must be less than total data rows ({total_rows})"
        )

    # Determine effective trading dates
    if trading_start_date is None:
        # Need at least lookback_period rows before trading can start
        if total_rows <= lookback_period:
            raise DateRangeError(
                f"Need at least {lookback_period + 1} rows for lookback_period={lookback_period}"
            )
        effective_start = data.index[lookback_period]
    else:
        effective_start = trading_start_date

    if trading_end_date is None:
        effective_end = data_end
    else:
        effective_end = trading_end_date

    # Validate start date
    if effective_start < data_start:
        raise DateRangeError(
            f"Trading start date ({effective_start}) is before data start ({data_start})"
        )

    if effective_start > data_end:
        raise DateRangeError(
            f"Trading start date ({effective_start}) is after data end ({data_end})"
        )

    # Validate end date
    if effective_end > data_end:
        raise DateRangeError(f"Trading end date ({effective_end}) is after data end ({data_end})")

    if effective_end < data_start:
        raise DateRangeError(
            f"Trading end date ({effective_end}) is before data start ({data_start})"
        )

    # Validate start < end
    if effective_start >= effective_end:
        raise DateRangeError(
            f"Trading start date ({effective_start}) must be before "
            f"trading end date ({effective_end})"
        )

    # Ensure enough data for lookback before trading starts
    start_idx = data.index.get_indexer([effective_start], method="nearest")[0]
    if start_idx < lookback_period:
        raise DateRangeError(
            f"Not enough data before trading start date for lookback_period={lookback_period}. "
            f"Need at least {lookback_period} rows before {effective_start}, "
            f"but only have {start_idx} rows."
        )


def validate_strategies(strategies: List[Strategy]) -> None:
    """Validate list of strategies.

    :param strategies: List of strategies to validate
    """
    if not strategies:
        raise StrategyError("Must provide at least one strategy")

    for i, strategy in enumerate(strategies):
        # Check inherits from base Strategy
        if not isinstance(strategy, Strategy):
            raise StrategyError(
                f"Strategy at index {i} does not inherit from base Strategy class. "
                f"Got type: {type(strategy)}"
            )

        # Check has predict method
        if not hasattr(strategy, "predict") or not callable(getattr(strategy, "predict")):
            raise StrategyError(
                f"Strategy '{strategy.get_name()}' does not have callable predict() method"
            )

        # Check name is valid
        name = strategy.get_name()
        if not name or not isinstance(name, str):
            raise StrategyError(
                f"Strategy at index {i} has invalid name: {name}. Must be non-empty string."
            )

    # Check for duplicate names
    strategy_names = [s.get_name() for s in strategies]
    duplicate_names = [name for name in strategy_names if strategy_names.count(name) > 1]

    if duplicate_names:
        raise StrategyError(
            f"Duplicate strategy names found: {set(duplicate_names)}. "
            f"All strategies must have unique names."
        )
