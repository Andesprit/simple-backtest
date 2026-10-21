"""Input validation with custom exceptions."""

import warnings
from datetime import datetime
from pathlib import Path
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


class InsufficientHistoryError(StrategyError):
    """A strategy requires more history than the engine will provide."""

    pass


class StrategyExecutionError(StrategyError, RuntimeError):
    """A strategy callback or order failed during a backtest."""

    def __init__(
        self,
        strategy_name: str,
        timestamp: datetime,
        stage: str,
        cause: Exception,
        hint: str = "",
    ) -> None:
        self.strategy_name = strategy_name
        self.timestamp = timestamp
        self.stage = stage
        self.cause = cause
        self.hint = hint
        super().__init__(f"Strategy '{strategy_name}' {stage} failed at {timestamp}: {cause}{hint}")

    def __reduce__(self):
        """Preserve contextual constructor arguments across worker processes."""
        serialized_cause = RuntimeError(f"{type(self.cause).__name__}: {self.cause}")
        return (
            type(self),
            (self.strategy_name, self.timestamp, self.stage, serialized_cause, self.hint),
        )


def _normalize_ohlcv_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Return an independent frame with unambiguous canonical OHLCV names."""
    data = data.copy(deep=True)
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

    # Reject duplicates before normalization so no price series is silently selected.
    if data.columns.duplicated().any():
        duplicates = data.columns[data.columns.duplicated()].tolist()
        raise DataValidationError(f"DataFrame has duplicate column names: {duplicates}")

    column_mapping = {}
    for canonical in ("Open", "High", "Low", "Close", "Volume"):
        matches = [
            col
            for col in data.columns
            if isinstance(col, str) and col.casefold() == canonical.casefold()
        ]
        if len(matches) > 1:
            raise DataValidationError(
                f"Ambiguous columns for '{canonical}': {matches}. "
                "Remove or rename the extra columns so only one matches this OHLCV name."
            )
        if matches:
            column_mapping[matches[0]] = canonical
    data.rename(columns=column_mapping, inplace=True)

    return data


def validate_dataframe(
    data: pd.DataFrame, strict: bool = True, require_volume: bool = False
) -> pd.DataFrame:
    """Validate data and return a normalized copy without changing caller data.

    OHLCV names are matched case-insensitively. Ambiguous aliases (for example
    Close and close) are rejected. Date, gap and adjustment fixes are suggestions
    only; validation never applies them automatically.
    The engine accepts one OHLC(V) series at a time. Its accounting model is
    long-only and cash-funded; asset-specific mechanics are not inferred here.

    :param data: DataFrame to validate (OHLCV format)
    :param strict: If True, raise errors; if False, warn only
    :param require_volume: If True, Volume column is required; if False, it's optional
    """
    if not isinstance(data, pd.DataFrame):
        path = repr(str(data)) if isinstance(data, (str, Path)) else "'prices.csv'"
        raise DataValidationError(
            f"Expected pandas DataFrame, got {type(data).__name__}. "
            "Load the CSV yourself (replace 'Date' with your date column).\n"
            "→ Fix:\n"
            f"  df = pd.read_csv({path}, index_col='Date', parse_dates=True).sort_index().ffill().dropna()\n"
            "Review gaps before using df.ffill(); use df.dropna() to omit missing bars instead. "
            "Open/High/Low/Close are required; OHLCV names are case-insensitive."
        )

    if data.empty:
        raise DataValidationError(
            "DataFrame is empty (0 rows).\n"
            "\n"
            "→ Fix: Ensure you loaded data correctly:\n"
            "  print(df.head())  # Check if data loaded\n"
            "  print(len(df))    # Check number of rows"
        )

    source_columns = data.columns
    flatten_columns = isinstance(source_columns, pd.MultiIndex)
    data = _normalize_ohlcv_columns(data)
    if flatten_columns:
        source_columns = source_columns.droplevel(1)
    case_mapping = {
        old: new
        for old, new in zip(source_columns, data.columns)
        if isinstance(old, str) and old != new
    }

    # Check required columns
    required_columns = ["Open", "High", "Low", "Close"]
    if require_volume:
        required_columns.append("Volume")

    missing_columns = [col for col in required_columns if col not in data.columns]
    date_column = (
        next(
            (
                col
                for col in data.columns
                if isinstance(col, str) and col.casefold() in ("date", "datetime", "timestamp")
            ),
            None,
        )
        if not isinstance(data.index, pd.DatetimeIndex)
        else None
    )
    dates = pd.Index(data[date_column]) if date_column is not None else data.index
    try:
        parsed_dates = pd.to_datetime(dates, errors="coerce")
        unsorted = not parsed_dates.is_monotonic_increasing
    except (TypeError, ValueError):
        # Exotic non-datetime indexes still get the structural DataValidationError below.
        unsorted = False
    adj_close = (
        next(
            (col for col in data.columns if isinstance(col, str) and col.casefold() == "adj close"),
            None,
        )
        if "Close" in missing_columns
        else None
    )
    available = [col for col in required_columns if col in data.columns]
    if adj_close is not None:
        available.append(adj_close)
    missing_counts = data[available].isna().sum()
    nan_columns = missing_counts[missing_counts > 0].index.tolist()

    problems = []
    fix = "df"
    if flatten_columns:
        fix += ".droplevel(1, axis='columns')"
    if case_mapping:
        fix += f".rename(columns={case_mapping!r})"
    if not isinstance(data.index, pd.DatetimeIndex):
        problems.append(f"DataFrame index must be DatetimeIndex, got {type(data.index).__name__}.")
        if date_column is not None:
            problems.append(f"Use date column {date_column!r} as the index.")
            fix += f".assign(**{{{date_column!r}: lambda frame: pd.to_datetime(frame[{date_column!r}])}}).set_index({date_column!r})"
        elif isinstance(data.index, pd.RangeIndex):
            fix = "pd.read_csv('prices.csv', index_col='Date', parse_dates=True)"
            problems.append(
                "Load the date column as the index; replace 'Date' with its actual name."
            )
        else:
            fix += ".set_axis(pd.to_datetime(df.index), axis='index')"
    if missing_columns:
        problems.append(
            f"Missing required columns: {missing_columns}. "
            f"Required columns: {required_columns}. Your columns: {list(data.columns)}."
        )
        if adj_close is not None:
            fix += f".rename(columns={{{adj_close!r}: 'Close'}})"
            problems.append(
                f"Rename {adj_close!r} to 'Close' only if this is the price series you intend. "
                "Open/High/Low remain unadjusted by this rename; use consistently adjusted OHLC "
                "to avoid mixing adjusted Close with unadjusted prices."
            )
        if any(col != "Close" or adj_close is None for col in missing_columns):
            problems.append(
                "Supply the missing OHLCV columns or rename the corresponding source columns before running again."
            )
    if unsorted:
        problems.append("DataFrame dates must be sorted in ascending order.")
        fix += ".sort_index()"
    if nan_columns:
        locations = {
            col: [str(date) for date in dates[data[col].isna().to_numpy()][:3]]
            for col in nan_columns
        }
        problems.append(
            f"Found NaN values in columns: {nan_columns}. "
            f"NaN counts: {missing_counts[missing_counts > 0].to_dict()}. "
            f"First missing dates per column (up to 3): {locations}."
        )
        fixed_nan_columns = ["Close" if col == adj_close else col for col in nan_columns]
        fix += f".ffill().dropna(subset={fixed_nan_columns!r})"
        problems.append(
            "Review gaps before using df.ffill(); it assumes the previous value is valid. "
            "The final dropna removes leading gaps that cannot be forward-filled. "
            "Alternatively, use df.dropna() to omit missing bars."
        )
    if problems:
        message = "\n".join(problems) + f"\n→ Fix:\n  df = {fix}"
        if strict or missing_columns:
            raise DataValidationError(message)
        warnings.warn(message, UserWarning)

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

    # Check for infinite values
    inf_counts = data[cols_to_validate].isin([float("inf"), float("-inf")]).sum()
    if inf_counts.any():
        raise DataValidationError(
            f"DataFrame contains infinite values:\n{inf_counts[inf_counts > 0]}"
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
    broken_rules = pd.DataFrame(
        {
            "High below Low": data["High"] < data["Low"],
            "Open below Low": data["Open"] < data["Low"],
            "Open above High": data["Open"] > data["High"],
            "Close below Low": data["Close"] < data["Low"],
            "Close above High": data["Close"] > data["High"],
        }
    )
    invalid_ohlc = broken_rules.any(axis=1)

    if invalid_ohlc.any():
        details = [
            f"DataFrame has {invalid_ohlc.sum()} rows with invalid OHLC relationships.",
            "Broken rules (counts may overlap); first dates per rule (up to 3):",
        ]
        for rule, mask in broken_rules.items():
            if mask.any():
                first_dates = [str(date) for date in data.index[mask][:3]]
                details.append(f"  {rule}: {mask.sum()} rows; dates: {first_dates}.")
        if broken_rules[["Close below Low", "Close above High"]].any().any():
            details.append(
                "If Close came from 'Adj Close', it may be mixed with unadjusted Open/High/Low. "
                "In that case, supply unadjusted Close, or supply consistently adjusted OHLC "
                "from your data provider. Renaming alone cannot recover the missing raw Close "
                "or adjustment factors."
            )
        details.append(
            "Check these rows against your source data and correct the prices. "
            "If this followed a column-wise fill, reload the original data and consider "
            "omitting incomplete bars with df.dropna(subset=['Open', 'High', 'Low', 'Close']); "
            "filling individual prices can place them outside that day's High/Low range."
        )
        message = "\n".join(details)
        if strict:
            raise DataValidationError(message)
        else:
            warnings.warn(message, UserWarning)

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

    return data


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
        fix = (
            f"Reduce lookback_period to lookback_period={total_rows - 2} if the strategy's "
            "required_history allows, or supply more data."
            if total_rows >= 3
            else "Supply more data; even lookback_period=1 needs at least 3 rows."
        )
        raise DateRangeError(
            f"lookback_period ({lookback_period}) must be less than total data rows ({total_rows}). "
            f"Need at least {lookback_period + 2} rows for this lookback and two trading dates. {fix}"
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
        hint = ""
        if total_rows == lookback_period + 1:
            fix = (
                f"Reduce lookback_period to lookback_period={total_rows - 2} if the strategy's "
                "required_history allows, or supply more data."
                if total_rows >= 3
                else "Supply more data; even lookback_period=1 needs at least 3 rows."
            )
            hint = (
                f" Data has {total_rows} rows; lookback_period={lookback_period} leaves only "
                f"one trading date. Need at least {lookback_period + 2} rows. {fix}"
            )
        raise DateRangeError(
            f"Trading start date ({effective_start}) must be before "
            f"trading end date ({effective_end}).{hint}"
        )

    # Ensure enough data for lookback before trading starts
    start_idx = data.index.searchsorted(effective_start, side="left")
    end_idx = data.index.searchsorted(effective_end, side="right")
    if start_idx >= end_idx:
        raise DateRangeError("Requested date interval contains no trading bars")
    if start_idx < lookback_period:
        raise DateRangeError(
            f"Not enough data before trading start date for lookback_period={lookback_period}. "
            f"Need at least {lookback_period} rows before {effective_start}, "
            f"but only have {start_idx} rows."
        )


def validate_strategies(strategies: List[Strategy], lookback_period: int | None = None) -> None:
    """Validate list of strategies.

    :param strategies: List of strategies to validate
    :param lookback_period: Maximum history rows supplied to each strategy
    """
    if isinstance(strategies, Strategy):
        raise TypeError("Pass a list of strategy instances: backtest.run([strategy]).")
    if isinstance(strategies, type) and issubclass(strategies, Strategy):
        raise TypeError(
            f"Pass a list containing an instance: backtest.run([{strategies.__name__}(...)])."
        )
    if not strategies:
        raise StrategyError("Must provide at least one strategy")

    for i, strategy in enumerate(strategies):
        if isinstance(strategy, type) and issubclass(strategy, Strategy):
            raise StrategyError(
                f"Strategy at index {i} is a class. Create an instance with "
                f"{strategy.__name__}(...) and pass it in a list to run()."
            )
        # Check inherits from base Strategy
        if not isinstance(strategy, Strategy):
            raise StrategyError(
                f"Strategy at index {i} does not inherit from base Strategy class. "
                f"Got type: {type(strategy)}"
            )

        if not all(
            hasattr(strategy, attribute)
            for attribute in ("_name", "_portfolio_state", "_random_seed", "rng")
        ):
            raise AttributeError(
                f"{type(strategy).__name__} is missing Strategy initialization. "
                "Call super().__init__() from your strategy's __init__()."
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
        if name == "benchmark":
            raise StrategyError("Strategy name 'benchmark' is reserved for the internal benchmark")

        required_history = strategy.required_history
        if (
            isinstance(required_history, bool)
            or not isinstance(required_history, int)
            or required_history < 0
        ):
            raise StrategyError(
                f"Strategy '{name}' required_history must be a non-negative integer"
            )
        if lookback_period is not None and required_history > lookback_period:
            raise InsufficientHistoryError(
                f"Strategy '{name}' requires {required_history} history rows, "
                f"but lookback_period is {lookback_period}"
            )

    # Check for duplicate names
    strategy_names = [s.get_name() for s in strategies]
    duplicate_names = [name for name in strategy_names if strategy_names.count(name) > 1]

    if duplicate_names:
        raise StrategyError(
            f"Duplicate strategy names found: {set(duplicate_names)}. "
            f"All strategies must have unique names."
        )
