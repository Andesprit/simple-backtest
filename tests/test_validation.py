"""Tests for validation utilities."""

from datetime import timedelta

import pandas as pd
import pytest

from simple_backtest.strategy.base import Strategy
from simple_backtest.utils.validation import (
    BacktestError,
    DataValidationError,
    DateRangeError,
    StrategyError,
    validate_dataframe,
    validate_date_range,
    validate_strategies,
)


class DummyStrategy(Strategy):
    """Dummy strategy for testing."""

    def predict(self, data, trade_history):
        return {"signal": "hold", "size": 0, "order_ids": None}


@pytest.fixture
def valid_dataframe():
    """Create valid OHLCV dataframe."""
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0] * 100,
            "High": [110.0] * 100,
            "Low": [90.0] * 100,
            "Close": [105.0] * 100,
            "Volume": [1000000] * 100,
        },
        index=dates,
    )


class TestValidateDataframe:
    """Tests for validate_dataframe function."""

    def test_valid_dataframe(self, valid_dataframe):
        """Test that valid dataframe passes validation."""
        # Should not raise
        validate_dataframe(valid_dataframe)

    def test_empty_dataframe(self):
        """Test that empty dataframe raises error."""
        df = pd.DataFrame()

        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_missing_required_columns(self, valid_dataframe):
        """Test that missing required columns raise error."""
        # Remove Close column
        df = valid_dataframe.drop(columns=["Close"])

        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_non_datetime_index(self, valid_dataframe):
        """Test that non-datetime index raises error."""
        df = valid_dataframe.reset_index(drop=True)

        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_unsorted_index(self, valid_dataframe):
        """Test that unsorted index raises error."""
        df = valid_dataframe.iloc[::-1]  # Reverse order

        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_duplicate_dates(self, valid_dataframe):
        """Test that duplicate dates raise error."""
        # Create dataframe with duplicate dates
        dates = list(valid_dataframe.index[:50]) + list(valid_dataframe.index[:50])
        df = pd.DataFrame(
            {
                "Open": [100.0] * 100,
                "High": [110.0] * 100,
                "Low": [90.0] * 100,
                "Close": [105.0] * 100,
                "Volume": [1000000] * 100,
            },
            index=pd.DatetimeIndex(dates),
        )

        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_missing_values(self, valid_dataframe):
        """Test that missing values raise error."""
        df = valid_dataframe.copy()
        df.loc[df.index[10], "Close"] = None

        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_infinite_values(self, valid_dataframe):
        """Test that infinite values raise error."""
        df = valid_dataframe.copy()
        df.loc[df.index[10], "Close"] = float("inf")

        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_negative_prices(self, valid_dataframe):
        """Test that negative prices raise error."""
        df = valid_dataframe.copy()
        df.loc[df.index[10], "Close"] = -100.0

        with pytest.raises(DataValidationError):
            validate_dataframe(df)

    def test_zero_prices_rejected(self, valid_dataframe):
        df = valid_dataframe.copy()
        df.loc[df.index[10], ["Open", "High", "Low", "Close"]] = 0.0

        with pytest.raises(DataValidationError, match="strictly positive"):
            validate_dataframe(df)

    def test_infinite_optional_volume_rejected(self, valid_dataframe):
        df = valid_dataframe.astype({"Volume": float})
        df.loc[df.index[10], "Volume"] = float("inf")

        with pytest.raises(DataValidationError, match="infinite"):
            validate_dataframe(df)

    def test_missing_volume_allowed(self, valid_dataframe):
        """Test that missing volume column is allowed."""
        df = valid_dataframe.drop(columns=["Volume"])

        # Should not raise (Volume is optional)
        validate_dataframe(df)

    def test_negative_volume_raises_error(self, valid_dataframe):
        """Test that negative volume raises error when volume is required."""
        df = valid_dataframe.copy()
        df.loc[df.index[10], "Volume"] = -1000

        # Volume only checked when require_volume=True
        with pytest.raises(DataValidationError, match="negative values"):
            validate_dataframe(df, require_volume=True)


class TestValidateDateRange:
    """Tests for validate_date_range function."""

    def test_valid_date_range(self, valid_dataframe):
        """Test validation with valid date range."""
        start = valid_dataframe.index[30]
        end = valid_dataframe.index[50]

        # Should not raise (lookback_period=20 means we need 20 rows before start)
        validate_date_range(
            valid_dataframe, trading_start_date=start, trading_end_date=end, lookback_period=20
        )

    def test_no_dates_provided(self, valid_dataframe):
        """Test validation with no dates (should use full range)."""
        # Should not raise (lookback_period=20, auto dates)
        validate_date_range(
            valid_dataframe, trading_start_date=None, trading_end_date=None, lookback_period=20
        )

    def test_start_date_before_data(self, valid_dataframe):
        """Test that start date before data raises error."""
        start = valid_dataframe.index[0] - timedelta(days=10)

        with pytest.raises(DateRangeError):
            validate_date_range(
                valid_dataframe, trading_start_date=start, trading_end_date=None, lookback_period=10
            )

    def test_end_date_after_data(self, valid_dataframe):
        """Test that end date after data raises error."""
        end = valid_dataframe.index[-1] + timedelta(days=10)

        with pytest.raises(DateRangeError):
            validate_date_range(
                valid_dataframe, trading_start_date=None, trading_end_date=end, lookback_period=10
            )

    def test_start_after_end(self, valid_dataframe):
        """Test that start after end raises error."""
        start = valid_dataframe.index[50]
        end = valid_dataframe.index[10]

        with pytest.raises(DateRangeError):
            validate_date_range(
                valid_dataframe, trading_start_date=start, trading_end_date=end, lookback_period=5
            )

    def test_only_start_date(self, valid_dataframe):
        """Test validation with only start date."""
        start = valid_dataframe.index[30]

        # Should not raise (need lookback rows before start)
        validate_date_range(
            valid_dataframe, trading_start_date=start, trading_end_date=None, lookback_period=20
        )

    def test_only_end_date(self, valid_dataframe):
        """Test validation with only end date."""
        end = valid_dataframe.index[50]

        # Should not raise (auto start after lookback)
        validate_date_range(
            valid_dataframe, trading_start_date=None, trading_end_date=end, lookback_period=20
        )


class TestValidateStrategies:
    """Tests for validate_strategies function."""

    def test_valid_single_strategy(self):
        """Test validation with single strategy."""
        strategies = [DummyStrategy()]

        # Should not raise
        validate_strategies(strategies)

    def test_valid_multiple_strategies(self):
        """Test validation with multiple strategies."""
        strategies = [
            DummyStrategy(name="Strategy1"),
            DummyStrategy(name="Strategy2"),
            DummyStrategy(name="Strategy3"),
        ]

        # Should not raise
        validate_strategies(strategies)

    def test_empty_list(self):
        """Test that empty strategy list raises error."""
        with pytest.raises(StrategyError, match="Must provide at least one strategy"):
            validate_strategies([])

    def test_not_list(self):
        """Test that non-list raises error."""
        with pytest.raises(TypeError):  # Will raise TypeError for non-iterable
            validate_strategies(DummyStrategy())

    def test_non_strategy_instance(self):
        """Test that non-Strategy instance raises error."""
        strategies = [DummyStrategy(), "not a strategy"]

        with pytest.raises(StrategyError, match="does not inherit from base Strategy class"):
            validate_strategies(strategies)

    def test_duplicate_strategy_names(self):
        """Test that duplicate names raise error."""
        strategies = [
            DummyStrategy(name="SameName"),
            DummyStrategy(name="SameName"),
        ]

        with pytest.raises(StrategyError, match="Duplicate strategy names"):
            validate_strategies(strategies)

    def test_duplicate_default_names(self):
        """Test that duplicate default names raise error."""
        strategies = [
            DummyStrategy(),  # Default name: DummyStrategy
            DummyStrategy(),  # Default name: DummyStrategy
        ]

        with pytest.raises(StrategyError, match="Duplicate strategy names"):
            validate_strategies(strategies)


class TestExceptionClasses:
    """Tests for custom exception classes."""

    def test_backtest_error_inheritance(self):
        """Test that BacktestError inherits from Exception."""
        error = BacktestError("test message")
        assert isinstance(error, Exception)
        assert str(error) == "test message"

    def test_data_validation_error_inheritance(self):
        """Test that DataValidationError inherits from BacktestError."""
        error = DataValidationError("test message")
        assert isinstance(error, BacktestError)
        assert isinstance(error, Exception)

    def test_date_range_error_inheritance(self):
        """Test that DateRangeError inherits from BacktestError."""
        error = DateRangeError("test message")
        assert isinstance(error, BacktestError)
        assert isinstance(error, Exception)

    def test_strategy_error_inheritance(self):
        """Test that StrategyError inherits from BacktestError."""
        error = StrategyError("test message")
        assert isinstance(error, BacktestError)
        assert isinstance(error, Exception)
