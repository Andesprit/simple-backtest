"""Tests for execution price utilities."""

import pandas as pd
import pytest

from simple_backtest.utils.execution import (
    create_execution_price_extractor,
    get_execution_price,
    get_typical_price,
    get_vwap,
    validate_ohlcv_row,
)


@pytest.fixture
def sample_row():
    """Create sample OHLCV row."""
    return pd.Series(
        {
            "Open": 100.0,
            "High": 110.0,
            "Low": 90.0,
            "Close": 105.0,
            "Volume": 1000000,
        }
    )


class TestGetExecutionPrice:
    """Tests for get_execution_price function."""

    def test_open_price(self, sample_row):
        """Test extracting open price."""
        price = get_execution_price(sample_row, method="open")
        assert price == 100.0

    def test_close_price(self, sample_row):
        """Test extracting close price."""
        price = get_execution_price(sample_row, method="close")
        assert price == 105.0

    def test_vwap_price(self, sample_row):
        """The legacy VWAP name warns because OHLCV cannot provide true VWAP."""
        with pytest.warns(DeprecationWarning, match="typical"):
            price = get_execution_price(sample_row, method="vwap")

        # VWAP = (High + Low + Close) / 3
        expected = (110.0 + 90.0 + 105.0) / 3
        assert price == pytest.approx(expected)

    def test_typical_price(self, sample_row):
        """Typical price is named honestly and does not depend on volume."""
        price = get_execution_price(sample_row, method="typical")

        assert price == pytest.approx((110.0 + 90.0 + 105.0) / 3)
        assert price == get_typical_price(sample_row)

    def test_invalid_method(self, sample_row):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid execution price method"):
            get_execution_price(sample_row, method="invalid")

    def test_custom_method(self, sample_row):
        """Test custom price extraction function."""

        def custom_extractor(row):
            """Use average of open and close."""
            return (row["Open"] + row["Close"]) / 2

        price = get_execution_price(sample_row, method="custom", custom_func=custom_extractor)

        expected = (100.0 + 105.0) / 2
        assert price == pytest.approx(expected)

    def test_custom_without_function(self, sample_row):
        """Test that custom method without function raises error."""
        with pytest.raises(ValueError, match="custom_func must be provided"):
            get_execution_price(sample_row, method="custom")


class TestCreateExecutionPriceExtractor:
    """Tests for create_execution_price_extractor function."""

    def test_create_open_extractor(self, sample_row):
        """Test creating open price extractor."""
        extractor = create_execution_price_extractor(method="open")

        price = extractor(sample_row)
        assert price == 100.0

    def test_create_close_extractor(self, sample_row):
        """Test creating close price extractor."""
        extractor = create_execution_price_extractor(method="close")

        price = extractor(sample_row)
        assert price == 105.0

    def test_create_vwap_extractor(self, sample_row):
        """The legacy extractor remains available during its deprecation window."""
        with pytest.warns(DeprecationWarning, match="typical"):
            extractor = create_execution_price_extractor(method="vwap")

        price = extractor(sample_row)
        expected = (110.0 + 90.0 + 105.0) / 3
        assert price == pytest.approx(expected)

    def test_create_typical_extractor(self, sample_row):
        extractor = create_execution_price_extractor(method="typical")

        assert extractor(sample_row) == pytest.approx((110.0 + 90.0 + 105.0) / 3)

    def test_create_custom_extractor(self, sample_row):
        """Test creating custom price extractor."""

        def custom_func(row):
            """Use high price."""
            return row["High"]

        extractor = create_execution_price_extractor(method="custom", custom_func=custom_func)

        price = extractor(sample_row)
        assert price == 110.0

    def test_extractor_is_callable(self, sample_row):
        """Test that returned extractor is callable."""
        extractor = create_execution_price_extractor(method="open")

        assert callable(extractor)
        assert extractor(sample_row) == 100.0

    def test_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid execution price method"):
            create_execution_price_extractor(method="invalid")

    def test_extractor_with_dataframe(self):
        """Test extractor works with DataFrame row."""
        dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [110, 111, 112, 113, 114],
                "Low": [90, 91, 92, 93, 94],
                "Close": [105, 106, 107, 108, 109],
                "Volume": [1000000] * 5,
            },
            index=dates,
        )

        extractor = create_execution_price_extractor(method="close")

        # Test on different rows
        assert extractor(df.iloc[0]) == 105.0
        assert extractor(df.iloc[1]) == 106.0
        assert extractor(df.iloc[4]) == 109.0

    def test_custom_without_function_raises_error(self):
        """Test that custom method without function raises error."""
        with pytest.raises(ValueError, match="custom_func must be provided"):
            create_execution_price_extractor(method="custom")


class TestVWAPEdgeCases:
    """Tests for the deprecated pseudo-VWAP helper."""

    def test_vwap_without_volume_column(self):
        """Test VWAP calculation when Volume column is missing (e.g., forex)."""
        row = pd.Series(
            {
                "Open": 100.0,
                "High": 110.0,
                "Low": 90.0,
                "Close": 105.0,
                # No Volume column
            }
        )

        with pytest.warns(DeprecationWarning, match="typical"):
            price = get_vwap(row)
        expected = (110.0 + 90.0 + 105.0) / 3
        assert price == pytest.approx(expected)

    def test_vwap_with_zero_volume(self):
        """Test VWAP calculation when volume is zero."""
        row = pd.Series(
            {
                "Open": 100.0,
                "High": 110.0,
                "Low": 90.0,
                "Close": 105.0,
                "Volume": 0,
            }
        )

        with pytest.warns(DeprecationWarning, match="typical"):
            price = get_vwap(row)
        expected = (110.0 + 90.0 + 105.0) / 3
        assert price == pytest.approx(expected)


class TestValidateOHLCVRow:
    """Tests for validate_ohlcv_row function."""

    def test_valid_row(self):
        """Test validation passes for valid row."""
        row = pd.Series(
            {
                "Open": 100.0,
                "High": 110.0,
                "Low": 90.0,
                "Close": 105.0,
                "Volume": 1000000,
            }
        )

        # Should not raise
        validate_ohlcv_row(row)

    def test_missing_required_columns(self):
        """Test that missing columns raise error."""
        row = pd.Series({"Open": 100.0, "High": 110.0})

        with pytest.raises(KeyError, match="Missing required OHLC columns"):
            validate_ohlcv_row(row)

    def test_missing_volume_when_required(self):
        """Test that missing volume raises error when required."""
        row = pd.Series(
            {
                "Open": 100.0,
                "High": 110.0,
                "Low": 90.0,
                "Close": 105.0,
            }
        )

        with pytest.raises(KeyError, match="Missing required OHLC columns"):
            validate_ohlcv_row(row, require_volume=True)

    def test_negative_prices_raise_error(self):
        """Test that negative prices raise error."""
        row = pd.Series(
            {
                "Open": -100.0,
                "High": 110.0,
                "Low": 90.0,
                "Close": 105.0,
            }
        )

        with pytest.raises(ValueError, match="must be non-negative"):
            validate_ohlcv_row(row)

    def test_negative_volume_raises_error(self):
        """Test that negative volume raises error."""
        row = pd.Series(
            {
                "Open": 100.0,
                "High": 110.0,
                "Low": 90.0,
                "Close": 105.0,
                "Volume": -1000,
            }
        )

        with pytest.raises(ValueError, match="Volume must be non-negative"):
            validate_ohlcv_row(row)

    def test_high_less_than_low_raises_error(self):
        """Test that High < Low raises error."""
        row = pd.Series(
            {
                "Open": 100.0,
                "High": 80.0,  # High < Low!
                "Low": 90.0,
                "Close": 85.0,
            }
        )

        with pytest.raises(ValueError, match="High .* must be >= Low"):
            validate_ohlcv_row(row)

    def test_open_outside_range_raises_error(self):
        """Test that Open outside Low-High range raises error."""
        row = pd.Series(
            {
                "Open": 115.0,  # Above High!
                "High": 110.0,
                "Low": 90.0,
                "Close": 105.0,
            }
        )

        with pytest.raises(ValueError, match="Open .* must be between Low .* and High"):
            validate_ohlcv_row(row)

    def test_close_outside_range_raises_error(self):
        """Test that Close outside Low-High range raises error."""
        row = pd.Series(
            {
                "Open": 100.0,
                "High": 110.0,
                "Low": 90.0,
                "Close": 120.0,  # Above High!
            }
        )

        with pytest.raises(ValueError, match="Close .* must be between Low .* and High"):
            validate_ohlcv_row(row)
