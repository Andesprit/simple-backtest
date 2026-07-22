"""Tests for commission utility functions (legacy)."""

import pytest

from simple_backtest.config import BacktestConfig
from simple_backtest.utils.commission import (
    create_custom_commission,
    flat_commission,
    get_commission_calculator,
    percentage_commission,
    tiered_commission,
)


class TestPercentageCommissionFunction:
    """Tests for percentage_commission function."""

    def test_basic_calculation(self):
        """Test basic percentage commission calculation."""
        commission = percentage_commission(shares=100, price=50, rate=0.001)
        assert commission == 5.0  # 100 * 50 * 0.001 = 5.0

    def test_zero_rate(self):
        """Test with zero rate."""
        commission = percentage_commission(shares=100, price=50, rate=0.0)
        assert commission == 0.0

    def test_fractional_shares(self):
        """Test with fractional shares."""
        commission = percentage_commission(shares=10.5, price=100, rate=0.002)
        assert commission == pytest.approx(2.1)  # 10.5 * 100 * 0.002 = 2.1


class TestFlatCommissionFunction:
    """Tests for flat_commission function."""

    def test_basic_calculation(self):
        """Test flat commission (ignores shares and price)."""
        commission = flat_commission(shares=100, price=50, flat_fee=9.99)
        assert commission == 9.99

    def test_zero_fee(self):
        """Test with zero fee."""
        commission = flat_commission(shares=100, price=50, flat_fee=0.0)
        assert commission == 0.0

    def test_different_trade_sizes(self):
        """Test that commission is same regardless of trade size."""
        fee = 5.0
        assert flat_commission(10, 100, fee) == fee
        assert flat_commission(1000, 100, fee) == fee


class TestTieredCommissionFunction:
    """Tests for tiered_commission function."""

    def test_first_tier(self):
        """Test commission in first tier."""
        tiers = [(1000, 0.002), (5000, 0.001), (float("inf"), 0.0005)]
        commission = tiered_commission(shares=5, price=100, tiers=tiers)
        # $500 in first tier: 500 * 0.002 = 1.0
        assert commission == pytest.approx(1.0)

    def test_second_tier(self):
        """Test commission spanning multiple tiers."""
        tiers = [(1000, 0.002), (5000, 0.001), (float("inf"), 0.0005)]
        commission = tiered_commission(shares=20, price=100, tiers=tiers)
        # $2000: $0-1000 @ 0.002 + $1000-2000 @ 0.001
        # = (1000 * 0.002) + (1000 * 0.001) = 2.0 + 1.0 = 3.0
        assert commission == pytest.approx(3.0)

    def test_third_tier(self):
        """Test commission in highest tier."""
        tiers = [(1000, 0.002), (5000, 0.001), (float("inf"), 0.0005)]
        commission = tiered_commission(shares=60, price=100, tiers=tiers)
        # $6000: $0-1000 @ 0.002 + $1000-5000 @ 0.001 + $5000-6000 @ 0.0005
        # = (1000 * 0.002) + (4000 * 0.001) + (1000 * 0.0005)
        # = 2.0 + 4.0 + 0.5 = 6.5
        assert commission == pytest.approx(6.5)

    def test_empty_tiers(self):
        """An incomplete schedule is rejected instead of undercharging."""
        with pytest.raises(ValueError, match="final"):
            tiered_commission(shares=10, price=100, tiers=[])


class TestGetCommissionCalculator:
    """Tests for get_commission_calculator factory function."""

    def test_percentage_type(self):
        """Test creating percentage commission calculator."""
        config = BacktestConfig(
            initial_capital=10000,
            lookback_period=30,
            commission_type="percentage",
            commission_value=0.001,
        )
        calculator = get_commission_calculator(config)

        # Test calculation
        commission = calculator(100, 50)
        assert commission == 5.0

    def test_flat_type(self):
        """Test creating flat commission calculator."""
        config = BacktestConfig(
            initial_capital=10000,
            lookback_period=30,
            commission_type="flat",
            commission_value=9.99,
        )
        calculator = get_commission_calculator(config)

        # Test calculation
        commission = calculator(100, 50)
        assert commission == 9.99

    def test_tiered_type_with_list(self):
        """Test creating tiered commission calculator with list."""
        tiers = [(1000, 0.002), (5000, 0.001), (float("inf"), 0.0005)]
        config = BacktestConfig(
            initial_capital=10000,
            lookback_period=30,
            commission_type="tiered",
            commission_value=tiers,
        )
        calculator = get_commission_calculator(config)

        # Test calculation in first tier
        commission = calculator(5, 100)  # $500
        assert commission == pytest.approx(1.0)

    def test_tiered_type_with_single_value(self):
        """Test that tiered requires list of tuples."""
        # Pydantic validates commission_value must be list for tiered
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError):
            BacktestConfig(
                initial_capital=10000,
                lookback_period=30,
                commission_type="tiered",
                commission_value=0.001,  # Should be list, not float
            )

    def test_custom_type(self):
        """Custom mode requires the callable accepted by Backtest."""
        config = BacktestConfig(
            initial_capital=10000,
            lookback_period=30,
            commission_type="custom",
            commission_value=0.0,
        )
        with pytest.raises(ValueError, match="commission_calculator must be provided"):
            get_commission_calculator(config)

    def test_invalid_type(self):
        """Test that invalid commission type raises validation error."""
        # Pydantic validates commission_type at config creation
        from pydantic_core import ValidationError

        with pytest.raises(ValidationError):
            BacktestConfig(
                initial_capital=10000,
                lookback_period=30,
                commission_type="invalid_type",  # Not a valid literal
                commission_value=0.001,
            )


class TestCreateCustomCommission:
    """Tests for create_custom_commission wrapper."""

    def test_valid_custom_function(self):
        """Test wrapping valid custom commission function."""

        def my_commission(shares, price):
            return shares * price * 0.001

        wrapped = create_custom_commission(my_commission)
        commission = wrapped(100, 50)
        assert commission == 5.0

    def test_negative_commission_raises_error(self):
        """Test that negative commission raises error."""

        def bad_commission(shares, price):
            return -10.0

        wrapped = create_custom_commission(bad_commission)

        with pytest.raises(ValueError, match="Commission must be non-negative"):
            wrapped(100, 50)

    def test_zero_commission_allowed(self):
        """Test that zero commission is allowed."""

        def zero_commission(shares, price):
            return 0.0

        wrapped = create_custom_commission(zero_commission)
        commission = wrapped(100, 50)
        assert commission == 0.0

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "free"])
    def test_non_finite_or_non_numeric_commission_raises_error(self, value):
        """Custom callbacks cannot inject invalid accounting values."""
        wrapped = create_custom_commission(lambda shares, price: value)

        with pytest.raises(ValueError, match="finite number"):
            wrapped(100, 50)

    def test_complex_custom_logic(self):
        """Test wrapping complex custom commission logic."""

        def tiered_min_commission(shares, price):
            """Commission with minimum fee."""
            base = shares * price * 0.001
            return max(base, 5.0)

        wrapped = create_custom_commission(tiered_min_commission)

        # Small trade should use minimum
        assert wrapped(1, 100) == 5.0

        # Large trade should use percentage
        assert wrapped(100, 100) == 10.0
