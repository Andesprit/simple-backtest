"""Tests for commission modules."""

import pytest

from simple_backtest.commission import (
    Commission,
    FlatCommission,
    PercentageCommission,
    TieredCommission,
)


class TestPercentageCommission:
    """Tests for PercentageCommission."""

    def test_initialization(self):
        """Test commission initialization."""
        commission = PercentageCommission(rate=0.001)
        assert commission.rate == 0.001
        assert "Percentage" in commission.get_name()

    def test_invalid_rate(self):
        """Test that negative rate raises error."""
        with pytest.raises(ValueError, match="Commission rate must be non-negative"):
            PercentageCommission(rate=-0.001)

    @pytest.mark.parametrize("rate", [float("nan"), float("inf")])
    def test_nonfinite_rate(self, rate):
        with pytest.raises(ValueError, match="finite"):
            PercentageCommission(rate=rate)

    def test_calculate(self):
        """Test commission calculation."""
        commission = PercentageCommission(rate=0.001)  # 0.1%

        # $10,000 trade @ 0.1% = $10
        result = commission.calculate(shares=100, price=100)
        assert result == 10.0

    def test_calculate_zero_rate(self):
        """Test commission with zero rate."""
        commission = PercentageCommission(rate=0.0)

        result = commission.calculate(shares=100, price=100)
        assert result == 0.0

    def test_calculate_fractional_shares(self):
        """Test commission with fractional shares."""
        commission = PercentageCommission(rate=0.001)

        # 10.5 shares @ $50 = $525, commission = $0.525
        result = commission.calculate(shares=10.5, price=50)
        assert result == pytest.approx(0.525)

    def test_callable(self):
        """Test that commission can be called as a function."""
        commission = PercentageCommission(rate=0.001)

        # Should work as callable
        result = commission(100, 100)
        assert result == 10.0


class TestFlatCommission:
    """Tests for FlatCommission."""

    def test_initialization(self):
        """Test commission initialization."""
        commission = FlatCommission(fee=5.0)
        assert commission.fee == 5.0
        assert "Flat" in commission.get_name()

    def test_invalid_fee(self):
        """Test that negative fee raises error."""
        with pytest.raises(ValueError, match="Commission fee must be non-negative"):
            FlatCommission(fee=-5.0)

    @pytest.mark.parametrize("fee", [float("nan"), float("inf")])
    def test_nonfinite_fee(self, fee):
        with pytest.raises(ValueError, match="finite"):
            FlatCommission(fee=fee)

    def test_calculate(self):
        """Test commission calculation."""
        commission = FlatCommission(fee=5.0)

        # Always returns flat fee regardless of trade size
        assert commission.calculate(shares=100, price=100) == 5.0
        assert commission.calculate(shares=10, price=100) == 5.0
        assert commission.calculate(shares=1000, price=100) == 5.0

    def test_calculate_zero_fee(self):
        """Test commission with zero fee."""
        commission = FlatCommission(fee=0.0)

        result = commission.calculate(shares=100, price=100)
        assert result == 0.0

    def test_callable(self):
        """Test that commission can be called as a function."""
        commission = FlatCommission(fee=5.0)

        result = commission(100, 100)
        assert result == 5.0


class TestTieredCommission:
    """Tests for TieredCommission."""

    def test_initialization(self):
        """Test commission initialization."""
        tiers = [(1000, 0.002), (5000, 0.001), (float("inf"), 0.0005)]
        commission = TieredCommission(tiers=tiers)
        assert commission.tiers == tiers
        assert "Tiered" in commission.get_name()

    def test_invalid_tiers_empty(self):
        """Test that empty tiers raises error."""
        with pytest.raises(ValueError, match="Tiers list cannot be empty"):
            TieredCommission(tiers=[])

    def test_invalid_tiers_negative_rate(self):
        """Test that negative rate raises error."""
        with pytest.raises(ValueError, match="rate must be non-negative"):
            TieredCommission(tiers=[(1000, -0.001)])

    def test_invalid_tiers_order(self):
        """Test that out-of-order thresholds raise error."""
        with pytest.raises(ValueError, match="thresholds must be in ascending order"):
            TieredCommission(tiers=[(5000, 0.001), (1000, 0.002)])

    def test_invalid_tiers_require_terminal_rate(self):
        """Schedules must specify how values above the final threshold are charged."""
        with pytest.raises(ValueError, match="must end"):
            TieredCommission(tiers=[(1000, 0.002), (5000, 0.001)])

    @pytest.mark.parametrize(
        "tiers",
        [
            [(float("nan"), 0.1), (float("inf"), 0.01)],
            [(1000, float("nan")), (float("inf"), 0.01)],
            [(1000, 0.1), (float("inf"), float("inf"))],
        ],
    )
    def test_nonfinite_tier_values(self, tiers):
        with pytest.raises(ValueError, match="finite"):
            TieredCommission(tiers)

    def test_calculate_first_tier(self):
        """Test commission in first tier."""
        # Tiers: $0-1000 @ 0.2%, $1000-5000 @ 0.1%, $5000+ @ 0.05%
        tiers = [(1000, 0.002), (5000, 0.001), (float("inf"), 0.0005)]
        commission = TieredCommission(tiers=tiers)

        # $500 trade falls entirely in first tier @ 0.2%
        result = commission.calculate(shares=10, price=50)
        assert result == pytest.approx(1.0)  # $500 * 0.002 = $1.00

    def test_calculate_second_tier(self):
        """Test commission in second tier."""
        tiers = [(1000, 0.002), (5000, 0.001), (float("inf"), 0.0005)]
        commission = TieredCommission(tiers=tiers)

        # $2000 trade: $0-1000 @ 0.2% + $1000-2000 @ 0.1%
        result = commission.calculate(shares=20, price=100)
        expected = (1000 * 0.002) + (1000 * 0.001)  # $2 + $1 = $3
        assert result == pytest.approx(expected)

    def test_calculate_third_tier(self):
        """Test commission in third tier."""
        tiers = [(1000, 0.002), (5000, 0.001), (float("inf"), 0.0005)]
        commission = TieredCommission(tiers=tiers)

        # $6000 trade: $0-1000 @ 0.2% + $1000-5000 @ 0.1% + $5000-6000 @ 0.05%
        result = commission.calculate(shares=60, price=100)
        expected = (1000 * 0.002) + (4000 * 0.001) + (1000 * 0.0005)  # $2 + $4 + $0.50 = $6.50
        assert result == pytest.approx(expected)

    def test_calculate_at_boundary(self):
        """Test commission exactly at tier boundary."""
        tiers = [(1000, 0.002), (5000, 0.001), (float("inf"), 0.0005)]
        commission = TieredCommission(tiers=tiers)

        # $1000 trade should fall entirely in first tier
        result = commission.calculate(shares=10, price=100)
        assert result == pytest.approx(2.0)  # $1000 * 0.002 = $2

    def test_callable(self):
        """Test that commission can be called as a function."""
        tiers = [(1000, 0.002), (float("inf"), 0.001)]
        commission = TieredCommission(tiers=tiers)

        # $500 trade in first tier
        result = commission(10, 50)
        assert result == pytest.approx(1.0)  # $500 * 0.002 = $1


class TestCommissionBase:
    """Tests for Commission base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that Commission cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Commission()

    def test_custom_commission_implementation(self):
        """Test creating custom commission class."""

        class CustomCommission(Commission):
            """Custom commission with minimum fee."""

            def __init__(self, rate=0.001, min_fee=1.0):
                super().__init__(name="CustomMin")
                self.rate = rate
                self.min_fee = min_fee

            def calculate(self, shares, price):
                """Calculate commission with minimum."""
                commission = shares * price * self.rate
                return max(commission, self.min_fee)

        commission = CustomCommission(rate=0.001, min_fee=5.0)
        assert commission.get_name() == "CustomMin"

        # Small trade should use minimum fee
        assert commission.calculate(shares=1, price=100) == 5.0

        # Large trade should use percentage
        assert commission.calculate(shares=100, price=100) == 10.0

    def test_commission_validation(self):
        """Test that commission validates negative results."""

        class BadCommission(Commission):
            """Commission that returns negative value."""

            def calculate(self, shares, price):
                return -10.0  # Invalid!

        commission = BadCommission()

        with pytest.raises(ValueError, match="Commission must be non-negative"):
            commission(100, 100)
