"""Tests for BacktestConfig."""

from datetime import datetime

import pytest

from simple_backtest.config.settings import BacktestConfig


def test_config_defaults():
    """Test default configuration values."""
    config = BacktestConfig()

    assert config.initial_capital == 1000.0
    assert config.lookback_period == 30
    assert config.commission_type == "percentage"
    assert config.commission_value == 0.001
    assert config.execution_price == "open"
    assert config.parallel_execution is True
    assert config.n_jobs == -1
    assert config.risk_free_rate == 0.0


def test_config_custom_values():
    """Test custom configuration values."""
    config = BacktestConfig(
        initial_capital=10000.0,
        lookback_period=50,
        commission_type="flat",
        commission_value=10.0,
        execution_price="close",
        n_jobs=4,
    )

    assert config.initial_capital == 10000.0
    assert config.lookback_period == 50
    assert config.commission_type == "flat"
    assert config.n_jobs == 4


def test_config_invalid_n_jobs():
    """Test invalid n_jobs value."""
    with pytest.raises(ValueError, match="n_jobs must be"):
        BacktestConfig(n_jobs=0)

    with pytest.raises(ValueError, match="n_jobs must be"):
        BacktestConfig(n_jobs=-2)


def test_config_invalid_date_range():
    """Test invalid date range."""
    with pytest.raises(ValueError, match="must be before"):
        BacktestConfig(
            trading_start_date=datetime(2020, 12, 31),
            trading_end_date=datetime(2020, 1, 1),
        )


def test_config_validate_against_data():
    """Test data validation."""
    config = BacktestConfig(
        lookback_period=50,
        trading_start_date=datetime(2020, 3, 1),
        trading_end_date=datetime(2020, 12, 31),
    )

    # Valid data
    config.validate_against_data(
        data_start=datetime(2020, 1, 1),
        data_end=datetime(2021, 1, 1),
        total_rows=365,
    )

    # Invalid: lookback too large
    with pytest.raises(ValueError, match="lookback_period"):
        config.validate_against_data(
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2020, 12, 31),
            total_rows=40,  # Less than lookback
        )

    # Invalid: trading end after data end
    with pytest.raises(ValueError, match="after data end"):
        config.validate_against_data(
            data_start=datetime(2020, 1, 1),
            data_end=datetime(2020, 6, 1),
            total_rows=365,
        )


def test_config_validation_constraints():
    """Test Pydantic validation constraints."""
    # Negative initial capital
    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=-100.0)

    # Zero lookback period
    with pytest.raises(ValueError):
        BacktestConfig(lookback_period=0)

    # Negative commission
    with pytest.raises(ValueError):
        BacktestConfig(commission_value=-0.01)

    # Invalid commission type
    with pytest.raises(ValueError):
        BacktestConfig(commission_type="invalid")

    # Invalid execution price
    with pytest.raises(ValueError):
        BacktestConfig(execution_price="invalid")

    with pytest.raises(ValueError):
        BacktestConfig(commission_value=float("nan"))

    with pytest.raises(ValueError):
        BacktestConfig(commission_value=float("inf"))

    with pytest.raises(ValueError):
        BacktestConfig(initial_capital=float("inf"))

    with pytest.raises(ValueError):
        BacktestConfig(risk_free_rate=float("nan"))


def test_config_rejects_unknown_or_removed_options():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        BacktestConfig(enable_caching=True)


def test_config_rejects_mixed_timezone_boundaries():
    config = BacktestConfig(trading_start_date=datetime(2020, 2, 1))

    with pytest.raises(ValueError, match="timezone awareness"):
        config.validate_against_data(
            data_start=datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
            data_end=datetime.fromisoformat("2021-01-01T00:00:00+00:00"),
            total_rows=365,
        )
