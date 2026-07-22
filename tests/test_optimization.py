"""Tests for optimization modules."""

import pandas as pd
import pytest

from simple_backtest import BacktestConfig
from simple_backtest.optimization import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    WalkForwardOptimizer,
)
from simple_backtest.strategy.base import Strategy
from simple_backtest.strategy.moving_average import MovingAverageStrategy
from simple_backtest.utils.validation import StrategyExecutionError


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    data = pd.DataFrame(
        {
            "Open": [100 + (i % 20) for i in range(200)],
            "High": [105 + (i % 20) for i in range(200)],
            "Low": [95 + (i % 20) for i in range(200)],
            "Close": [100 + (i % 20) for i in range(200)],
            "Volume": [1000000] * 200,
        },
        index=dates,
    )
    return data


@pytest.fixture
def backtest_config():
    """Create basic backtest configuration."""
    return BacktestConfig.default(initial_capital=10000, lookback_period=30)


@pytest.fixture
def param_space():
    """Create parameter space for testing."""
    return {
        "short_window": [5, 10],
        "long_window": [20, 30],
        "shares": [10],
    }


class TestGridSearchOptimizer:
    """Tests for GridSearchOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = GridSearchOptimizer(verbose=False)
        assert optimizer.get_name() == "GridSearch"
        assert optimizer.verbose is False

    def test_custom_name(self):
        """Test custom optimizer name."""
        optimizer = GridSearchOptimizer(name="MyGridSearch")
        assert optimizer.get_name() == "MyGridSearch"

    def test_optimize_returns_dataframe(self, sample_data, backtest_config, param_space):
        """Test that optimize returns a DataFrame."""
        optimizer = GridSearchOptimizer(verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert isinstance(results, pd.DataFrame)
        assert not results.empty

    def test_optimize_tests_all_combinations(self, sample_data, backtest_config, param_space):
        """Test that optimizer tests all parameter combinations."""
        optimizer = GridSearchOptimizer(verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        # Should test 2 * 2 * 1 = 4 combinations
        assert len(results) == 4

    def test_optimize_includes_parameters(self, sample_data, backtest_config, param_space):
        """Test that results include parameter columns."""
        optimizer = GridSearchOptimizer(verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert "short_window" in results.columns
        assert "long_window" in results.columns
        assert "shares" in results.columns

    def test_optimize_includes_metrics(self, sample_data, backtest_config, param_space):
        """Test that results include metric columns."""
        optimizer = GridSearchOptimizer(verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        # Check for common metrics
        assert "sharpe_ratio" in results.columns
        assert "total_return" in results.columns
        assert "max_drawdown" in results.columns

    def test_optimize_sorted_by_metric(self, sample_data, backtest_config, param_space):
        """Test that results are sorted by target metric."""
        optimizer = GridSearchOptimizer(verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        # Check that results are sorted in descending order
        sharpe_values = results["sharpe_ratio"].tolist()
        assert sharpe_values == sorted(sharpe_values, reverse=True)

    def test_optimize_minimizes_risk_metrics(self, sample_data, backtest_config, param_space):
        optimizer = GridSearchOptimizer(verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="max_drawdown",
        )

        assert results["max_drawdown"].tolist() == sorted(results["max_drawdown"])

    def test_optimize_empty_param_space(self, sample_data, backtest_config):
        """Test behavior with empty parameter space."""
        optimizer = GridSearchOptimizer(verbose=False)

        # Empty param space results in 1 result (single combination with no parameters)
        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space={},
            metric="sharpe_ratio",
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 1  # Single combination (no parameters to vary)

    def test_optimize_invalid_metric(self, sample_data, backtest_config, param_space):
        """Test behavior with invalid metric name."""
        optimizer = GridSearchOptimizer(verbose=False)

        with pytest.raises(ValueError, match="Metric 'nonexistent_metric'"):
            optimizer.optimize(
                data=sample_data,
                config=backtest_config,
                strategy_class=MovingAverageStrategy,
                param_space=param_space,
                metric="nonexistent_metric",
            )

    def test_optimizer_does_not_swallow_strategy_runtime_errors(self, sample_data, backtest_config):
        class ExplodingStrategy(Strategy):
            def predict(self, data, trade_history):
                raise RuntimeError("broken strategy")

        optimizer = GridSearchOptimizer(verbose=False)

        with pytest.raises(StrategyExecutionError, match="broken strategy"):
            optimizer.optimize(
                data=sample_data,
                config=backtest_config,
                strategy_class=ExplodingStrategy,
                param_space={},
            )

    def test_optimizer_records_invalid_parameter_combinations(self, sample_data, backtest_config):
        optimizer = GridSearchOptimizer(verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space={"short_window": [5, 30], "long_window": [20], "shares": [1]},
        )

        assert len(results) == 1
        assert len(optimizer.failures) == 1
        assert optimizer.failures[0]["parameters"]["short_window"] == 30


class TestRandomSearchOptimizer:
    """Tests for RandomSearchOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = RandomSearchOptimizer(n_iter=10, random_state=42, verbose=False)
        assert optimizer.get_name() == "RandomSearch"
        assert optimizer.n_iter == 10
        assert optimizer.random_state == 42
        assert optimizer.verbose is False

    def test_custom_name(self):
        """Test custom optimizer name."""
        optimizer = RandomSearchOptimizer(n_iter=10, name="MyRandomSearch")
        assert optimizer.get_name() == "MyRandomSearch"

    def test_optimize_returns_dataframe(self, sample_data, backtest_config, param_space):
        """Test that optimize returns a DataFrame."""
        optimizer = RandomSearchOptimizer(n_iter=5, random_state=42, verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert isinstance(results, pd.DataFrame)
        assert not results.empty

    def test_optimize_respects_n_iter(self, sample_data, backtest_config, param_space):
        """Test that optimizer tests n_iter combinations."""
        optimizer = RandomSearchOptimizer(n_iter=3, random_state=42, verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        # Should test exactly 3 combinations
        assert len(results) == 3

    def test_optimize_reproducibility(self, sample_data, backtest_config, param_space):
        """Test that random_state ensures reproducibility."""
        # Note: Due to random.seed() being global, we just test that results are consistent
        optimizer1 = RandomSearchOptimizer(n_iter=5, random_state=42, verbose=False)

        results1 = optimizer1.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        # Reset random state and run again
        optimizer2 = RandomSearchOptimizer(n_iter=5, random_state=42, verbose=False)

        results2 = optimizer2.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        # Results should have same number of rows
        assert len(results1) == len(results2)
        assert len(results1) == 5

    def test_optimize_includes_metrics(self, sample_data, backtest_config, param_space):
        """Test that results include metric columns."""
        optimizer = RandomSearchOptimizer(n_iter=5, random_state=42, verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert "sharpe_ratio" in results.columns
        assert "total_return" in results.columns

    def test_optimize_sorted_by_metric(self, sample_data, backtest_config, param_space):
        """Test that results are sorted by target metric."""
        optimizer = RandomSearchOptimizer(n_iter=5, random_state=42, verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        # Check that results are sorted in descending order
        sharpe_values = results["sharpe_ratio"].tolist()
        assert sharpe_values == sorted(sharpe_values, reverse=True)


class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer."""

    def test_initialization(self):
        """Test optimizer initialization."""
        optimizer = WalkForwardOptimizer(train_size=0.7, verbose=False)
        assert optimizer.get_name() == "WalkForward"
        assert optimizer.train_size == 0.7
        assert optimizer.n_splits == 3
        assert optimizer.verbose is False

    def test_invalid_train_size(self):
        """Test that invalid train_size raises error."""
        with pytest.raises(ValueError, match="train_size must be between 0 and 1"):
            WalkForwardOptimizer(train_size=1.5)

        with pytest.raises(ValueError, match="train_size must be between 0 and 1"):
            WalkForwardOptimizer(train_size=0.0)

    def test_invalid_n_splits(self):
        with pytest.raises(ValueError, match="n_splits must be positive"):
            WalkForwardOptimizer(n_splits=0)

    def test_custom_name(self):
        """Test custom optimizer name."""
        optimizer = WalkForwardOptimizer(train_size=0.7, name="MyWalkForward")
        assert optimizer.get_name() == "MyWalkForward"

    def test_optimize_returns_dataframe(self, sample_data, backtest_config, param_space):
        """Test that optimize returns a DataFrame."""
        optimizer = WalkForwardOptimizer(train_size=0.7, verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert isinstance(results, pd.DataFrame)
        assert not results.empty

    def test_optimize_includes_train_metrics(self, sample_data, backtest_config, param_space):
        """Test that results include train metrics."""
        optimizer = WalkForwardOptimizer(train_size=0.7, verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert "train_sharpe_ratio" in results.columns
        assert "train_total_return" in results.columns

    def test_optimize_includes_test_metrics(self, sample_data, backtest_config, param_space):
        """Test that results include test metrics."""
        optimizer = WalkForwardOptimizer(train_size=0.7, verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert "test_sharpe_ratio" in results.columns
        assert "test_total_return" in results.columns

    def test_optimize_includes_difference(self, sample_data, backtest_config, param_space):
        """Test that results include train/test difference."""
        optimizer = WalkForwardOptimizer(train_size=0.7, verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert "sharpe_ratio_diff" in results.columns

    def test_optimize_returns_chronological_out_of_sample_folds(
        self, sample_data, backtest_config, param_space
    ):
        optimizer = WalkForwardOptimizer(train_size=0.7, verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert results["fold"].tolist() == [1, 2, 3]
        assert results["test_start"].is_monotonic_increasing
        assert results["test_end"].is_monotonic_increasing
        assert (results["train_end"] < results["test_start"]).all()
        assert results.loc[0, "train_start"] == sample_data.index[0]
        assert results.loc[2, "test_end"] == sample_data.index[-1]

    def test_optimize_with_custom_base_optimizer(self, sample_data, backtest_config, param_space):
        """Test using custom base optimizer."""
        base_optimizer = RandomSearchOptimizer(n_iter=3, random_state=42, verbose=False)
        optimizer = WalkForwardOptimizer(
            train_size=0.7, base_optimizer=base_optimizer, verbose=False
        )

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert isinstance(results, pd.DataFrame)
        assert len(results) == 3

    def test_train_test_split(self, sample_data, backtest_config, param_space):
        """Test that data is properly split into train/test."""
        optimizer = WalkForwardOptimizer(train_size=0.6, verbose=False)

        results = optimizer.optimize(
            data=sample_data,
            config=backtest_config,
            strategy_class=MovingAverageStrategy,
            param_space=param_space,
            metric="sharpe_ratio",
        )

        assert "train_sharpe_ratio" in results.columns
        assert "test_sharpe_ratio" in results.columns
        assert results["train_end"].tolist() == [
            sample_data.index[119],
            sample_data.index[145],
            sample_data.index[172],
        ]
        assert results["test_start"].tolist() == [
            sample_data.index[120],
            sample_data.index[146],
            sample_data.index[173],
        ]
