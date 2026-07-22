"""Tests for results module."""

import pandas as pd
import pytest

from simple_backtest.core.results import BacktestResults, StrategyResult


@pytest.fixture
def sample_strategy_data():
    """Create sample strategy result data."""
    dates = pd.date_range(start="2020-01-01", periods=10, freq="D")
    return {
        "metrics": {
            "total_return": 25.5,
            "cagr": 10.2,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 1.8,
            "max_drawdown": 12.3,
            "volatility": 15.0,
            "total_trades": 10,
            "win_rate": 60.0,
            "profit_factor": 2.5,
        },
        "portfolio_values": pd.Series(
            [10000, 10100, 10200, 10150, 10300, 10400, 10350, 10500, 10600, 10700], index=dates
        ),
        "trade_history": [
            {"timestamp": dates[0], "signal": "buy", "shares": 10, "price": 100, "commission": 1.0},
            {
                "timestamp": dates[5],
                "signal": "sell",
                "shares": 10,
                "price": 105,
                "commission": 1.0,
            },
        ],
        "returns": pd.Series(
            [0.0, 0.01, 0.0099, -0.0049, 0.0148, 0.0097, -0.0048, 0.0145, 0.0095, 0.0094],
            index=dates,
        ),
    }


@pytest.fixture
def sample_results_dict(sample_strategy_data):
    """Create sample results dictionary with multiple strategies."""
    return {
        "Strategy1": sample_strategy_data.copy(),
        "Strategy2": {
            "metrics": {
                "total_return": 30.0,
                "cagr": 12.0,
                "sharpe_ratio": 1.8,
                "sortino_ratio": 2.0,
                "max_drawdown": 10.0,
                "volatility": 12.0,
                "total_trades": 8,
                "win_rate": 70.0,
                "profit_factor": 3.0,
            },
            "portfolio_values": sample_strategy_data["portfolio_values"] * 1.1,
            "trade_history": sample_strategy_data["trade_history"],
            "returns": sample_strategy_data["returns"] * 1.1,
        },
        "benchmark": sample_strategy_data.copy(),
    }


class TestStrategyResult:
    """Tests for StrategyResult class."""

    def test_initialization(self, sample_strategy_data):
        """Test StrategyResult initialization."""
        result = StrategyResult("TestStrategy", sample_strategy_data)

        assert result.name == "TestStrategy"
        assert result.metrics["sharpe_ratio"] == 1.5
        assert isinstance(result.portfolio_values, pd.Series)
        assert len(result.trade_history) == 2
        assert isinstance(result.returns, pd.Series)

    def test_get_metric(self, sample_strategy_data):
        """Test getting specific metric."""
        result = StrategyResult("TestStrategy", sample_strategy_data)

        assert result.get_metric("sharpe_ratio") == 1.5
        assert result.get_metric("total_return") == 25.5

    def test_get_metric_not_found(self, sample_strategy_data):
        """Test getting non-existent metric raises error."""
        result = StrategyResult("TestStrategy", sample_strategy_data)

        with pytest.raises(KeyError, match="Metric 'nonexistent' not found"):
            result.get_metric("nonexistent")

    def test_dict_style_access(self, sample_strategy_data):
        """Test dict-style access to attributes."""
        result = StrategyResult("TestStrategy", sample_strategy_data)

        assert result["metrics"]["sharpe_ratio"] == 1.5
        assert isinstance(result["portfolio_values"], pd.Series)
        assert len(result["trade_history"]) == 2

    def test_to_dict(self, sample_strategy_data):
        """Test converting to dictionary."""
        result = StrategyResult("TestStrategy", sample_strategy_data)

        result_dict = result._to_dict()

        assert "metrics" in result_dict
        assert "portfolio_values" in result_dict
        assert "trade_history" in result_dict
        assert "returns" in result_dict

    def test_repr(self, sample_strategy_data):
        """Test string representation."""
        result = StrategyResult("TestStrategy", sample_strategy_data)

        repr_str = repr(result)

        assert "StrategyResult" in repr_str
        assert "TestStrategy" in repr_str
        assert "1.50" in repr_str  # Sharpe ratio
        assert "25.50" in repr_str  # Total return

    def test_plot_equity_curve_honors_show_drawdown(self, sample_strategy_data):
        result = StrategyResult("TestStrategy", sample_strategy_data)

        with_drawdown = result.plot_equity_curve(show_drawdown=True)
        without_drawdown = result.plot_equity_curve(show_drawdown=False)

        assert len(with_drawdown.data) == 2
        assert len(without_drawdown.data) == 1


class TestBacktestResults:
    """Tests for BacktestResults class."""

    def test_initialization(self, sample_results_dict):
        """Test BacktestResults initialization."""
        results = BacktestResults(sample_results_dict)

        assert len(results) == 2  # Strategy1 and Strategy2 (benchmark separate)
        assert results.benchmark is not None
        assert results.benchmark.name == "benchmark"

    def test_get_strategy(self, sample_results_dict):
        """Test getting specific strategy."""
        results = BacktestResults(sample_results_dict)

        strategy1 = results.get_strategy("Strategy1")

        assert isinstance(strategy1, StrategyResult)
        assert strategy1.name == "Strategy1"
        assert strategy1.metrics["sharpe_ratio"] == 1.5

    def test_get_strategy_not_found(self, sample_results_dict):
        """Test getting non-existent strategy raises error."""
        results = BacktestResults(sample_results_dict)

        with pytest.raises(KeyError, match="Strategy 'NonExistent' not found"):
            results.get_strategy("NonExistent")

    def test_list_strategies(self, sample_results_dict):
        """Test listing strategy names."""
        results = BacktestResults(sample_results_dict)

        strategies = results.list_strategies()

        assert len(strategies) == 2
        assert "Strategy1" in strategies
        assert "Strategy2" in strategies
        assert "benchmark" not in strategies  # Benchmark excluded

    def test_compare(self, sample_results_dict):
        """Test comparing strategies."""
        results = BacktestResults(sample_results_dict)

        comparison = results.compare(
            metrics=["total_return", "sharpe_ratio", "max_drawdown"],
            include_benchmark=True,
        )

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 3  # 2 strategies + benchmark
        assert "total_return" in comparison.columns
        assert "sharpe_ratio" in comparison.columns
        assert "max_drawdown" in comparison.columns

    def test_compare_without_benchmark(self, sample_results_dict):
        """Test comparing strategies without benchmark."""
        results = BacktestResults(sample_results_dict)

        comparison = results.compare(
            metrics=["total_return", "sharpe_ratio"],
            include_benchmark=False,
        )

        assert len(comparison) == 2  # Only strategies
        assert "benchmark" not in comparison.index

    def test_compare_sorts_risk_metric_ascending(self, sample_results_dict):
        results = BacktestResults(sample_results_dict)

        comparison = results.compare(metrics=["max_drawdown"], include_benchmark=False)

        assert comparison.index.tolist() == ["Strategy2", "Strategy1"]

    def test_compare_default_metrics(self, sample_results_dict):
        """Test compare with default metrics."""
        results = BacktestResults(sample_results_dict)

        comparison = results.compare()

        # Should include default metrics
        assert "total_return" in comparison.columns
        assert "sharpe_ratio" in comparison.columns
        assert "sortino_ratio" in comparison.columns
        assert "max_drawdown" in comparison.columns

    def test_best_strategy(self, sample_results_dict):
        """Test finding best strategy."""
        results = BacktestResults(sample_results_dict)

        best = results.best_strategy(metric="sharpe_ratio")

        assert isinstance(best, StrategyResult)
        assert best.name == "Strategy2"  # Strategy2 has sharpe_ratio of 1.8
        assert best.metrics["sharpe_ratio"] == 1.8

    def test_worst_strategy(self, sample_results_dict):
        """Test finding worst strategy."""
        results = BacktestResults(sample_results_dict)

        worst = results.worst_strategy(metric="sharpe_ratio")

        assert isinstance(worst, StrategyResult)
        assert worst.name == "Strategy1"  # Strategy1 has sharpe_ratio of 1.5
        assert worst.metrics["sharpe_ratio"] == 1.5

    def test_best_and_worst_strategy_respect_metric_direction(self, sample_results_dict):
        results = BacktestResults(sample_results_dict)

        assert results.best_strategy(metric="max_drawdown").name == "Strategy2"
        assert results.worst_strategy(metric="max_drawdown").name == "Strategy1"

    def test_best_strategy_rejects_unknown_metric(self, sample_results_dict):
        results = BacktestResults(sample_results_dict)

        with pytest.raises(KeyError, match="Metric 'not_a_metric'"):
            results.best_strategy(metric="not_a_metric")

    def test_best_strategy_no_strategies(self):
        """Test best_strategy with no strategies raises error."""
        results = BacktestResults(
            {
                "benchmark": {
                    "metrics": {},
                    "portfolio_values": pd.Series(),
                    "trade_history": [],
                    "returns": pd.Series(),
                }
            }
        )

        with pytest.raises(ValueError, match="No strategies available"):
            results.best_strategy()

    def test_dict_style_access(self, sample_results_dict):
        """Test dict-style access for backward compatibility."""
        results = BacktestResults(sample_results_dict)

        strategy1 = results["Strategy1"]

        assert isinstance(strategy1, dict)
        assert "metrics" in strategy1
        assert strategy1["metrics"]["sharpe_ratio"] == 1.5

    def test_len(self, sample_results_dict):
        """Test length returns number of strategies."""
        results = BacktestResults(sample_results_dict)

        assert len(results) == 2  # Excludes benchmark

    def test_iter(self, sample_results_dict):
        """Test iterating over strategy names."""
        results = BacktestResults(sample_results_dict)

        names = list(results)

        assert len(names) == 2
        assert "Strategy1" in names
        assert "Strategy2" in names
        assert "benchmark" not in names

    def test_items(self, sample_results_dict):
        """Test items() for backward compatibility."""
        results = BacktestResults(sample_results_dict)

        items = list(results.items())

        assert len(items) == 3  # Includes benchmark
        names = [name for name, _ in items]
        assert "Strategy1" in names
        assert "Strategy2" in names
        assert "benchmark" in names

    def test_keys(self, sample_results_dict):
        """Test keys() for backward compatibility."""
        results = BacktestResults(sample_results_dict)

        keys = list(results.keys())

        assert len(keys) == 3  # Includes benchmark
        assert "Strategy1" in keys
        assert "Strategy2" in keys
        assert "benchmark" in keys

    def test_values(self, sample_results_dict):
        """Test values() for backward compatibility."""
        results = BacktestResults(sample_results_dict)

        values = list(results.values())

        assert len(values) == 3  # Includes benchmark
        assert all(isinstance(v, dict) for v in values)
        assert all("metrics" in v for v in values)

    def test_repr(self, sample_results_dict):
        """Test string representation."""
        results = BacktestResults(sample_results_dict)

        repr_str = repr(results)

        assert "BacktestResults" in repr_str
        assert "2 strategies" in repr_str
