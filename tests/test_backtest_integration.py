"""Integration tests for the complete backtest workflow."""

import pandas as pd
import pytest

from simple_backtest import (
    Backtest,
    BacktestConfig,
    BuyAndHoldStrategy,
    DCAStrategy,
    MovingAverageStrategy,
    Strategy,
)


@pytest.fixture
def sample_data():
    """Create sample market data for testing."""
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    # Create trending data
    base_price = 100
    prices = [base_price + i * 0.5 + (i % 20) * 2 for i in range(200)]

    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.05 for p in prices],
            "Low": [p * 0.95 for p in prices],
            "Close": prices,
            "Volume": [1000000] * 200,
        },
        index=dates,
    )


class TestFullBacktestWorkflow:
    """Tests for complete backtest workflows."""

    def test_single_strategy_backtest(self, sample_data):
        """Test running backtest with single strategy."""
        config = BacktestConfig.default(initial_capital=10000, lookback_period=30)
        strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=10)

        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run([strategy])

        # Verify results structure
        assert results is not None
        assert len(results) == 1
        assert strategy.get_name() in results.list_strategies()

        # Verify metrics are calculated
        strategy_result = results.get_strategy(strategy.get_name())
        assert "total_return" in strategy_result.metrics
        assert "sharpe_ratio" in strategy_result.metrics
        assert "total_trades" in strategy_result.metrics

        # Verify benchmark is included
        assert results.benchmark is not None

    def test_multiple_strategies_backtest(self, sample_data):
        """Test running backtest with multiple strategies."""
        config = BacktestConfig.default(initial_capital=10000, lookback_period=30)

        strategies = [
            MovingAverageStrategy(short_window=10, long_window=30, shares=10, name="MA_Fast"),
            MovingAverageStrategy(short_window=20, long_window=50, shares=10, name="MA_Slow"),
            BuyAndHoldStrategy(shares=50, name="BuyHold"),
            DCAStrategy(investment_amount=500, interval_days=30, name="DCA"),
        ]

        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run(strategies)

        # Verify all strategies ran
        assert len(results) == 4
        for strategy in strategies:
            assert strategy.get_name() in results.list_strategies()

    def test_backtest_with_custom_date_range(self, sample_data):
        """Test backtest with custom date range."""
        start_date = sample_data.index[50]
        end_date = sample_data.index[150]

        config = BacktestConfig(
            initial_capital=10000,
            lookback_period=30,
            trading_start_date=start_date,
            trading_end_date=end_date,
        )

        strategy = BuyAndHoldStrategy(shares=50)
        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run([strategy])

        # Verify backtest used correct date range
        strategy_result = results.get_strategy(strategy.get_name())
        assert len(strategy_result.portfolio_values) <= (150 - 50 + 1)

    def test_backtest_with_percentage_commission(self, sample_data):
        """Test backtest with percentage commission."""
        config = BacktestConfig(
            initial_capital=10000,
            lookback_period=30,
            commission_type="percentage",
            commission_value=0.001,  # 0.1%
        )

        strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=10)
        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run([strategy])

        # Verify trades have commission deducted
        strategy_result = results.get_strategy(strategy.get_name())
        if strategy_result.trade_history:
            for trade in strategy_result.trade_history:
                assert "commission" in trade
                assert trade["commission"] > 0

    def test_backtest_with_flat_commission(self, sample_data):
        """Test backtest with flat commission."""
        config = BacktestConfig(
            initial_capital=10000,
            lookback_period=30,
            commission_type="flat",
            commission_value=5.0,  # $5 per trade
        )

        strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=10)
        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run([strategy])

        # Verify flat commission is applied
        strategy_result = results.get_strategy(strategy.get_name())
        if strategy_result.trade_history:
            for trade in strategy_result.trade_history:
                assert trade["commission"] == 5.0

    def test_backtest_with_parallel_execution(self, sample_data):
        """Test backtest with parallel execution enabled."""
        config = BacktestConfig(
            initial_capital=10000,
            lookback_period=30,
            parallel_execution=True,
            n_jobs=2,
        )

        strategies = [
            MovingAverageStrategy(short_window=10, long_window=30, shares=10, name=f"MA_{i}")
            for i in range(4)
        ]

        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run(strategies)

        # Verify all strategies completed
        assert len(results) == 4

    def test_backtest_with_sequential_execution(self, sample_data):
        """Test backtest with sequential execution."""
        config = BacktestConfig(
            initial_capital=10000,
            lookback_period=30,
            parallel_execution=False,
        )

        strategies = [
            MovingAverageStrategy(short_window=10, long_window=30, shares=10, name=f"MA_{i}")
            for i in range(3)
        ]

        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run(strategies)

        # Verify all strategies completed
        assert len(results) == 3

    def test_backtest_results_comparison(self, sample_data):
        """Test results comparison functionality."""
        config = BacktestConfig.default(initial_capital=10000, lookback_period=30)

        strategies = [
            MovingAverageStrategy(short_window=10, long_window=30, shares=10, name="Fast"),
            MovingAverageStrategy(short_window=20, long_window=50, shares=10, name="Slow"),
        ]

        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run(strategies)

        # Test comparison
        comparison = results.compare(metrics=["total_return", "sharpe_ratio"])

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) >= 2  # At least 2 strategies
        assert "total_return" in comparison.columns
        assert "sharpe_ratio" in comparison.columns

    def test_backtest_best_strategy(self, sample_data):
        """Test finding best strategy."""
        config = BacktestConfig.default(initial_capital=10000, lookback_period=30)

        strategies = [
            MovingAverageStrategy(short_window=10, long_window=30, shares=10, name="Fast"),
            BuyAndHoldStrategy(shares=50, name="BuyHold"),
        ]

        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run(strategies)

        # Test best strategy
        best = results.best_strategy(metric="total_return")

        assert best is not None
        assert best.name in ["Fast", "BuyHold"]
        assert "total_return" in best.metrics

    def test_backtest_with_insufficient_lookback(self, sample_data):
        """Test backtest with lookback period larger than data."""
        # Use only 50 data points
        small_data = sample_data.head(50)

        config = BacktestConfig(
            initial_capital=10000,
            lookback_period=100,  # Larger than data
        )

        # Should raise validation error
        with pytest.raises(Exception):  # DataValidationError or similar
            Backtest(data=small_data, config=config)

    def test_backtest_with_zero_commission(self, sample_data):
        """Test backtest with zero commission."""
        config = BacktestConfig.zero_commission(initial_capital=10000, lookback_period=30)

        strategy = MovingAverageStrategy(short_window=10, long_window=30, shares=10)
        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run([strategy])

        # Verify no commission charged
        strategy_result = results.get_strategy(strategy.get_name())
        if strategy_result.trade_history:
            for trade in strategy_result.trade_history:
                assert trade["commission"] == 0.0

    def test_backtest_portfolio_values_continuity(self, sample_data):
        """Test that portfolio values are continuous."""
        config = BacktestConfig.default(initial_capital=10000, lookback_period=30)
        strategy = BuyAndHoldStrategy(shares=50)

        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run([strategy])

        strategy_result = results.get_strategy(strategy.get_name())
        portfolio_values = strategy_result.portfolio_values

        # Verify no missing dates in portfolio values
        assert portfolio_values.index.is_monotonic_increasing
        assert len(portfolio_values) > 0

        # Verify values are reasonable
        assert all(portfolio_values >= 0)
        assert portfolio_values.iloc[0] >= config.initial_capital * 0.5  # Not too low

    def test_backtest_returns_calculation(self, sample_data):
        """Test that returns are calculated correctly."""
        config = BacktestConfig.default(initial_capital=10000, lookback_period=30)
        strategy = BuyAndHoldStrategy(shares=50)

        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run([strategy])

        strategy_result = results.get_strategy(strategy.get_name())
        returns = strategy_result.returns

        # Verify returns are calculated
        assert len(returns) > 0

        # Verify returns are reasonable (between -100% and +100% per day)
        assert all(returns >= -1.0)
        assert all(returns <= 1.0)


class TestBacktestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_backtest_with_single_data_point(self):
        """Test backtest with minimal data."""
        dates = pd.date_range(start="2020-01-01", periods=1, freq="D")
        data = pd.DataFrame(
            {
                "Open": [100],
                "High": [105],
                "Low": [95],
                "Close": [100],
                "Volume": [1000000],
            },
            index=dates,
        )

        config = BacktestConfig(initial_capital=10000, lookback_period=1)
        strategy = BuyAndHoldStrategy(shares=10)

        # Should handle gracefully or raise appropriate error
        try:
            backtest = Backtest(data=data, config=config)
            results = backtest.run([strategy])
            # If it succeeds, verify minimal results
            assert len(results) >= 0
        except Exception as e:
            # Expected to fail with insufficient data
            assert "data" in str(e).lower() or "lookback" in str(e).lower()

    def test_backtest_with_flat_prices(self):
        """Test backtest with flat prices (no returns)."""
        dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
        data = pd.DataFrame(
            {
                "Open": [100] * 100,
                "High": [105] * 100,
                "Low": [95] * 100,
                "Close": [100] * 100,  # Flat prices
                "Volume": [1000000] * 100,
            },
            index=dates,
        )

        config = BacktestConfig.default(initial_capital=10000, lookback_period=30)
        strategy = BuyAndHoldStrategy(shares=50)

        backtest = Backtest(data=data, config=config)

        results = backtest.run([strategy])

        assert len(results) == 1
        assert results.benchmark.metrics["beta"] == 0.0

    def test_backtest_strategy_state_isolation(self, sample_data):
        """Test that strategies don't share state."""
        config = BacktestConfig.default(initial_capital=10000, lookback_period=30)

        # Run same strategy twice with different names
        strategies = [
            MovingAverageStrategy(short_window=10, long_window=30, shares=10, name="MA_1"),
            MovingAverageStrategy(short_window=10, long_window=30, shares=10, name="MA_2"),
        ]

        backtest = Backtest(data=sample_data, config=config)
        results = backtest.run(strategies)

        # Strategies should have identical results (state isolation verified)
        result1 = results.get_strategy("MA_1")
        result2 = results.get_strategy("MA_2")

        assert result1.metrics["total_return"] == result2.metrics["total_return"]
        assert len(result1.trade_history) == len(result2.trade_history)

    def test_final_value_includes_last_day_commission(self):
        """End-of-period equity reflects fills and costs on the final bar."""

        class LastDayBuyer(Strategy):
            def predict(self, data, trade_history):
                return self.buy(1) if self._portfolio_state["is_last_day"] else self.hold()

        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        prices = [100.0, 101.0, 102.0, 103.0]
        data = pd.DataFrame(
            {"Open": prices, "High": prices, "Low": prices, "Close": prices}, index=dates
        )
        config = BacktestConfig(
            initial_capital=1000,
            lookback_period=1,
            commission_type="flat",
            commission_value=10.0,
            parallel_execution=False,
        )

        result = Backtest(data, config).run([LastDayBuyer()]).get_strategy("LastDayBuyer")

        assert result.metrics["final_value"] == pytest.approx(990.0)
        assert result.portfolio_values.iloc[-1] == pytest.approx(990.0)

    def test_custom_commission_requires_a_callable(self, sample_data):
        """Custom commission mode never silently becomes zero-cost trading."""
        config = BacktestConfig(commission_type="custom", commission_value=0.0)

        with pytest.raises(ValueError, match="commission_calculator must be provided"):
            Backtest(sample_data, config)

    def test_custom_execution_price_uses_supplied_extractor(self, sample_data):
        """The main Backtest API supports its advertised custom execution mode."""
        config = BacktestConfig(
            lookback_period=5,
            execution_price="custom",
            parallel_execution=False,
        )
        result = (
            Backtest(
                sample_data,
                config,
                execution_price_extractor=lambda row: float(row["Low"]),
            )
            .run([BuyAndHoldStrategy(shares=1)])
            .get_strategy("BuyAndHold")
        )

        assert result.trade_history[0]["price"] == sample_data.iloc[5]["Low"]

    def test_strategy_errors_raise_by_default(self, sample_data):
        """A broken strategy cannot silently produce a valid-looking result."""

        class BrokenStrategy(Strategy):
            def predict(self, data, trade_history):
                raise RuntimeError("broken prediction")

        config = BacktestConfig(lookback_period=5, parallel_execution=False)

        with pytest.raises(RuntimeError, match="BrokenStrategy.*broken prediction"):
            Backtest(sample_data, config).run([BrokenStrategy()])

    def test_continue_error_policy_returns_structured_diagnostics(self, sample_data):
        """Opt-in continuation records every strategy failure."""

        class BrokenStrategy(Strategy):
            def predict(self, data, trade_history):
                raise RuntimeError("broken prediction")

        config = BacktestConfig(
            lookback_period=5,
            parallel_execution=False,
            error_policy="continue",
        )
        result = (
            Backtest(sample_data, config).run([BrokenStrategy()]).get_strategy("BrokenStrategy")
        )

        assert result.errors
        assert result.errors[0]["stage"] == "prediction"
        assert result.errors[0]["error_type"] == "RuntimeError"
        assert result.errors[0]["message"] == "broken prediction"

    def test_progress_output_is_opt_in(self, sample_data, capsys):
        """Library calls remain quiet unless progress output is requested."""
        config = BacktestConfig(lookback_period=5, parallel_execution=False)

        Backtest(sample_data, config).run([BuyAndHoldStrategy(shares=1)])

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_repeated_runs_reuse_an_immutable_benchmark(self, sample_data):
        """Repeated runs avoid recomputing the same benchmark and isolate results."""

        class NeverTradeStrategy(Strategy):
            def predict(self, data, trade_history):
                return self.hold()

        calls = 0

        def commission(shares, price):
            nonlocal calls
            calls += 1
            return 0.0

        config = BacktestConfig.default(
            commission_type="custom",
            commission_value=0.0,
            parallel_execution=False,
        )
        backtest = Backtest(sample_data, config, commission_calculator=commission)

        first = backtest.run([NeverTradeStrategy(name="first")])
        calls_after_first_run = calls
        first.benchmark.portfolio_values.iloc[0] = -1
        second = backtest.run([NeverTradeStrategy(name="second")])

        assert calls == calls_after_first_run
        assert second.benchmark.portfolio_values.iloc[0] >= 0

    def test_portfolio_helpers_are_unavailable_after_run(self, sample_data):
        strategy = BuyAndHoldStrategy(shares=1)
        config = BacktestConfig.default(parallel_execution=False)

        Backtest(sample_data, config).run([strategy])

        with pytest.raises(RuntimeError, match="only be called inside predict"):
            strategy.get_cash()

    def test_strategy_cannot_mutate_engine_trade_history(self, sample_data):
        class MutatingStrategy(Strategy):
            def __init__(self):
                super().__init__()
                self.callback_saw_position = False

            def predict(self, data, trade_history):
                if trade_history:
                    trade_history[0]["price"] = -1
                    trade_history.append({"signal": "fake"})
                return self.buy(1) if not self.has_position() else self.hold()

            def on_trade_executed(self, trade_info):
                self.callback_saw_position = self.has_position()
                trade_info["price"] = -1
                trade_info["positions"].clear()

        strategy = MutatingStrategy()
        config = BacktestConfig.default(parallel_execution=False)

        result = Backtest(sample_data, config).run([strategy]).get_strategy("MutatingStrategy")

        assert strategy.callback_saw_position is True
        assert len(result.trade_history) == 1
        assert result.trade_history[0]["price"] > 0
        assert result.trade_history[0]["positions"]

    def test_parallel_strategy_errors_remain_contextual(self, sample_data):
        class BrokenStrategy(Strategy):
            def predict(self, data, trade_history):
                raise RuntimeError("parallel failure")

        config = BacktestConfig.default(parallel_execution=True, n_jobs=2)
        strategies = [BrokenStrategy(name="broken-1"), BrokenStrategy(name="broken-2")]

        with pytest.raises(RuntimeError, match="parallel failure"):
            Backtest(sample_data, config).run(strategies)
