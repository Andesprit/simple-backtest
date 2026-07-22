"""Backtesting engine with parallelization support."""

from copy import deepcopy
from datetime import datetime
from math import isfinite
from typing import Any, Callable, Dict, List

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.core.portfolio import Portfolio
from simple_backtest.core.results import BacktestResults
from simple_backtest.metrics.calculator import calculate_metrics
from simple_backtest.strategy.base import Strategy
from simple_backtest.utils.commission import create_custom_commission, get_commission_calculator
from simple_backtest.utils.execution import create_execution_price_extractor
from simple_backtest.utils.logger import get_logger
from simple_backtest.utils.validation import (
    StrategyExecutionError,
    validate_dataframe,
    validate_date_range,
    validate_strategies,
)

# Initialize logger
logger = get_logger(__name__)


class Backtest:
    """Backtesting engine with parallel strategy execution support."""

    def __init__(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        commission_calculator: Callable[[float, float], float] | None = None,
        execution_price_extractor: Callable[[pd.Series], float] | None = None,
    ):
        """Initialize backtest engine.

        :param data: OHLCV DataFrame with DatetimeIndex
        :param config: Backtest configuration
        :param commission_calculator: Deterministic, side-effect-free custom commission
            function (uses config if None)
        """
        # Copy before normalization so validation never mutates caller-owned data.
        self.data = data.copy(deep=True)
        validate_dataframe(self.data, strict=True)
        self.config = config

        # Validate config against data
        if isinstance(self.data.index, pd.DatetimeIndex):
            config.validate_against_data(
                data_start=self.data.index[0],
                data_end=self.data.index[-1],
                total_rows=len(self.data),
            )

        # Determine trading range
        self._setup_trading_range()

        # Validate date range
        validate_date_range(
            self.data,
            self.trading_start_date,
            self.trading_end_date,
            self.config.lookback_period,
        )

        # Setup commission calculator
        if commission_calculator is None:
            self.commission_calculator = get_commission_calculator(config)
        else:
            self.commission_calculator = create_custom_commission(commission_calculator)

        # Setup execution price extractor
        self.price_extractor = create_execution_price_extractor(
            method=config.execution_price,
            custom_func=execution_price_extractor,
        )
        self.periods_per_year = config.periods_per_year or self._infer_periods_per_year()
        self._benchmark_results: Dict[str, Any] | None = None

    def _setup_trading_range(self) -> None:
        """Set trading date range from config and data."""
        if self.config.trading_start_date is None:
            # Start after lookback period
            start_idx = self.config.lookback_period
            self.trading_start_date = self.data.index[start_idx]
        else:
            self.trading_start_date = self.config.trading_start_date

        if self.config.trading_end_date is None:
            self.trading_end_date = self.data.index[-1]
        else:
            self.trading_end_date = self.config.trading_end_date

        # Get trading data slice
        self.trading_data = self.data.loc[self.trading_start_date : self.trading_end_date]

    def _infer_periods_per_year(self) -> int:
        """Infer an annualization factor from the data's median spacing."""
        if len(self.data.index) < 2 or not isinstance(self.data.index, pd.DatetimeIndex):
            return 252

        median_spacing = self.data.index.to_series().diff().dropna().median()
        seconds = median_spacing.total_seconds()
        if seconds <= 0:
            return 252

        has_weekends = bool((self.data.index.dayofweek >= 5).any())
        active_days = 365 if has_weekends else 252
        periods = round(active_days * 86400 / seconds)
        return max(1, periods)

    def _extract_price(self, row: pd.Series) -> float:
        """Extract and validate an execution price at the engine boundary."""
        price = float(self.price_extractor(row))
        if not isfinite(price) or price <= 0:
            raise ValueError(f"Execution price must be finite and positive, got {price}")
        return price

    def run(self, strategies: List[Strategy]) -> BacktestResults:
        """Run backtest for all strategies.

        :param strategies: List of strategies to backtest
        :return: BacktestResults object with methods for accessing and comparing results
        """
        # Validate strategies
        validate_strategies(strategies)

        # Create benchmark
        if self._benchmark_results is None:
            self._benchmark_results = self._run_benchmark()
        benchmark_results = deepcopy(self._benchmark_results)
        benchmark_values = benchmark_results["portfolio_values"]

        # Reset strategies
        for strategy in strategies:
            strategy.reset_state()

        try:
            if self.config.parallel_execution and len(strategies) > 1:
                n_jobs = self.config.n_jobs if self.config.n_jobs != -1 else -1
                strategy_results = Parallel(n_jobs=n_jobs)(
                    delayed(self._run_single_strategy)(strategy, benchmark_values)
                    for strategy in strategies
                )
            else:
                strategy_results = []
                iterator = (
                    tqdm(strategies, desc="Running strategies")
                    if self.config.show_progress
                    else strategies
                )
                for strategy in iterator:
                    strategy_results.append(self._run_single_strategy(strategy, benchmark_values))
        finally:
            # Portfolio helpers are only valid while predict() is executing.
            for strategy in strategies:
                strategy._portfolio_state = None

        # Combine results
        results = {"benchmark": benchmark_results}
        for strategy, result in zip(strategies, strategy_results):
            results[strategy.get_name()] = result

        return BacktestResults(results)

    def _run_single_strategy(
        self,
        strategy: Strategy,
        benchmark_values: pd.Series,
    ) -> Dict[str, Any]:
        """Run backtest for single strategy.

        :param strategy: Strategy to backtest
        :return: Results dict with metrics, portfolio_values, trade_history, returns
        """
        # Create portfolio
        portfolio = Portfolio(self.config.initial_capital)

        # Track portfolio values over time
        portfolio_values = []
        timestamps = []
        exposure = []
        errors: List[Dict[str, Any]] = []
        state_snapshot = portfolio.get_state_snapshot()
        strategy_trade_history: List[Dict[str, Any]] = []

        # Get trading date range
        start_idx = self.data.index.get_indexer([self.trading_start_date], method="nearest")[0]
        end_idx = self.data.index.get_indexer([self.trading_end_date], method="nearest")[0]

        # Progress bar (only for non-parallel execution)
        iterator = range(start_idx, end_idx + 1)
        if not self.config.parallel_execution and self.config.show_progress:
            iterator = tqdm(
                iterator,
                desc=f"Backtesting {strategy.get_name()}",
                leave=False,
            )

        # Main backtest loop
        for i in iterator:
            current_date = self.data.index[i]
            current_row = self.data.iloc[i]

            # Extract lookback window
            lookback_start = max(0, i - self.config.lookback_period)
            lookback_data = self.data.iloc[lookback_start:i]

            current_price = self._extract_price(current_row)
            portfolio_value = portfolio.get_portfolio_value(current_price)

            if len(lookback_data) >= self.config.lookback_period:
                prediction: Dict[str, Any] | None = None
                try:
                    strategy._portfolio_state = {
                        **state_snapshot,
                        "portfolio_value": portfolio_value,
                        "current_price": current_price,
                        "is_last_day": i == end_idx,
                    }
                    prediction = strategy.predict(
                        lookback_data,
                        strategy_trade_history,
                    )
                    strategy.validate_prediction(prediction)
                except Exception as error:
                    self._handle_strategy_error(
                        strategy,
                        current_date,
                        "prediction",
                        error,
                        errors,
                    )

                if prediction is not None:
                    signal = prediction["signal"]
                    size = prediction["size"]
                    trade_info = None

                    try:
                        if signal == "buy" and size > 0:
                            commission = self.commission_calculator(size, current_price)
                            if portfolio.can_afford(size, current_price, commission):
                                trade_info = portfolio.execute_buy(
                                    shares=size,
                                    price=current_price,
                                    commission=commission,
                                    timestamp=current_date,
                                )
                        elif signal == "sell" and size > 0:
                            if portfolio.get_total_shares() >= size:
                                commission = self.commission_calculator(size, current_price)
                                trade_info = portfolio.execute_sell(
                                    shares=size,
                                    price=current_price,
                                    commission=commission,
                                    timestamp=current_date,
                                    order_ids=prediction.get("order_ids"),
                                )
                    except Exception as error:
                        self._handle_strategy_error(
                            strategy,
                            current_date,
                            "execution",
                            error,
                            errors,
                        )

                    if trade_info is not None:
                        # Refresh isolated strategy views only when accounting changes.
                        state_snapshot = portfolio.get_state_snapshot()
                        strategy_trade_history = portfolio.get_trade_history()
                        strategy._portfolio_state = {
                            **state_snapshot,
                            "portfolio_value": portfolio.get_portfolio_value(current_price),
                            "current_price": current_price,
                            "is_last_day": i == end_idx,
                        }
                        try:
                            strategy.on_trade_executed(deepcopy(trade_info))
                        except Exception as error:
                            self._handle_strategy_error(
                                strategy,
                                current_date,
                                "trade callback",
                                error,
                                errors,
                            )

            # Metrics use end-of-period equity after every fill and commission.
            portfolio_values.append(portfolio.get_portfolio_value(current_price))
            timestamps.append(current_date)
            exposure.append(portfolio.get_total_shares() > 0)

        # Create portfolio values series
        portfolio_series = pd.Series(portfolio_values, index=timestamps)

        # Calculate returns
        returns = portfolio_series.pct_change().dropna()

        benchmark_for_period = benchmark_values.reindex(timestamps)
        exposure_series = pd.Series(exposure, index=timestamps, dtype=bool)

        # Calculate metrics
        metrics = calculate_metrics(
            trade_history=portfolio.get_trade_history(),
            portfolio_values=portfolio_series,
            benchmark_values=benchmark_for_period,
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=self.periods_per_year,
            exposure=exposure_series,
        )

        return {
            "metrics": metrics,
            "portfolio_values": portfolio_series,
            "trade_history": portfolio.get_trade_history(),
            "returns": returns,
            "errors": errors,
        }

    def _handle_strategy_error(
        self,
        strategy: Strategy,
        timestamp: datetime,
        stage: str,
        error: Exception,
        errors: List[Dict[str, Any]],
    ) -> None:
        """Raise a contextual error or append a structured diagnostic."""
        wrapped = StrategyExecutionError(strategy.get_name(), timestamp, stage, error)
        if self.config.error_policy == "raise":
            raise wrapped from error

        errors.append(
            {
                "strategy": strategy.get_name(),
                "timestamp": timestamp,
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )
        logger.warning(str(wrapped))

    def _run_benchmark(self) -> Dict[str, Any]:
        """Run buy-and-hold benchmark."""
        # Create portfolio
        portfolio = Portfolio(self.config.initial_capital)

        # Get first trading date
        start_idx = self.data.index.get_indexer([self.trading_start_date], method="nearest")[0]
        end_idx = self.data.index.get_indexer([self.trading_end_date], method="nearest")[0]

        first_date = self.data.index[start_idx]
        first_row = self.data.iloc[start_idx]
        first_price = self._extract_price(first_row)

        # Find the maximum affordable quantity for any non-negative commission model.
        low = 0.0
        high = self.config.initial_capital / first_price
        for _ in range(80):
            candidate = (low + high) / 2
            commission = self.commission_calculator(candidate, first_price)
            if candidate * first_price + commission <= self.config.initial_capital:
                low = candidate
            else:
                high = candidate
        max_shares = low

        if max_shares > 0:
            commission = self.commission_calculator(max_shares, first_price)
            if portfolio.can_afford(max_shares, first_price, commission):
                portfolio.execute_buy(
                    shares=max_shares,
                    price=first_price,
                    commission=commission,
                    timestamp=first_date,
                )

        # Track portfolio values
        portfolio_values = []
        timestamps = []
        exposure = []

        for i in range(start_idx, end_idx + 1):
            current_date = self.data.index[i]
            current_row = self.data.iloc[i]
            current_price = self._extract_price(current_row)

            portfolio_value = portfolio.get_portfolio_value(current_price)
            portfolio_values.append(portfolio_value)
            timestamps.append(current_date)
            exposure.append(portfolio.get_total_shares() > 0)

        # Create series
        portfolio_series = pd.Series(portfolio_values, index=timestamps)
        returns = portfolio_series.pct_change().dropna()

        # Benchmark metrics compare the benchmark with the same marked equity series.
        metrics = calculate_metrics(
            trade_history=portfolio.get_trade_history(),
            portfolio_values=portfolio_series,
            benchmark_values=portfolio_series,  # Compare to itself
            initial_capital=self.config.initial_capital,
            risk_free_rate=self.config.risk_free_rate,
            periods_per_year=self.periods_per_year,
            exposure=pd.Series(exposure, index=timestamps, dtype=bool),
        )

        return {
            "metrics": metrics,
            "portfolio_values": portfolio_series,
            "trade_history": portfolio.get_trade_history(),
            "returns": returns,
            "errors": [],
        }
