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
        self.config = config
        validate_dataframe(
            self.data,
            strict=True,
            require_volume=config.max_volume_participation is not None,
        )

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
        """Infer annual periods from the observed samples over calendar time."""
        if len(self.data.index) < 2 or not isinstance(self.data.index, pd.DatetimeIndex):
            return 252

        elapsed_seconds = (self.data.index[-1] - self.data.index[0]).total_seconds()
        if elapsed_seconds <= 0:
            return 252

        elapsed_years = elapsed_seconds / (365.25 * 86400)
        periods = round((len(self.data.index) - 1) / elapsed_years)
        return max(1, periods)

    def _extract_price(self, row: pd.Series) -> float:
        """Extract and validate an execution price at the engine boundary."""
        price = float(self.price_extractor(row))
        if not isfinite(price) or price <= 0:
            raise ValueError(f"Execution price must be finite and positive, got {price}")
        return price

    def _fill_price(self, reference_price: float, side: str) -> float:
        """Apply deterministic adverse spread and slippage to a reference price."""
        adverse_fraction = (self.config.slippage_bps + self.config.spread_bps / 2) / 10_000
        multiplier = 1 + adverse_fraction if side == "buy" else 1 - adverse_fraction
        price = reference_price * multiplier
        if not isfinite(price) or price <= 0:
            raise ValueError(f"Simulated {side} fill price must be positive, got {price}")
        return price

    def _cap_order_size(self, row: pd.Series, requested_size: float) -> float:
        """Clamp an order to the configured fraction of bar volume."""
        if self.config.max_volume_participation is None:
            return requested_size
        available_size = float(row["Volume"]) * self.config.max_volume_participation
        return min(requested_size, available_size)

    def _max_affordable_quantity(self, cash: float, price: float) -> float:
        """Find the largest quantity affordable under the commission function."""
        low = 0.0
        high = cash / price
        for _ in range(80):
            candidate = (low + high) / 2
            commission = self.commission_calculator(candidate, price)
            if candidate * price + commission <= cash:
                low = candidate
            else:
                high = candidate
        return low

    def _execute_order(
        self,
        portfolio: Portfolio,
        prediction: Dict[str, Any],
        row: pd.Series,
        reference_price: float,
        timestamp: datetime,
    ) -> Dict[str, Any] | None:
        """Execute one validated prediction under configured simulation constraints."""
        signal = prediction["signal"]
        requested_size = prediction["size"]
        size = self._cap_order_size(row, requested_size)
        if signal == "hold" or size <= 0:
            return None

        fill_price = self._fill_price(reference_price, signal)
        commission = self.commission_calculator(size, fill_price)
        trade_info = None
        if signal == "buy" and portfolio.can_afford(size, fill_price, commission):
            trade_info = portfolio.execute_buy(
                shares=size,
                price=fill_price,
                commission=commission,
                timestamp=timestamp,
            )
        elif signal == "sell" and portfolio.get_total_shares() >= size:
            trade_info = portfolio.execute_sell(
                shares=size,
                price=fill_price,
                commission=commission,
                timestamp=timestamp,
                order_ids=prediction.get("order_ids"),
            )

        if trade_info is not None and size < requested_size:
            trade_info["requested_shares"] = requested_size
        return trade_info

    def run(self, strategies: List[Strategy]) -> BacktestResults:
        """Run backtest for all strategies.

        :param strategies: List of strategies to backtest
        :return: BacktestResults object with methods for accessing and comparing results
        """
        # Validate strategies
        validate_strategies(strategies, self.config.lookback_period)

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

        def notify_trade(
            trade_info: Dict[str, Any],
            current_date: datetime,
            current_price: float,
            is_last_day: bool,
        ) -> None:
            nonlocal state_snapshot, strategy_trade_history
            state_snapshot = portfolio.get_state_snapshot()
            strategy_trade_history = portfolio.get_trade_history()
            strategy._portfolio_state = {
                **state_snapshot,
                "portfolio_value": portfolio.get_portfolio_value(current_price),
                "current_price": current_price,
                "timestamp": current_date,
                "is_last_day": is_last_day,
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
                        "timestamp": current_date,
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
                    trade_info = None

                    try:
                        trade_info = self._execute_order(
                            portfolio,
                            prediction,
                            current_row,
                            current_price,
                            current_date,
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
                        notify_trade(trade_info, current_date, current_price, i == end_idx)

            if i == end_idx and self.config.final_liquidation and portfolio.get_total_shares() > 0:
                try:
                    liquidation = self._execute_order(
                        portfolio,
                        {
                            "signal": "sell",
                            "size": portfolio.get_total_shares(),
                            "order_ids": None,
                        },
                        current_row,
                        current_price,
                        current_date,
                    )
                    if liquidation is not None:
                        liquidation["forced_liquidation"] = True
                        notify_trade(liquidation, current_date, current_price, True)
                except Exception as error:
                    self._handle_strategy_error(
                        strategy,
                        current_date,
                        "final liquidation",
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

        # Track portfolio values
        portfolio_values = []
        timestamps = []
        exposure = []

        for i in range(start_idx, end_idx + 1):
            current_date = self.data.index[i]
            current_row = self.data.iloc[i]
            current_price = self._extract_price(current_row)

            should_accumulate = not (i == end_idx and self.config.final_liquidation)
            can_buy_this_bar = i == start_idx
            if should_accumulate and can_buy_this_bar and portfolio.cash > 0:
                buy_price = self._fill_price(current_price, "buy")
                affordable_size = self._max_affordable_quantity(portfolio.cash, buy_price)
                buy_size = self._cap_order_size(current_row, affordable_size)
                if buy_size > 0:
                    commission = self.commission_calculator(buy_size, buy_price)
                    if portfolio.can_afford(buy_size, buy_price, commission):
                        trade = portfolio.execute_buy(
                            shares=buy_size,
                            price=buy_price,
                            commission=commission,
                            timestamp=current_date,
                        )
                        if buy_size < affordable_size:
                            trade["requested_shares"] = affordable_size

            if i == end_idx and self.config.final_liquidation:
                sell_size = self._cap_order_size(current_row, portfolio.get_total_shares())
                if sell_size > 0:
                    sell_price = self._fill_price(current_price, "sell")
                    commission = self.commission_calculator(sell_size, sell_price)
                    trade = portfolio.execute_sell(
                        shares=sell_size,
                        price=sell_price,
                        commission=commission,
                        timestamp=current_date,
                    )
                    trade["forced_liquidation"] = True

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
