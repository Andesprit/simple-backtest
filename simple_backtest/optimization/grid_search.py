"""Grid search optimizer - exhaustive parameter search."""

import itertools
import math
from typing import Any, Dict, List, Type

import pandas as pd
from tqdm import tqdm

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.core.backtest import Backtest
from simple_backtest.optimization.base import Optimizer
from simple_backtest.strategy.base import Strategy
from simple_backtest.utils.logger import get_logger
from simple_backtest.utils.validation import InsufficientHistoryError

# Initialize logger
logger = get_logger(__name__)


class GridSearchOptimizer(Optimizer):
    """Grid search optimizer - tests all parameter combinations.

    Exhaustively searches through all possible parameter combinations.
    Best for small parameter spaces.

    Example:
        optimizer = GridSearchOptimizer()
        results = optimizer.optimize(
            data=data,
            config=config,
            strategy_class=MovingAverageStrategy,
            param_space={
                'short_window': [5, 10, 15],
                'long_window': [20, 30, 40],
                'shares': [10]
            },
            metric='sharpe_ratio'
        )
    """

    def __init__(self, verbose: bool = True, name: str | None = None):
        """Initialize grid search optimizer.

        :param verbose: Show progress bar
        :param name: Optimizer name
        """
        super().__init__(name=name or "GridSearch")
        self.verbose = verbose

    def optimize(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        strategy_class: Type[Strategy],
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """Run grid search optimization.

        :param data: OHLCV DataFrame with DatetimeIndex
        :param config: Backtest configuration
        :param strategy_class: Strategy class to optimize
        :param param_space: Dict of param_name: [values] to search
        :param metric: Metric to optimize
        :return: DataFrame of results sorted by metric
        """
        results = []
        param_space = self._prepare_search(param_space)
        param_names = list(param_space.keys())
        backtest = Backtest(data, config)
        combination_count = math.prod(len(values) for values in param_space.values())
        param_combinations = itertools.product(*param_space.values())

        if self.verbose:
            logger.info(f"Testing {combination_count} parameter combinations...")

        # Iterate through all combinations
        iterator = (
            tqdm(param_combinations, total=combination_count, desc="Grid Search")
            if self.verbose
            else param_combinations
        )

        for params in iterator:
            self.summary["attempted_evaluations"] += 1
            self.summary["unique_candidates"] += 1
            param_dict = dict(zip(param_names, params))

            try:
                strategy = strategy_class(**param_dict)
            except (TypeError, ValueError) as error:
                self._record_failure(param_dict, error)
                if self.verbose:
                    logger.warning(f"Invalid params {param_dict}: {error}")
                continue

            try:
                metrics = self._run_backtest(data, config, strategy, backtest=backtest)
            except InsufficientHistoryError as error:
                self._record_failure(param_dict, error)
                if self.verbose:
                    logger.warning(f"Invalid history for params {param_dict}: {error}")
                continue
            results.append({**param_dict, **metrics})

        return self._finish_search(
            results,
            metric,
            {
                **backtest.metadata,
                "method": "grid",
                "strategy_class": f"{strategy_class.__module__}.{strategy_class.__qualname__}",
                "parameter_space": param_space,
                "available_candidates": combination_count,
            },
        )
