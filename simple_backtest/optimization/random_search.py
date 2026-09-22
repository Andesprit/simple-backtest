"""Random search optimizer - samples random parameter combinations."""

import math
import random
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


class RandomSearchOptimizer(Optimizer):
    """Random search optimizer - randomly samples parameter space.

    Faster than grid search for large parameter spaces.
    Samples n_iter random combinations instead of testing all.

    Example:
        optimizer = RandomSearchOptimizer(n_iter=50, random_state=42)
        results = optimizer.optimize(
            data=data,
            config=config,
            strategy_class=MovingAverageStrategy,
            param_space={
                'short_window': list(range(5, 21)),
                'long_window': list(range(20, 61)),
                'shares': [10]
            },
            metric='sharpe_ratio'
        )
    """

    def __init__(
        self,
        n_iter: int = 100,
        random_state: int | None = None,
        verbose: bool = True,
        name: str | None = None,
        *,
        repeats: int = 1,
    ):
        """Initialize random search optimizer.

        :param n_iter: Maximum number of unique combinations to test
        :param random_state: Random seed for reproducibility
        :param verbose: Show progress bar
        :param name: Optimizer name
        :param repeats: Independent seeded evaluations per candidate (default one)
        """
        super().__init__(name=name or "RandomSearch")
        for label, value in (("n_iter", n_iter), ("repeats", repeats)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        self.repeats = repeats
        self.n_iter = n_iter
        self.random_state = random_state
        self.verbose = verbose

        self._random = random.Random(random_state)

    def optimize(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        strategy_class: Type[Strategy],
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """Run random search optimization.

        :param data: OHLCV DataFrame with DatetimeIndex
        :param config: Backtest configuration
        :param strategy_class: Strategy class to optimize
        :param param_space: Dict of param_name: [values] to sample from
        :param metric: Metric to optimize
        :return: DataFrame of results sorted by metric
        """
        results = []
        param_space = self._prepare_search(param_space)
        if self.repeats > 1 and {"trial", "simulation_seed"}.intersection(param_space):
            raise ValueError(
                "trial and simulation_seed are reserved columns for repeated evaluations"
            )
        if self.random_state is not None:
            self._random.seed(self.random_state)
        param_names = list(param_space)
        combinations = math.prod(len(values) for values in param_space.values())
        indices = self._random.sample(range(combinations), min(self.n_iter, combinations))
        self.summary["unique_candidates"] = len(indices)
        backtest = Backtest(data, config)
        iterator = tqdm(indices, desc="Random Search") if self.verbose else indices

        for index in iterator:
            param_dict = {}
            for name in reversed(param_names):
                index, offset = divmod(index, len(param_space[name]))
                param_dict[name] = param_space[name][offset]
            for trial in range(self.repeats):
                self.summary["attempted_evaluations"] += 1
                seed = config.random_seed
                trial_config = config
                if self.repeats > 1:
                    seed = self._random.randrange(2**32)
                    trial_config = config.model_copy(update={"random_seed": seed})
                try:
                    strategy = strategy_class(**param_dict)
                except (TypeError, ValueError) as error:
                    self._record_failure(param_dict, error)
                    continue
                try:
                    metrics = self._run_backtest(
                        data,
                        trial_config,
                        strategy,
                        backtest=backtest if self.repeats == 1 else None,
                    )
                except InsufficientHistoryError as error:
                    self._record_failure(param_dict, error)
                    continue
                row = {**param_dict, **metrics}
                if self.repeats > 1:
                    row.update(trial=trial + 1, simulation_seed=seed)
                results.append(row)

        return self._finish_search(
            results,
            metric,
            {
                **backtest.metadata,
                "method": "random",
                "strategy_class": f"{strategy_class.__module__}.{strategy_class.__qualname__}",
                "parameter_space": param_space,
                "random_state": self.random_state,
                "repeats": self.repeats,
                "available_candidates": combinations,
            },
        )
