"""Expanding-window, out-of-sample parameter optimization."""

from typing import Any, Dict, List, Type

import pandas as pd

from simple_backtest.config.settings import BacktestConfig
from simple_backtest.optimization.base import Optimizer
from simple_backtest.optimization.grid_search import GridSearchOptimizer
from simple_backtest.strategy.base import Strategy
from simple_backtest.utils.logger import get_logger

logger = get_logger(__name__)


class WalkForwardOptimizer(Optimizer):
    """Select parameters on expanding training windows and evaluate the winner OOS.

    Each output row is one chronological fold. Test metrics never participate in
    parameter selection or result ordering.
    """

    def __init__(
        self,
        train_size: float = 0.7,
        base_optimizer: Optimizer | None = None,
        n_splits: int = 3,
        verbose: bool = True,
        name: str | None = None,
    ):
        """Initialize an expanding-window optimizer.

        :param train_size: Fraction reserved for the initial training window
        :param base_optimizer: Optimizer used independently in every fold
        :param n_splits: Number of chronological out-of-sample folds
        :param verbose: Show progress information
        :param name: Optimizer name
        """
        super().__init__(name=name or "WalkForward")
        if not 0.0 < train_size < 1.0:
            raise ValueError(f"train_size must be between 0 and 1, got {train_size}")
        if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError(f"n_splits must be positive, got {n_splits}")

        self.train_size = train_size
        self.n_splits = n_splits
        self.base_optimizer = base_optimizer or GridSearchOptimizer(verbose=verbose)
        self.verbose = verbose

    def optimize(
        self,
        data: pd.DataFrame,
        config: BacktestConfig,
        strategy_class: Type[Strategy],
        param_space: Dict[str, List[Any]],
        metric: str = "sharpe_ratio",
    ) -> pd.DataFrame:
        """Run chronological walk-forward optimization.

        The initial training window expands after every fold. Evaluation data
        includes prior rows for lookback context, but trading starts at the first
        test timestamp.
        """
        split_idx = int(len(data) * self.train_size)
        remaining = len(data) - split_idx
        if split_idx <= config.lookback_period:
            raise ValueError("Initial training window must contain more rows than lookback_period")
        if remaining < self.n_splits:
            raise ValueError(
                f"Test region has {remaining} rows, fewer than n_splits={self.n_splits}"
            )

        fold_sizes = self._fold_sizes(remaining)
        param_names = list(param_space)
        rows: list[dict[str, Any]] = []
        test_start_idx = split_idx

        for fold_number, fold_size in enumerate(fold_sizes, start=1):
            test_end_idx = test_start_idx + fold_size
            train_data = data.iloc[:test_start_idx]
            test_data = data.iloc[test_start_idx:test_end_idx]

            if self.verbose:
                logger.info(
                    "Walk-forward fold %s/%s: train %s to %s, test %s to %s",
                    fold_number,
                    self.n_splits,
                    train_data.index[0],
                    train_data.index[-1],
                    test_data.index[0],
                    test_data.index[-1],
                )

            train_config = config.model_copy(
                update={"trading_start_date": None, "trading_end_date": None}
            )
            train_results = self.base_optimizer.optimize(
                data=train_data,
                config=train_config,
                strategy_class=strategy_class,
                param_space=param_space,
                metric=metric,
            )
            if train_results.empty:
                raise RuntimeError(f"No parameter combination succeeded in fold {fold_number}")
            if metric not in train_results:
                raise ValueError(f"Metric '{metric}' was not produced by the base optimizer")

            selected = train_results.iloc[0]
            # Selecting an entire pandas row can coerce integer parameters to
            # floats. Recover the original user-provided objects before
            # constructing the strategy.
            parameters = {
                name: next(
                    candidate for candidate in param_space[name] if candidate == selected[name]
                )
                for name in param_names
            }
            context_start = max(0, test_start_idx - config.lookback_period)
            evaluation_data = data.iloc[context_start:test_end_idx]
            evaluation_config = config.model_copy(
                update={
                    "trading_start_date": test_data.index[0],
                    "trading_end_date": test_data.index[-1],
                }
            )
            test_metrics = self._run_backtest(
                evaluation_data,
                evaluation_config,
                strategy_class(**parameters),
            )

            train_metrics = {
                key: value for key, value in selected.items() if key not in param_names
            }
            row = {
                "fold": fold_number,
                "train_start": train_data.index[0],
                "train_end": train_data.index[-1],
                "test_start": test_data.index[0],
                "test_end": test_data.index[-1],
                **parameters,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
            row[f"{metric}_diff"] = row[f"test_{metric}"] - row[f"train_{metric}"]
            rows.append(row)
            test_start_idx = test_end_idx

        return pd.DataFrame(rows)

    def _fold_sizes(self, test_rows: int) -> list[int]:
        """Split all test rows across folds, assigning remainder to later folds."""
        base_size, remainder = divmod(test_rows, self.n_splits)
        return [
            base_size + int(index >= self.n_splits - remainder) for index in range(self.n_splits)
        ]
