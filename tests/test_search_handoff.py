"""Handing a parameter search off to its evaluated winner, or explaining why there is none."""

import gc
import warnings
import weakref

import numpy as np
import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig, BuyAndHoldStrategy, MovingAverageStrategy
from simple_backtest.optimization import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    WalkForwardOptimizer,
)
from simple_backtest.optimization.base import Optimizer
from simple_backtest.strategy.base import Strategy
from simple_backtest.utils.validation import InsufficientHistoryError, StrategyExecutionError

SPACE = {"short_window": [3, 4], "long_window": [8, 10], "shares": [1]}


@pytest.fixture
def data():
    rng = np.random.default_rng(34)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, 100)))
    index = pd.date_range("2024-01-01", periods=100)
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close}, index=index
    )


@pytest.fixture
def config():
    return BacktestConfig.zero_commission(
        initial_capital=1000, lookback_period=12, parallel_execution=False, risk_free_rate=0
    )


@pytest.fixture
def runs(monkeypatch):
    """Count every Backtest.run call."""
    count = {"n": 0}
    original = Backtest.run

    def counted(self, strategies):
        count["n"] += 1
        return original(self, strategies)

    monkeypatch.setattr(Backtest, "run", counted)
    return count


class Coin(Strategy):
    def __init__(self, p=0.5, shares=1, name=None):
        super().__init__(name=name or f"Coin_{p}_{shares}")
        self.p, self.shares = p, shares

    def predict(self, data, trade_history):
        if self.rng.random() < self.p:
            return self.buy(self.shares)
        return self.sell(self.shares) if self.has_position() else self.hold()


def fills(result):
    return [(t["signal"], t["price"], t["shares"]) for t in result.trade_history]


@pytest.mark.parametrize("metric", ["total_return", "volatility", "max_drawdown"])
@pytest.mark.parametrize(
    "optimizer", [GridSearchOptimizer(verbose=False), RandomSearchOptimizer(verbose=False)]
)
def test_winner_is_the_evaluated_top_row_with_typed_parameters(
    data, config, runs, optimizer, metric
):
    ranked = optimizer.optimize(data, config, MovingAverageStrategy, SPACE, metric)
    evaluations = runs["n"]
    parameters, result = optimizer.best_parameters, optimizer.best_result

    assert runs["n"] == evaluations
    assert list(parameters) == list(SPACE)
    assert all(type(value) is int for value in parameters.values())
    assert parameters == {name: ranked.iloc[0][name] for name in SPACE}
    assert result.metrics[metric] == ranked.iloc[0][metric]
    fresh = Backtest(data, config).run([MovingAverageStrategy(**parameters)])
    fresh = fresh.get_strategy(result.name)
    assert result.returns.equals(fresh.returns)
    assert fills(result) == fills(fresh)
    text = str(optimizer)
    assert f"best {metric}" in text
    assert ("higher" if metric == "total_return" else "lower") in text
    assert "short_window=" in text and "optimizer.best_result" in text
    assert repr(optimizer).endswith("')")


def test_repeated_trial_is_the_exact_selected_trial(config, data, runs):
    optimizer = RandomSearchOptimizer(n_iter=3, repeats=4, random_state=7, verbose=False)
    ranked = optimizer.optimize(
        data, config, Coin, {"p": [0.1, 0.3, 0.6], "shares": [1, 2]}, "total_return"
    )
    assert runs["n"] == 12
    result, row = optimizer.best_result, ranked.iloc[0]
    assert runs["n"] == 12
    seed = int(row["simulation_seed"])
    assert result.metadata["random_seed"] == seed
    assert result.metrics["total_return"] == row["total_return"]
    assert list(optimizer.best_parameters) == ["p", "shares"]
    replay = config.model_copy(update={"random_seed": seed})
    fresh = Backtest(data, replay).run([Coin(**optimizer.best_parameters)])
    fresh = fresh.get_strategy(result.name)
    assert result.returns.equals(fresh.returns) and fills(result) == fills(fresh)
    assert f"selected trial {int(row['trial'])} (simulation_seed={seed})" in str(optimizer)
    assert "not an average" in str(optimizer)


class Scored(GridSearchOptimizer):
    """Custom objective from the hook: ties, infinity and NaN by short_window."""

    scores = {3: 1.0, 4: float("nan"), 5: float("inf"), 6: float("inf")}

    def _run_backtest(self, data, config, strategy, backtest=None):
        metrics = dict(super()._run_backtest(data, config, strategy, backtest))
        metrics["custom"] = self.scores[strategy.short_window]
        return metrics


def test_custom_objective_ties_infinity_and_nan_follow_ranking(data, config):
    optimizer = Scored(verbose=False)
    space = {"short_window": [3, 4, 5, 6], "long_window": [8], "shares": [1]}
    ranked = optimizer.optimize(data, config, MovingAverageStrategy, space, "custom")
    assert ranked.short_window.tolist() == [5, 6, 3]
    assert optimizer.best_parameters["short_window"] == 5
    assert optimizer.best_result.name == "MA_5_8"
    assert "best custom = inf" in str(optimizer)
    assert "1 undefined custom" in str(optimizer)


def test_empty_parameter_space_selects_strategy_defaults(data, config):
    optimizer = GridSearchOptimizer(verbose=False)
    optimizer.optimize(data, config, BuyAndHoldStrategy, {}, "total_return")
    assert optimizer.best_parameters == {}
    assert optimizer.best_result.name == BuyAndHoldStrategy().get_name()
    assert "parameters: strategy defaults" in str(optimizer)


def test_only_the_best_result_is_retained(data, config, monkeypatch):
    seen = []
    original = Optimizer._run_backtest

    def tracked(self, *args, **kwargs):
        metrics = original(self, *args, **kwargs)
        seen.append(weakref.ref(self._last_result))
        return metrics

    monkeypatch.setattr(Optimizer, "_run_backtest", tracked)
    optimizer = GridSearchOptimizer(verbose=False)
    optimizer.optimize(data, config, MovingAverageStrategy, SPACE, "total_return")
    gc.collect()
    assert len(seen) == 4
    assert [ref() for ref in seen if ref() is not None] == [optimizer.best_result]


def test_metrics_only_hook_keeps_parameters_and_never_inherits_a_result(data, config):
    class Mixed(GridSearchOptimizer):
        def _run_backtest(self, data, config, strategy, backtest=None):
            if strategy.short_window == 3:
                super()._run_backtest(data, config, strategy, backtest)
                raise InsufficientHistoryError("needs more history")
            return {"total_return": float(strategy.long_window)}

    optimizer = Mixed(verbose=False)
    optimizer.optimize(data, config, MovingAverageStrategy, SPACE, "total_return")
    assert optimizer.best_parameters == {"short_window": 4, "long_window": 10, "shares": 1}
    with pytest.raises(ValueError, match="returned metrics without keeping"):
        optimizer.best_result
    assert "inspect: best_parameters, optimizer.failures" in str(optimizer)


def test_no_winner_states_explain_and_never_expose_a_previous_winner(data, config, runs):
    optimizer = GridSearchOptimizer(verbose=False)
    with pytest.raises(ValueError, match="No search has run yet"):
        optimizer.best_parameters

    optimizer.optimize(data, config, MovingAverageStrategy, SPACE, "total_return")
    with pytest.raises(ValueError, match="All 4 candidate evaluations failed"):
        optimizer.optimize(data, config, MovingAverageStrategy, {"short": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="optimizer.failures"):
        optimizer.best_result

    optimizer.optimize(data, config, MovingAverageStrategy, SPACE, "total_return")
    before = runs["n"]
    with pytest.raises(ValueError, match="Did you mean 'sharpe_ratio'"):
        optimizer.optimize(data, config, MovingAverageStrategy, SPACE, "sharpe")
    assert runs["n"] - before == 1
    with pytest.raises(ValueError, match="stopped with an error"):
        optimizer.best_parameters

    optimizer.optimize(data, config, MovingAverageStrategy, SPACE, "total_return")
    flat = data.assign(Open=100.0, High=100.0, Low=100.0, Close=100.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ranked = optimizer.optimize(flat, config, MovingAverageStrategy, SPACE, "sortino_ratio")
    assert ranked.empty and ranked.attrs["search"]["undefined_objectives"] == 4
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "No candidate had a defined sortino_ratio (4 evaluated, 4 undefined" in message
    assert "metric='total_return'" in message
    assert message in str(optimizer)
    with pytest.raises(ValueError, match="defined sortino_ratio"):
        optimizer.best_result


def test_walk_forward_points_to_its_out_of_sample_report(data, config):
    optimizer = WalkForwardOptimizer(train_size=0.6, n_splits=2, verbose=False)
    optimizer.optimize(data, config, MovingAverageStrategy, SPACE, "total_return")
    assert len(optimizer.report.folds) == 2
    with pytest.raises(ValueError, match="optimizer.report.folds"):
        optimizer.best_result
    assert "out-of-sample" in str(optimizer)


class Explodes(Strategy):
    def __init__(self, explode=False):
        super().__init__()
        self.explode = explode

    def predict(self, data, trade_history):
        if self.explode:
            raise RuntimeError("indicator failed on this candidate")
        return self.hold()


@pytest.mark.parametrize(
    "optimizer",
    [
        GridSearchOptimizer(verbose=False),
        RandomSearchOptimizer(n_iter=2, random_state=1, verbose=False),
    ],
)
def test_search_aborted_after_a_success_selects_nothing(data, config, runs, optimizer):
    optimizer.optimize(data, config, MovingAverageStrategy, SPACE, "total_return")
    with pytest.raises(StrategyExecutionError, match="indicator failed"):
        optimizer.optimize(data, config, Explodes, {"explode": [False, True]}, "total_return")
    after_abort = runs["n"]
    assert f"{optimizer.get_name()}: The last search stopped with an error" in str(optimizer)
    for attribute in ("best_parameters", "best_result"):
        with pytest.raises(ValueError, match="stopped with an error"):
            getattr(optimizer, attribute)
    assert runs["n"] == after_abort

    optimizer.optimize(data, config, Explodes, {"explode": [False]}, "total_return")
    assert optimizer.best_parameters == {"explode": False}
    assert optimizer.best_result.name == "Explodes"
