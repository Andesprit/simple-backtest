"""Grouped per-combination evidence of a parameter search, without changing its winner."""

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from simple_backtest import Backtest, BacktestConfig, MovingAverageStrategy
from simple_backtest.optimization import (
    GridSearchOptimizer,
    RandomSearchOptimizer,
    WalkForwardOptimizer,
)
from simple_backtest.strategy.base import Strategy
from simple_backtest.utils.validation import StrategyExecutionError


class Entry(Strategy):
    """Buys once: steady always 2 shares, lottery 10 shares with probability 1/4."""

    def __init__(self, mode="steady", name=None):
        if mode not in ("steady", "lottery"):
            raise ValueError("mode must be steady or lottery")
        super().__init__(name=name)
        self.mode, self.entered = mode, False

    def predict(self, data, trade_history):
        if self.entered:
            return self.hold()
        self.entered = True
        shares = 2 if self.mode == "steady" else int(self.rng.choice([0, 0, 0, 10]))
        return self.buy(shares) if shares else self.hold()


class Weighted(Strategy):
    def __init__(self, weights=(1,), tag=None, name=None):
        super().__init__(name=name)
        self.weights = weights

    def predict(self, data, trade_history):
        return self.hold() if trade_history else self.buy(sum(self.weights))


@pytest.fixture
def data():
    close = np.linspace(100, 125, 80)
    return pd.DataFrame(
        {"Open": close, "High": close + 2, "Low": close - 2, "Close": close},
        index=pd.date_range("2024-01-01", periods=80),
    )


@pytest.fixture
def config():
    return BacktestConfig.zero_commission(
        initial_capital=10000,
        lookback_period=3,
        parallel_execution=False,
        risk_free_rate=0,
        random_seed=7,
    )


@pytest.fixture
def runs(monkeypatch):
    count = {"n": 0}
    original = Backtest.run

    def counted(self, strategies):
        count["n"] += 1
        return original(self, strategies)

    monkeypatch.setattr(Backtest, "run", counted)
    return count


MODES = {"mode": ["steady", "lottery", "invalid"]}


def repeated(data, config):
    search = RandomSearchOptimizer(n_iter=3, repeats=12, random_state=42, verbose=False)
    ranked = search.optimize(data, config, Entry, MODES, metric="total_return")
    return search, ranked


def test_repeated_trials_show_best_versus_typical_with_exact_accounting(data, config, runs):
    search, ranked = repeated(data, config)
    reference, reference_ranked = repeated(data, config)
    evaluations = runs["n"]
    summary = search.trial_summary().set_index("mode")
    figure = search.plot_trials()
    assert runs["n"] == evaluations  # inspection and plot never simulate

    totals = summary[["attempted", "defined", "undefined", "failed"]].sum()
    assert totals["attempted"] == search.summary["attempted_evaluations"] == 36
    assert totals["defined"] + totals["undefined"] == search.summary["successful_evaluations"]
    assert totals["undefined"] == search.summary["undefined_objectives"] == 0
    assert totals["failed"] == len(search.failures) == 12
    lottery, steady, invalid = summary.loc["lottery"], summary.loc["steady"], summary.loc["invalid"]
    assert lottery["total_return_max"] == pytest.approx(2.405063, abs=1e-6)
    assert lottery["total_return_median"] == 0
    assert (
        steady["total_return_min"]
        == steady["total_return_max"]
        == pytest.approx(0.481013, abs=1e-6)
    )
    assert invalid["failed"] == 12 and np.isnan(invalid["total_return_median"])
    assert "all 12 evaluations failed" in invalid["note"]
    assert summary["contains_selected"].tolist() == [False, False, True]
    best = search._best
    assert lottery["selected_trial"] == (
        f"trial {best['trial']}, simulation_seed={best['simulation_seed']}, "
        f"total_return={best['score']:.6g}"
    )
    assert search.trial_summary().attrs["trial_summary"]["unit"] == "%"
    assert "trial_summary()" in str(search)

    # The winner and ranked output are exactly those of an identical search.
    pd.testing.assert_frame_equal(ranked, reference_ranked)
    assert ranked.attrs == reference_ranked.attrs
    assert search.best_parameters == reference.best_parameters
    assert search.best_result.metadata["random_seed"] == ranked.loc[0, "simulation_seed"]
    assert (best["trial"], best["simulation_seed"]) == (
        reference._best["trial"],
        reference._best["simulation_seed"],
    )

    box, star = figure.data
    assert len(box.y) == 24 and star.y == (best["score"],)
    assert "In-sample" in figure.layout.title.text
    assert "12 failed" in figure.layout.xaxis.ticktext[0]


def test_failed_repeats_keep_their_trial_and_seed_without_changing_seed_order(data, config):
    search, ranked = repeated(data, config)
    failed = [record for record in search._evaluations if record[1] == "failed"]
    assert [record[3] for record in failed] == list(range(1, 13))
    assert all(isinstance(record[4], int) for record in failed)
    seeds = [record[4] for record in search._evaluations]
    assert len(set(seeds)) == 36
    assert set(ranked["simulation_seed"]) <= set(seeds)


def test_all_undefined_objectives_remain_visible(data, config):
    flat = data.assign(Open=100.0, High=102.0, Low=98.0, Close=100.0)
    search = RandomSearchOptimizer(n_iter=2, repeats=3, random_state=1, verbose=False)
    with pytest.warns(UserWarning, match="No candidate had a defined sortino_ratio"):
        ranked = search.optimize(
            flat, config, Entry, {"mode": ["steady", "lottery"]}, "sortino_ratio"
        )
    assert ranked.empty
    summary = search.trial_summary()
    assert summary["undefined"].tolist() == [3, 3]
    assert not summary["contains_selected"].any()
    assert summary["note"].str.contains("no defined sortino_ratio").all()
    figure = search.plot_trials()
    assert len(figure.data) == 1 and len(figure.data[0].y) == 0
    assert len(figure.layout.annotations) == 2


def test_integer_grid_minimized_objective_keeps_types_and_winner(config):
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 1, 160))
    data = pd.DataFrame(
        {"Open": close, "High": close + 1, "Low": close - 1, "Close": close},
        index=pd.date_range("2024-01-01", periods=160),
    )
    grid = GridSearchOptimizer(verbose=False)
    space = {"short_window": [5, 10, 30], "long_window": [10, 20], "shares": [10]}
    ranked = grid.optimize(
        data,
        config.model_copy(update={"lookback_period": 30}),
        MovingAverageStrategy,
        space,
        metric="max_drawdown",
    )
    summary = grid.trial_summary()
    assert len(summary) == 6 and summary["attempted"].eq(1).all()
    assert summary["failed"].sum() == len(grid.failures) == 3
    assert {type(value) for value in summary["short_window"]} == {int}
    selected = summary[summary["contains_selected"]].iloc[0]
    assert {k: selected[k] for k in space} == grid.best_parameters
    assert selected["max_drawdown_min"] == ranked.loc[0, "max_drawdown"]
    assert selected["selected_trial"].startswith("max_drawdown=")
    assert summary.attrs["trial_summary"]["direction"] == "lower is better"
    defined = summary[summary["defined"] == 1]
    assert defined["note"].eq("single observation").all()


def test_categorical_unhashable_values_group_by_position_and_stay_detached(data, config):
    weights = [[1, 2], (1, 2)]
    space = {"weights": weights, "tag": [None, {"a": 1}]}
    grid = GridSearchOptimizer(verbose=False)
    grid.optimize(data, config, Weighted, space, metric="total_return")
    summary = grid.trial_summary()
    assert len(summary) == 4
    assert summary["weights"].tolist() == [[1, 2], [1, 2], (1, 2), (1, 2)]
    # Mutating the caller space or a returned value never changes the evidence.
    weights[0].append(99)
    space["tag"][1]["a"] = 2
    summary.loc[0, "weights"].append(7)
    again = grid.trial_summary()
    assert again["weights"].tolist() == [[1, 2], [1, 2], (1, 2), (1, 2)]
    assert again["tag"].tolist()[1] == {"a": 1}
    assert again["contains_selected"].sum() == 1
    ticks = grid.plot_trials().layout.xaxis.ticktext
    assert len(ticks) == 4 and len(set(ticks)) == 4


def test_empty_parameter_space_is_one_default_observation(data, config):
    grid = GridSearchOptimizer(verbose=False)
    grid.optimize(data, config, Entry, {}, metric="total_return")
    summary = grid.trial_summary()
    assert len(summary) == 1 and summary.loc[0, "attempted"] == 1
    assert summary.loc[0, "contains_selected"]
    assert "strategy defaults" in grid.plot_trials().layout.xaxis.ticktext[0]


def test_unbounded_objective_is_kept_and_labelled(data, config):
    grid = GridSearchOptimizer(verbose=False)
    grid.optimize(data, config, Entry, {"mode": ["steady"]}, metric="total_return")
    key = grid._evaluations[0][0]
    grid._evaluations[0] = (key, "defined", float("inf"), None, None)
    grid._best["score"] = float("inf")
    summary = grid.trial_summary()
    assert summary.loc[0, "total_return_max"] == float("inf")
    figure = grid.plot_trials()
    assert len(figure.data) == 1 and len(figure.data[0].y) == 0
    assert "unbounded" in figure.layout.annotations[0].text


def test_no_search_abort_all_failed_and_rerun_never_show_stale_evidence(data, config):
    grid = GridSearchOptimizer(verbose=False)
    with pytest.raises(ValueError, match="No search has run yet"):
        grid.trial_summary()
    grid.optimize(data, config, Entry, {"mode": ["steady"]}, metric="total_return")
    assert len(grid.trial_summary()) == 1

    class Boom(Entry):
        def predict(self, data, trade_history):
            if self.mode == "lottery":
                raise RuntimeError("boom")
            return super().predict(data, trade_history)

    with pytest.raises(StrategyExecutionError):
        grid.optimize(data, config, Boom, {"mode": ["steady", "lottery"]}, metric="total_return")
    with pytest.raises(ValueError, match="stopped with an error"):
        grid.trial_summary()
    with pytest.raises(ValueError, match="stopped with an error"):
        grid.plot_trials()

    with pytest.raises(ValueError, match="All 2 candidate evaluations failed"):
        grid.optimize(data, config, Entry, {"mode": ["x", "y"]}, metric="total_return")
    summary = grid.trial_summary()
    assert summary["failed"].tolist() == [1, 1] and not summary["contains_selected"].any()


def test_walk_forward_keeps_fold_specific_guidance(data, config):
    walk = WalkForwardOptimizer(train_size=0.5, n_splits=2, verbose=False)
    walk.optimize(data, config, Entry, {"mode": ["steady"]}, metric="total_return")
    with pytest.raises(ValueError, match="separately in each fold"):
        walk.trial_summary()


def test_custom_optimizers_without_evidence_are_refused_not_guessed(data, config):
    class MetricsOnly(GridSearchOptimizer):
        def _run_backtest(self, data, config, strategy, backtest=None):
            return Backtest(data, config).run([strategy]).get_strategy(strategy.get_name()).metrics

    class Silent(GridSearchOptimizer):
        def _consider(self, parameters, metrics, metric, **trial):
            pass

    hook = MetricsOnly(verbose=False)
    hook.optimize(data, config, Entry, {"mode": ["steady", "lottery"]}, metric="total_return")
    assert hook.trial_summary()["attempted"].tolist() == [1, 1]  # metrics-only hook still works

    silent = Silent(verbose=False)
    silent.optimize(data, config, Entry, {"mode": ["steady"]}, metric="total_return")
    with pytest.raises(ValueError, match="recorded per-evaluation evidence for 0 of 1"):
        silent.trial_summary()


def test_compact_evidence_keeps_one_full_result(data, config):
    search, _ = repeated(data, config)
    assert all(len(record) == 5 for record in search._evaluations)
    retained = [r for r in search._evaluations for v in r if type(v).__name__ == "StrategyResult"]
    assert not retained and search._last_result is None
    size = sum(sys.getsizeof(record) for record in search._evaluations)
    assert size / len(search._evaluations) < 200
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        search.trial_summary()
