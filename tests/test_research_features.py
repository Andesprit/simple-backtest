"""Numerical and end-to-end contracts for inspectable research experiments."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from simple_backtest import (
    Backtest,
    BacktestConfig,
    BuyAndHoldStrategy,
    DCAStrategy,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    Strategy,
    WalkForwardOptimizer,
)
from simple_backtest.metrics.definitions import calculate_sortino_ratio
from simple_backtest.metrics.objectives import rank_results


def candles(rows=24):
    prices = np.resize([100.0, 110.0, 95.0, 115.0], rows)
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 5,
            "Low": prices - 5,
            "Close": prices + 2,
            "Volume": np.full(rows, 100.0),
        },
        index=pd.date_range("2024-01-01", periods=rows, tz="UTC"),
    )


def config(**kwargs):
    return BacktestConfig(
        **(
            dict(
                initial_capital=1000,
                lookback_period=1,
                commission_type="flat",
                commission_value=1.0,
                parallel_execution=False,
                periods_per_year=252,
                random_seed=42,
            )
            | kwargs
        )
    )


class Budget(Strategy):
    def __init__(self, amount=1000):
        super().__init__()
        self.amount = amount

    def predict(self, data, trade_history):
        return self.hold() if self.has_position() else self.buy_budget(self.amount)


class RandomStrategy(Strategy):
    def __init__(self, shares=1):
        super().__init__()
        self.shares = shares

    def predict(self, data, trade_history):
        if self.rng.random() < 0.5:
            return self.sell_all() if self.has_position() else self.buy(self.shares)
        return self.hold()


def test_sortino_uses_rms_shortfall_over_all_observations():
    assert calculate_sortino_ratio(pd.Series([0.02, -0.01, -0.01, 0.02])) == pytest.approx(
        11.2249721603
    )
    # One loss has a well-defined downside deviation, unlike a sample standard deviation.
    assert calculate_sortino_ratio(pd.Series([0.02, -0.01]), periods_per_year=2) == pytest.approx(1)
    assert math.isnan(calculate_sortino_ratio(pd.Series([0.01, 0.02])))
    assert math.isnan(calculate_sortino_ratio(pd.Series(dtype=float)))


def test_sortino_subtracts_target_before_computing_shortfall():
    # A 21% annual target is 10% per period with two periods per year.
    assert calculate_sortino_ratio(
        pd.Series([0.2, 0]), risk_free_rate=0.21, periods_per_year=2
    ) == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize(
    "commission",
    [
        {"commission_type": "flat", "commission_value": 1.0},
        {"commission_type": "percentage", "commission_value": 0.01},
        {"commission_type": "tiered", "commission_value": [(500.0, 0.002), (float("inf"), 0.001)]},
    ],
)
def test_budget_includes_execution_gap_fees_and_slippage(commission):
    result = (
        Backtest(candles(), config(**commission, slippage_bps=20, spread_bps=10))
        .run([Budget()])
        .get_strategy("Budget")
    )
    trade = result.trade_history[0]
    assert trade["price"] == pytest.approx(110 * 1.0025)
    assert trade["shares"] * trade["price"] + trade["commission"] <= 1000
    assert trade["cash"] == pytest.approx(0, abs=1e-9)
    assert result.order_outcomes[0]["status"] == "filled"
    assert result.order_outcomes[0]["requested_budget"] == 1000
    assert result.order_outcomes[0]["requested_shares"] is None


def test_dca_can_invest_all_available_cash_including_fees():
    result = (
        Backtest(candles(), config())
        .run([DCAStrategy(investment_amount=1000)])
        .get_strategy("DCA_7d")
    )
    assert result.trade_history[0]["shares"] == pytest.approx(999 / 110)
    assert result.trade_history[0]["cash"] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("amount", [-1, float("nan"), float("inf"), True])
def test_invalid_budgets_are_programming_errors(amount):
    with pytest.raises(RuntimeError, match="budget must"):
        Backtest(candles(), config()).run([Budget(amount)])


def test_order_outcomes_explain_cash_volume_and_position_rejections():
    class Orders(Strategy):
        def predict(self, data, trade_history):
            step = data.index[-1].day
            return {1: self.buy(1000), 2: self.sell(1), 3: self.buy(1), 4: self.buy(20)}.get(
                step, self.hold()
            )

    data = candles()
    data.loc[data.index[3], "Volume"] = 0
    result = (
        Backtest(data, config(initial_capital=10, max_volume_participation=0.1))
        .run([Orders()])
        .get_strategy("Orders")
    )
    assert [o["reason"] for o in result.order_outcomes] == [
        "insufficient_cash",
        "insufficient_shares",
        "zero_volume",
        "insufficient_cash",
    ]
    assert not result.errors
    partial = (
        Backtest(candles(), config(max_volume_participation=0.01))
        .run([Budget()])
        .get_strategy("Budget")
    )
    assert partial.order_outcomes[0]["status"] == "partial"
    assert partial.order_outcomes[0]["reason"] == "volume_limit"
    assert partial.order_outcomes[0]["filled_shares"] == 1


def test_budget_larger_than_cash_is_explicitly_partial():
    result = Backtest(candles(), config()).run([Budget(2000)]).get_strategy("Budget")
    assert result.order_outcomes[0]["status"] == "partial"
    assert result.order_outcomes[0]["reason"] == "insufficient_cash"


@pytest.mark.parametrize(
    "optimizer",
    [
        GridSearchOptimizer(verbose=False),
        RandomSearchOptimizer(n_iter=10, random_state=42, verbose=False),
    ],
)
def test_search_only_evaluates_unique_candidates_and_validates_empty_lists(optimizer):
    result = optimizer.optimize(candles(), config(), BuyAndHoldStrategy, {"shares": [1, 1, 2]})
    assert len(result) == 2
    assert optimizer.summary["attempted_evaluations"] == 2
    assert optimizer.summary["unique_candidates"] == 2
    assert result.attrs["search"]["available_candidates"] == 2
    with pytest.raises(ValueError, match="non-empty"):
        optimizer.optimize(candles(), config(), BuyAndHoldStrategy, {"shares": []})


@pytest.mark.parametrize("field,value", [("n_iter", 0), ("n_iter", True), ("repeats", -1)])
def test_random_search_rejects_invalid_counts(field, value):
    with pytest.raises(ValueError, match="positive integer"):
        RandomSearchOptimizer(**{field: value})


def test_repeated_trials_are_explicit_and_reproducible():
    optimizer = RandomSearchOptimizer(n_iter=10, repeats=3, random_state=42, verbose=False)
    first = optimizer.optimize(candles(), config(), RandomStrategy, {"shares": [1, 2]})
    second = optimizer.optimize(candles(), config(), RandomStrategy, {"shares": [1, 2]})
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 6
    assert first.simulation_seed.nunique() == 6
    assert optimizer.summary["unique_candidates"] == 2
    assert optimizer.summary["attempted_evaluations"] == 6


def test_undefined_objectives_are_not_selected():
    data = candles().assign(Open=100.0, High=100.0, Low=100.0, Close=100.0)
    settings = config(commission_value=0)
    result = Backtest(data, settings).run([BuyAndHoldStrategy(shares=1)])
    for choose in (result.best_strategy, result.worst_strategy):
        with pytest.raises(ValueError, match="undefined"):
            choose("sortino_ratio")
    optimizer = GridSearchOptimizer(verbose=False)
    assert optimizer.optimize(
        data, settings, BuyAndHoldStrategy, {"shares": [1, 2]}, "sortino_ratio"
    ).empty
    assert optimizer.summary["undefined_objectives"] == 2


def test_rankings_exclude_nan_but_preserve_infinity_and_risk_direction():
    values = pd.DataFrame(
        {
            "profit_factor": [float("nan"), 2.0, float("inf")],
            "max_drawdown": [float("nan"), 2.0, 5.0],
        }
    )
    assert rank_results(values, "profit_factor").profit_factor.tolist() == [float("inf"), 2.0]
    assert rank_results(values, "max_drawdown").max_drawdown.tolist() == [2.0, 5.0]


def test_search_provenance_is_detached_from_mutable_parameters():
    class ParameterStrategy(Strategy):
        def __init__(self, settings):
            super().__init__()
            self.settings = settings

        def predict(self, data, trade_history):
            return self.hold()

    settings = {"thresholds": [1, 2]}
    result = GridSearchOptimizer(verbose=False).optimize(
        candles(), config(), ParameterStrategy, {"settings": [settings]}
    )
    settings["thresholds"].append(3)
    assert result.attrs["search"]["parameter_space"]["settings"] == [{"thresholds": [1, 2]}]


def test_metadata_is_detached_and_exports_are_strict_json(tmp_path):
    data = candles()
    settings = config()
    engine = Backtest(data, settings)
    settings.initial_capital = 5
    result = engine.run([Budget()]).get_strategy("Budget")
    assert result.metadata["config"]["initial_capital"] == 1000
    assert result.metadata["strategy"]["parameters"]["amount"] == 1000
    assert result.metadata["random_seed"] == 42
    assert result.metadata["data"]["timezone"] == "UTC"
    mutated = result.metadata
    mutated["config"]["initial_capital"] = 0
    assert result.metadata["config"]["initial_capital"] == 1000
    assert (
        engine.metadata["data"]["fingerprint"]
        == Backtest(data.copy(), config()).metadata["data"]["fingerprint"]
    )
    data.loc[data.index[-1], "Close"] += 1
    assert (
        engine.metadata["data"]["fingerprint"]
        != Backtest(data, config()).metadata["data"]["fingerprint"]
    )
    result.metrics["sortino_ratio"] = float("nan")
    result.metrics["profit_factor"] = float("inf")
    path = tmp_path / "experiment.json"
    result.export_json(path)
    exported = json.loads(path.read_text(), parse_constant=lambda value: pytest.fail(value))
    assert exported["schema_version"] == 1
    assert exported["metrics"]["sortino_ratio"] is None
    assert exported["metrics"]["profit_factor"] == "inf"
    assert len(exported["portfolio_values"]["values"]) == len(result.portfolio_values)
    assert exported["order_outcomes"][0]["status"] == "filled"


def test_strategy_rng_replays_across_runs_and_parallel_execution():
    engine = Backtest(candles(), config())
    first = engine.run([RandomStrategy()]).get_strategy("RandomStrategy")
    second = engine.run([RandomStrategy()]).get_strategy("RandomStrategy")
    pd.testing.assert_series_equal(first.portfolio_values, second.portfolio_values)
    parallel = (
        Backtest(candles(), config(parallel_execution=True, n_jobs=2))
        .run([RandomStrategy(), BuyAndHoldStrategy()])
        .get_strategy("RandomStrategy")
    )
    pd.testing.assert_series_equal(first.portfolio_values, parallel.portfolio_values)


def test_walk_forward_report_preserves_independent_fold_equity_and_trades(tmp_path):
    optimizer = WalkForwardOptimizer(train_size=0.5, n_splits=3, verbose=False)
    report = optimizer.optimize_report(candles(), config(), BuyAndHoldStrategy, {"shares": [1, 2]})
    assert len(report.folds) == 3
    assert report.aggregate_metrics["test_bars"] == 12
    assert report.aggregate_metrics["total_trades"] == 3
    for fold, row in zip(report.folds, report.summary.itertuples()):
        assert fold.result.portfolio_values.iloc[0] == 1000
        assert fold.result.trade_history[0]["timestamp"] == row.test_start
        assert fold.result.returns.index[-1] == row.test_end
        assert fold.parameters["shares"] == row.shares
        assert fold.result.metadata["strategy"]["parameters"]["shares"] == row.shares
    assert report.aggregate_metrics["mean_fold_return"] == pytest.approx(
        report.summary.test_total_return.mean()
    )
    report.export_json(tmp_path / "walk-forward.json")
    payload = json.loads((tmp_path / "walk-forward.json").read_text())
    assert payload["capital_policy"] == payload["position_policy"] == "reset"
    assert len(payload["folds"]) == 3
