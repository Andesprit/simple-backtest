# Reproducible research

This example runs entirely offline. It records execution outcomes, searches
unique parameter combinations with explicit repeated trials, and saves both
single-run and walk-forward reports.

```python
import numpy as np
import pandas as pd

from simple_backtest import (
    Backtest, BacktestConfig, RandomSearchOptimizer, Strategy, WalkForwardOptimizer,
)


class RandomRebalance(Strategy):
    def __init__(self, probability=0.5):
        super().__init__()
        self.probability = probability

    def predict(self, data, trade_history):
        if self.rng.random() >= self.probability:
            return self.hold()
        return self.sell_all() if self.has_position() else self.buy_budget(250)


rng = np.random.default_rng(7)
prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
data = pd.DataFrame(
    {"Open": prices, "High": prices * 1.02, "Low": prices * 0.98,
     "Close": prices * 1.001, "Volume": 1000.0},
    index=pd.date_range("2024-01-01", periods=120, tz="UTC"),
)
config = BacktestConfig(
    initial_capital=1000, lookback_period=1, commission_type="flat",
    commission_value=1, random_seed=42, periods_per_year=365,
    parallel_execution=False,
)
results = Backtest(data, config).run([RandomRebalance()])
strategy = results.get_strategy("RandomRebalance")
print(pd.DataFrame(strategy.order_outcomes).head())
print(strategy.metadata["data"]["fingerprint"])
results.export_json("experiment.json")  # All strategies and the benchmark.
strategy.export_json("strategy.json")   # One strategy, including its provenance.

search = RandomSearchOptimizer(n_iter=10, repeats=3, random_state=42, verbose=False)
ranked = search.optimize(
    data, config, RandomRebalance, {"probability": [0.25, 0.5, 0.75]},
    metric="total_return",
)
assert search.summary["unique_candidates"] == 3
assert search.summary["attempted_evaluations"] == 9
print(ranked[["probability", "trial", "simulation_seed", "total_return"]])
search.export_json(ranked, "search.json")

walk_forward = WalkForwardOptimizer(train_size=0.5, n_splits=3, verbose=False)
report = walk_forward.optimize_report(
    data, config, RandomRebalance, {"probability": [0.25, 0.5, 0.75]},
    metric="total_return",
)
print(report.summary)
print(report.aggregate_metrics)
first_fold = report.folds[0]
assert first_fold.result.portfolio_values.iloc[0] == config.initial_capital
print(first_fold.parameters, first_fold.result.trade_history[:1])
report.export_json("walk-forward.json")
```

## Seeds and recorded inputs

`random_seed` resets each strategy's local NumPy `self.rng` before every run,
including worker processes. Call `super().reset_state()` in overrides. It does
not seed Python's global `random`, global NumPy state, or external estimators;
configure those explicitly. Strategies sharing a seed start with the same RNG
sequence. Random search's `random_state` controls candidate sampling and trial
seeds; `repeats > 1` replaces the configured simulation seed for each trial.
Each trial is returned separately, not averaged before ranking.

Result metadata includes copied configuration, strategy class and constructor
parameters, package/Python/NumPy/pandas versions, seed, input-data fingerprint,
timezone, and effective annualization. The fingerprint covers the normalized
input data, including context bars and feature columns. Retain the source data
and code alongside exports; a fingerprint alone cannot reconstruct them.
Treat an initialized engine's `data` and `config` as immutable; construct a new
engine when changing inputs.

The default `get_parameters()` captures constructor parameters stored as
attributes with the same names. Override it to return portable values when your
strategy renames parameters or accepts `**kwargs`. Uncaptured parameters are
marked explicitly. Estimators exposing `get_params()` have their parameters
recorded; arbitrary objects are described by type. Custom callbacks are
identified by name, not serialized as executable code. This metadata supports
auditing and comparison, not automatic recreation of arbitrary Python objects.

## Fold accounting

Training expands chronologically; each test region is evaluated only after its
parameters are selected. Earlier bars supply lookback context, never test fills.
Cash, positions, strategy state, and RNG reset at the start of every fold.
Open positions are marked at the final close unless `final_liquidation=True`,
which attempts closure using the normal fees and per-order volume limits.

`report.folds` retains complete strategy and benchmark results. Aggregate metrics
include counts, mean/median/worst fold returns, and maximum fold drawdown. Returns
and drawdowns are percentages. Fold averages are unweighted. Do not concatenate
the equity curves and interpret their reset balances as one investable account.
The existing `optimize()` DataFrame API remains available; its detailed report
is also stored in `optimizer.report`.

## Export format

Exports use `schema_version: 1` and strict JSON. Series store aligned `index` and
`values` arrays; DataFrames add `columns`. Timestamps use ISO 8601, undefined
numbers use `null`, and positive/negative infinity use `"inf"` / `"-inf"`.
Trade IDs identify individual fills and may differ across otherwise identical
seeded runs. Search exports include provenance and candidate counts from
`ranked.attrs`; save with the optimizer's exporter to preserve those attributes.
