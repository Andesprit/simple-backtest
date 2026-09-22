# Changelog

All notable user-facing changes are documented here.

## [Unreleased]

## [0.5.0] - 2026-09-22

### Fixed

- Sortino uses root mean square shortfall across all observations rather than
  the sample standard deviation of negative returns.
- Strategy portfolio helpers and sizing use the prior close, preventing execution
  prices from leaking into predictions. Fills still use the configured price.
- Equity uses each bar's close for both strategies and the benchmark.
- First-bar costs and returns are included in drawdown and risk calculations.
- Invalid predictions cannot execute under `error_policy="continue"`.
- Date bounds select only bars within the requested interval and reject empty intervals.
- Trade notifications copy only the new record instead of the complete history.
- The ML tutorial now supplies enough history to train, excludes the unknown final
  training label, and asserts that models are trained.

### Changed

- DCA invests a total budget including fees, sized at the execution price.
- Sortino is undefined (`NaN`) when downside deviation is zero. Rankings exclude
  undefined objectives, and best/worst selection rejects all-undefined inputs.
- Random searches sample unique candidates without replacement; both search
  implementations deduplicate parameter values and reject empty candidate lists.
- `portfolio_values` now includes starting cash at the last timestamp before
  trading. `returns` includes the first trading bar; exposure excludes the baseline.
- Trade records include `total_shares`. Full `positions` snapshots are now opt-in:
  set `BacktestConfig(record_position_snapshots=True)` or pass that keyword to
  `Portfolio` to retain the earlier trade-record detail.
- `buy_cash()` and `buy_percent()` produce quantities using the prior close.
  Execution gaps and costs may cause those fixed-quantity orders to be rejected.
- Releases must pass the complete reusable CI workflow, including notebook
  execution, and publish its validated distributions without rebuilding them.

### Added

- `Strategy.buy_budget()` and structured order outcomes for fills, partial fills,
  and normal rejections, separate from strategy execution errors.
- Explicit repeated random-search trials, local seeded strategy RNGs, and
  attempted/unique/successful/failed evaluation counts.
- Detailed walk-forward reports with per-fold equity, trades, selected parameters,
  benchmarks, and aggregate statistics under explicit reset-per-fold accounting.
- Experiment provenance and versioned JSON exports for backtest results,
  parameter searches, and walk-forward reports.
- Generated cash, quantity, FIFO cost, and equity conservation tests across
  fee models, volume constraints, partial sales, and liquidation policies.
- An offline README quick start checked against its documented output.
- Real execution of all six notebooks with deterministic synthetic data in CI.
- A repeatable accumulation benchmark and regression coverage for timing,
  valuation, boundaries, rejected predictions, and history copying.

### Migration notes

- DCA's `investment_amount` is now the total spending cap, including fees.
  Use `buy_budget()` for cash-capped orders; `buy_cash()` still fixes quantity
  using the prior close.
- Equity has one extra pre-trading cash observation. Align per-bar analytics
  with `returns`, which now includes the first trading bar.
- Enable `record_position_snapshots=True` if consuming `positions` in trades.
- Handle `NaN` Sortino values when there is no downside. Rankings omit undefined
  objectives; selecting from all-undefined values raises an error.
- Random searches no longer repeat candidates implicitly. Use `repeats=` for
  independent trials and `self.rng` with `random_seed` for reproducible strategies.
- Walk-forward folds reset cash and positions; their aggregate report describes
  independent tests, not a continuously funded portfolio.

## [0.4.0] - 2026-07-28

### Added

- Deterministic adverse slippage and spread simulation through `slippage_bps`
  and `spread_bps`; both default to zero.
- Optional per-order bar-volume caps through `max_volume_participation`.
- An engine-level `final_liquidation` policy applied consistently to strategies
  and the internal benchmark.
- Explicit `Strategy.required_history` validation so optimizers do not rank
  candidates that can never receive enough data.
- Hand-calculated accounting, timing, visualization, notebook, and installed
  wheel verification.

### Changed

- Built-in Buy and Hold and DCA strategies no longer force a final-bar sale.
  Set `final_liquidation=True` when realized end-of-test positions are required.
- Annualization is inferred from observed samples per calendar year. Set
  `periods_per_year` explicitly for short or irregular datasets.
- `execution_price="typical"` now names the OHLC typical-price calculation
  accurately.
- DCA contribution intervals advance only after a purchase actually fills.
- Reusing a seeded `RandomSearchOptimizer` now reproduces its samples.

### Fixed

- Reject the reserved strategy name `benchmark` instead of silently losing
  strategy results.
- Apply configured execution costs and participation constraints consistently
  to strategy and benchmark fills.
- Validate built distributions by importing and running them outside the source
  checkout.

### Deprecated

- `execution_price="vwap"` remains as an alias for `"typical"` through 0.4.x.
  A single OHLC(V) bar cannot provide true VWAP.

### Removed

- The previously exposed caching utility, removed during the reliability
  rewrite after 0.3.0, is not part of the 0.4.0 API.

### Migration notes

- Replace `execution_price="vwap"` with `"typical"`.
- If a built-in strategy must close at the end, set
  `BacktestConfig(..., final_liquidation=True)`.
- Ensure `lookback_period >= strategy.required_history`; for
  `MovingAverageStrategy`, this means at least `long_window`.
