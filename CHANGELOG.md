# Changelog

All notable user-facing changes are documented here.

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
