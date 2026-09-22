# Simulation scope and release readiness

Simple Backtest is strictly a historical backtesting library. It does not
connect to brokers, route live orders, manage credentials, or autonomously
trade. Those capabilities are intentionally outside the project.

## What 0.5.0 is prepared to claim

- Deterministic, prior-bar strategy evaluation for one long-only,
  cash-funded instrument.
- FIFO position accounting with entry and exit commission allocation.
- Explicit mark-to-market or final-liquidation policy.
- Optional deterministic spread, slippage, and static bar-volume constraints.
- Reproducible optimizers, chronological walk-forward evaluation, and
  consistently annualized metrics.
- Offline tests for numerical accounting, timing, plots, notebooks, packaging,
  and installed-wheel behavior.

These properties make the package suitable for reproducible research and
simulation under its documented assumptions. They do not make a strategy safe
or likely to perform in the future.

## What the simulation does not establish

- Input features are free of look-ahead, survivorship, selection, or corporate
  action bias. The engine hides the execution bar, but it cannot audit columns
  supplied by the user.
- Results survive regime changes, multiple-testing bias, or live market
  microstructure.
- A static basis-point cost and volume cap reproduce queues, latency, partial
  fills across bars, exchange rejections, halts, or order types.
- Taxes, dividends, splits, borrow, margin, leverage, funding, FX conversion,
  contract multipliers, or multi-asset portfolio interactions are modeled.

## Decisions outside this library

Anyone considering real-money activity must independently choose and validate
broker/exchange integrations, position and loss limits, kill switches,
reconciliation, monitoring, incident response, audit trails, security controls,
market-data licensing, and applicable legal, tax, and regulatory obligations.
Simple Backtest neither supplies nor validates those systems.

## Equity and signal timing

`predict()` receives prior bars only. Portfolio helpers use the last visible
close, including the reference price for fixed-quantity sizing. Execution uses
the configured bar price, while reported equity uses that bar's close. Price
gaps and costs can cause a sized order to be rejected for insufficient cash.
An explicit `buy_budget()` instead sizes at execution with a total spending cap
including costs. DCA uses this budget behavior. Order outcomes make ordinary
cash, holdings, and volume rejections visible separately from programming errors.

The equity series includes a cash-only baseline at the last timestamp before
trading, followed by one close valuation per trading bar. This includes entry
costs in first-period returns and drawdowns. The baseline is excluded from
exposure time and is not a trading bar. Date limits constrain actual fills;
neither boundary is rounded to a bar outside the requested interval.

Trade records are compact by default. Full open-position snapshots require
`record_position_snapshots=True` and can use quadratic storage when many lots
accumulate. These snapshots are historical copies, not live portfolio references.

## Research reports

Walk-forward reports retain independent test folds, each starting with fresh
cash and positions. Aggregate statistics summarize those folds; they do not
simulate one continuous portfolio. Seeded strategies must use `self.rng` for
the engine's replay guarantee. External random sources need their own seeds.

Versioned JSON exports include configuration, strategy parameters, seeds,
annualization, timezone, and a normalized input-data fingerprint. They preserve
results for inspection, but do not package source data or executable callbacks.
See [Reproducible research](REPRODUCIBLE_RESEARCH.md) for a tested offline example
and the precise capture and fold-accounting rules.

The honest interpretation is narrow: the library estimates what a rule would
have done on the supplied historical data under the configured simulation
assumptions.
