# Simulation scope and release readiness

Simple Backtest is strictly a historical backtesting library. It does not
connect to brokers, route live orders, manage credentials, or autonomously
trade. Those capabilities are intentionally outside the project.

## What 0.4.0 is prepared to claim

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

The honest interpretation is narrow: the library estimates what a rule would
have done on the supplied historical data under the configured simulation
assumptions.
