<div align="center">

# Simple Backtest

**A small, transparent backtesting framework for long-only strategies**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/LGuillermoAngaritaG/simple-backtest/actions/workflows/ci.yml/badge.svg)](https://github.com/LGuillermoAngaritaG/simple-backtest/actions/workflows/ci.yml)

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Examples](#-examples)

</div>

---

## 📖 About

Simple Backtest is a Python framework designed to make backtesting trading strategies straightforward and accessible. Whether you're testing a simple moving average crossover or a complex machine learning model, Simple Backtest provides the tools you need.

**Key Philosophy**: Bring your own data from any library, API, or file. The
package deliberately provides no market-data source. Inherit from `Strategy`,
`Commission`, or `Optimizer` to supply custom behavior; the built-in classes
and examples are starting points.

Simple Backtest is strictly a simulation library. It does not connect to brokers,
route orders, manage live risk, or operate trading accounts. Read the
[simulation scope and limitations](docs/SIMULATION_SCOPE.md) before using its
results to inform financial decisions.

## ✨ Features

<table>
<tr>
<td width="50%">

### 🚀 Performance
- **Parallel Execution**: Test multiple strategies simultaneously
- **Optimized Core**: Fast backtesting engine with efficient portfolio tracking
- **Efficient Accounting**: Constant-time position totals and optional strategy parallelism

</td>
<td width="50%">

### 📊 Analytics
- **20+ Metrics**: Sharpe, Sortino, Calmar, Win Rate, etc.
- **Benchmark Comparison**: Alpha, Beta, Information Ratio
- **Interactive Visualizations**: Plotly-powered charts

</td>
</tr>
<tr>
<td width="50%">

### 🎯 Design
- **Clean Architecture**: Strategy Pattern for extensibility
- **Type Safety**: Pydantic validation for configurations
- **Explicit Scope**: One long-only, cash-funded instrument per backtest

</td>
<td width="50%">

### 🔧 Flexibility
- **Custom Strategies**: Easy inheritance model
- **Commission Models**: Percentage, flat, tiered, custom
- **Parameter Optimization**: Grid search, random search, walk-forward

</td>
</tr>
</table>

### Supported Assets

Works with one OHLC(V) price series at a time. The accounting model supports
long-only, cash-funded spot instruments; it does not model short selling,
margin, leverage, borrowing, contract multipliers, funding, or FX conversion.

| Asset Type | Support | Notes |
|------------|---------|-------|
| 📈 **Stocks** | ✅ Supported | Long-only; fractional or whole shares |
| ₿ **Spot crypto** | ✅ Supported | Long-only fractional units |
| 📊 **ETFs** | ✅ Supported | Same accounting as stocks |
| 💱 **Forex** | ⚠️ Price signals only | No lots, leverage, rollover, or currency conversion |
| 🛢️ **Commodities** | ⚠️ Price signals only | No physical/contract mechanics |
| 📉 **Futures** | ❌ Accounting unsupported | No margin, multipliers, expiry, or roll logic |
| 📊 **Options** | ❌ No | Requires Greeks, strikes, expiration |

## 📓 Examples

### Interactive Notebooks

Explore comprehensive examples in Jupyter notebooks. Click "Open in Colab" to run them directly in your browser:

| Notebook | Description | Colab Link |
|----------|-------------|------------|
| **01_basic_usage.ipynb** | Introduction, data loading, commission setup, strategy comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/01_basic_usage.ipynb) |
| **02_candle_strategies.ipynb** | Candlestick patterns (Engulfing, Hammer, Doji, etc.) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/02_candle_strategies.ipynb) |
| **03_ta_strategies.ipynb** | Technical indicators (RSI, MACD, Bollinger Bands, etc.) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/03_ta_strategies.ipynb) |
| **04_ml_strategies.ipynb** | Machine learning strategies (Logistic Regression, Random Forest, Gradient Boosting) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/04_ml_strategies.ipynb) |
| **05_commission_usage.ipynb** | Commission models comparison and custom implementations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/05_commission_usage.ipynb) |
| **06_advanced_optimization.ipynb** | Grid search, random search, walk-forward optimization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LGuillermoAngaritaG/simple-backtest/blob/main/notebooks/06_advanced_optimization.ipynb) |


## 📦 Installation

```bash
# Using pip
pip install simple-backtest

# Using uv (recommended)
uv add simple-backtest

# From source
git clone https://github.com/LGuillermoAngaritaG/simple-backtest.git
cd simple-backtest
uv sync --extra dev --extra notebooks
```

**Requirements**: Python 3.10+

## 🚀 Quick Start

This example runs offline after installing `simple-backtest`. Its synthetic prices
make the output reproducible; they are not historical investment performance.

```python
import pandas as pd
from simple_backtest import Backtest, BacktestConfig, BuyAndHoldStrategy

data = pd.DataFrame(
    {
        "Open": [100.0, 100.0, 100.0, 100.0, 100.0],
        "High": [130.0, 130.0, 130.0, 130.0, 130.0],
        "Low": [90.0, 90.0, 90.0, 90.0, 90.0],
        "Close": [100.0, 110.0, 120.0, 115.0, 125.0],
    },
    index=pd.date_range("2024-01-01", periods=5),
)
config = BacktestConfig.zero_commission(
    initial_capital=1000, lookback_period=1, parallel_execution=False
)
strategy = BuyAndHoldStrategy(shares=1)
result = Backtest(data, config).run([strategy]).get_strategy("BuyAndHold")
print(f"Final value: ${result.metrics['final_value']:.2f}")
print(f"Total return: {result.metrics['total_return']:.2f}%")
```

```text
Final value: $1025.00
Total return: 2.50%
```

The strategy buys one share at the second bar's $100 open. The remaining $900
cash plus that share at the final $125 close gives $1,025. Replace `data` with
your own OHLC DataFrame; add Volume when using volume participation limits.
For moving-average examples below, supply enough rows for their lookback windows.

To use your own CSV with a `Date` column:

```python
data = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)
```

The engine accepts OHLCV names in any capitalization on an internal copy. It
requires sorted dates, positive numeric OHLC prices and no missing prices. For
partly missing bars, review and omit incomplete rows with
`data.dropna(subset=['Open', 'High', 'Low', 'Close'])`; filling individual prices
can put them outside that day's High/Low range. If every gap row is missing all
four prices, you may choose `data.sort_index().ffill().dropna(subset=['Open', 'High', 'Low', 'Close'])`
to carry whole bars forward instead, after reviewing whether that assumption is appropriate.
If your only closing-price column is `Adj Close`, supply unadjusted `Close` or
consistently adjusted OHLC from your provider. Rename `Adj Close` to `Close` only
when Open/High/Low are already adjusted to the same basis; a rename cannot recover
missing raw prices or adjustment factors. When both closing-price columns are
present, the engine uses `Close` unchanged.
`validate_dataframe(data)` returns a normalized copy and leaves `data` unchanged.

## 📚 Documentation

### Creating a Custom Strategy

Implement your own strategy by inheriting from `Strategy` and defining the `predict()` method:

```python
from simple_backtest import Strategy

class MyStrategy(Strategy):
    """Custom trading strategy."""

    # The engine and optimizers reject configurations with shorter lookbacks.
    required_history = 20

    def __init__(self, threshold=100, name=None):
        super().__init__(name=name or "MyStrategy")
        self.threshold = threshold

    def predict(self, data, trade_history):
        """Generate trading signal.

        Args:
            data: OHLCV DataFrame with lookback window
            trade_history: List of past trades

        Returns:
            Dict with keys: signal ("buy"/"hold"/"sell"), size, order_ids
        """
        current_price = data['Close'].iloc[-1]

        # Simple logic: buy below threshold, sell above
        if current_price < self.threshold and not self.has_position():
            return self.buy(10)  # Buy 10 shares
        elif current_price > self.threshold * 1.2 and self.has_position():
            return self.sell_all()  # Sell all positions
        else:
            return self.hold()  # Do nothing
```

**Strategy Helper Methods:**
- `self.has_position()` - Check if holding any shares
- `self.get_position()` - Get current share count
- `self.get_cash()` - Get available cash
- `self.get_portfolio_value()` - Get total portfolio value
- `self.buy(shares)` - Return buy signal
- `self.sell(shares)` - Return sell signal
- `self.sell_all()` - Sell all positions
- `self.buy_percent(percent)` - Buy shares worth % of portfolio
- `self.buy_cash(amount)` - Buy shares worth specific amount
- `self.buy_budget(amount)` - Spend at most this amount, including fees and slippage

### Configuration Presets

Quick configurations for common scenarios:

```python
from simple_backtest import BacktestConfig

# Zero commission (for testing)
config = BacktestConfig.zero_commission(initial_capital=10000)

# Dense bar-data preset (not a latency/order-book HFT simulator)
config = BacktestConfig.high_frequency(initial_capital=100000)

# Swing trading (longer lookback, typical retail commission)
config = BacktestConfig.swing_trading(initial_capital=10000)

# Low percentage commission preset (0.01%)
config = BacktestConfig.low_commission(initial_capital=10000)
```

### Comparing Multiple Strategies

Print the result to see returns, risk and trading activity together. This example
uses reproducible synthetic prices; replace `data` with your own OHLC DataFrame.

```python
import numpy as np
import pandas as pd
from simple_backtest import (
    Backtest,
    BacktestConfig,
    MovingAverageStrategy,
    BuyAndHoldStrategy,
    DCAStrategy
)

rng = np.random.default_rng(0)
close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 300)))
data = pd.DataFrame(
    {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close},
    index=pd.date_range("2023-01-01", periods=300),
)

# Create strategies
strategies = [
    MovingAverageStrategy(short_window=10, long_window=30, shares=10),
    BuyAndHoldStrategy(shares=50),
    DCAStrategy(investment_amount=500, interval_days=30)
]

# Run backtest
config = BacktestConfig.default(initial_capital=10000, parallel_execution=False)
backtest = Backtest(data, config)
results = backtest.run(strategies)

print(results)
```

```text
Strategy    Return    CAGR  Max Drawdown  Sharpe  Trades
--------------------------------------------------------
benchmark    7.53%  10.32%        14.38%    0.49       1
BuyAndHold   3.65%   4.97%         7.76%    0.33       1
MA_10_30    -0.77%  -1.04%         2.46%   -1.70      12
DCA_30d     -2.89%  -3.89%         5.35%   -1.00       9

Best total return: BuyAndHold (3.65%). It trailed benchmark by 3.88 percentage points.
```

"Best total return" ranks the strategies by return, excluding the benchmark; it
does not rank their risk. The notebook display renders the same table, verdict
and any rejected-order explanations. Printing an individual strategy result
also shows its final value and win statistics; win rate and profit factor are
`n/a` when no trades have closed.

For a numeric DataFrame, a different ranking metric, or an equity chart:

```python
comparison = results.compare()  # Original numeric metrics, without display formatting
best = results.best_strategy('sharpe_ratio')
print(f"Best: {best.name} (Sharpe: {best.metrics['sharpe_ratio']:.2f})")
results.plot_comparison().show()
```

To see each strategy's fills on the price chart, pass the same price DataFrame
you gave to `Backtest`:

```python
results.get_strategy('BuyAndHold').plot_trades(data).show()  # One strategy

for name, fig in results.plot_trades(data).items():  # Every strategy, without the benchmark
    fig.show()
```

Each chart covers the whole trading period, including bars after the last fill.
A strategy with no fills still gets a chart. A note below the chart says whether
the strategy placed no orders, its orders were rejected (with the main reason and
a sizing hint), or it raised errors (full details stay in `result.errors`). If
the price data misses some backtest dates, the chart title says so; if it covers
none of them, `plot_trades` raises a `ValueError`.

### Sharing a Report

Write everything above into one HTML file that opens offline in any browser,
without Python:

```python
results.export_html("report.html", price_data=data)  # Returns the written path
```

The report has the comparison table and verdict, what to check for each
strategy (no orders, rejected orders, errors), its key metrics and first fills,
the equity and drawdown chart, one trade chart per strategy, and the recorded
assumptions: capital, input and trading dates, execution price, costs, seed and
data fingerprint. Values that differ between strategies are listed per strategy;
a value an older result never recorded shows as "not recorded".
`price_data` is optional: without it the report skips the trade charts and says
how to add them. A single result works the same way, for example
`optimizer.best_result.export_html("best.html", price_data=data)`.

The tables are plain HTML, so the answers stay readable even if the charts do
not load. Exporting never re-runs a strategy. Long error messages and long
fill lists are shortened; `results.export_json(path)` keeps every record. The
file embeds the Plotly chart library once (about 4.8 MB), so it needs no
internet connection; each extra chart adds only its data (tens of KB here).
The numbers are simulated on your data, not a forecast; a run picked by a
parameter search is in-sample.

### Parameter Optimization

Find strategy parameters with a built-in optimizer, then inspect the run that won:

```python
import numpy as np
import pandas as pd
from simple_backtest import BacktestConfig, GridSearchOptimizer, MovingAverageStrategy

rng = np.random.default_rng(42)
close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, 250)))
data = pd.DataFrame(
    {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close},
    index=pd.date_range("2024-01-01", periods=250),
)

optimizer = GridSearchOptimizer(verbose=False)
results = optimizer.optimize(
    data=data,
    config=BacktestConfig.default(lookback_period=60),
    strategy_class=MovingAverageStrategy,
    param_space={"short_window": [5, 10, 20], "long_window": [30, 60], "shares": [10]},
    metric="sharpe_ratio",
)
print(results.head(5))         # Ranked candidates, best first

print(optimizer)               # Objective, winner, score, and evaluated/failed counts
best = optimizer.best_result   # The winning run itself, not a new simulation
print(best)                    # Also best.plot_trades(data), best.export_html("best.html", price_data=data)
optimizer.best_parameters      # {'short_window': ..., 'long_window': ..., 'shares': 10}
```

`best_parameters` keeps the exact values from `param_space` (integers stay
integers), so `MovingAverageStrategy(**optimizer.best_parameters)` is ready for
the next experiment. Copying a row of `results` instead adds metric columns and
turns integers into floats.

If no candidate has a defined objective (for example `sortino_ratio` on flat
prices with `risk_free_rate=0`), `optimize` returns an empty DataFrame and warns
once with the counts; retry with `metric="total_return"`. `print(optimizer)`
gives the same explanation, and `best_result` raises a `ValueError` instead of
returning an old winner. A misspelled objective stops after one evaluation and
suggests the closest metric name. Failed candidates stay in `optimizer.failures`.

**Available Optimizers:**
- `GridSearchOptimizer` - Exhaustive search (best for small spaces)
- `RandomSearchOptimizer` - Random sampling (faster for large spaces)
- `WalkForwardOptimizer` - Expanding training windows with chronological out-of-sample folds

Set each strategy's `required_history` to the minimum number of rows its
indicators need. Optimizers record parameter combinations that exceed
`lookback_period` as failed candidates instead of running invalid simulations.
Random search samples without replacement, up to `n_iter` unique combinations;
duplicate parameter values are removed and empty candidate lists are rejected.
Use an explicit `random_state` for reproducible candidate selection. Set
`BacktestConfig(random_seed=42)` and use `self.rng` in stochastic strategies.
`RandomSearchOptimizer(repeats=3, random_state=42)` runs three independently
seeded trials per candidate and includes `trial` and `simulation_seed` columns.
Repeated trials are ranked individually, so inspect their distribution before
choosing parameters: with repeats, `best_result` is the single best in-sample
trial, and `print(optimizer)` names its trial and seed. `optimizer.summary` reports attempted evaluations, unique
candidates, failures, and excluded undefined objectives.

To see the evidence behind the winner, group the trials per tested combination
(no new simulations):

```python
summary = optimizer.trial_summary()  # One row per tested combination
print(summary)                       # attempted/defined/undefined/failed, <metric>_min/_median/_max
optimizer.plot_trials().show()       # Every observed trial per combination; star = selected trial
```

For example, a strategy that wins big in 1 of 4 seeds can rank first
(`total_return_max` 2.41%) while its `total_return_median` is 0%, next to a
steady combination at 0.48% in every repeat. `contains_selected` marks the
combination holding the winning trial and `selected_trial` names that single
trial; the aggregates never pick a different winner. Failed and undefined
evaluations keep their rows and a `note` (also after a search where every
candidate failed). Values are in-sample observations in the objective's units
(`summary.attrs["trial_summary"]` gives unit and direction), not a robustness
estimate. The full per-trial rows stay in the ranked DataFrame, failures in
`optimizer.failures`, and the winning run in `optimizer.best_result`.
Combinations are grouped as the search deduplicated them, so values that
compare equal (such as `1`, `1.0` and `True`) count as one.

`WalkForwardOptimizer.optimize()` still returns a chronological DataFrame.
It has no single winner: `best_result` points you to `optimizer.report`.
Use `optimize_report()` for selected parameters, strategy and benchmark equity,
trades, and diagnostics for each fold. Every test fold starts with fresh initial
capital and empty positions; aggregate metrics summarize independent folds and
do not represent a continuous portfolio.

See the executable [research guide](docs/REPRODUCIBLE_RESEARCH.md) for seeded
experiments, detailed walk-forward reports, and JSON exports.

### Custom Commission Models

Create custom commission structures:

```python
from simple_backtest import Commission

class TieredWithMinimum(Commission):
    """Tiered commission with minimum fee."""

    def __init__(self):
        super().__init__(name="TieredMin")

    def calculate(self, shares, price):
        trade_value = shares * price

        if trade_value < 1000:
            commission = max(trade_value * 0.002, 1.0)  # 0.2%, min $1
        elif trade_value < 10000:
            commission = trade_value * 0.001  # 0.1%
        else:
            commission = trade_value * 0.0005  # 0.05%

        return commission

# Pass custom behavior explicitly to Backtest
from simple_backtest import Backtest, BacktestConfig

config = BacktestConfig.default(
    commission_type="custom",
    commission_value=0.0,
)
backtest = Backtest(data, config, commission_calculator=TieredWithMinimum())
results = backtest.run([strategy])
```

Custom commission callbacks must be deterministic and side-effect free because
the engine evaluates them for benchmark affordability as well as strategy fills.
Budget sizing uses bisection and requires total cost (quantity times price plus
commission) to be nondecreasing with quantity. The built-in fee models satisfy
this requirement.

For a custom execution price, set `execution_price="custom"` and pass
`execution_price_extractor=` to `Backtest`. Strategy exceptions raise with
strategy/date/stage context by default; use `error_policy="continue"` only when
you intentionally want structured diagnostics in `StrategyResult.errors`.

### Execution and Timing Assumptions

Signals receive only rows strictly before the execution bar. Inside `predict()`,
portfolio valuation and the `buy_cash()` / `buy_percent()` sizing helpers use the
last available close. They never use the execution bar's price. These helpers
produce fixed share quantities: price gaps, fees, or slippage can make an order
unaffordable, in which case it is rejected. Trade callbacks receive completed fills. Orders fill at the
configured bar price (`open`, `close`, `typical`, or `custom`). The legacy
`vwap` option remains as a deprecated alias for the OHLC typical price
`(high + low + close) / 3`; one OHLCV bar is not enough to calculate true VWAP.

Use `buy_budget(amount)` to request a spending cap instead of a fixed quantity.
The engine sizes it at execution, including commission, spread, and slippage,
without exposing the execution price to `predict()`. Cash and volume limits can
reduce the fill. DCA uses budget orders, so its `investment_amount` now includes
fees and can invest the remaining cash without an avoidable rejection.

`StrategyResult.order_outcomes` records each non-hold attempt with its requested
shares or budget, filled shares, `filled` / `partial` / `rejected` status, and a
reason such as `insufficient_cash`, `insufficient_shares`, `zero_volume`,
`volume_limit`, or `zero_size`. Rejections are ordinary execution outcomes;
invalid predictions and callback failures follow `error_policy` and appear in
`errors` when continuing is enabled.

Execution realism is deterministic and opt-in:

```python
config = BacktestConfig.default(
    slippage_bps=5,                 # adverse to both buys and sells
    spread_bps=10,                  # half-spread applied in each direction
    max_volume_participation=0.05,  # at most 5% of the execution bar's volume
    final_liquidation=True,         # attempt to close on the final bar
)
```

When volume participation caps an order, only the capped quantity is filled and
the unfilled remainder is cancelled; the engine does not maintain resting
orders. `final_liquidation=False` is the default, so open positions remain
marked to the final close. Enabling it applies the same cost and volume rules to
both strategies and the benchmark.

Equity is marked at each bar's Close, independently of the chosen execution
price. `portfolio_values` starts with a cash-only baseline at the timestamp of
the last bar before trading. `returns` therefore includes the first trading
bar's costs and price movement. The baseline is excluded from exposure time.
Only bars inside the requested trading interval can execute orders.

Trade records contain cash and `total_shares` after each fill. Set
`record_position_snapshots=True` to also include the full `positions` dictionary
in each trade. This optional detail can require quadratic storage for strategies
that accumulate many lots; the default keeps records compact.

Annualized metrics infer observations per year from the data's timestamp span.
For short, irregular, or mixed-frequency data, set `periods_per_year`
explicitly. DCA intervals are measured from successful fills, not rejected or
zero-sized attempts.

### Logging Control

Control framework verbosity:

```python
from simple_backtest.utils import setup_logging, disable_logging, enable_debug_logging
import logging

# Default: WARNING level (minimal output)

# For verbose output during optimization
setup_logging(level=logging.INFO)

# For debugging issues
enable_debug_logging()

# To suppress all output
disable_logging()
```

## 📊 Performance Metrics

The framework calculates 20+ metrics automatically:

Sortino uses the root mean square shortfall below the periodic risk-free target,
including zero shortfalls for observations above target. With no downside its
denominator is zero and the ratio is `NaN`. Optimizer rankings and best/worst
selection exclude undefined (`NaN`) objectives; selection raises if all values
are undefined. Legitimate infinite metrics, such as profit factor with gains
and no losses, remain rankable.

### Returns
- Total Return (%)
- CAGR (Compound Annual Growth Rate)

### Risk Metrics
- Volatility (annualized standard deviation)
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk-adjusted return)
- Calmar Ratio (return vs max drawdown)
- Max Drawdown (%)
- Max Drawdown Duration

### Trade Statistics
- Total Trades
- Win Rate (%)
- Profit Factor
- Trade Expectancy
- Average Win / Average Loss

### Benchmark Comparison
- Alpha (excess return vs benchmark)
- Beta (correlation with benchmark)
- Information Ratio


## 🛠️ Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/LGuillermoAngaritaG/simple-backtest.git
cd simple-backtest

# Install with uv (recommended)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=simple_backtest

# Run specific test file
uv run pytest tests/test_strategy.py

# Run specific test
uv run pytest tests/test_strategy.py::test_strategy_initialization
```

### Executable Examples and Performance

```bash
uv sync --extra dev --extra notebooks
uv run python scripts/smoke_notebooks.py
uv run python scripts/benchmark_accumulation.py
```

The notebook check executes all six tutorials on seeded synthetic candles,
skipping only tagged installation cells. Executed notebooks, including failure
output, are saved to `build/notebooks/`. CI runs these checks before producing
release artifacts. Publishing reuses the full CI workflow for the release commit
and uploads those validated artifacts to PyPI.

### Code Quality

```bash
# Lint code
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Format code
uv run ruff format .

# Run pre-commit hooks
pre-commit run --all-files
```

### Pre-commit Hooks

Pre-commit hooks automatically run linting, formatting, and tests on commit:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🤝 Contributing

Contributions are welcome! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Run tests**: `uv run pytest`
5. **Run linting**: `uv run ruff check . && uv run ruff format --check .`
6. **Commit your changes**: `git commit -m "Add amazing feature"`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Pydantic](https://docs.pydantic.dev/) for configuration validation
- Uses [Plotly](https://plotly.com/) for interactive visualizations
- Parallel processing with [Joblib](https://joblib.readthedocs.io/)
- Testing with [Pytest](https://docs.pytest.org/)
- Code quality with [Ruff](https://github.com/astral-sh/ruff)

## 📬 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/LGuillermoAngaritaG/simple-backtest/issues)
- **Discussions**: [GitHub Discussions](https://github.com/LGuillermoAngaritaG/simple-backtest/discussions)
- **Email**: guille2005_13@hotmail.com
