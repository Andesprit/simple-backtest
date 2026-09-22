"""Measure accumulation scaling: uv run python scripts/benchmark_accumulation.py."""

import argparse
from statistics import median
from time import perf_counter

import pandas as pd

from simple_backtest import Backtest, BacktestConfig, Strategy


class Accumulate(Strategy):
    """Add one lot per trading bar."""

    def predict(self, data, trade_history):
        return self.buy(1)


def main() -> None:
    """Report median runtime without imposing a hardware-dependent test threshold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", nargs="+", type=int, default=[100, 200, 400, 1000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--snapshots", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or any(bars < 2 for bars in args.bars):
        parser.error("repeats must be positive and bars must be at least 2")
    for bars in args.bars:
        data = pd.DataFrame(
            {column: [10.0] * (bars + 1) for column in ["Open", "High", "Low", "Close"]},
            index=pd.date_range("2020-01-01", periods=bars + 1),
        )
        config = BacktestConfig.zero_commission(
            initial_capital=10.0 * (bars + 1),
            lookback_period=1,
            parallel_execution=False,
            record_position_snapshots=args.snapshots,
        )
        timings = []
        for _ in range(args.repeats):
            started = perf_counter()
            result = Backtest(data, config).run([Accumulate()]).get_strategy("Accumulate")
            timings.append(perf_counter() - started)
            assert len(result.trade_history) == bars
            assert result.metrics["final_value"] == config.initial_capital
        print(f"{bars:>6} bars: {median(timings):.4f}s median; snapshots={args.snapshots}")


if __name__ == "__main__":
    main()
