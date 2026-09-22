"""Verify an installed wheel outside the source checkout."""

from importlib.metadata import version
from math import isclose, isfinite
from pathlib import Path

import pandas as pd

import simple_backtest
from simple_backtest import Backtest, BacktestConfig, BuyAndHoldStrategy
from simple_backtest.visualization.plotter import plot_equity_curve


def main() -> None:
    """Run an import and behavior smoke test against site-packages."""
    package_path = Path(simple_backtest.__file__).resolve()
    if "site-packages" not in package_path.parts:
        raise RuntimeError(f"Expected installed wheel import, got {package_path}")
    if simple_backtest.__version__ != "0.5.0":
        raise RuntimeError(f"Unexpected installed version {simple_backtest.__version__}")
    if version("simple-backtest") != simple_backtest.__version__:
        raise RuntimeError("Wheel metadata and package API versions do not match")

    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = [100.0, 101.0, 102.0, 103.0]
    data = pd.DataFrame(
        {
            "Open": prices,
            "High": [price + 5 for price in prices],
            "Low": prices,
            "Close": [price + 5 for price in prices],
            "Volume": [1000.0] * len(prices),
        },
        index=dates,
    )
    results = Backtest(
        data,
        BacktestConfig.zero_commission(
            initial_capital=1000, lookback_period=1, parallel_execution=False
        ),
    ).run([BuyAndHoldStrategy(shares=1)])
    metric = results.get_strategy("BuyAndHold").metrics["total_return"]
    if not isfinite(metric):
        raise RuntimeError(f"Wheel smoke produced non-finite total_return: {metric}")
    strategy = results.get_strategy("BuyAndHold")
    if strategy.portfolio_values.iloc[0] != 1000 or not isclose(metric, 0.7):
        raise RuntimeError("Wheel smoke failed the starting-cash and closing-equity oracle")
    if not plot_equity_curve(results).data:
        raise RuntimeError("Wheel smoke produced an empty equity plot")

    print(f"validated {simple_backtest.__version__} from {package_path}")


if __name__ == "__main__":
    main()
