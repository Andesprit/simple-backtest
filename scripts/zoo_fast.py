"""Run the strategy zoo on one bar file at short intervals: uv run python scripts/zoo_fast.py.

strategy_zoo.py builds 1h, 4h and 1d bars over five years. Minute bars over
five years are too many for its slower indicators, so this driver scores one
CSV at the given intervals over a shorter window split into two halves.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from binance_klines import resample
from strategy_zoo import ZOO, label, simulate, stats


def main() -> None:
    """Print a summary and write one row per run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--intervals", nargs="+", default=["5m"])
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--start", default="2024-10-01")
    parser.add_argument("--middle", default="2025-10-01")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    start, middle = pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.middle, tz="UTC")
    base = pd.read_csv(args.csv, index_col="Date", parse_dates=["Date"])
    rows = []
    for interval in args.intervals:
        d = base if interval == "1m" else resample(base, interval.replace("m", "min"))
        d = d[d.index >= start - pd.Timedelta(days=30)]
        per_year = len(d) / ((d.index[-1] - d.index[0]).days / 365.25)
        windows = {
            "full": d.index >= start,
            "h1": (d.index >= start) & (d.index < middle),
            "h2": d.index >= middle,
        }
        entries = [("benchmark", "Buy and hold", lambda d: pd.Series(1.0, index=d.index), [()], [])]
        for family, name, build, presets, names in entries + ZOO:
            if family == "calendar":
                continue
            for params in presets:
                position = build(d, *params)
                position[d.index.to_series().shift(-1) < start] = 0.0
                returns, held = simulate(position, d, args.fee)
                rows.append(
                    {
                        "id": f"{name} ({label(params, names)}) {interval}",
                        "family": family,
                        "interval": interval,
                    }
                    | {
                        f"{w}_{k}": v
                        for w, m in windows.items()
                        for k, v in stats(returns[m], held[m], per_year).items()
                    }
                )
        print(f"{interval}: done", flush=True)
    table = pd.DataFrame(rows)
    hold = table[table["family"] == "benchmark"].set_index("interval")
    table["beats_hold"] = (
        (table["h1_sharpe"] > table["interval"].map(hold["h1_sharpe"]))
        & (table["h2_sharpe"] > table["interval"].map(hold["h2_sharpe"]))
        & (table["family"] != "benchmark")
    )
    table["positive"] = (
        (table["h1_ret"] > 0) & (table["h2_ret"] > 0) & (table["family"] != "benchmark")
    )
    table.round(4).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
