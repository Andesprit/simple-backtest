"""Walk-forward test of the intraday grids: uv run python scripts/walk_forward.py.

Every step months, each strategy re-picks its parameters by Sharpe on the
trailing lookback window and trades the next step months with them. The
stitched out-of-sample returns are compared with buy-and-hold over the same
months. 4h and 1d bars are built from the 1h CSV.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from binance_klines import resample
from research_intraday import GRIDS

BARS_PER_YEAR = {"1h": 24 * 365, "4h": 6 * 365, "1d": 365}


def bar_returns(position: pd.Series, data: pd.DataFrame, fee: float) -> pd.Series:
    """Per-bar net return of a close-decided position filled at the next open."""
    held = position.shift(1).fillna(0.0)
    bar = (data["Open"].shift(-1) / data["Open"] - 1).fillna(0.0)
    return held * bar - held.diff().abs().fillna(held.iloc[0]) * fee


def sharpe(returns: pd.Series, per_year: int) -> float:
    """Annualized Sharpe of per-bar returns; zero when flat."""
    std = returns.std()
    return returns.mean() / std * np.sqrt(per_year) if std > 0 else 0.0


def score(returns: pd.Series, per_year: int) -> dict:
    """Total return, Sharpe and max drawdown of a return series."""
    equity = (1 + returns).cumprod()
    return {
        "return": equity.iloc[-1] - 1,
        "sharpe": sharpe(returns, per_year),
        "max_dd": (equity / equity.cummax() - 1).min(),
    }


def walk(data: pd.DataFrame, per_year: int, fee: float, lookback: int, step: int) -> pd.DataFrame:
    """Return one row per strategy plus buy-and-hold over the stitched test months."""
    starts = pd.date_range(pd.Timestamp("2022-01-01", tz="UTC"), data.index[-1], freq=f"{step}MS")
    hold = bar_returns(pd.Series(1.0, index=data.index), data, 0.0)
    rows = []
    for name, (build, grid) in GRIDS.items():
        candidates = [bar_returns(build(data, **params), data, fee) for params in grid]
        pieces, picks = [], []
        for start in starts:
            end = start + pd.DateOffset(months=step)
            train = (data.index >= start - pd.DateOffset(months=lookback)) & (data.index < start)
            test = (data.index >= start) & (data.index < end)
            best = max(range(len(grid)), key=lambda i: sharpe(candidates[i][train], per_year))
            pieces.append(candidates[best][test])
            picks.append(str(grid[best]))
        stitched = pd.concat(pieces)
        rows.append({"strategy": name, "picks": len(set(picks))} | score(stitched, per_year))
    span = data.index >= starts[0]
    rows.append({"strategy": "buy_hold", "picks": 0} | score(hold[span], per_year))
    return pd.DataFrame(rows)


def main() -> None:
    """Print walk-forward results for 1h, 4h and 1d bars at each fee."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--lookback", type=int, default=24, help="training months")
    parser.add_argument("--step", type=int, default=6, help="months traded per pick")
    args = parser.parse_args()

    hourly = pd.read_csv(args.data / "BTCUSDT-1h.csv", index_col="Date", parse_dates=["Date"])
    frames = {"1h": hourly, "4h": resample(hourly, "4h"), "1d": resample(hourly, "1D")}
    for interval, data in frames.items():
        for fee in [0.001, 0.00075]:
            table = walk(data, BARS_PER_YEAR[interval], fee, args.lookback, args.step)
            print(f"\n{interval}, fee {fee:.4%}")
            print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
