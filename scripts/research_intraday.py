"""Grid-test intraday BTC strategies: uv run python scripts/research_intraday.py.

Reads the CSVs written by binance_klines.py. Each strategy emits a long/flat
position from bar closes; the position fills at the next bar's open and earns
open-to-open returns. Every position change pays the fee once. The loop is
vectorized because the event-driven engine is too slow for a grid over
700,000 five-minute bars.

Parameters are picked on the train window by Sharpe and judged on the test
window, which the selection never sees.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

BARS_PER_YEAR = {"1h": 24 * 365, "10m": 6 * 24 * 365, "5m": 12 * 24 * 365}


def rsi(close: pd.Series, length: int) -> pd.Series:
    """Wilder RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def latch(enter: pd.Series, leave: pd.Series) -> pd.Series:
    """Hold 1 from an entry bar until an exit bar; exits win ties."""
    state = pd.Series(np.nan, index=enter.index)
    state[enter] = 1.0
    state[leave] = 0.0
    return state.ffill().fillna(0.0)


def ma_cross(data: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Long while the fast SMA is above the slow SMA."""
    close = data["Close"]
    return (close.rolling(fast).mean() > close.rolling(slow).mean()).astype(float)


def rsi_revert(data: pd.DataFrame, length: int, low: int, high: int) -> pd.Series:
    """Buy when RSI drops below low, sell when it rises above high."""
    value = rsi(data["Close"], length)
    return latch(value < low, value > high)


def bollinger(data: pd.DataFrame, length: int, width: float) -> pd.Series:
    """Buy a close below the lower band, sell a close above the middle band."""
    close = data["Close"]
    mid = close.rolling(length).mean()
    lower = mid - width * close.rolling(length).std()
    return latch(close < lower, close > mid)


def breakout(data: pd.DataFrame, enter: int, leave: int) -> pd.Series:
    """Buy a close above the prior enter-bar high, sell below the prior leave-bar low."""
    close = data["Close"]
    top = data["High"].rolling(enter).max().shift(1)
    bottom = data["Low"].rolling(leave).min().shift(1)
    return latch(close > top, close < bottom)


# Lookbacks are in bars of the interval under test.

GRIDS = {
    "ma_cross": (
        ma_cross,
        [
            {"fast": f, "slow": s}
            for f, s in itertools.product([5, 10, 20, 50], [50, 100, 200, 400])
            if f < s
        ],
    ),
    "rsi_revert": (
        rsi_revert,
        [
            {"length": n, "low": lo, "high": hi}
            for n, lo, hi in itertools.product([7, 14, 28], [20, 25, 30], [50, 60, 70])
        ],
    ),
    "bollinger": (
        bollinger,
        [
            {"length": n, "width": w}
            for n, w in itertools.product([20, 50, 100], [1.5, 2.0, 2.5, 3.0])
        ],
    ),
    "breakout": (
        breakout,
        [
            {"enter": e, "leave": x}
            for e, x in itertools.product([20, 50, 100, 200], [10, 20, 50, 100])
        ],
    ),
}


def evaluate(position: pd.Series, data: pd.DataFrame, fee: float, per_year: int) -> dict:
    """Score a close-decided position that fills at the next open."""
    held = position.shift(1).fillna(0.0)
    bar = data["Open"].shift(-1) / data["Open"] - 1
    turns = held.diff().abs().fillna(held.iloc[0])
    returns = (held * bar.fillna(0.0) - turns * fee).to_numpy()
    equity = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(equity)
    std = returns.std()
    return {
        "return": equity[-1] - 1,
        "max_dd": (equity / peak - 1).min(),
        "sharpe": returns.mean() / std * np.sqrt(per_year) if std > 0 else 0.0,
        "trades": int(turns.sum() // 2),
        "exposure": held.mean(),
    }


def buy_hold(data: pd.DataFrame, fee: float, per_year: int) -> dict:
    """Score holding for the whole window with one entry fee."""
    position = pd.Series(1.0, index=data.index)
    return evaluate(position, data, fee, per_year) | {"trades": 0}


def run(data_dir: Path, split: str, fees: list[float]) -> pd.DataFrame:
    """Grid every strategy on every interval and fee; return one row per run."""
    rows = []
    for interval, per_year in BARS_PER_YEAR.items():
        data = pd.read_csv(
            data_dir / f"BTCUSDT-{interval}.csv", index_col="Date", parse_dates=["Date"]
        )
        windows = {"train": data[data.index < split], "test": data[data.index >= split]}
        for name, (build, grid) in GRIDS.items():
            for params in grid:
                # Indicators warm up on the full series, so the test window starts with live state.

                position = build(data, **params)
                for window, frame in windows.items():
                    for fee in fees:
                        stats = evaluate(position.loc[frame.index], frame, fee, per_year)
                        rows.append(
                            {
                                "interval": interval,
                                "strategy": name,
                                "params": str(params),
                                "window": window,
                                "fee": fee,
                            }
                            | stats
                        )
        for window, frame in windows.items():
            for fee in fees:
                rows.append(
                    {
                        "interval": interval,
                        "strategy": "buy_hold",
                        "params": "{}",
                        "window": window,
                        "fee": fee,
                    }
                    | buy_hold(frame, fee, per_year)
                )
        print(f"{interval}: done")
    return pd.DataFrame(rows)


def main() -> None:
    """Write the full grid to data/research.csv and print the train winners."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--split", default="2024-07-01")
    args = parser.parse_args()

    results = run(args.data, args.split, [0.001, 0.00075])
    results.to_csv(args.data / "research.csv", index=False)


if __name__ == "__main__":
    main()
