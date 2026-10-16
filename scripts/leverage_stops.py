"""Backtest leveraged strategies with a hard stop-loss: uv run python scripts/leverage_stops.py.

A strategy's long/flat signal opens and closes positions at the next bar's
open, as in strategy_zoo.py. A stop sits a fixed price distance below each
entry. When a bar's low touches it, the position closes at the stop, or at the
open when the bar gaps through it. After a stop the strategy waits for its
signal to turn off and on again before it re-enters. Leverage scales every
price move, fee and financing charge.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from binance_klines import resample
from strategy_zoo import START, ZOO

BUILDERS = {name: build for _, name, build, _, _ in ZOO}

# USD minus EUR policy rate in percent a year, roughly, for overnight financing.

RATE_GAP = {2020: 0.4, 2021: 0.5, 2022: 1.6, 2023: 1.7, 2024: 1.4, 2025: 1.9, 2026: 1.5}
MARKUP = 1.0

CASES = [
    ("EURUSD", "Connors RSI2", "4h", (100, 5, 5)),
    ("USDEUR", "RSI reversion", "1d", (7, 20, 60)),
    ("EURUSD", "RSI reversion", "1d", (7, 20, 60)),
    ("USDEUR", "Lowest close", "1d", (10,)),
]


def carry(symbol: str, years: np.ndarray) -> np.ndarray:
    """Yearly financing cost on notional for holding the quoted currency."""
    gap = np.array([RATE_GAP[y] for y in years]) / 100
    return gap + MARKUP / 100 if symbol == "EURUSD" else MARKUP / 100 - gap


def simulate(
    d: pd.DataFrame, signal: pd.Series, symbol: str, leverage: float, fee: float, stop: float | None
) -> dict:
    """Walk bars once, applying signals at opens and the stop inside each bar."""
    o, h, low, c = (d[k].to_numpy() for k in ["Open", "High", "Low", "Close"])
    want = signal.shift(1).fillna(0.0).to_numpy()
    live = d.index >= START
    step = np.diff(d.index.asi8, append=d.index.asi8[-1]) / 1e9 / (365.25 * 86400)
    cost = carry(symbol, d.index.year.to_numpy())
    equity, peak, worst = 1.0, 1.0, 0.0
    inside, stopped, mark, stop_px = False, False, 0.0, 0.0
    trades = stops = 0
    for i in range(len(o)):
        if not live[i]:
            continue
        if stopped and want[i] == 0:
            stopped = False
        if inside and want[i] == 0:
            equity *= 1 + leverage * (o[i] / mark - 1) - leverage * fee
            inside = False
        elif not inside and want[i] == 1 and not stopped:
            equity *= 1 - leverage * fee
            inside, mark, trades = True, o[i], trades + 1
            stop_px = o[i] * (1 - stop) if stop else 0.0
        if inside:
            if stop and low[i] <= stop_px:
                fill = min(o[i], stop_px)
                equity *= 1 + leverage * (fill / mark - 1) - leverage * fee
                inside, stopped, stops = False, True, stops + 1
            else:
                equity *= 1 + leverage * (c[i] / mark - 1) - leverage * cost[i] * step[i]
                mark = c[i]
        equity = max(equity, 0.0)
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1)
    years = live.sum() / (len(d) / ((d.index[-1] - d.index[0]).days / 365.25))
    return {
        "return": equity - 1,
        "cagr": equity ** (1 / years) - 1 if equity > 0 else -1.0,
        "max_dd": worst,
        "trades": trades,
        "stops": stops,
    }


def main() -> None:
    """Print each case at 1x and at the chosen leverage across stop distances."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--fee", type=float, default=0.0001)
    args = parser.parse_args()

    rows = []
    for symbol, name, interval, params in CASES:
        hourly = pd.read_csv(args.data / f"{symbol}-1h.csv", index_col="Date", parse_dates=["Date"])
        d = hourly if interval == "1h" else resample(hourly, {"4h": "4h", "1d": "1D"}[interval])
        signal = BUILDERS[name](d, *params)
        for leverage, stop in [(1, None), (args.leverage, None)] + [
            (args.leverage, s) for s in (0.005, 0.01, 0.02)
        ]:
            stats = simulate(d, signal, symbol, leverage, args.fee, stop)
            rows.append(
                {
                    "case": f"{name} {params} {symbol} {interval}",
                    "leverage": leverage,
                    "stop": f"{stop:.1%}" if stop else "none",
                }
                | stats
            )
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
