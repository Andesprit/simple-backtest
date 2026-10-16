"""Backtest a zoo of 100+ long/flat BTC strategies: uv run python scripts/strategy_zoo.py.

Every strategy is a fixed preset chosen before looking at results. Each runs on
1h, 4h and 1d bars built from the 1h CSV and is scored on the last five years,
Oct 2021 to Sep 2026, and on each half of that span. Earlier bars only warm
up the indicators.
Signals use bar closes and fill at the next open; every buy and sell pays the
fee. Writes data/zoo-SYMBOL.csv (one row per run) and data/zoo_curves-SYMBOL.csv
(weekly equity).

Families come from common technical-analysis references and from published
BTC studies: moving-average and breakout trend rules, oscillator mean
reversion, volume confirmation and intraday seasonality (Quantpedia's
21:00-23:00 UTC window).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from binance_klines import resample

START = pd.Timestamp("2021-10-01", tz="UTC")
MIDDLE = pd.Timestamp("2024-04-01", tz="UTC")


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def wilder(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1 / n, adjust=False).mean()


def atr(d: pd.DataFrame, n: int) -> pd.Series:
    prev = d["Close"].shift(1)
    true_range = pd.concat(
        [d["High"] - d["Low"], (d["High"] - prev).abs(), (d["Low"] - prev).abs()], axis=1
    )
    return wilder(true_range.max(axis=1), n)


def rsi(x: pd.Series, n: int) -> pd.Series:
    delta = x.diff()
    return 100 - 100 / (1 + wilder(delta.clip(lower=0), n) / wilder(-delta.clip(upper=0), n))


def stoch(d: pd.DataFrame, n: int) -> pd.Series:
    low, high = d["Low"].rolling(n).min(), d["High"].rolling(n).max()
    return 100 * (d["Close"] - low) / (high - low)


def latch(enter: pd.Series, leave: pd.Series) -> pd.Series:
    """Hold from an entry bar until an exit bar; exits win ties."""
    state = pd.Series(np.nan, index=enter.index)
    state[enter.fillna(False).astype(bool)] = 1.0
    state[leave.fillna(False).astype(bool)] = 0.0
    return state.ffill().fillna(0.0)


def above(a: pd.Series, b: pd.Series | float) -> pd.Series:
    return (a > b).astype(float)


# Trend rules.


def sma_cross(d, fast, slow):
    return above(sma(d["Close"], fast), sma(d["Close"], slow))


def ema_cross(d, fast, slow):
    return above(ema(d["Close"], fast), ema(d["Close"], slow))


def price_sma(d, n):
    return above(d["Close"], sma(d["Close"], n))


def price_ema(d, n):
    return above(d["Close"], ema(d["Close"], n))


def triple_ma(d, fast, mid, slow):
    c = d["Close"]
    return ((sma(c, fast) > sma(c, mid)) & (sma(c, mid) > sma(c, slow))).astype(float)


def macd(d, fast, slow, signal):
    line = ema(d["Close"], fast) - ema(d["Close"], slow)
    return above(line, ema(line, signal))


def macd_zero(d, fast, slow, signal):
    line = ema(d["Close"], fast) - ema(d["Close"], slow)
    return ((line > 0) & (line > ema(line, signal))).astype(float)


def donchian(d, enter, leave):
    top = d["High"].rolling(enter).max().shift(1)
    bottom = d["Low"].rolling(leave).min().shift(1)
    return latch(d["Close"] > top, d["Close"] < bottom)


def keltner_break(d, n, mult):
    mid = ema(d["Close"], n)
    return latch(d["Close"] > mid + mult * atr(d, n), d["Close"] < mid)


def bollinger_break(d, n, width):
    mid = sma(d["Close"], n)
    return latch(d["Close"] > mid + width * d["Close"].rolling(n).std(), d["Close"] < mid)


def supertrend(d, n, mult):
    """Long while close stays above the ratcheting ATR band."""
    mid = ((d["High"] + d["Low"]) / 2).to_numpy()
    band = (mult * atr(d, n)).to_numpy()
    close = d["Close"].to_numpy()
    upper, lower = mid + band, mid - band
    out = np.zeros(len(close))
    trend, up, low = 0, upper[0], lower[0]
    for i in range(1, len(close)):
        if np.isnan(band[i]):
            continue
        up = upper[i] if upper[i] < up or close[i - 1] > up else up
        low = lower[i] if lower[i] > low or close[i - 1] < low else low
        if close[i] > up:
            trend = 1
        elif close[i] < low:
            trend = 0
        out[i] = trend
    return pd.Series(out, index=d.index)


def adx_di(d, n, threshold):
    up, down = d["High"].diff(), -d["Low"].diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=d.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=d.index)
    rng = atr(d, n)
    pdi, mdi = 100 * wilder(plus, n) / rng, 100 * wilder(minus, n) / rng
    adx = wilder(100 * (pdi - mdi).abs() / (pdi + mdi), n)
    return ((pdi > mdi) & (adx > threshold)).astype(float)


def aroon(d, n):
    up = d["High"].rolling(n + 1).apply(np.argmax, raw=True) / n
    down = d["Low"].rolling(n + 1).apply(np.argmin, raw=True) / n
    return above(up, down)


def ichimoku(d, tenkan, kijun, senkou):
    def mid(k):
        return (d["High"].rolling(k).max() + d["Low"].rolling(k).min()) / 2

    t, k = mid(tenkan), mid(kijun)
    span_a, span_b = ((t + k) / 2).shift(kijun), mid(senkou).shift(kijun)
    cloud = pd.concat([span_a, span_b], axis=1).max(axis=1)
    return ((d["Close"] > cloud) & (t > k)).astype(float)


def psar(d, step, cap):
    """Parabolic SAR, long while price sits above the stop."""
    high, low = d["High"].to_numpy(), d["Low"].to_numpy()
    out = np.zeros(len(high))
    long, stop, extreme, af = True, low[0], high[0], step
    for i in range(1, len(high)):
        stop = stop + af * (extreme - stop)
        if long:
            if low[i] < stop:
                long, stop, extreme, af = False, extreme, low[i], step
            elif high[i] > extreme:
                extreme, af = high[i], min(af + step, cap)
        else:
            if high[i] > stop:
                long, stop, extreme, af = True, extreme, high[i], step
            elif low[i] < extreme:
                extreme, af = low[i], min(af + step, cap)
        out[i] = float(long)
    return pd.Series(out, index=d.index)


def momentum(d, n):
    return above(d["Close"].pct_change(n), 0)


def roc_threshold(d, n, enter, leave):
    roc = d["Close"].pct_change(n)
    return latch(roc > enter, roc < leave)


def hull_slope(d, n):
    c = d["Close"]

    def wma(x, k):
        weights = np.arange(1, k + 1)
        return x.rolling(k).apply(lambda w: np.dot(w, weights) / weights.sum(), raw=True)

    hull = wma(2 * wma(c, n // 2) - wma(c, n), int(np.sqrt(n)))
    return above(hull.diff(), 0)


def regression_slope(d, n):
    x = np.arange(n) - (n - 1) / 2
    slope = d["Close"].rolling(n).apply(lambda w: np.dot(w, x), raw=True)
    return above(slope, 0)


def heikin_ashi(d, streak):
    ha_close = (d["Open"] + d["High"] + d["Low"] + d["Close"]) / 4
    ha_open = ha_close.copy()
    values, opens = ha_close.to_numpy(), d["Open"].to_numpy()
    out = np.empty(len(values))
    out[0] = opens[0]
    for i in range(1, len(values)):
        out[i] = (out[i - 1] + values[i - 1]) / 2
    ha_open[:] = out
    green = (ha_close > ha_open).astype(int)
    run = green.groupby((green != green.shift()).cumsum()).cumsum()
    return latch(run >= streak, green == 0)


def vortex(d, n):
    rng = atr(d, 1).rolling(n).sum()
    plus = (d["High"] - d["Low"].shift(1)).abs().rolling(n).sum() / rng
    minus = (d["Low"] - d["High"].shift(1)).abs().rolling(n).sum() / rng
    return above(plus, minus)


def trix(d, n):
    triple = ema(ema(ema(d["Close"], n), n), n)
    return above(triple.pct_change(), 0)


def chandelier(d, enter, n, mult):
    """Enter on a new high, exit when close drops mult ATR below the highest high."""
    top = d["High"].rolling(enter).max().shift(1)
    stop = d["High"].rolling(n).max() - mult * atr(d, n)
    return latch(d["Close"] > top, d["Close"] < stop)


def turtle(d, enter, leave):
    return donchian(d, enter, leave)


def dual_momentum(d, n, ma):
    return ((d["Close"].pct_change(n) > 0) & (d["Close"] > sma(d["Close"], ma))).astype(float)


def kama(d, n, fast, slow):
    c = d["Close"]
    change = (c - c.shift(n)).abs()
    noise = c.diff().abs().rolling(n).sum()
    sc = ((change / noise) * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    values, ratio = c.to_numpy(), sc.fillna(0).to_numpy()
    out = np.empty(len(values))
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = out[i - 1] + ratio[i] * (values[i] - out[i - 1])
    return above(c, pd.Series(out, index=c.index))


def vol_target_trend(d, n, vol_n):
    """Price above SMA, but only when realized volatility is below its own median."""
    vol = d["Close"].pct_change().rolling(vol_n).std()
    calm = vol < vol.rolling(vol_n * 10).median()
    return ((d["Close"] > sma(d["Close"], n)) & calm).astype(float)


# Mean reversion rules.


def rsi_revert(d, n, low, high):
    value = rsi(d["Close"], n)
    return latch(value < low, value > high)


def connors_rsi2(d, trend, low, exit_ma):
    c = d["Close"]
    return latch((c > sma(c, trend)) & (rsi(c, 2) < low), c > sma(c, exit_ma))


def bollinger_revert(d, n, width):
    mid = sma(d["Close"], n)
    return latch(d["Close"] < mid - width * d["Close"].rolling(n).std(), d["Close"] > mid)


def stoch_revert(d, n, low, high):
    k = stoch(d, n).rolling(3).mean()
    return latch(k < low, k > high)


def williams_r(d, n, low, high):
    r = stoch(d, n) - 100
    return latch(r < low, r > high)


def cci_revert(d, n, low):
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    dev = tp.rolling(n).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    cci = (tp - sma(tp, n)) / (0.015 * dev)
    return latch(cci < low, cci > 0)


def zscore_revert(d, n, enter):
    z = (d["Close"] - sma(d["Close"], n)) / d["Close"].rolling(n).std()
    return latch(z < -enter, z > 0)


def ibs(d, low, high):
    value = (d["Close"] - d["Low"]) / (d["High"] - d["Low"])
    return latch(value < low, value > high)


def lowest_close(d, n):
    c = d["Close"]
    return latch(c <= c.rolling(n).min(), c > d["High"].shift(1))


def trend_dip(d, trend, n, low):
    """Buy RSI dips only above the trend SMA, sell on RSI recovery or trend loss."""
    c = d["Close"]
    up = c > sma(c, trend)
    value = rsi(c, n)
    return latch(up & (value < low), (value > 65) | ~up)


def mfi_revert(d, n, low, high):
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    flow = tp * d["Volume"]
    pos = flow.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    neg = flow.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    mfi = 100 - 100 / (1 + pos / neg)
    return latch(mfi < low, mfi > high)


# Momentum oscillators.


def rsi_momentum(d, n, enter, leave):
    value = rsi(d["Close"], n)
    return latch(value > enter, value < leave)


def stoch_momentum(d, n):
    k = stoch(d, n)
    return latch((k > 80) & (k.shift(1) <= 80), k < 50)


# Volume rules.


def obv_trend(d, n):
    obv = (np.sign(d["Close"].diff()).fillna(0) * d["Volume"]).cumsum()
    return above(obv, sma(obv, n))


def volume_breakout(d, n, mult):
    top = d["High"].rolling(n).max().shift(1)
    loud = d["Volume"] > mult * sma(d["Volume"], n)
    return latch((d["Close"] > top) & loud, d["Close"] < sma(d["Close"], n))


def rolling_vwap(d, n):
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    vwap = (tp * d["Volume"]).rolling(n).sum() / d["Volume"].rolling(n).sum()
    return above(d["Close"], vwap)


def chaikin_flow(d, n):
    mult = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / (d["High"] - d["Low"])
    cmf = (mult.fillna(0) * d["Volume"]).rolling(n).sum() / d["Volume"].rolling(n).sum()
    return above(cmf, 0)


# Calendar and volatility rules.


def hour_window(d, start, end):
    """Hold through UTC hours start..end-1; only meaningful on 1h bars."""
    hour = d.index.hour

    # The position decided at a bar's close covers the next bar.

    nxt = (hour + 1) % 24
    return pd.Series(((nxt >= start) & (nxt < end)).astype(float), index=d.index)


def weekday_hold(d, days):
    """Hold through the given UTC weekdays (0 is Monday)."""
    nxt = (d.index + (d.index[1] - d.index[0])).weekday
    return pd.Series(np.isin(nxt, days).astype(float), index=d.index)


def range_breakout(d, k):
    """Larry Williams breakout: hold one bar after close clears open plus k times the prior range."""
    trigger = d["Open"] + k * (d["High"] - d["Low"]).shift(1)
    return above(d["Close"], trigger)


def squeeze(d, n):
    """Enter when Bollinger width hits a low and price breaks up; exit below the mid band."""
    mid = sma(d["Close"], n)
    width = d["Close"].rolling(n).std() / mid
    tight = width < width.rolling(n * 5).quantile(0.2)
    return latch(
        tight.shift(1) & (d["Close"] > mid + 2 * d["Close"].rolling(n).std()), d["Close"] < mid
    )


ZOO = [
    ("trend", "SMA cross", sma_cross, [(10, 50), (20, 100), (50, 200)], ["fast", "slow"]),
    ("trend", "EMA cross", ema_cross, [(12, 26), (20, 50), (50, 200)], ["fast", "slow"]),
    ("trend", "Price over SMA", price_sma, [(20,), (50,), (100,), (200,)], ["n"]),
    ("trend", "Price over EMA", price_ema, [(20,), (50,), (200,)], ["n"]),
    ("trend", "Triple MA", triple_ma, [(10, 30, 100), (20, 50, 200)], ["fast", "mid", "slow"]),
    ("trend", "MACD signal", macd, [(12, 26, 9), (24, 52, 18)], ["fast", "slow", "signal"]),
    (
        "trend",
        "MACD above zero",
        macd_zero,
        [(12, 26, 9), (24, 52, 18)],
        ["fast", "slow", "signal"],
    ),
    (
        "trend",
        "Donchian",
        donchian,
        [(20, 10), (50, 25), (100, 50), (200, 100)],
        ["enter", "leave"],
    ),
    ("trend", "Turtle", turtle, [(20, 10), (55, 20)], ["enter", "leave"]),
    ("trend", "Keltner breakout", keltner_break, [(20, 1.5), (50, 2.0)], ["n", "mult"]),
    ("trend", "Bollinger breakout", bollinger_break, [(20, 2.0), (50, 2.0)], ["n", "width"]),
    ("trend", "Supertrend", supertrend, [(10, 3.0), (20, 3.0), (50, 4.0)], ["n", "mult"]),
    ("trend", "ADX DI", adx_di, [(14, 20), (14, 25), (28, 20)], ["n", "threshold"]),
    ("trend", "Aroon", aroon, [(25,), (50,)], ["n"]),
    ("trend", "Ichimoku", ichimoku, [(9, 26, 52), (20, 60, 120)], ["tenkan", "kijun", "senkou"]),
    ("trend", "Parabolic SAR", psar, [(0.02, 0.2), (0.01, 0.1)], ["step", "cap"]),
    ("trend", "Momentum", momentum, [(24,), (50,), (100,), (200,)], ["n"]),
    (
        "trend",
        "ROC threshold",
        roc_threshold,
        [(20, 0.02, -0.02), (50, 0.05, -0.03)],
        ["n", "enter", "leave"],
    ),
    ("trend", "Hull MA slope", hull_slope, [(20,), (50,), (100,)], ["n"]),
    ("trend", "Regression slope", regression_slope, [(20,), (50,), (100,)], ["n"]),
    ("trend", "Heikin Ashi", heikin_ashi, [(2,), (3,)], ["streak"]),
    ("trend", "Vortex", vortex, [(14,), (28,)], ["n"]),
    ("trend", "TRIX", trix, [(15,), (30,)], ["n"]),
    ("trend", "Chandelier", chandelier, [(20, 22, 3.0), (50, 50, 3.0)], ["enter", "n", "mult"]),
    ("trend", "Dual momentum", dual_momentum, [(50, 100), (100, 200)], ["n", "ma"]),
    ("trend", "KAMA", kama, [(10, 2, 30), (20, 2, 30)], ["n", "fast", "slow"]),
    ("trend", "Calm trend", vol_target_trend, [(50, 20), (100, 30)], ["n", "vol_n"]),
    (
        "reversion",
        "RSI reversion",
        rsi_revert,
        [(14, 30, 70), (14, 25, 60), (28, 30, 55), (7, 20, 60)],
        ["n", "low", "high"],
    ),
    (
        "reversion",
        "Connors RSI2",
        connors_rsi2,
        [(200, 10, 5), (100, 5, 5)],
        ["trend", "low", "exit_ma"],
    ),
    (
        "reversion",
        "Bollinger reversion",
        bollinger_revert,
        [(20, 2.0), (20, 2.5), (50, 2.0)],
        ["n", "width"],
    ),
    (
        "reversion",
        "Stochastic reversion",
        stoch_revert,
        [(14, 20, 80), (14, 10, 70)],
        ["n", "low", "high"],
    ),
    (
        "reversion",
        "Williams %R",
        williams_r,
        [(14, -90, -20), (28, -80, -30)],
        ["n", "low", "high"],
    ),
    ("reversion", "CCI reversion", cci_revert, [(20, -100), (20, -150)], ["n", "low"]),
    (
        "reversion",
        "Z-score reversion",
        zscore_revert,
        [(20, 2.0), (50, 2.0), (50, 2.5)],
        ["n", "enter"],
    ),
    ("reversion", "IBS", ibs, [(0.2, 0.8), (0.1, 0.7)], ["low", "high"]),
    ("reversion", "Lowest close", lowest_close, [(5,), (10,)], ["n"]),
    ("reversion", "Trend dip", trend_dip, [(200, 14, 40), (100, 7, 30)], ["trend", "n", "low"]),
    ("reversion", "MFI reversion", mfi_revert, [(14, 20, 80), (14, 30, 70)], ["n", "low", "high"]),
    (
        "momentum",
        "RSI momentum",
        rsi_momentum,
        [(14, 55, 45), (14, 60, 50), (28, 55, 50)],
        ["n", "enter", "leave"],
    ),
    ("momentum", "Stochastic breakout", stoch_momentum, [(14,), (50,)], ["n"]),
    ("volume", "OBV trend", obv_trend, [(20,), (50,), (100,)], ["n"]),
    ("volume", "Volume breakout", volume_breakout, [(20, 1.5), (50, 2.0)], ["n", "mult"]),
    ("volume", "Rolling VWAP", rolling_vwap, [(24,), (100,)], ["n"]),
    ("volume", "Chaikin money flow", chaikin_flow, [(20,), (50,)], ["n"]),
    ("calendar", "Hour window", hour_window, [(21, 23), (0, 8), (13, 21)], ["start", "end"]),
    ("calendar", "Weekday hold", weekday_hold, [((4,),), ((5, 6),), ((0, 1, 2, 3),)], ["days"]),
    ("volatility", "Range breakout", range_breakout, [(0.5,), (0.7,)], ["k"]),
    ("volatility", "Bollinger squeeze", squeeze, [(20,), (50,)], ["n"]),
]


def runs_of(held: np.ndarray, returns: np.ndarray) -> tuple[int, float]:
    """Count round trips and their win rate from the held series and bar returns."""
    edges = np.diff(np.concatenate([[0], held, [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts) == 0:
        return 0, 0.0
    log = np.log1p(returns)
    cum = np.concatenate([[0], np.cumsum(log)])
    gains = cum[ends] - cum[starts]
    return len(starts), float((gains > 0).mean())


def stats(returns: pd.Series, held: pd.Series, per_year: int) -> dict:
    """Headline metrics for one window."""
    r = returns.to_numpy()
    equity = np.cumprod(1 + r)
    years = len(r) / per_year
    std = r.std()
    down = r[r < 0].std()
    max_dd = float((equity / np.maximum.accumulate(equity) - 1).min())
    cagr = equity[-1] ** (1 / years) - 1
    trades, win = runs_of(held.to_numpy(), r)
    return {
        "ret": float(equity[-1] - 1),
        "cagr": float(cagr),
        "sharpe": float(r.mean() / std * np.sqrt(per_year)) if std > 0 else 0.0,
        "sortino": float(r.mean() / down * np.sqrt(per_year)) if down > 0 else 0.0,
        "max_dd": max_dd,
        "calmar": float(cagr / -max_dd) if max_dd < 0 else 0.0,
        "trades": trades,
        "win": win,
        "exposure": float(held.mean()),
    }


def simulate(position: pd.Series, d: pd.DataFrame, fee: float) -> tuple[pd.Series, pd.Series]:
    """Per-bar net returns and the held position."""
    held = position.shift(1).fillna(0.0).clip(0, 1)
    bar = (d["Open"].shift(-1) / d["Open"] - 1).fillna(0.0)
    return held * bar - held.diff().abs().fillna(held.iloc[0]) * fee, held


def label(params: tuple, names: list[str]) -> str:
    return ", ".join(f"{n}={v}" for n, v in zip(names, params))


def main() -> None:
    """Run every preset on every interval and write the dashboard data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args()

    hourly = pd.read_csv(
        args.data / f"{args.symbol}-1h.csv", index_col="Date", parse_dates=["Date"]
    )
    frames = {"1h": hourly, "4h": resample(hourly, "4h"), "1d": resample(hourly, "1D")}
    weeks = pd.date_range(START, hourly.index[-1], freq="W-SUN")
    rows, curves = [], {}
    for interval, d in frames.items():
        # Count bars per year from the data, since forex skips weekends.

        per_year = len(d) / ((d.index[-1] - d.index[0]).days / 365.25)
        windows = {
            "full": d.index >= START,
            "h1": (d.index >= START) & (d.index < MIDDLE),
            "h2": d.index >= MIDDLE,
        }
        entries = [("benchmark", "Buy and hold", lambda d: pd.Series(1.0, index=d.index), [()], [])]
        for family, name, build, presets, names in entries + ZOO:
            if family == "calendar" and name == "Hour window" and interval != "1h":
                continue
            for params in presets:
                position = build(d, *params)

                # Flatten the warm-up years; the first fill can land at START's open.

                position[d.index.to_series().shift(-1) < START] = 0.0
                returns, held = simulate(position, d, args.fee)
                key = (
                    f"{name} ({label(params, names)}) {interval}" if names else f"{name} {interval}"
                )
                equity = (1 + returns[windows["full"]]).cumprod()
                curve = equity.reindex(weeks, method="ffill").fillna(1.0)
                curves[key] = [round(v, 4) for v in curve]
                live = returns[windows["full"]]
                yearly = (1 + live).groupby(live.index.year).prod() - 1
                rows.append(
                    {
                        "id": key,
                        "family": family,
                        "strategy": name,
                        "params": label(params, names),
                        "interval": interval,
                        **{
                            f"{w}_{k}": v
                            for w, mask in windows.items()
                            for k, v in stats(returns[mask], held[mask], per_year).items()
                        },
                        "years": {int(y): round(float(v), 4) for y, v in yearly.items()},
                    }
                )
        print(f"{interval}: {sum(r['interval'] == interval for r in rows)} runs")

    table = pd.DataFrame(rows)
    years = pd.DataFrame(list(table.pop("years"))).add_prefix("y")
    table = pd.concat([table, years], axis=1)

    # A run beats holding when its Sharpe tops buy-and-hold's on the same bars in both halves.

    hold = table[table["family"] == "benchmark"].set_index("interval")
    table["beats_hold"] = np.where(
        (table["h1_sharpe"] > table["interval"].map(hold["h1_sharpe"]))
        & (table["h2_sharpe"] > table["interval"].map(hold["h2_sharpe"]))
        & (table["family"] != "benchmark"),
        "Yes",
        "No",
    )
    table.round(4).to_csv(args.data / f"zoo-{args.symbol}.csv", index=False)
    long = pd.DataFrame(curves, index=[str(w.date()) for w in weeks]).rename_axis("week")
    long = long.reset_index().melt(id_vars="week", var_name="id", value_name="equity")
    long[["id", "week", "equity"]].to_csv(args.data / f"zoo_curves-{args.symbol}.csv", index=False)
    print(f"{len(ZOO)} families, {sum(len(z[3]) for z in ZOO)} strategies, {len(rows)} runs")


if __name__ == "__main__":
    main()
