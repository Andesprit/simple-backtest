"""Turn Binance EURUSDT bars into a forex-like EUR/USD series: uv run python scripts/forex_prep.py.

EURUSDT trades around the clock on a crypto venue, so it carries weekend bars
and occasional spikes when USDT or the order book wobbles. This script drops
the forex weekend (Friday 21:00 to Sunday 21:00 UTC), replaces closes more
than 1% from their 25-bar median, and clips wicks to 1% beyond the bar body.
It writes EURUSD-1h.csv (long euro) and USDEUR-1h.csv (the inverse, long dollar).
"""

import argparse
from pathlib import Path

import pandas as pd


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Drop weekend bars and damp spikes that real EUR/USD never prints."""
    day, hour = data.index.weekday, data.index.hour
    weekend = ((day == 4) & (hour >= 21)) | (day == 5) | ((day == 6) & (hour < 21))
    data = data[~weekend].copy()
    median = data["Close"].rolling(25, center=True, min_periods=1).median()
    for column in ["Open", "Close"]:
        spike = (data[column] / median - 1).abs() > 0.01
        data.loc[spike, column] = median[spike]
    body_high = data[["Open", "Close"]].max(axis=1)
    body_low = data[["Open", "Close"]].min(axis=1)
    data["High"] = data["High"].clip(lower=body_high, upper=body_high * 1.01)
    data["Low"] = data["Low"].clip(lower=body_low * 0.99, upper=body_low)
    return data


def invert(data: pd.DataFrame) -> pd.DataFrame:
    """Quote the pair the other way, so long means long dollar."""
    return pd.DataFrame(
        {
            "Open": 1 / data["Open"],
            "High": 1 / data["Low"],
            "Low": 1 / data["High"],
            "Close": 1 / data["Close"],
            "Volume": data["Volume"],
        },
        index=data.index,
    )


def main() -> None:
    """Write both quotes of the cleaned pair next to the source CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    args = parser.parse_args()

    raw = pd.read_csv(args.data / "EURUSDT-1h.csv", index_col="Date", parse_dates=["Date"])
    eurusd = clean(raw)
    eurusd.to_csv(args.data / "EURUSD-1h.csv")
    invert(eurusd).to_csv(args.data / "USDEUR-1h.csv")
    print(f"{len(raw)} bars in, {len(eurusd)} out")


if __name__ == "__main__":
    main()
