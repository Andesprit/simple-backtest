"""Download Binance spot klines: uv run python scripts/binance_klines.py.

Fetches monthly archives from data.binance.vision and writes one CSV per
interval to ``data/`` (git-ignored). The package itself ships no data source;
this script lives outside it on purpose.
"""

import argparse
import io
import zipfile
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pandas as pd

ARCHIVE = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/{interval}/{name}.zip"
COLUMNS = ["open_time", "Open", "High", "Low", "Close", "Volume", "close_time"]


def months(start: str, end: date) -> list[str]:
    """List YYYY-MM strings from start through the last complete month before end."""
    last = pd.Period(end, freq="M") - 1
    return [str(p) for p in pd.period_range(start, last, freq="M")]


def parse_archive(raw: bytes) -> pd.DataFrame:
    """Parse one Binance kline zip into an OHLCV frame indexed by UTC open time."""
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        with archive.open(archive.namelist()[0]) as handle:
            frame = pd.read_csv(handle, header=None, usecols=range(7), names=COLUMNS)

    # Some archives carry a header row; drop it.

    frame = frame[pd.to_numeric(frame["open_time"], errors="coerce").notna()]
    open_time = frame["open_time"].astype("int64")

    # Spot archives switch from milliseconds to microseconds in 2025.

    unit = "us" if open_time.iloc[0] > 10**14 else "ms"
    frame.index = pd.to_datetime(open_time, unit=unit, utc=True).rename("Date")
    return frame[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def download(symbol: str, interval: str, start: str, today: date) -> pd.DataFrame:
    """Fetch and concatenate every monthly archive for one interval."""
    frames = []
    for month in months(start, today):
        name = f"{symbol}-{interval}-{month}"
        url = ARCHIVE.format(symbol=symbol, interval=interval, name=name)
        try:
            with urlopen(url, timeout=60) as response:
                frames.append(parse_archive(response.read()))
        except HTTPError as error:
            raise SystemExit(f"{url}: HTTP {error.code}") from error
        print(f"{name}: {len(frames[-1])} bars")
    data = pd.concat(frames)
    return data[~data.index.duplicated()].sort_index()


def resample(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Merge bars into a coarser interval, such as 5m into 10m."""
    ohlcv = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    return data.resample(rule, label="left", closed="left").agg(ohlcv).dropna()


def load(path: Path) -> pd.DataFrame:
    """Read a CSV written by this script back into an OHLCV frame."""
    return pd.read_csv(path, index_col="Date", parse_dates=["Date"])


def main() -> None:
    """Write data/{symbol}-{interval}.csv for 1h, 5m and the derived 10m."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", default="2020-01")
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True)

    today = date.today()
    for interval in ["1h", "5m"]:
        data = download(args.symbol, interval, args.start, today)
        data.to_csv(args.out / f"{args.symbol}-{interval}.csv")
        print(f"{interval}: {len(data)} bars, {data.index[0]} to {data.index[-1]}")
        if interval == "5m":
            ten = resample(data, "10min")
            ten.to_csv(args.out / f"{args.symbol}-10m.csv")
            print(f"10m: {len(ten)} bars")


if __name__ == "__main__":
    main()
