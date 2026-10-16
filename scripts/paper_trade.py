"""Paper-trade one strategy on live BTCUSDT bars: uv run python scripts/paper_trade.py.

Polls Binance's public market-data mirror after each bar closes, recomputes the
strategy's position from closed bars, and simulates a fill at the next bar's
open with the same fee model as research_intraday.py. It places no orders and
holds no API keys. Every decision appends a row to the log CSV, and the
portfolio state survives restarts through that log.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from research_intraday import GRIDS  # noqa: E402

KLINES = "https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
SECONDS = {"5m": 300, "10m": 600, "1h": 3600, "4h": 14400, "1d": 86400}


def fetch(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Return closed bars as an OHLCV frame indexed by UTC open time."""

    # Binance has no 10m klines; build them from 5m.

    source = "5m" if interval == "10m" else interval
    count = limit * 2 if interval == "10m" else limit
    with urlopen(
        KLINES.format(symbol=symbol, interval=source, limit=min(count, 1000)), timeout=30
    ) as response:
        rows = json.load(response)
    frame = pd.DataFrame(
        [row[:7] for row in rows],
        columns=["open_time", "Open", "High", "Low", "Close", "Volume", "close_time"],
    )
    frame.index = pd.to_datetime(frame["open_time"], unit="ms", utc=True).rename("Date")
    frame = frame[["Open", "High", "Low", "Close", "Volume"]].astype(float)
    now = pd.Timestamp.now(tz="UTC")
    frame = frame[frame.index + pd.Timedelta(seconds=SECONDS[source]) <= now]
    if interval == "10m":
        ohlcv = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        grouped = frame.resample("10min", label="left", closed="left")
        frame = grouped.agg(ohlcv)[grouped.size() == 2]
    return frame


def resume(log: Path, cash: float) -> tuple[float, float, str | None]:
    """Read cash, units and the last decided bar from the log, or start fresh."""
    if not log.exists():
        return cash, 0.0, None
    last = pd.read_csv(log).iloc[-1]
    return float(last["cash"]), float(last["units"]), str(last["bar"])


def step(
    args: argparse.Namespace, cash: float, units: float, last_bar: str | None
) -> tuple[float, float, str | None]:
    """Decide on the newest closed bar and simulate any fill at the live price."""
    build, _ = GRIDS[args.strategy]
    data = fetch(args.symbol, args.interval, args.history)
    bar = str(data.index[-1])
    if bar == last_bar:
        return cash, units, last_bar
    want = float(build(data, **json.loads(args.params)).iloc[-1])

    # The next bar has opened, so its open is the latest trade price.

    with urlopen(KLINES.format(symbol=args.symbol, interval="1m", limit=1), timeout=30) as response:
        price = float(json.load(response)[-1][1])
    action = "hold"
    if want == 1.0 and units == 0.0:
        units, cash, action = cash * (1 - args.fee) / price, 0.0, "buy"
    elif want == 0.0 and units > 0.0:
        cash, units, action = units * price * (1 - args.fee), 0.0, "sell"
    row = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bar": bar,
        "close": data["Close"].iloc[-1],
        "signal": want,
        "action": action,
        "price": price,
        "cash": cash,
        "units": units,
        "equity": cash + units * price,
    }
    pd.DataFrame([row]).to_csv(args.log, mode="a", header=not args.log.exists(), index=False)
    print(json.dumps(row))
    return cash, units, bar


def main() -> None:
    """Loop forever, or once with --once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="ma_cross", choices=sorted(GRIDS))
    parser.add_argument(
        "--params", default='{"fast": 10, "slow": 200}', help="JSON keyword arguments"
    )
    parser.add_argument("--interval", default="4h", choices=sorted(SECONDS))
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--cash", type=float, default=10_000.0)
    parser.add_argument("--history", type=int, default=500, help="closed bars fed to the strategy")
    parser.add_argument("--log", type=Path, default=Path("data/paper_trades.csv"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.log.parent.mkdir(exist_ok=True)

    cash, units, last_bar = resume(args.log, args.cash)
    while True:
        try:
            cash, units, last_bar = step(args, cash, units, last_bar)
        except OSError as error:
            print(f"fetch failed, retrying next bar: {error}", file=sys.stderr)
        if args.once:
            return

        # Wake a few seconds after the next bar boundary.

        period = SECONDS[args.interval]
        time.sleep(period - time.time() % period + 5)


if __name__ == "__main__":
    main()
