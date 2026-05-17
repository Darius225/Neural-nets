"""Download intraday OHLCV bars from Binance's public klines endpoint.

Used to compute proper Realized Variance (Andersen et al. 2003): for
each calendar day, sum the squared intraday log returns to estimate
that day's true volatility. This estimate is far less noisy than the
single-day |log_return| we've been using.

Output: a parquet file (smaller than CSV, faster to load) with one row
per intraday bar.

Usage:
    python scripts/fetch_binance_intraday.py ETHUSDT \\
        --interval 5m --start 2023-01-01 --end 2026-05-18
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

BINANCE_URL = "https://api.binance.com/api/v3/klines"
DEFAULT_OUT_DIR = Path("stock_market_data/crypto/intraday")

# How many milliseconds each interval covers (for cursor stepping).
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
}


def _to_ms(date_str: str) -> int:
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def fetch(symbol: str, interval: str, start_ms: int, end_ms: int, sleep_s: float = 0.2):
    """Page through klines until end_ms. Returns a list of raw kline rows."""
    if interval not in INTERVAL_MS:
        raise SystemExit(f"unknown interval {interval}; allowed: {list(INTERVAL_MS)}")
    rows = []
    cursor = start_ms
    n_calls = 0
    while cursor < end_ms:
        url = f"{BINANCE_URL}?symbol={symbol}&interval={interval}&startTime={cursor}&limit=1000"
        with urllib.request.urlopen(url, timeout=30) as resp:
            chunk = json.load(resp)
        n_calls += 1
        if not chunk:
            break
        rows.extend(c for c in chunk if c[0] < end_ms)
        last_open_ms = chunk[-1][0]
        if last_open_ms >= end_ms or last_open_ms + INTERVAL_MS[interval] == cursor:
            break
        cursor = last_open_ms + INTERVAL_MS[interval]
        time.sleep(sleep_s)
        if n_calls % 25 == 0:
            done_days = (cursor - start_ms) / (24 * 60 * 60 * 1000)
            print(f"  ... {n_calls} calls, ~{done_days:.0f} days fetched", flush=True)
    return rows


def to_dataframe(rows: list) -> pd.DataFrame:
    """Convert raw kline rows into a typed pandas DataFrame indexed by open time."""
    df = pd.DataFrame(
        rows,
        columns=[
            "open_ms",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "close_ms",
            "quote_vol",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)
    df["time"] = pd.to_datetime(df["open_ms"], unit="ms", utc=True)
    df = df.set_index("time")
    return df[["Open", "High", "Low", "Close", "Volume"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("symbol")
    p.add_argument("--interval", default="5m", choices=list(INTERVAL_MS))
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2026-05-18")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    out_path = (
        Path(args.out) if args.out else DEFAULT_OUT_DIR / f"{args.symbol}_{args.interval}.parquet"
    )

    print(f"fetching {args.symbol} {args.interval} {args.start} .. {args.end}")
    t0 = time.time()
    rows = fetch(args.symbol, args.interval, _to_ms(args.start), _to_ms(args.end))
    if not rows:
        raise SystemExit("Binance returned no data")

    df = to_dataframe(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(
        f"  {len(df):,} bars in {time.time() - t0:.1f}s "
        f"({df.index[0].date()} .. {df.index[-1].date()})"
    )
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
