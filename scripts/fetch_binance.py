"""Download daily OHLCV from Binance's public REST API as a Kaggle-format CSV.

A fallback for when yfinance is rate-limited or otherwise unreachable.
The Binance ``/api/v3/klines`` endpoint is unauthenticated, returns up
to 1000 candles per call, and covers most major crypto pairs back to
their listing date.

Output schema matches the Kaggle S&P 500 dump consumed by
``src.data.load_csv(with_dates=True)`` — same columns in the same order
with the same DD-MM-YYYY date format — so the existing pipeline can
load it without modification.

Usage:
    python scripts/fetch_binance.py ETHUSDT --start 2018-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List

BINANCE_URL = "https://api.binance.com/api/v3/klines"
DEFAULT_OUT_DIR = Path("stock_market_data/crypto/csv")


def _to_ms(date_str: str) -> int:
    """ISO date string -> milliseconds since epoch (UTC)."""
    return int(datetime.strptime(date_str, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_klines(symbol: str, start_ms: int, end_ms: int, sleep_s: float = 0.2) -> List[list]:
    """Page through ``/klines`` until ``end_ms`` is reached or data dries up."""
    rows: List[list] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (f"{BINANCE_URL}?symbol={symbol}&interval=1d"
               f"&startTime={cursor}&limit=1000")
        with urllib.request.urlopen(url, timeout=20) as resp:
            chunk = json.load(resp)
        if not chunk:
            break
        rows.extend(c for c in chunk if c[0] < end_ms)
        last_open_ms = chunk[-1][0]
        if last_open_ms >= end_ms or last_open_ms + 86_400_000 == cursor:
            break
        cursor = last_open_ms + 86_400_000   # next day
        time.sleep(sleep_s)
    return rows


def write_csv(rows: List[list], path: Path) -> None:
    """Match the Kaggle dump's column order + DD-MM-YYYY date format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Low", "Open", "Volume", "High", "Close", "Adjusted Close"])
        for r in rows:
            open_ms = int(r[0])
            d = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc)
            date = d.strftime("%d-%m-%Y")
            open_, high, low, close = (float(r[1]), float(r[2]), float(r[3]), float(r[4]))
            volume = float(r[5])
            w.writerow([date, low, open_, volume, high, close, close])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("symbol", help="Binance symbol, e.g. ETHUSDT, BTCUSDT")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-01-01")
    p.add_argument("--out", default=None,
                   help="output CSV path; defaults to stock_market_data/crypto/csv/<symbol>.csv")
    args = p.parse_args()

    start_ms = _to_ms(args.start)
    end_ms = _to_ms(args.end)
    out_path = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"{args.symbol}.csv"

    print(f"fetching {args.symbol} {args.start} .. {args.end} from Binance...")
    rows = fetch_klines(args.symbol, start_ms, end_ms)
    print(f"  {len(rows)} daily candles received")

    if not rows:
        raise SystemExit("Binance returned no data — check symbol or date range")

    write_csv(rows, out_path)
    print(f"  wrote {out_path}")
    first_date = datetime.fromtimestamp(rows[0][0] / 1000, tz=timezone.utc).date()
    last_date = datetime.fromtimestamp(rows[-1][0] / 1000, tz=timezone.utc).date()
    print(f"  range: {first_date} .. {last_date}")


if __name__ == "__main__":
    main()
