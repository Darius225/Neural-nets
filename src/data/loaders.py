"""Data sources and lightweight slicing.

Two ways to get OHLCV into memory:
  - Local CSVs from the Kaggle S&P 500 dump (DD-MM-YYYY date format).
  - Live history via ``yfinance`` (DatetimeIndex returned directly).

``slice_by_date`` is here too because it's the natural companion to
loading — once the index is a ``DatetimeIndex`` you almost always want
to carve out a calendar window before doing anything else.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Optional

import pandas as pd

FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_CSV_GLOB = "stock_market_data/sp500/csv/*.csv"
# Kaggle S&P 500 dump stores dates as DD-MM-YYYY. yfinance returns
# DatetimeIndex. We accept either.
DEFAULT_DATE_FORMAT = "%d-%m-%Y"


def discover_csv_paths(pattern: str = DEFAULT_CSV_GLOB) -> Dict[str, str]:
    """Map ticker symbol -> CSV path for every file matching ``pattern``."""
    paths: Dict[str, str] = {}
    for file in glob.glob(pattern, recursive=True):
        ticker = os.path.splitext(os.path.basename(file))[0]
        paths[ticker] = file
    return paths


def load_csv(path: str, with_dates: bool = False) -> pd.DataFrame:
    """Load OHLCV from a Kaggle-format CSV.

    With ``with_dates=True``, also parses the ``Date`` column (DD-MM-YYYY)
    and sets it as the index — needed for date-range filtering.
    """
    if not with_dates:
        return pd.read_csv(path, usecols=FEATURE_COLUMNS)
    df = pd.read_csv(path, usecols=["Date"] + FEATURE_COLUMNS)
    df["Date"] = pd.to_datetime(df["Date"], format=DEFAULT_DATE_FORMAT)
    return df.set_index("Date").sort_index()


def slice_by_date(df: pd.DataFrame, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Return rows of ``df`` (DatetimeIndex assumed) within [start, end]."""
    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]
    return df


def load_yfinance(ticker: str, period: str = "max") -> pd.DataFrame:
    """Fetch historical OHLCV via yfinance. Lazy import keeps it optional."""
    import yfinance as yf

    df = yf.Ticker(ticker).history(period=period)
    return df[FEATURE_COLUMNS]
