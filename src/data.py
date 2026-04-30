"""Data loading and preparation for the stock predictor.

Two sources are supported:
  - Local CSVs under ``stock_market_data/sp500/csv/`` (Kaggle S&P 500 dump).
  - Live data via ``yfinance``.

Provides both a random-split ``prepare_dataset`` (for the original
demo) and a calendar-date split ``prepare_train_test_split`` (for the
regime-shift experiments — fits the scaler on training data only).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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


@dataclass
class Dataset:
    """Bundle of arrays ready to feed a Conv1D model."""

    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    scaler: MinMaxScaler

    @property
    def input_shape(self) -> int:
        return self.X_train.shape[1]


@dataclass
class TrainTestSplit:
    """Train/test split with everything needed to compute proper metrics.

    ``y_test_prev`` is the previous trading day's close at each test
    timestep — the input to the persistence baseline and to directional
    accuracy. ``train_close_min``/``max`` are the price range seen during
    training, for out-of-range detection on the test set.
    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    y_test_prev: np.ndarray
    test_index: pd.Index
    scaler: MinMaxScaler
    train_close_min: float
    train_close_max: float

    @property
    def input_shape(self) -> int:
        return self.X_train.shape[1]


def prepare_train_test_split(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: Optional[str] = None,
) -> TrainTestSplit:
    """Build a train/test pair using explicit calendar dates.

    The scaler is fit on training data **only** (no leak into test).
    ``df`` must have a DatetimeIndex (use ``load_csv(path, with_dates=True)``).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex — use load_csv(..., with_dates=True)")

    df = df.sort_index()
    train_df = slice_by_date(df, end=train_end)
    test_df = slice_by_date(df, start=test_start, end=test_end)

    if len(train_df) < 2 or len(test_df) < 2:
        raise ValueError(f"Not enough rows: train={len(train_df)}, test={len(test_df)}")

    scaler = MinMaxScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])

    train_scaled = scaler.transform(train_df[FEATURE_COLUMNS])
    test_scaled = scaler.transform(test_df[FEATURE_COLUMNS])

    X_train = train_scaled[:-1][..., np.newaxis]
    y_train = train_df["Close"].values[1:]

    X_test = test_scaled[:-1][..., np.newaxis]
    y_test = test_df["Close"].values[1:]
    y_test_prev = test_df["Close"].values[:-1]
    test_index = test_df.index[1:]

    return TrainTestSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_test_prev=y_test_prev,
        test_index=test_index,
        scaler=scaler,
        train_close_min=float(train_df["Close"].min()),
        train_close_max=float(train_df["Close"].max()),
    )


def prepare_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    scaler: Optional[MinMaxScaler] = None,
) -> Dataset:
    """Scale features and build the next-day-close supervised problem."""
    scaler = scaler if scaler is not None else MinMaxScaler()
    features = df[FEATURE_COLUMNS]

    X_full = scaler.fit_transform(features)[:-1]
    y_full = features["Close"].values[1:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=test_size, shuffle=False
    )
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    return Dataset(X_train, X_val, y_train, y_val, scaler)
