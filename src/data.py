"""Data loading and preparation for the stock predictor.

Two sources are supported:
  - Local CSVs under ``stock_market_data/sp500/csv/`` (Kaggle S&P 500 dump).
  - Live data via ``yfinance``.

The pipeline is the same regardless of source: take OHLCV columns, scale
with ``MinMaxScaler``, and produce ``(X, y)`` where each row in ``X`` is
one trading day's OHLCV and ``y`` is the *next* day's close.
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


def discover_csv_paths(pattern: str = DEFAULT_CSV_GLOB) -> Dict[str, str]:
    """Map ticker symbol -> CSV path for every file matching ``pattern``."""
    paths: Dict[str, str] = {}
    for file in glob.glob(pattern, recursive=True):
        ticker = os.path.splitext(os.path.basename(file))[0]
        paths[ticker] = file
    return paths


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, usecols=FEATURE_COLUMNS)


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


def prepare_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    scaler: Optional[MinMaxScaler] = None,
) -> Dataset:
    """Scale features and build the next-day-close supervised problem.

    Each input row is one day's OHLCV; the target is the *next* day's
    raw (unscaled) close. We deliberately train on raw target values to
    keep MAE/MAPE interpretable in dollars.
    """
    scaler = scaler if scaler is not None else MinMaxScaler()
    features = df[FEATURE_COLUMNS]

    X_full = scaler.fit_transform(features)[:-1]
    y_full = features["Close"].values[1:]

    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=test_size, shuffle=False
    )
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    return Dataset(X_train, X_val, y_train, y_val, scaler)
