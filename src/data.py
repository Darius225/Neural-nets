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
from typing import Callable, Dict, Optional, Tuple

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


@dataclass
class WindowedReturnsSplit:
    """Train/val/test for the windowed-returns pipeline.

    Each input is a ``(window_size, n_features)`` slice; each target is
    the *next-day simple return* ``(close[t+1] - close[t]) / close[t]``.
    The window is z-scored per-window per-feature so the model is
    invariant to absolute price level — handles regime shifts naturally.

    To recover a predicted price from a predicted return:
        predicted_close[t+1] = close_at_t * (1 + predicted_return)
    """
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    close_at_t_test: np.ndarray
    actual_close_test: np.ndarray
    test_index: pd.Index
    train_close_min: float
    train_close_max: float
    window_size: int
    n_features: int

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.window_size, self.n_features


def _zscore_window(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardise each feature within a single window (axis=0)."""
    mu = window.mean(axis=0, keepdims=True)
    sigma = window.std(axis=0, keepdims=True)
    return (window - mu) / (sigma + eps)


def _build_windows(features: np.ndarray, closes: np.ndarray, window_size: int):
    """Sliding windows + per-window z-score. Returns X, y, close_at_t."""
    n = len(features)
    if n < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} rows, got {n}")

    n_windows = n - window_size
    X = np.empty((n_windows, window_size, features.shape[1]), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.float32)
    close_at_t = np.empty(n_windows, dtype=np.float32)

    for i in range(n_windows):
        window = features[i : i + window_size]
        X[i] = _zscore_window(window)
        close_t = closes[i + window_size - 1]
        close_t1 = closes[i + window_size]
        y[i] = (close_t1 - close_t) / close_t
        close_at_t[i] = close_t
    return X, y, close_at_t


def prepare_windowed_returns_split(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: Optional[str] = None,
    window_size: int = 30,
    internal_val_fraction: float = 0.15,
    feature_builder: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> WindowedReturnsSplit:
    """Build a (train, internal_val, test) split for windowed-returns training.

    Internal val is the last ``internal_val_fraction`` of pre-``train_end``
    data — used for early stopping. Test is everything in
    [``test_start``, ``test_end``].

    ``feature_builder`` is an optional callable that takes the OHLCV
    DataFrame and returns a new DataFrame of derived features (e.g.
    :func:`src.features.build_technical_features`). When given, the
    model trains on those features instead of raw OHLCV. NaNs from the
    warmup period are dropped.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have a DatetimeIndex — use load_csv(..., with_dates=True)")

    df = df.sort_index()

    if feature_builder is not None:
        features_full = feature_builder(df)
        usable = features_full.dropna().index
        df = df.loc[usable]
        feat_df = features_full.loc[usable]
    else:
        feat_df = df[FEATURE_COLUMNS]

    train_full = slice_by_date(df, end=train_end)
    test_df = slice_by_date(df, start=test_start, end=test_end)

    if len(train_full) < window_size + 10 or len(test_df) < window_size + 2:
        raise ValueError(
            f"Not enough rows for window={window_size}: "
            f"train={len(train_full)}, test={len(test_df)}"
        )

    train_features = feat_df.loc[train_full.index].values.astype(np.float32)
    train_closes = train_full["Close"].values.astype(np.float32)
    test_features = feat_df.loc[test_df.index].values.astype(np.float32)
    test_closes = test_df["Close"].values.astype(np.float32)

    X_full, y_full, _ = _build_windows(train_features, train_closes, window_size)

    split = int(len(X_full) * (1 - internal_val_fraction))
    X_train, X_val = X_full[:split], X_full[split:]
    y_train, y_val = y_full[:split], y_full[split:]

    X_test, y_test, close_at_t_test = _build_windows(test_features, test_closes, window_size)
    actual_close_test = close_at_t_test * (1 + y_test)
    test_index = test_df.index[window_size:]

    return WindowedReturnsSplit(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        close_at_t_test=close_at_t_test,
        actual_close_test=actual_close_test,
        test_index=test_index,
        train_close_min=float(train_full["Close"].min()),
        train_close_max=float(train_full["Close"].max()),
        window_size=window_size,
        n_features=train_features.shape[1],
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
