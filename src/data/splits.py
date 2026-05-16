"""Functions that build the supervised splits.

Four prepare_* functions, one per pipeline version, all returning the
dataclasses defined in :mod:`src.schemas.splits`:

  - ``prepare_dataset``                 -> :class:`Dataset` (v1)
  - ``prepare_train_test_split``        -> :class:`TrainTestSplit` (v1 crisis)
  - ``prepare_windowed_returns_split``  -> :class:`WindowedReturnsSplit` (v2/v3/v5)
  - ``prepare_multi_ticker_split``      -> :class:`MultiTickerSplit` (v4)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ..schemas.splits import (
    Dataset,
    MultiTickerSplit,
    TrainTestSplit,
    WindowedReturnsSplit,
)
from .loaders import FEATURE_COLUMNS, load_csv, slice_by_date


# -------------------------------- helpers ----------------------------------


def _zscore_window(window: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardise each feature within a single window (axis=0)."""
    mu = window.mean(axis=0, keepdims=True)
    sigma = window.std(axis=0, keepdims=True)
    return (window - mu) / (sigma + eps)


def _build_windows(features: np.ndarray, closes: np.ndarray, window_size: int,
                   horizon: int = 1):
    """Sliding windows + per-window z-score. Returns X, y, close_at_t.

    ``horizon`` is the forecast horizon in days — ``y[i]`` is the simple
    return ``(close[t+H] - close[t]) / close[t]`` where ``t`` is the
    last day in window ``i`` and ``H`` is ``horizon``. Default 1 day.
    Longer horizons reduce noise and weaken the persistence baseline.
    """
    n = len(features)
    needed = window_size + horizon
    if n < needed:
        raise ValueError(f"Need at least {needed} rows for window={window_size}, "
                         f"horizon={horizon}; got {n}")

    n_windows = n - window_size - horizon + 1
    X = np.empty((n_windows, window_size, features.shape[1]), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.float32)
    close_at_t = np.empty(n_windows, dtype=np.float32)

    for i in range(n_windows):
        window = features[i : i + window_size]
        X[i] = _zscore_window(window)
        close_t = closes[i + window_size - 1]
        close_th = closes[i + window_size - 1 + horizon]
        y[i] = (close_th - close_t) / close_t   # H-day simple return
        close_at_t[i] = close_t
    return X, y, close_at_t


# ------------------------------ prepare_* ----------------------------------


def prepare_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    scaler: Optional[MinMaxScaler] = None,
) -> Dataset:
    """Scale features and build the next-day-close supervised problem.

    Used by the v1 pipeline only. The scaler is fit on the WHOLE
    dataframe, which leaks val-set statistics — kept for backwards
    compatibility with the original notebook flow. Use
    :func:`prepare_train_test_split` for any honest backtest.
    """
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

    # Next-day-close targets: input row i predicts the close of row i+1.
    X_train = train_scaled[:-1][..., np.newaxis]
    y_train = train_df["Close"].values[1:]

    X_test = test_scaled[:-1][..., np.newaxis]
    y_test = test_df["Close"].values[1:]
    y_test_prev = test_df["Close"].values[:-1]
    test_index = test_df.index[1:]  # the dates being predicted

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


def prepare_windowed_returns_split(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: Optional[str] = None,
    window_size: int = 30,
    internal_val_fraction: float = 0.15,
    feature_builder: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    horizon: int = 1,
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

    ``horizon`` (default 1) sets the forecast horizon in days. With
    ``horizon=5`` the model predicts the 5-day cumulative simple return
    instead of next-day. Persistence becomes a weaker baseline at
    longer horizons.
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

    needed = window_size + horizon
    if len(train_full) < needed + 10 or len(test_df) < needed:
        raise ValueError(
            f"Not enough rows for window={window_size}, horizon={horizon}: "
            f"train={len(train_full)}, test={len(test_df)}"
        )

    train_features = feat_df.loc[train_full.index].values.astype(np.float32)
    train_closes = train_full["Close"].values.astype(np.float32)
    test_features = feat_df.loc[test_df.index].values.astype(np.float32)
    test_closes = test_df["Close"].values.astype(np.float32)

    X_full, y_full, _ = _build_windows(train_features, train_closes, window_size, horizon=horizon)

    split = int(len(X_full) * (1 - internal_val_fraction))
    X_train, X_val = X_full[:split], X_full[split:]
    y_train, y_val = y_full[:split], y_full[split:]

    X_test, y_test, close_at_t_test = _build_windows(
        test_features, test_closes, window_size, horizon=horizon,
    )
    actual_close_test = close_at_t_test * (1 + y_test)
    test_index = test_df.index[window_size + horizon - 1 : window_size + horizon - 1 + len(X_test)]

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


def prepare_multi_ticker_split(
    csv_paths: Dict[str, str],
    train_end: str,
    test_start: str,
    test_end: Optional[str],
    test_tickers: list,
    window_size: int = 30,
    internal_val_fraction: float = 0.15,
    feature_builder: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    min_pre_train_rows: int = 200,
    max_train_tickers: Optional[int] = None,
    target_clip: Optional[float] = 0.2,
    verbose: bool = True,
) -> MultiTickerSplit:
    """Build a combined multi-ticker training set + per-ticker test sets.

    For each ticker in ``csv_paths`` that has enough pre-``train_end``
    history, run the v3 windowed-returns pipeline and contribute its
    windows to the combined ``X_train`` / ``X_val``. For each
    ``test_tickers`` ticker, also store a per-ticker
    :class:`WindowedReturnsSplit` we'll evaluate the trained model on.
    """
    train_X_chunks, train_y_chunks = [], []
    val_X_chunks, val_y_chunks = [], []
    per_ticker_test: Dict[str, WindowedReturnsSplit] = {}
    train_tickers: list = []
    skipped = 0

    items = list(csv_paths.items())
    if max_train_tickers is not None:
        # Always keep the test tickers, then pad with others up to the cap.
        kept = [(t, p) for t, p in items if t in test_tickers]
        others = [(t, p) for t, p in items if t not in test_tickers]
        kept.extend(others[: max_train_tickers - len(kept)])
        items = kept

    for ticker, path in items:
        try:
            df = load_csv(path, with_dates=True)
            pre_train = slice_by_date(df, end=train_end)
            if len(pre_train) < min_pre_train_rows:
                skipped += 1
                continue
            split = prepare_windowed_returns_split(
                df, train_end=train_end, test_start=test_start, test_end=test_end,
                window_size=window_size, internal_val_fraction=internal_val_fraction,
                feature_builder=feature_builder,
            )
            train_X_chunks.append(split.X_train)
            train_y_chunks.append(split.y_train)
            val_X_chunks.append(split.X_val)
            val_y_chunks.append(split.y_val)
            train_tickers.append(ticker)
            if ticker in test_tickers:
                per_ticker_test[ticker] = split
        except Exception as exc:
            skipped += 1
            if verbose:
                print(f"  [skip {ticker}] {exc}")

    if not train_X_chunks:
        raise RuntimeError("No tickers had enough data to build a training set.")

    X_train = np.concatenate(train_X_chunks, axis=0)
    y_train = np.concatenate(train_y_chunks, axis=0)
    X_val = np.concatenate(val_X_chunks, axis=0)
    y_val = np.concatenate(val_y_chunks, axis=0)

    # Clip extreme target returns. Daily moves > target_clip are almost
    # always data artefacts (splits / IPO / bad rows) and they trash MSE
    # training.
    if target_clip is not None:
        n_clipped_train = int(((np.abs(y_train) > target_clip)).sum())
        n_clipped_val = int(((np.abs(y_val) > target_clip)).sum())
        y_train = np.clip(y_train, -target_clip, target_clip)
        y_val = np.clip(y_val, -target_clip, target_clip)
        if verbose and (n_clipped_train or n_clipped_val):
            print(f"  clipped {n_clipped_train} train + {n_clipped_val} val targets "
                  f"to +-{target_clip} (likely splits/IPO data artefacts)")

    if verbose:
        print(f"  built combined training set: {len(X_train):,} train windows, "
              f"{len(X_val):,} val windows from {len(train_tickers)} tickers "
              f"({skipped} skipped)")
        print(f"  per-ticker test sets ready for: {sorted(per_ticker_test)}")

    return MultiTickerSplit(
        X_train=X_train, X_val=X_val,
        y_train=y_train, y_val=y_val,
        per_ticker_test=per_ticker_test,
        n_features=X_train.shape[2],
        window_size=window_size,
        train_tickers=train_tickers,
        test_tickers=[t for t in test_tickers if t in per_ticker_test],
    )
