"""Supervised splits used by the four pipeline versions.

Four split shapes, increasing in sophistication:

  - ``Dataset`` + ``prepare_dataset``                — random shuffle-free
    train/val split on per-day OHLCV (v1 pipeline).
  - ``TrainTestSplit`` + ``prepare_train_test_split`` — calendar-date
    split, scaler fit on train only (v1 crisis test).
  - ``WindowedReturnsSplit`` + ``prepare_windowed_returns_split`` —
    30-day windows, per-window z-score, next-day return target
    (v2 / v3 / v5 pipelines, optionally with technical features).
  - ``MultiTickerSplit`` + ``prepare_multi_ticker_split`` — windows
    from many tickers stacked into one combined training set (v4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .loaders import FEATURE_COLUMNS, load_csv, slice_by_date


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
    accuracy. ``train_close_min`` / ``max`` are the price range seen
    during training, for out-of-range detection on the test set.
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
    close_at_t_test: np.ndarray   # close[t] for each test sample
    actual_close_test: np.ndarray  # close[t+1] — what we're predicting
    test_index: pd.Index
    train_close_min: float
    train_close_max: float
    window_size: int
    n_features: int

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.window_size, self.n_features


@dataclass
class MultiTickerSplit:
    """Combined training set across many tickers + per-ticker test sets.

    The premise: each ticker is just one realisation of "how stocks
    behave". Train on samples from many tickers to learn the common
    structure, then evaluate per-ticker on the held-out crisis window.

    All inputs use the same windowed-returns shape (per-window z-score),
    so a single CNN trained on the combined set can score any test
    ticker.
    """
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    per_ticker_test: Dict[str, "WindowedReturnsSplit"]
    n_features: int
    window_size: int
    train_tickers: list
    test_tickers: list

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.window_size, self.n_features


# -------------------------------- helpers ----------------------------------


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

    n_windows = n - window_size  # last window has no next-day target
    X = np.empty((n_windows, window_size, features.shape[1]), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.float32)
    close_at_t = np.empty(n_windows, dtype=np.float32)

    for i in range(n_windows):
        window = features[i : i + window_size]
        X[i] = _zscore_window(window)
        close_t = closes[i + window_size - 1]
        close_t1 = closes[i + window_size]
        y[i] = (close_t1 - close_t) / close_t   # simple return
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
