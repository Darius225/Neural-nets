"""Schemas for the supervised-data splits used by the four pipeline versions.

These are the *shapes* the data takes after preparation. The functions
that produce them live in :mod:`src.data.splits`; keeping the
dataclasses here makes the data-flow contract easy to read in one
file without the windowing / target-building logic in the way.

We keep these as plain dataclasses rather than Pydantic because every
field is a numpy array — Pydantic's per-field validation has nothing
to check on an ndarray and would only add construction overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass
class Dataset:
    """Bundle of arrays ready to feed a Conv1D model (v1 pipeline)."""

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
    """Train/val/test for the windowed-returns pipeline (v2 onwards).

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
    close_at_t_test: np.ndarray  # close[t] for each test sample
    actual_close_test: np.ndarray  # close[t+1] — what we're predicting
    test_index: pd.Index
    train_close_min: float
    train_close_max: float
    window_size: int
    n_features: int

    @property
    def input_shape(self) -> tuple[int, int]:
        return self.window_size, self.n_features


@dataclass
class MultiTickerSplit:
    """Combined training set across many tickers + per-ticker test sets (v4)."""

    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    per_ticker_test: dict[str, WindowedReturnsSplit]
    n_features: int
    window_size: int
    train_tickers: list
    test_tickers: list

    @property
    def input_shape(self) -> tuple[int, int]:
        return self.window_size, self.n_features
