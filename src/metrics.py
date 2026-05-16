"""Metric functions for next-day-close stock prediction.

Going beyond MAE/MAPE because, for stocks, those numbers can look great
while the model is useless — a "predict tomorrow = today" baseline is
remarkably hard to beat. The metrics here are designed to expose that.

The result schema lives in :class:`src.schemas.metrics.PredictionMetrics`.
This module hosts the *functions* that compute it.

Key concept: the **skill score** vs naive persistence is the single most
honest measure. ``skill_vs_persistence(y_true, y_pred, y_prev) > 0`` means
the model is genuinely doing better than just echoing yesterday's close.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .schemas.metrics import PredictionMetrics


def naive_persistence_forecast(y_prev: np.ndarray) -> np.ndarray:
    """The do-nothing baseline: pred[t+1] = close[t]. This is what every
    stock predictor must beat to justify its existence."""
    return np.asarray(y_prev, dtype=float)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float:
    """% of days where sign(pred - prev) matches sign(actual - prev).

    The relevant question is not "is the predicted value close" but
    "did we get the direction right" — that's what determines whether
    you'd make money trading on the prediction.
    """
    true_dir = np.sign(np.asarray(y_true) - np.asarray(y_prev))
    pred_dir = np.sign(np.asarray(y_pred) - np.asarray(y_prev))
    # Days with no change in either count as ties — exclude them.
    mask = (true_dir != 0) & (pred_dir != 0)
    if not mask.any():
        return float("nan")
    return float((true_dir[mask] == pred_dir[mask]).mean() * 100)


def skill_score(mse_model: float, mse_baseline: float) -> float:
    """Murphy skill score: 1 - MSE_model / MSE_baseline.

    > 0 : model beats baseline (the more, the better).
    = 0 : model is as good as baseline.
    < 0 : model is *worse* than just echoing yesterday's close.
    """
    if mse_baseline <= 0:
        return float("nan")
    return 1.0 - mse_model / mse_baseline


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: Optional[np.ndarray] = None,
    train_min: Optional[float] = None,
    train_max: Optional[float] = None,
) -> PredictionMetrics:
    """Compute the full metric bundle.

    ``y_prev`` is the previous day's close at each timestep — needed for
    directional accuracy and the persistence skill score. Pass ``None``
    to skip those.

    ``train_min`` / ``train_max`` are the price range seen during
    training; if both given, we report what fraction of test prices fell
    outside that range — a useful regime-shift indicator.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_pred - y_true

    mae = float(np.mean(np.abs(residual)))
    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs(residual / y_true)) * 100)
    mean_signed = float(np.mean(residual))
    worst = float(np.max(np.abs(residual)))

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    dir_acc = float("nan")
    skill = None
    if y_prev is not None:
        y_prev = np.asarray(y_prev, dtype=float)
        dir_acc = directional_accuracy(y_true, y_pred, y_prev)
        mse_baseline = float(np.mean((y_prev - y_true) ** 2))
        skill = skill_score(mse, mse_baseline)

    out_of_range = None
    if train_min is not None and train_max is not None:
        out_of_range = float(((y_true < train_min) | (y_true > train_max)).mean() * 100)

    return PredictionMetrics(
        n=len(y_true),
        mae=mae,
        rmse=rmse,
        mape=mape,
        r2=r2,
        directional_accuracy=dir_acc,
        mean_signed_error=mean_signed,
        worst_day_error=worst,
        skill_vs_persistence=skill,
        out_of_train_range_pct=out_of_range,
    )
