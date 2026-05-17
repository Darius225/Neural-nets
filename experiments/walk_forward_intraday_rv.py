"""Walk-forward volatility prediction with **proper intraday Realized Variance**.

Current walk_forward_vol_eth.py uses ``y = |log_return[t+1]|`` as the
target — a single noisy estimate of the next day's volatility. This
script replaces it with the gold-standard estimator from Andersen,
Bollerslev, Diebold & Labys (2003):

    RV[t+1] = sqrt( sum of (5-min log returns)^2 over day t+1 )

That's the *integrated variance* of the underlying price process,
estimated from 288 intraday observations per day. It's vastly smoother
than |log_return| (which is one sample of a noisy random variable),
and the GARCH / HAR-RV literature consistently reports skill jumps of
+0.15 to +0.30 from this change alone.

We compare directly against the daily-target baseline by using the
same architecture, same windowing, same walk-forward folds. The only
thing that changes is what the model is trying to predict.

Run:
    python experiments/walk_forward_intraday_rv.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.data import load_csv
from src.data.splits import _zscore_window
from src.features import build_technical_features
from src.models import build_returns_cnn
from src.schemas.configs import ReturnsCNNConfig

INTRADAY_PARQUET = Path("stock_market_data/crypto/intraday/ETHUSDT_5m.parquet")
DAILY_CSV = Path("stock_market_data/crypto/csv/ETHUSDT.csv")

WINDOW_SIZE = 30
EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 64
SEED = 42

CONFIG = ReturnsCNNConfig(
    conv1_filters=64,
    conv1_kernel=3,
    conv2_filters=48,
    conv2_kernel=4,
    dense_units=64,
    dropout=0.3,
    activation="relu",
    huber_delta=0.01,
    learning_rate=0.002,
)

# Test folds — limited by intraday data coverage (2024-01-01 onwards).
FOLDS = [
    ("2025-H1", "2024-01-01", "2024-12-31", "2025-01-01", "2025-06-30"),
    ("2025-H2", "2024-01-01", "2025-06-30", "2025-07-01", "2025-12-31"),
    ("2026", "2024-01-01", "2025-12-31", "2026-01-01", "2026-05-17"),
]


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def compute_daily_rv(intraday_df: pd.DataFrame) -> pd.Series:
    """For each calendar day, compute the realised volatility from
    intraday returns:

        RV[d] = sqrt( sum( log_return_5m^2 ) ) over day d

    Returns a Series indexed by calendar date (DatetimeIndex tz-naive).
    """
    closes = intraday_df["Close"].astype(np.float64)
    intraday_returns = np.log(closes / closes.shift(1))
    sq = intraday_returns**2
    # Group by calendar day (use UTC date)
    daily_rv = sq.groupby(intraday_df.index.normalize()).sum().pipe(np.sqrt).astype(np.float32)
    daily_rv.index = daily_rv.index.tz_localize(None)
    daily_rv.name = "RV"
    return daily_rv


def build_rv_windows(features: np.ndarray, rv: np.ndarray, window_size: int):
    """Same windowing as build_vol_windows but the target is RV[t+1] and
    the persistence baseline is RV[t]."""
    n = len(features)
    needed = window_size + 1
    if n < needed:
        raise ValueError(f"Need at least {needed} rows, got {n}")
    n_windows = n - window_size
    X = np.empty((n_windows, window_size, features.shape[1]), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.float32)
    rv_at_t = np.empty(n_windows, dtype=np.float32)
    for i in range(n_windows):
        X[i] = _zscore_window(features[i : i + window_size])
        y[i] = rv[i + window_size]  # tomorrow's RV (target)
        rv_at_t[i] = rv[i + window_size - 1]  # today's RV (persistence baseline)
    return X, y, rv_at_t


def windows_in_range(features_df: pd.DataFrame, rv_series: pd.Series, start: str, end: str):
    common = features_df.index.intersection(rv_series.index)
    mask = (common >= pd.to_datetime(start)) & (common <= pd.to_datetime(end))
    idx = common[mask]
    if len(idx) < WINDOW_SIZE + 1:
        return None
    f = features_df.loc[idx].values.astype(np.float32)
    r = rv_series.loc[idx].values.astype(np.float32)
    return build_rv_windows(f, r, WINDOW_SIZE)


def run_fold(features_df, rv_series, tr_start, tr_end, te_start, te_end):
    train_pack = windows_in_range(features_df, rv_series, tr_start, tr_end)
    test_pack = windows_in_range(features_df, rv_series, te_start, te_end)
    if train_pack is None or test_pack is None:
        return None
    X_tr_all, y_tr_all, _ = train_pack
    X_te, y_te, rv_at_t = test_pack

    cut = int(len(X_tr_all) * 0.85)
    X_tr, X_val = X_tr_all[:cut], X_tr_all[cut:]
    y_tr, y_val = y_tr_all[:cut], y_tr_all[cut:]

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_returns_cnn(WINDOW_SIZE, X_tr.shape[2], config=CONFIG)
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
    )
    pred = np.maximum(model.predict(X_te, verbose=0).flatten(), 0.0)

    mse_model = float(np.mean((pred - y_te) ** 2))
    mse_persist = float(np.mean((rv_at_t - y_te) ** 2))
    skill = 1.0 - mse_model / mse_persist if mse_persist > 0 else float("nan")
    ic = float(np.corrcoef(pred, y_te)[0, 1]) if pred.std() > 0 and y_te.std() > 0 else float("nan")
    return {
        "n_train": len(X_tr_all),
        "n_test": len(X_te),
        "ic": ic,
        "skill": skill,
        "mae_model": float(np.mean(np.abs(pred - y_te))),
        "mae_persist": float(np.mean(np.abs(rv_at_t - y_te))),
    }


def t_stat(values):
    arr = np.array([v for v in values if not math.isnan(v)])
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    return float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr))))


def main() -> None:
    for path in (INTRADAY_PARQUET, DAILY_CSV):
        if not path.exists():
            raise SystemExit(
                f"missing {path}.\n"
                f"  - intraday parquet: scripts/fetch_binance_intraday.py ETHUSDT --interval 5m\n"
                f"  - daily CSV:        scripts/fetch_binance.py ETHUSDT"
            )

    t0 = time.time()
    print("walk-forward intraday-RV volatility prediction (ETHUSDT)")
    print("target: realised vol from 5-min returns, baseline: yesterday's RV\n")

    print("loading intraday parquet...")
    intraday = pd.read_parquet(INTRADAY_PARQUET)
    print(f"  {len(intraday):,} bars, {intraday.index[0]} .. {intraday.index[-1]}")

    rv_series = compute_daily_rv(intraday)
    print(
        f"  computed {len(rv_series)} daily RV values "
        f"(mean {rv_series.mean() * 100:.3f}%, max {rv_series.max() * 100:.3f}%)"
    )

    print("loading daily CSV for features...")
    daily_df = load_csv(str(DAILY_CSV), with_dates=True)
    features_df = build_technical_features(daily_df).dropna()
    print(f"  {features_df.shape[1]} features over {len(features_df)} days")

    print(f"\n{'fold':<9}{'n_tr':>7}{'n_te':>7}{'IC':>9}{'skill':>9}{'MAE_m%':>10}{'MAE_p%':>10}")
    print("-" * 61)

    rows = []
    for label, ts, te, vs, ve in FOLDS:
        r = run_fold(features_df, rv_series, ts, te, vs, ve)
        if r is None:
            print(f"{label:<9}  [skip — not enough data]")
            continue
        rows.append({"fold": label, **r})
        print(
            f"{label:<9}{r['n_train']:>7}{r['n_test']:>7}"
            f"{r['ic']:>+9.4f}{r['skill']:>+9.4f}"
            f"{r['mae_model'] * 100:>10.4f}{r['mae_persist'] * 100:>10.4f}"
        )

    if not rows:
        return

    skills = [r["skill"] for r in rows]
    print("-" * 61)
    print(f"{'mean':<23}{np.mean([r['ic'] for r in rows]):>+9.4f}{np.mean(skills):>+9.4f}")

    t = t_stat(skills)
    sig = (
        "  p<0.01"
        if abs(t) >= 9.92
        else "  p<0.05"
        if abs(t) >= 4.30
        else "  p<0.10"
        if abs(t) >= 2.92
        else "  NS"
    )
    print(f"\nskill mean = {np.mean(skills):+.4f}, t-stat = {t:+.2f}{sig}  (n={len(skills)} folds)")

    print()
    if np.mean(skills) > 0.45:
        print("  -> Intraday RV target lifts skill into the HAR-RV literature")
        print("     range (+0.40 to +0.65). The 5-min returns give a far less")
        print("     noisy ground truth than single-day |log_return|.")
    elif np.mean(skills) > 0.30:
        print("  -> Intraday RV target provides a modest improvement over")
        print("     daily |log_return|. Headroom for HAR-RV features next.")
    else:
        print("  -> Surprisingly little improvement. Check baseline definition")
        print("     or sample size.")

    print(f"\ntotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
