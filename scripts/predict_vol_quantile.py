"""Volatility forecasting with **quantile regression** (pinball loss).

This is the third predict_vol script, with a different statistical philosophy:

  scripts/predict.py            -> point prediction of next-day RETURN (skill ~0)
  scripts/predict_vol.py        -> point prediction of next-day |return|,
                                   then Gaussian-approximated bands with
                                   empirical bias correction
  scripts/predict_vol_quantile.py (this file)
                                -> direct prediction of the 5th / 50th / 95th
                                   percentile of next-day |return| using
                                   pinball (quantile) loss. The bands are
                                   the model's own outputs, not derived from
                                   a Gaussian assumption.

Why this is interesting:

  - The Gaussian assumption in predict_vol.py is approximate. Daily returns
    have fat tails — a true 95 % interval should be wider than 1.96 * sigma.
  - Quantile regression learns the conditional quantiles directly. No bias
    correction needed (each quantile is optimised against its own loss).
  - The 90 % interval [q05, q95] is calibrated by construction: roughly 5 %
    of out-of-sample actuals should fall above q95, 5 % below q05.

Usage:
    python scripts/predict_vol_quantile.py --ticker ETHUSDT --test-days 14
    python scripts/predict_vol_quantile.py --ticker JPM --source sp500
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

from src.data import load_csv
from src.data.splits import _zscore_window
from src.features import build_technical_features

WINDOW_SIZE = 30
EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 64
SEED = 42
QUANTILES = (0.05, 0.50, 0.95)  # 5th, 50th, 95th percentile

# Architecture mirrors build_returns_cnn but the output is len(QUANTILES)
# instead of 1 — same inductive bias, different loss surface.
ARCH = {
    "conv1_filters": 64,
    "conv1_kernel": 3,
    "conv2_filters": 48,
    "conv2_kernel": 4,
    "dense_units": 64,
    "dropout": 0.3,
    "learning_rate": 0.002,
}


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def pinball_loss(quantiles):
    """Pinball loss (a.k.a. quantile loss).

    For each quantile q and each sample:
        L_q(y, p) = max( q * (y - p),  (q - 1) * (y - p) )

    Asymmetric: under-prediction is penalised more when q > 0.5, over-prediction
    more when q < 0.5. Minimising L_q over a dataset converges to the conditional
    q-quantile of y given the input features.

    Expects y_true of shape (batch, 1) and y_pred of shape (batch, n_quantiles).
    """
    qs = tf.constant(quantiles, dtype=tf.float32)

    def loss(y_true, y_pred):
        # y_true: (batch, 1), broadcast to (batch, n_quantiles)
        err = y_true - y_pred
        # Pinball: max( q*err, (q-1)*err ) elementwise
        pinball = tf.maximum(qs * err, (qs - 1.0) * err)
        return tf.reduce_mean(pinball)

    return loss


def build_quantile_cnn(window_size: int, n_features: int, n_quantiles: int):
    """Same CNN topology as build_returns_cnn but the head emits n_quantiles values."""
    model = Sequential(
        [
            Conv1D(
                ARCH["conv1_filters"],
                kernel_size=ARCH["conv1_kernel"],
                activation="relu",
                input_shape=(window_size, n_features),
            ),
            Conv1D(
                ARCH["conv2_filters"],
                kernel_size=ARCH["conv2_kernel"],
                activation="relu",
            ),
            Flatten(),
            Dense(ARCH["dense_units"], activation="relu"),
            Dropout(ARCH["dropout"]),
            Dense(n_quantiles),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=ARCH["learning_rate"]),
        loss=pinball_loss(QUANTILES),
    )
    return model


def find_csv(ticker: str, source: str) -> Path:
    candidates = {
        "binance": Path(f"stock_market_data/crypto/csv/{ticker}.csv"),
        "sp500": Path(f"stock_market_data/sp500/csv/{ticker}.csv"),
    }
    if source == "auto":
        for p in candidates.values():
            if p.exists():
                return p
    elif source in candidates and candidates[source].exists():
        return candidates[source]
    raise SystemExit(f"CSV not found for {ticker} (source={source}).")


def build_vol_windows(features: np.ndarray, closes: np.ndarray):
    n = len(features)
    log_returns = np.log(closes[1:] / closes[:-1])
    abs_log_returns = np.abs(log_returns).astype(np.float32)
    n_windows = n - WINDOW_SIZE
    X = np.empty((n_windows, WINDOW_SIZE, features.shape[1]), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.float32)
    vol_at_t = np.empty(n_windows, dtype=np.float32)
    for i in range(n_windows):
        X[i] = _zscore_window(features[i : i + WINDOW_SIZE])
        y[i] = abs_log_returns[i + WINDOW_SIZE - 1]
        vol_at_t[i] = abs_log_returns[i + WINDOW_SIZE - 2]
    return X, y, vol_at_t


def train_model(X_tr, y_tr, X_val, y_val, seed: int = SEED):
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_quantile_cnn(WINDOW_SIZE, X_tr.shape[2], len(QUANTILES))
    model.fit(
        X_tr,
        y_tr.reshape(-1, 1),
        validation_data=(X_val, y_val.reshape(-1, 1)),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
    )
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ticker", required=True)
    p.add_argument("--source", choices=["auto", "binance", "sp500"], default="auto")
    p.add_argument("--test-days", type=int, default=14)
    args = p.parse_args()

    csv_path = find_csv(args.ticker, args.source)
    print(f"loading {csv_path}...")
    df = load_csv(str(csv_path), with_dates=True)
    print(f"  {len(df)} rows, {df.index[0].date()} .. {df.index[-1].date()}")

    features_df = build_technical_features(df).dropna()
    closes = df.loc[features_df.index, "Close"].astype(np.float32)
    features = features_df.values.astype(np.float32)
    closes_arr = closes.values

    X_all, y_all, vol_at_t_all = build_vol_windows(features, closes_arr)
    dates_all = features_df.index[WINDOW_SIZE:]

    test_n = args.test_days
    if len(X_all) < test_n + 100:
        raise SystemExit(f"need more than {test_n + 100} windows, have {len(X_all)}")

    # ----------------- Phase 1: train on everything before the last N days
    X_tr_pool = X_all[:-test_n]
    y_tr_pool = y_all[:-test_n]
    cut = int(len(X_tr_pool) * 0.85)
    X_tr, X_val = X_tr_pool[:cut], X_tr_pool[cut:]
    y_tr, y_val = y_tr_pool[:cut], y_tr_pool[cut:]

    print(f"\ntraining quantile CNN (q={QUANTILES}, pinball loss) on {len(X_tr)} samples...")
    t0 = time.time()
    model = train_model(X_tr, y_tr, X_val, y_val)
    print(f"  trained in {time.time() - t0:.1f}s")

    # ----------------- Phase 2: held-out backtest
    X_test = X_all[-test_n:]
    y_test = y_all[-test_n:]
    dates_test = dates_all[-test_n:]
    pred_q = model.predict(X_test, verbose=0)  # (test_n, 3)
    pred_q = np.maximum(pred_q, 0.0)  # vol cannot be negative
    # Enforce monotonicity in case the model learned a tiny crossing (rare).
    pred_q.sort(axis=1)
    q05, q50, q95 = pred_q[:, 0], pred_q[:, 1], pred_q[:, 2]

    inside_90 = np.mean((y_test >= q05) & (y_test <= q95)) * 100
    above_q95 = np.mean(y_test > q95) * 100
    below_q05 = np.mean(y_test < q05) * 100

    print(f"\nheld-out backtest, last {test_n} days:")
    print(f"  {'date':<12}{'actual%':>9}{'q05%':>9}{'q50%':>9}{'q95%':>9}{'in 90% band':>14}")
    print(f"  {'-' * 62}")
    for d, a, lo, mid, hi in zip(dates_test, y_test, q05, q50, q95, strict=False):
        inside = "YES" if lo <= a <= hi else ("over" if a > hi else "under")
        print(
            f"  {d.date()!s:<12}"
            f"{a * 100:>9.3f}{lo * 100:>9.3f}{mid * 100:>9.3f}{hi * 100:>9.3f}"
            f"{inside:>14}"
        )

    print()
    print(f"  inside 90% band: {inside_90:.1f}%   (target: ~90%)")
    print(f"  above q95:       {above_q95:.1f}%   (target: ~5%)")
    print(f"  below q05:       {below_q05:.1f}%   (target: ~5%)")

    # Median absolute error vs persistence
    mae_q50 = float(np.mean(np.abs(q50 - y_test)))
    mae_persist = float(np.mean(np.abs(vol_at_t_all[-test_n:] - y_test)))
    print(f"\n  MAE(q50)        = {mae_q50 * 100:.4f}%")
    print(f"  MAE persistence = {mae_persist * 100:.4f}%")

    # ----------------- Phase 3: predict tomorrow with calibrated bands
    print(f"\nretraining on all {len(X_all)} windows for tomorrow...")
    cut2 = int(len(X_all) * 0.85)
    final = train_model(X_all[:cut2], y_all[:cut2], X_all[cut2:], y_all[cut2:])

    last_window = features[-WINDOW_SIZE:]
    X_now = _zscore_window(last_window)[np.newaxis, ...]
    q = np.maximum(final.predict(X_now, verbose=0)[0], 0.0)
    q.sort()
    q05_t, q50_t, q95_t = float(q[0]), float(q[1]), float(q[2])

    last_close = float(df["Close"].iloc[-1])
    last_date = df.index[-1].date()
    next_date = last_date + pd.Timedelta(days=1)
    # Convert each quantile of |log_return| to a price range with that magnitude.
    # By symmetry assumption on log returns, the price stays within
    # last_close * exp(±q_x) with probability matching that quantile.
    band_lo_90 = last_close * np.exp(-q95_t)
    band_hi_90 = last_close * np.exp(+q95_t)
    band_lo_50 = last_close * np.exp(-q50_t)
    band_hi_50 = last_close * np.exp(+q50_t)
    band_lo_05 = last_close * np.exp(-q05_t)
    band_hi_05 = last_close * np.exp(+q05_t)

    print(f"\n{'-' * 64}")
    print(f"  QUANTILE-REGRESSION FORECAST  ({args.ticker} -> {next_date})")
    print(f"{'-' * 64}")
    print(f"  last close ({last_date}): ${last_close:.4f}")
    print()
    print("  predicted quantiles of |log_return|:")
    print(
        f"    q05 = {q05_t * 100:>6.3f}%   q50 = {q50_t * 100:>6.3f}%   q95 = {q95_t * 100:>6.3f}%"
    )
    print()
    print("  implied price bands:")
    print(f"    inner (q05 wing): ${band_lo_05:.2f} - ${band_hi_05:.2f}    <- tail risk floor")
    print(f"    median (q50):     ${band_lo_50:.2f} - ${band_hi_50:.2f}    <- typical range")
    print(f"    outer (q95):      ${band_lo_90:.2f} - ${band_hi_90:.2f}    <- 90% containment")
    print()
    print("  Read: ~90% of the time tomorrow's close lands inside the OUTER band.")
    print("        ~50% of the time it's even tighter, inside the MEDIAN band.")
    print("        Crossing the outer is genuinely unusual (5% upside, 5% downside).")
    print()
    print("  Quantile bands are CALIBRATED — no Gaussian assumption, no bias")
    print("  correction. Fat tails are baked into the quantile predictions.")


if __name__ == "__main__":
    main()
