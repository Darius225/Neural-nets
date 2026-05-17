"""Predict next-day volatility for any ticker + back-test on recent N days.

The companion to scripts/predict.py (which serves the return-prediction
model). This script serves the *volatility* model — the one that
actually has skill > 0.

Workflow:
  1. Load OHLCV from a local CSV (Binance crypto or Kaggle stocks).
  2. Train the CNN volatility model on everything *except* the last
     ``--test-days`` days.
  3. Roll forward one day at a time over those held-out days: predict
     tomorrow's |log_return|, then reveal the actual value, then move
     the window forward. Print a table.
  4. Train one more time on ALL available data and predict the next
     unobserved day's volatility.

Usage:
    python scripts/predict_vol.py --ticker ETHUSDT --test-days 14
    python scripts/predict_vol.py --ticker JPM --source sp500 --test-days 30
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
from tensorflow.keras.callbacks import EarlyStopping

from src.data import load_csv
from src.data.splits import _zscore_window
from src.features import build_technical_features
from src.models import build_returns_cnn
from src.schemas.configs import ReturnsCNNConfig

WINDOW_SIZE = 30
EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 64
SEED = 42

# Same ES-evolved config used in the walk-forward volatility experiments.
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


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


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
    raise SystemExit(
        f"CSV not found for {ticker} (source={source}). "
        f"Looked in {[str(p) for p in candidates.values()]}"
    )


def build_vol_windows(features: np.ndarray, closes: np.ndarray):
    """Sliding windows + per-window z-score, target = |log_return[t+1]|."""
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


def train_model(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
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
    return model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--ticker", required=True, help="ETHUSDT / BTCUSDT for crypto, JPM / AAPL / ... for stocks"
    )
    p.add_argument("--source", choices=["auto", "binance", "sp500"], default="auto")
    p.add_argument(
        "--test-days",
        type=int,
        default=14,
        help="how many recent days to back-test the model on (default 14)",
    )
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
    dates_all = features_df.index[WINDOW_SIZE:]  # dates the targets refer to

    test_n = args.test_days
    if len(X_all) < test_n + 100:
        raise SystemExit(f"need more than {test_n + 100} windows, have {len(X_all)}")

    # ------------------------------------------------------------------
    # Phase 1 — train on everything EXCEPT the last test_n windows
    # ------------------------------------------------------------------
    X_tr_pool = X_all[:-test_n]
    y_tr_pool = y_all[:-test_n]
    cut = int(len(X_tr_pool) * 0.85)
    X_tr, X_val = X_tr_pool[:cut], X_tr_pool[cut:]
    y_tr, y_val = y_tr_pool[:cut], y_tr_pool[cut:]

    print(f"\ntraining on {len(X_tr)} samples (last {test_n} days held out)...")
    t0 = time.time()
    model = train_model(X_tr, y_tr, X_val, y_val)
    print(f"  trained in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Phase 2 — predict each of the test_n held-out days
    # ------------------------------------------------------------------
    X_test = X_all[-test_n:]
    y_test = y_all[-test_n:]
    persist_test = vol_at_t_all[-test_n:]
    dates_test = dates_all[-test_n:]

    preds = np.maximum(model.predict(X_test, verbose=0).flatten(), 0.0)

    print(f"\nbacktest on last {test_n} held-out days:")
    print(
        f"  {'date':<12}{'pred_vol%':>10}{'actual%':>10}{'persist%':>10}"
        f"{'error_m':>10}{'error_p':>10}{'winner':>10}"
    )
    print(f"  {'-' * 82}")
    n_wins = 0
    for d, p_, a_, pe_ in zip(dates_test, preds, y_test, persist_test):
        err_m = abs(p_ - a_)
        err_p = abs(pe_ - a_)
        winner = "MODEL" if err_m < err_p else "persist"
        if winner == "MODEL":
            n_wins += 1
        print(
            f"  {d.date()!s:<12}{p_ * 100:>10.3f}{a_ * 100:>10.3f}{pe_ * 100:>10.3f}"
            f"{err_m * 100:>10.3f}{err_p * 100:>10.3f}{winner:>10}"
        )

    mae_m = float(np.mean(np.abs(preds - y_test)))
    mae_p = float(np.mean(np.abs(persist_test - y_test)))
    mse_m = float(np.mean((preds - y_test) ** 2))
    mse_p = float(np.mean((persist_test - y_test) ** 2))
    skill = 1.0 - mse_m / mse_p if mse_p > 0 else float("nan")

    print(f"\n  MAE model       = {mae_m * 100:.4f}%")
    print(f"  MAE persistence = {mae_p * 100:.4f}%")
    print(f"  skill score     = {skill:+.4f}")
    print(f"  model wins      = {n_wins}/{test_n} days")

    # ------------------------------------------------------------------
    # Phase 3 — retrain on ALL data and predict the NEXT day
    # ------------------------------------------------------------------
    print(f"\nretraining on all {len(X_all)} windows for tomorrow's prediction...")
    cut2 = int(len(X_all) * 0.85)
    X_tr2, X_val2 = X_all[:cut2], X_all[cut2:]
    y_tr2, y_val2 = y_all[:cut2], y_all[cut2:]
    final_model = train_model(X_tr2, y_tr2, X_val2, y_val2)

    last_window = features[-WINDOW_SIZE:]
    X_now = _zscore_window(last_window)[np.newaxis, ...]
    pred_tomorrow = float(np.maximum(final_model.predict(X_now, verbose=0)[0, 0], 0.0))

    last_date = df.index[-1].date()
    next_date = last_date + pd.Timedelta(days=1)
    last_close = float(df["Close"].iloc[-1])

    # Translate predicted |log_return| into a price range
    pred_up = last_close * np.exp(+pred_tomorrow)
    pred_down = last_close * np.exp(-pred_tomorrow)

    print(f"\n{'-' * 60}")
    print(f"  TOMORROW'S VOLATILITY PREDICTION ({args.ticker})")
    print(f"{'-' * 60}")
    print(f"  last close ({last_date}):        ${last_close:.4f}")
    print(f"  predicted |log_return| ({next_date}): {pred_tomorrow * 100:.3f}%")
    print(f"  implied 1-sigma range ({next_date}): ${pred_down:.4f} .. ${pred_up:.4f}")
    print()
    print("  This is a volatility forecast — it predicts MAGNITUDE of")
    print("  tomorrow's move, NOT direction. Skill above is over the past")
    print(f"  {test_n} days. Treat as risk / position-sizing input, not a")
    print("  trading signal.")


if __name__ == "__main__":
    main()
