"""Cross-crypto volatility prediction: 7 Binance pairs.

The walk_forward_vol_eth.py result (skill +0.34 across 6 folds on
ETH + BTC features) raised a question: is the volatility-prediction
finding specific to ETH/BTC, or does it generalise to other large-cap
crypto? This script answers it by running the same v3 pipeline on:

    ETHUSDT, BTCUSDT, SOLUSDT, BNBUSDT, AVAXUSDT, MATICUSDT, ADAUSDT

Each ticker is treated independently — no cross-asset features. For
each one, train on data up to 2024-12-31, predict 2025+ realised
|log_return| daily, and report the skill score vs persistence.

If skill clusters in the +0.20 to +0.35 range for every crypto pair,
the methodology lesson holds: volatility clustering is a universal
property of liquid crypto markets, not a quirk of one or two coins.

Run:
    python experiments/cross_crypto_vol.py
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
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.data import load_csv
from src.data.splits import _zscore_window
from src.features import build_technical_features
from src.models import build_returns_cnn
from src.schemas.configs import ReturnsCNNConfig

TICKERS = ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT", "MATICUSDT", "ADAUSDT"]
CSV_DIR = Path("stock_market_data/crypto/csv")
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-05-17"
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


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


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


def run_ticker(ticker: str):
    path = CSV_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = load_csv(str(path), with_dates=True)
    features_df = build_technical_features(df).dropna()
    closes = df.loc[features_df.index, "Close"].astype(np.float32).values
    features = features_df.values.astype(np.float32)

    # Cut into train + test by date
    train_mask = features_df.index <= TRAIN_END
    test_mask = (features_df.index >= TEST_START) & (features_df.index <= TEST_END)
    if train_mask.sum() < WINDOW_SIZE + 100 or test_mask.sum() < WINDOW_SIZE + 1:
        return None

    train_X, train_y, _ = build_vol_windows(features[train_mask], closes[train_mask])
    test_X, test_y, vol_at_t = build_vol_windows(features[test_mask], closes[test_mask])

    cut = int(len(train_X) * 0.85)
    X_tr, X_val = train_X[:cut], train_X[cut:]
    y_tr, y_val = train_y[:cut], train_y[cut:]

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_returns_cnn(WINDOW_SIZE, features.shape[1], config=CONFIG)
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
    )
    pred = np.maximum(model.predict(test_X, verbose=0).flatten(), 0.0)
    mse_model = float(np.mean((pred - test_y) ** 2))
    mse_persist = float(np.mean((vol_at_t - test_y) ** 2))
    skill = 1.0 - mse_model / mse_persist if mse_persist > 0 else float("nan")
    return {
        "ticker": ticker,
        "n_train_windows": len(train_X),
        "n_test_windows": len(test_X),
        "skill": skill,
        "mae_model": float(np.mean(np.abs(pred - test_y))),
        "mae_persist": float(np.mean(np.abs(vol_at_t - test_y))),
        "pred_mean": float(pred.mean()),
        "actual_mean": float(test_y.mean()),
    }


def main() -> None:
    t0 = time.time()
    print("cross-crypto volatility prediction")
    print(f"train end: {TRAIN_END}, test: {TEST_START} .. {TEST_END}\n")
    print(f"{'ticker':<10}{'n_train':>9}{'n_test':>8}{'skill':>9}{'MAE_m%':>9}{'MAE_p%':>9}")
    print("-" * 54)

    rows = []
    for t in TICKERS:
        r = run_ticker(t)
        if r is None:
            print(f"{t:<10}  [skip — missing or insufficient data]")
            continue
        rows.append(r)
        print(
            f"{r['ticker']:<10}{r['n_train_windows']:>9}{r['n_test_windows']:>8}"
            f"{r['skill']:>+9.4f}{r['mae_model'] * 100:>9.3f}{r['mae_persist'] * 100:>9.3f}"
        )

    if not rows:
        return

    skills = [r["skill"] for r in rows]
    print("-" * 54)
    print(f"{'mean':<27}{np.mean(skills):>+9.4f}")
    print(f"{'stdev':<27}{np.std(skills, ddof=1):>9.4f}")
    print(f"{'n>0':<27}{sum(1 for s in skills if s > 0):>3d}/{len(skills)}")

    t = float(np.mean(skills) / (np.std(skills, ddof=1) / math.sqrt(len(skills))))
    sig = (
        "  p<0.001"
        if abs(t) >= 4.6
        else "  p<0.01"
        if abs(t) >= 3.5
        else "  p<0.05"
        if abs(t) >= 2.4
        else "  NS"
    )
    print(f"\nt-stat vs zero: {t:+.2f}{sig}  (n={len(skills)} tickers)")

    print()
    if np.std(skills, ddof=1) < 0.10:
        print("  -> Skills cluster tightly across tickers. Volatility clustering")
        print("     is a universal property of liquid crypto, not coin-specific.")
        print("     Adding more tickers gives diminishing returns: the same")
        print("     underlying signal is being re-confirmed each time.")
    else:
        print("  -> Skills vary noticeably across tickers. Some coins have")
        print("     more predictable vol than others.")

    print(f"\ntotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
