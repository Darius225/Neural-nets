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


def build_vol_windows(features: np.ndarray, closes: np.ndarray, horizon: int = 1):
    """Sliding windows + per-window z-score.

    For ``horizon == 1`` the target is ``|log_return[t+1]|`` and the
    persistence baseline is yesterday's |log_return|.

    For ``horizon > 1`` the target is the realised volatility over the
    next ``h`` days:
        y[i] = sqrt(sum_{k=1..h} log_return[t+k]^2)
    and the baseline is the realised volatility over the *previous*
    ``h`` days. Bigger h = smoother target, weaker persistence
    baseline, lower-noise predictions.
    """
    n = len(features)
    log_returns = np.log(closes[1:] / closes[:-1]).astype(np.float32)  # length n-1
    sq = log_returns**2

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n < WINDOW_SIZE + horizon:
        raise ValueError(f"Need at least {WINDOW_SIZE + horizon} rows, got {n}")

    n_windows = n - WINDOW_SIZE - horizon + 1
    X = np.empty((n_windows, WINDOW_SIZE, features.shape[1]), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.float32)
    baseline = np.empty(n_windows, dtype=np.float32)

    for i in range(n_windows):
        X[i] = _zscore_window(features[i : i + WINDOW_SIZE])
        # Target window: next `horizon` log returns after the input window.
        # log_returns is indexed 0..n-2; the last day inside input window
        # corresponds to log_returns index (i + WINDOW_SIZE - 2).
        future_start = i + WINDOW_SIZE - 1
        y[i] = float(np.sqrt(sq[future_start : future_start + horizon].sum()))
        past_start = i + WINDOW_SIZE - 1 - horizon
        baseline[i] = float(np.sqrt(sq[past_start : past_start + horizon].sum()))
    return X, y, baseline


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
        help="how many recent days to back-test the 1-day model on (default 14)",
    )
    p.add_argument(
        "--horizons",
        default="1,5,10",
        help="comma-separated forecast horizons in days for the final fan forecast "
        "(default '1,5,10'). One model is retrained per horizon.",
    )
    args = p.parse_args()
    horizons = sorted({int(h) for h in args.horizons.split(",") if h.strip()})
    if not horizons or min(horizons) < 1:
        raise SystemExit("--horizons must be a comma list of positive integers, e.g. '1,5,10'")

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
    # Phase 3 — retrain on ALL data, predict each requested horizon
    # ------------------------------------------------------------------
    last_window = features[-WINDOW_SIZE:]
    X_now = _zscore_window(last_window)[np.newaxis, ...]
    last_date = df.index[-1].date()
    last_close = float(df["Close"].iloc[-1])

    # Multipliers that convert the model's predicted |log_return| / RV
    # into N-percent price bands. Assumes log-returns ~ N(0, sigma^2)
    # where the model's output is an estimate of E[|X|] = sigma * sqrt(2/pi).
    # So sigma_hat = pred * sqrt(pi/2). z-scores for two-sided bands:
    BANDS = [
        ("50%", 0.6745),
        ("68%", 1.0000),
        ("95%", 1.9600),
    ]
    EABS_TO_SIGMA = float(np.sqrt(np.pi / 2))  # ~1.2533

    print(f"\nretraining one model per horizon h in {horizons} for the fan forecast...")
    horizon_results = []
    for h in horizons:
        try:
            X_h, y_h, _ = build_vol_windows(features, closes_arr, horizon=h)
        except ValueError as exc:
            print(f"  [skip h={h}] {exc}")
            continue
        cut_h = int(len(X_h) * 0.85)
        model_h = train_model(X_h[:cut_h], y_h[:cut_h], X_h[cut_h:], y_h[cut_h:])

        # Empirical bias correction from the validation slice — fixes the
        # systematic under-prediction observed across all our experiments
        # (pred_mean / actual_mean ~ 0.7 on ETH; the model is conservative).
        val_pred = np.maximum(model_h.predict(X_h[cut_h:], verbose=0).flatten(), 0.0)
        val_actual = y_h[cut_h:]
        if val_pred.mean() > 0:
            calibration = float(val_actual.mean() / val_pred.mean())
        else:
            calibration = 1.0

        pred_raw = float(np.maximum(model_h.predict(X_now, verbose=0)[0, 0], 0.0))
        pred_cal = pred_raw * calibration  # bias-corrected
        sigma_hat = pred_cal * EABS_TO_SIGMA  # convert E[|X|] -> sigma

        target_date = last_date + pd.Timedelta(days=h)
        bands = {}
        for label, z in BANDS:
            bands[label] = (
                last_close * float(np.exp(-z * sigma_hat)),
                last_close * float(np.exp(+z * sigma_hat)),
            )

        horizon_results.append(
            {
                "h": h,
                "pred_raw": pred_raw,
                "calibration": calibration,
                "pred_cal": pred_cal,
                "sigma_hat": sigma_hat,
                "bands": bands,
                "target_date": target_date,
            }
        )
        print(
            f"  h={h:>2}d raw_pred={pred_raw * 100:>5.2f}%  "
            f"calibration={calibration:>4.2f}x  "
            f"sigma_hat={sigma_hat * 100:>5.2f}%  "
            f"(by {target_date})"
        )

    print(f"\n{'-' * 78}")
    print(f"  CALIBRATED FAN FORECAST ({args.ticker})")
    print(f"{'-' * 78}")
    print(f"  last close ({last_date}): ${last_close:.4f}")
    print("  bands derived from bias-corrected sigma, assuming Gaussian log-returns.\n")
    print(f"  {'horizon':<12}{'sigma':>10}{'50% band':>20}{'68% band':>20}{'95% band':>20}")
    print(f"  {'-' * 80}")
    for r in horizon_results:
        b50 = r["bands"]["50%"]
        b68 = r["bands"]["68%"]
        b95 = r["bands"]["95%"]
        print(
            f"  {r['h']:>2}-day fwd  {r['sigma_hat'] * 100:>8.2f}%"
            f"  ${b50[0]:>7.2f}-${b50[1]:>7.2f}"
            f"  ${b68[0]:>7.2f}-${b68[1]:>7.2f}"
            f"  ${b95[0]:>7.2f}-${b95[1]:>7.2f}"
        )

    print()
    print("  How to read this:")
    print("    50% band = most likely outcome (coin flip if price stays inside).")
    print("    68% band = standard 1-sigma; typical risk-management default.")
    print("    95% band = stress range; price exits with ~5% probability.")
    print()
    print("  Bias correction: validation set told us the raw CNN under-predicts")
    print("  magnitude by the factor shown above. The bands use the calibrated")
    print("  sigma, so they're closer to the empirical truth than the raw model.")
    print("  These remain MAGNITUDE forecasts — they bound the price, not direct it.")


if __name__ == "__main__":
    main()
