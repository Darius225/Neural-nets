"""Walk-forward validation for the ETH + BTC, 5-day horizon model.

The big honest question after evolve_eth_btc_5day.py: was the +0.077
correlation on 2025 a real signal or just one lucky out-of-sample
period? Walk-forward answers this by retraining across multiple
disjoint test years and asking whether the IC is stable.

Six folds, each "train on everything before, test on this year":

    train [2018-01 .. 2019-12]   test 2020
    train [2018-01 .. 2020-12]   test 2021
    train [2018-01 .. 2021-12]   test 2022   (LUNA / FTX shocks)
    train [2018-01 .. 2022-12]   test 2023   (post-crisis recovery)
    train [2018-01 .. 2023-12]   test 2024   (spot-ETF approval)
    train [2018-01 .. 2024-12]   test 2025+  (most recent)

Same model architecture across all folds — the ES-evolved config from
evolve_eth_btc_5day.py — so we test signal robustness, not search
robustness.

Aggregate: mean IC across folds, stdev, t-test against zero. If mean
is positive and stdev is small, the signal is real. If folds disagree
wildly, the 2025 result was a lucky draw.

Run:
    python experiments/walk_forward_eth.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.configs import ReturnsCNNConfig
from src.data import load_csv
from src.data.splits import _build_windows
from src.features import build_technical_features
from src.metrics import compute_metrics
from src.models import build_returns_cnn

ETH_CSV = Path("stock_market_data/crypto/csv/ETHUSDT.csv")
BTC_CSV = Path("stock_market_data/crypto/csv/BTCUSDT.csv")
WINDOW_SIZE = 30
HORIZON = 5
EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 64
SEED = 42

# ES-evolved config from evolve_eth_btc_5day.py — kept identical across
# folds so we test signal robustness, not search-loop robustness.
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

FOLDS = [
    ("2020", "2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2021", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2022", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2025+", "2018-01-01", "2024-12-31", "2025-01-01", "2026-04-30"),
]


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def build_combined(eth: pd.DataFrame, btc: pd.DataFrame):
    eth_f = build_technical_features(eth)
    btc_f = build_technical_features(btc).add_prefix("btc_")
    combined = eth_f.join(btc_f, how="inner").dropna()
    closes = eth.loc[combined.index, "Close"].astype(np.float32)
    return combined, closes


def windows_for_range(features: pd.DataFrame, closes: pd.Series, start: str, end: str):
    mask = (features.index >= pd.to_datetime(start)) & (features.index <= pd.to_datetime(end))
    f = features.loc[mask].values.astype(np.float32)
    c = closes.loc[mask].values.astype(np.float32)
    if len(f) < WINDOW_SIZE + HORIZON:
        return None
    X, y, c_at_t = _build_windows(f, c, WINDOW_SIZE, HORIZON)
    return X, y, c_at_t


def run_fold(features, closes, tr_start, tr_end, te_start, te_end):
    train_pack = windows_for_range(features, closes, tr_start, tr_end)
    test_pack = windows_for_range(features, closes, te_start, te_end)
    if train_pack is None or test_pack is None:
        return None
    X_tr_all, y_tr_all, _ = train_pack
    X_te, y_te, c_te = test_pack

    cut = int(len(X_tr_all) * 0.85)
    X_tr, X_val = X_tr_all[:cut], X_tr_all[cut:]
    y_tr, y_val = y_tr_all[:cut], y_tr_all[cut:]

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_returns_cnn(WINDOW_SIZE, features.shape[1], config=CONFIG)
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
    )
    pred = model.predict(X_te, verbose=0).flatten()
    pred_p = c_te * (1 + pred)
    actual_p = c_te * (1 + y_te)
    metrics = compute_metrics(actual_p, pred_p, y_prev=c_te)
    ic = float(np.corrcoef(pred, y_te)[0, 1]) if pred.std() > 0 and y_te.std() > 0 else float("nan")
    return {
        "n_train": len(X_tr_all),
        "n_test": len(X_te),
        "epochs": len(history.history["loss"]),
        "ic": ic,
        "skill": metrics.skill_vs_persistence,
        "dir_acc": metrics.directional_accuracy,
        "mae": metrics.mae,
        "pred_std_ratio": float(pred.std() / y_te.std()) if y_te.std() > 0 else float("nan"),
    }


def t_stat(ics):
    """One-sample t-statistic against zero. Not Bonferroni-corrected."""
    arr = np.array([x for x in ics if not math.isnan(x)])
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    return float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr))))


def main() -> None:
    for path in (ETH_CSV, BTC_CSV):
        if not path.exists():
            raise SystemExit(
                f"missing {path}. Run:\n"
                f"  python scripts/fetch_binance.py {path.stem} "
                f"--start 2018-01-01 --end 2026-05-17"
            )

    t0 = time.time()
    print("walk-forward validation: ETH-USD + BTC exogen, 5-day horizon")
    print(f"config: {CONFIG.model_dump()}\n")

    eth = load_csv(str(ETH_CSV), with_dates=True)
    btc = load_csv(str(BTC_CSV), with_dates=True)
    features, closes = build_combined(eth, btc)
    print(
        f"data: {features.shape[1]} features, "
        f"{features.index[0].date()}..{features.index[-1].date()}\n"
    )

    print(
        f"{'fold':<7}{'n_train':>8}{'n_test':>8}{'epochs':>8}"
        f"{'IC':>10}{'skill':>10}{'DirAcc%':>10}{'std_ratio':>11}{'MAE$':>10}"
    )
    print("-" * 82)

    rows = []
    for label, ts, te, vs, ve in FOLDS:
        r = run_fold(features, closes, ts, te, vs, ve)
        if r is None:
            print(f"{label:<7}  [skip — not enough data]")
            continue
        rows.append({"fold": label, **r})
        print(
            f"{label:<7}{r['n_train']:>8}{r['n_test']:>8}{r['epochs']:>8}"
            f"{r['ic']:>+10.4f}{r['skill']:>+10.4f}"
            f"{r['dir_acc']:>10.2f}{r['pred_std_ratio']:>11.3f}{r['mae']:>10.2f}"
        )

    if not rows:
        print("\nno folds ran.")
        return

    ics = [r["ic"] for r in rows]
    skills = [r["skill"] for r in rows]
    print("-" * 82)
    print(f"{'mean':<7}{'':>8}{'':>8}{'':>8}{np.mean(ics):>+10.4f}{np.mean(skills):>+10.4f}")
    print(
        f"{'stdev':<7}{'':>8}{'':>8}{'':>8}"
        f"{np.std(ics, ddof=1):>10.4f}{np.std(skills, ddof=1):>10.4f}"
    )
    print(
        f"{'n>0':<7}{'':>8}{'':>8}{'':>8}"
        f"{sum(1 for x in ics if x > 0):>10d}/{len(ics)}"
        f"{sum(1 for x in skills if x > 0):>9d}/{len(skills)}"
    )

    t = t_stat(ics)
    sig = ""
    if not math.isnan(t):
        sig = "  p<0.05" if abs(t) >= 2.57 else "  p<0.10" if abs(t) >= 2.02 else "  NS"
    print(
        f"\nIC mean = {np.mean(ics):+.4f}, "
        f"t-stat against zero = {t:+.2f}{sig}  (n={len(ics)} folds, df={len(ics) - 1})"
    )

    # Honest interpretation hint.
    print()
    if np.mean(ics) > 0.05 and abs(t) > 2.0:
        print("  -> consistent positive IC across folds. Signal looks real.")
    elif np.mean(ics) > 0 and sum(1 for x in ics if x > 0) >= len(ics) * 0.7:
        print("  -> mostly positive IC but borderline significance. Promising,")
        print("     would need more folds / longer history to confirm.")
    else:
        print("  -> IC is unstable across folds. The 2025 result was likely a")
        print("     lucky draw, not a persistent signal. Magnitude calibration")
        print("     or further model tweaks would be premature.")

    print(f"\ntotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
