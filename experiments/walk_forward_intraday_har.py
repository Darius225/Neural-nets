"""HAR-RV from intraday: features and target both at intraday precision.

walk_forward_intraday_rv.py tried to predict intraday-derived RV using
DAILY-derived features. That precision mismatch was the failure mode —
the model's inputs were noisier than the target it was asked to match.

This script fixes that. We compute the classical HAR components
(Corsi 2009) directly FROM the 5-min bars:

    rv_intra_1[d]  = sqrt( sum( log_return_5m^2 ) over day d )
    rv_intra_5[d]  = mean( rv_intra_1 over last 5 days )
    rv_intra_22[d] = mean( rv_intra_1 over last 22 days )

These three are the canonical predictors of next-day realised vol in
the literature. They join the existing 13 daily technical features for
a final feature set of 16 columns, and the target is the same
intraday-derived RV[t+1] used previously. Persistence baseline is
yesterday's RV.

Run:
    python experiments/walk_forward_intraday_har.py
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

FOLDS = [
    ("2020", "2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2021", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2022", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2025+", "2018-01-01", "2024-12-31", "2025-01-01", "2026-05-17"),
]


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def intraday_to_daily_rv(intraday_df: pd.DataFrame) -> pd.Series:
    """RV[d] = sqrt( sum( log_return_5m^2 ) over day d ). Returns
    a tz-naive Series indexed by date."""
    closes = intraday_df["Close"].astype(np.float64)
    intra_returns = np.log(closes / closes.shift(1))
    sq = intra_returns**2
    daily_rv = sq.groupby(intraday_df.index.normalize()).sum().pipe(np.sqrt)
    daily_rv.index = daily_rv.index.tz_localize(None)
    return daily_rv.astype(np.float32).rename("rv_intra_1")


def har_features_from_intraday(daily_rv: pd.Series) -> pd.DataFrame:
    """Three HAR-RV components at 1 / 5 / 22 day horizons, all
    *computed from intraday RV*. This is the proper HAR-RV input
    (Corsi 2009 + Andersen et al. 2003).
    """
    har = pd.DataFrame(index=daily_rv.index)
    har["rv_intra_1"] = daily_rv
    har["rv_intra_5"] = daily_rv.rolling(5).mean()
    har["rv_intra_22"] = daily_rv.rolling(22).mean()
    return har


def build_har_windows(features: np.ndarray, rv: np.ndarray, window_size: int):
    n = len(features)
    if n < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} rows, got {n}")
    n_windows = n - window_size
    X = np.empty((n_windows, window_size, features.shape[1]), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.float32)
    rv_at_t = np.empty(n_windows, dtype=np.float32)
    for i in range(n_windows):
        X[i] = _zscore_window(features[i : i + window_size])
        y[i] = rv[i + window_size]
        rv_at_t[i] = rv[i + window_size - 1]
    return X, y, rv_at_t


def windows_in_range(features_df, rv_series, start, end):
    common = features_df.index.intersection(rv_series.index)
    mask = (common >= pd.to_datetime(start)) & (common <= pd.to_datetime(end))
    idx = common[mask]
    if len(idx) < WINDOW_SIZE + 1:
        return None
    f = features_df.loc[idx].values.astype(np.float32)
    r = rv_series.loc[idx].values.astype(np.float32)
    return build_har_windows(f, r, WINDOW_SIZE)


def run_fold(features_df, rv_series, tr_start, tr_end, te_start, te_end):
    tr = windows_in_range(features_df, rv_series, tr_start, tr_end)
    te = windows_in_range(features_df, rv_series, te_start, te_end)
    if tr is None or te is None:
        return None
    X_tr_all, y_tr_all, _ = tr
    X_te, y_te, rv_at_t = te

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
                f"  intraday: scripts/fetch_binance_intraday.py ETHUSDT --interval 5m\n"
                f"  daily:    scripts/fetch_binance.py ETHUSDT"
            )

    t0 = time.time()
    print("walk-forward HAR-RV-from-intraday volatility prediction (ETHUSDT)")
    print("features: 13 daily technical + 3 intraday HAR (rv_intra_1/5/22)")
    print("target:   intraday-derived RV[t+1], baseline = RV[t]\n")

    print("loading intraday parquet + computing daily RV...")
    intraday = pd.read_parquet(INTRADAY_PARQUET)
    daily_rv = intraday_to_daily_rv(intraday)
    print(f"  {len(intraday):,} bars -> {len(daily_rv)} daily RV values")
    print(
        f"  RV stats: mean {daily_rv.mean() * 100:.3f}%, "
        f"max {daily_rv.max() * 100:.3f}%, "
        f"min {daily_rv.min() * 100:.4f}%"
    )

    print("\ncomputing HAR features from intraday RV (1/5/22-day horizons)...")
    har = har_features_from_intraday(daily_rv).dropna()
    print(f"  {har.shape[1]} HAR features over {len(har)} days")

    print("\nloading daily CSV and computing technical features...")
    daily_df = load_csv(str(DAILY_CSV), with_dates=True)
    tech = build_technical_features(daily_df).dropna()
    print(f"  {tech.shape[1]} daily technical features over {len(tech)} days")

    print("\njoining all features on intersection of dates...")
    combined = tech.join(har, how="inner").dropna()
    print(f"  combined: {combined.shape[1]} features over {len(combined)} days")

    print(f"\n{'fold':<9}{'n_tr':>7}{'n_te':>7}{'IC':>9}{'skill':>9}{'MAE_m%':>10}{'MAE_p%':>10}")
    print("-" * 61)

    rows = []
    for label, ts, te, vs, ve in FOLDS:
        r = run_fold(combined, daily_rv, ts, te, vs, ve)
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
    ics = [r["ic"] for r in rows]
    print("-" * 61)
    print(f"{'mean':<23}{np.mean(ics):>+9.4f}{np.mean(skills):>+9.4f}")

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

    # Direct comparison vs previous attempts
    print()
    print("------------- comparison with prior approaches -------------")
    print("  daily target, daily features (walk_forward_vol_eth.py):  skill +0.34")
    print("  intra target, daily features (walk_forward_intraday_rv): skill -0.35")
    print(
        f"  intra target, intra+daily feats (this file):              skill {np.mean(skills):+.4f}"
    )
    print()
    if np.mean(skills) > 0.3:
        print("  -> HAR features from intraday DO close the precision gap. The")
        print("     model now sees inputs at the same scale as the target it")
        print("     predicts, and beats persistence reliably.")
    elif np.mean(skills) > 0:
        print("  -> Positive skill regained vs prior intraday attempt, but")
        print("     short of the daily approach. Probably needs more training")
        print("     history (extend intraday fetch back to 2017-2018).")
    else:
        print("  -> Still losing to persistence. Intraday RV is genuinely a")
        print("     strong baseline; further work needed.")

    print(f"\ntotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
