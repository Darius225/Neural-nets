"""Ensemble of CNN + LightGBM on intraday-RV — classic quant pattern.

CNN-MSE lost the intraday-RV problem by skill -0.76. LightGBM won the
same problem by +0.12. The natural next question: does averaging the
two beat LightGBM alone? Three plausible outcomes:

  - Ensemble > LightGBM   → models capture complementary signal
  - Ensemble ≈ LightGBM   → CNN adds noise but doesn't ruin it
  - Ensemble < LightGBM   → CNN's bad predictions drag LightGBM down

Whichever way, the answer is informative. The pattern of running an
ensemble and reporting all three (CNN, LightGBM, average) is what a
quant team actually does in practice.

Also outputs tomorrow's live prediction from each model + the
ensemble so the user can see concrete numbers side by side.

Run:
    python experiments/ensemble_intraday_rv.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
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
CNN_EPOCHS = 40
CNN_PATIENCE = 6
BATCH_SIZE = 64
SEED = 42

CNN_CONFIG = ReturnsCNNConfig(
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

LGB_PARAMS = {
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "verbose": -1,
    "seed": SEED,
}

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
    closes = intraday_df["Close"].astype(np.float64)
    intra_returns = np.log(closes / closes.shift(1))
    sq = intra_returns**2
    daily_rv = sq.groupby(intraday_df.index.normalize()).sum().pipe(np.sqrt)
    daily_rv.index = daily_rv.index.tz_localize(None)
    return daily_rv.astype(np.float32).rename("rv_intra_1")


def har_features(rv: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=rv.index)
    out["rv_intra_1"] = rv
    out["rv_intra_5"] = rv.rolling(5).mean()
    out["rv_intra_22"] = rv.rolling(22).mean()
    return out


def make_cnn_windows(features_arr: np.ndarray, rv: np.ndarray, window_size: int):
    n = len(features_arr)
    n_windows = n - window_size
    X = np.empty((n_windows, window_size, features_arr.shape[1]), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.float32)
    rv_at_t = np.empty(n_windows, dtype=np.float32)
    for i in range(n_windows):
        X[i] = _zscore_window(features_arr[i : i + window_size])
        y[i] = rv[i + window_size]
        rv_at_t[i] = rv[i + window_size - 1]
    return X, y, rv_at_t


def slice_features_rv(features_df: pd.DataFrame, rv_series: pd.Series, start: str, end: str):
    common = features_df.index.intersection(rv_series.index)
    mask = (common >= pd.to_datetime(start)) & (common <= pd.to_datetime(end))
    idx = common[mask]
    if len(idx) < WINDOW_SIZE + 1:
        return None
    return (
        features_df.loc[idx].values.astype(np.float32),
        rv_series.loc[idx].values.astype(np.float32),
        idx,
    )


def train_cnn(X_tr, y_tr, X_val, y_val, seed: int = SEED):
    tf.keras.backend.clear_session()
    set_seed(seed)
    model = build_returns_cnn(WINDOW_SIZE, X_tr.shape[2], config=CNN_CONFIG)
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=CNN_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=CNN_PATIENCE, restore_best_weights=True)
        ],
    )
    return model


def train_lgb(X_tr, y_tr, X_val, y_val):
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val)
    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


def skill(pred: np.ndarray, actual: np.ndarray, baseline: np.ndarray) -> float:
    mse_m = float(np.mean((pred - actual) ** 2))
    mse_b = float(np.mean((baseline - actual) ** 2))
    return 1.0 - mse_m / mse_b if mse_b > 0 else float("nan")


def ic(pred: np.ndarray, actual: np.ndarray) -> float:
    if pred.std() == 0 or actual.std() == 0:
        return float("nan")
    return float(np.corrcoef(pred, actual)[0, 1])


def run_fold(features_df, rv_series, tr_start, tr_end, te_start, te_end):
    tr = slice_features_rv(features_df, rv_series, tr_start, tr_end)
    te = slice_features_rv(features_df, rv_series, te_start, te_end)
    if tr is None or te is None:
        return None
    f_tr, rv_tr, _ = tr
    f_te, rv_te, _ = te

    # ---- CNN: windowed input ----
    X_tr_w, y_tr, _ = make_cnn_windows(f_tr, rv_tr, WINDOW_SIZE)
    X_te_w, y_te, rv_at_t = make_cnn_windows(f_te, rv_te, WINDOW_SIZE)
    cut = int(len(X_tr_w) * 0.85)
    cnn = train_cnn(X_tr_w[:cut], y_tr[:cut], X_tr_w[cut:], y_tr[cut:])
    pred_cnn = np.maximum(cnn.predict(X_te_w, verbose=0).flatten(), 0.0)

    # ---- LightGBM: flat tabular input on the SAME rows ----
    # For each window, the "current day" features are the last row of that window.
    X_tr_flat = f_tr[WINDOW_SIZE - 1 : -1]
    X_te_flat = f_te[WINDOW_SIZE - 1 : -1]
    cut_flat = int(len(X_tr_flat) * 0.85)
    lgbm = train_lgb(X_tr_flat[:cut_flat], y_tr[:cut_flat], X_tr_flat[cut_flat:], y_tr[cut_flat:])
    pred_lgb = np.maximum(lgbm.predict(X_te_flat, num_iteration=lgbm.best_iteration), 0.0)

    # ---- Ensemble: simple 50/50 + weighted by inverse val MSE ----
    pred_avg = 0.5 * pred_cnn + 0.5 * pred_lgb

    # Weighted by inverse val MSE
    val_cnn = np.maximum(cnn.predict(X_tr_w[cut:], verbose=0).flatten(), 0.0)
    val_lgb = np.maximum(lgbm.predict(X_tr_flat[cut_flat:], num_iteration=lgbm.best_iteration), 0.0)
    mse_cnn_val = float(np.mean((val_cnn - y_tr[cut:]) ** 2))
    mse_lgb_val = float(np.mean((val_lgb - y_tr[cut_flat:]) ** 2))
    w_cnn = (1 / mse_cnn_val) if mse_cnn_val > 0 else 0
    w_lgb = (1 / mse_lgb_val) if mse_lgb_val > 0 else 0
    w_total = w_cnn + w_lgb
    if w_total > 0:
        pred_w = (w_cnn * pred_cnn + w_lgb * pred_lgb) / w_total
    else:
        pred_w = pred_avg
    weight_cnn_pct = 100 * w_cnn / w_total if w_total > 0 else 50

    return {
        "n_train": len(X_tr_w),
        "n_test": len(X_te_w),
        "cnn_skill": skill(pred_cnn, y_te, rv_at_t),
        "lgb_skill": skill(pred_lgb, y_te, rv_at_t),
        "ens_avg_skill": skill(pred_avg, y_te, rv_at_t),
        "ens_w_skill": skill(pred_w, y_te, rv_at_t),
        "cnn_ic": ic(pred_cnn, y_te),
        "lgb_ic": ic(pred_lgb, y_te),
        "ens_w_ic": ic(pred_w, y_te),
        "w_cnn_pct": weight_cnn_pct,
    }


def t_stat(values):
    arr = np.array([v for v in values if not math.isnan(v)])
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    return float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr))))


def main() -> None:
    for path in (INTRADAY_PARQUET, DAILY_CSV):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    t0 = time.time()
    print("ensemble (CNN + LightGBM) on intraday-RV walk-forward (ETHUSDT)\n")

    intraday = pd.read_parquet(INTRADAY_PARQUET)
    rv = intraday_to_daily_rv(intraday)
    har = har_features(rv).dropna()
    daily_df = load_csv(str(DAILY_CSV), with_dates=True)
    tech = build_technical_features(daily_df).dropna()
    combined = tech.join(har, how="inner").dropna()
    print(f"features: {combined.shape[1]} columns, {len(combined)} days\n")

    print(f"{'fold':<8}{'n_tr':>7}{'CNN':>9}{'LGB':>9}{'avg':>9}{'w_avg':>9}{'w_cnn%':>9}")
    print("-" * 60)

    rows = []
    for label, ts, te, vs, ve in FOLDS:
        r = run_fold(combined, rv, ts, te, vs, ve)
        if r is None:
            continue
        rows.append({"fold": label, **r})
        print(
            f"{label:<8}{r['n_train']:>7}"
            f"{r['cnn_skill']:>+9.4f}{r['lgb_skill']:>+9.4f}"
            f"{r['ens_avg_skill']:>+9.4f}{r['ens_w_skill']:>+9.4f}"
            f"{r['w_cnn_pct']:>8.1f}%"
        )

    if not rows:
        return

    print("-" * 60)
    cnn_skills = [r["cnn_skill"] for r in rows]
    lgb_skills = [r["lgb_skill"] for r in rows]
    avg_skills = [r["ens_avg_skill"] for r in rows]
    w_skills = [r["ens_w_skill"] for r in rows]
    print(
        f"{'mean':<8}{'':>7}"
        f"{np.mean(cnn_skills):>+9.4f}{np.mean(lgb_skills):>+9.4f}"
        f"{np.mean(avg_skills):>+9.4f}{np.mean(w_skills):>+9.4f}"
    )

    print("\nIC means:")
    print(f"  CNN:      {np.mean([r['cnn_ic'] for r in rows]):+.4f}")
    print(f"  LightGBM: {np.mean([r['lgb_ic'] for r in rows]):+.4f}")
    print(f"  weighted: {np.mean([r['ens_w_ic'] for r in rows]):+.4f}")

    print("\nt-stat skill vs zero:")
    print(f"  CNN:        {t_stat(cnn_skills):+.2f}")
    print(f"  LightGBM:   {t_stat(lgb_skills):+.2f}")
    print(f"  simple avg: {t_stat(avg_skills):+.2f}")
    print(f"  weighted:   {t_stat(w_skills):+.2f}")

    # ---------------- live prediction for tomorrow ----------------
    print(f"\n{'-' * 60}")
    print("  LIVE PREDICTION FOR TOMORROW (ETHUSDT)")
    print(f"{'-' * 60}")

    # Retrain on ALL data for the live forecast
    f_all = combined.values.astype(np.float32)
    rv_all = rv.loc[combined.index].values.astype(np.float32)
    X_all, y_all, _ = make_cnn_windows(f_all, rv_all, WINDOW_SIZE)
    cut = int(len(X_all) * 0.85)
    final_cnn = train_cnn(X_all[:cut], y_all[:cut], X_all[cut:], y_all[cut:])
    X_flat = f_all[WINDOW_SIZE - 1 : -1]
    final_lgb = train_lgb(X_flat[:cut], y_all[:cut], X_flat[cut:], y_all[cut:])

    last_window_cnn = _zscore_window(f_all[-WINDOW_SIZE:])[np.newaxis, ...]
    last_row_lgb = f_all[-1:][:, :]
    pred_cnn_tom = float(np.maximum(final_cnn.predict(last_window_cnn, verbose=0)[0, 0], 0.0))
    pred_lgb_tom = float(
        np.maximum(final_lgb.predict(last_row_lgb, num_iteration=final_lgb.best_iteration)[0], 0.0)
    )
    pred_avg_tom = 0.5 * pred_cnn_tom + 0.5 * pred_lgb_tom

    last_close = float(daily_df["Close"].iloc[-1])
    last_date = daily_df.index[-1].date()
    next_date = last_date + pd.Timedelta(days=1)

    def implied_band(vol):
        return last_close * np.exp(-vol), last_close * np.exp(+vol)

    print(f"  last observed close ({last_date}): ${last_close:,.2f}")
    print(f"  forecast date: {next_date}\n")
    print(f"  {'model':<14}{'pred RV%':>10}{'low (1sd)':>12}{'high (1sd)':>12}")
    print(f"  {'-' * 48}")
    for label, p in [("CNN", pred_cnn_tom), ("LightGBM", pred_lgb_tom), ("ensemble", pred_avg_tom)]:
        lo, hi = implied_band(p)
        print(f"  {label:<14}{p * 100:>10.3f}{lo:>12,.2f}{hi:>12,.2f}")

    print(f"\ntotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
