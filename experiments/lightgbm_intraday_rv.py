"""LightGBM rematch of the intraday-RV experiment that CNN-MSE lost 4 times.

walk_forward_intraday_har.py (final iteration) reported:
    intra target, intra HAR features, 8 years data:  skill -0.76

The diagnosis there was "CNN + MSE + smooth target + limited n = wrong
tool combination". This script tests that diagnosis by replacing the
CNN with LightGBM (gradient-boosted trees), keeping everything else
identical: same intraday-derived RV target, same persistence baseline,
same HAR-RV-from-intraday features, same 6 walk-forward folds.

LightGBM is the classical tabular workhorse: low parameter count,
strong built-in regularisation (L1/L2 + tree depth limits + bagging),
no per-window z-score needed. On Corsi-style HAR-RV problems the
literature routinely shows tree ensembles beat deep nets when n is in
the low thousands.

If LightGBM wins here, the diagnosis is confirmed: it was the model,
not the data.

Run:
    python experiments/lightgbm_intraday_rv.py
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

from src.data import load_csv
from src.features import build_technical_features

INTRADAY_PARQUET = Path("stock_market_data/crypto/intraday/ETHUSDT_5m.parquet")
DAILY_CSV = Path("stock_market_data/crypto/csv/ETHUSDT.csv")

# Same six folds as walk_forward_intraday_har.py for direct comparison.
FOLDS = [
    ("2020", "2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2021", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2022", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2025+", "2018-01-01", "2024-12-31", "2025-01-01", "2026-05-17"),
]

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
    "seed": 42,
}


def intraday_to_daily_rv(intraday_df: pd.DataFrame) -> pd.Series:
    closes = intraday_df["Close"].astype(np.float64)
    intra_returns = np.log(closes / closes.shift(1))
    sq = intra_returns**2
    daily_rv = sq.groupby(intraday_df.index.normalize()).sum().pipe(np.sqrt)
    daily_rv.index = daily_rv.index.tz_localize(None)
    return daily_rv.astype(np.float32).rename("rv_intra_1")


def har_features_from_intraday(daily_rv: pd.Series) -> pd.DataFrame:
    har = pd.DataFrame(index=daily_rv.index)
    har["rv_intra_1"] = daily_rv
    har["rv_intra_5"] = daily_rv.rolling(5).mean()
    har["rv_intra_22"] = daily_rv.rolling(22).mean()
    return har


def make_supervised(features_df: pd.DataFrame, rv: pd.Series, start: str, end: str):
    """Tabular X, y for LightGBM. X = features on day t, y = RV[t+1]."""
    common = features_df.index.intersection(rv.index)
    mask = (common >= pd.to_datetime(start)) & (common <= pd.to_datetime(end))
    idx = common[mask]
    if len(idx) < 30:
        return None
    # Shift y forward by 1 day to predict tomorrow's RV from today's features
    X = features_df.loc[idx].values[:-1]
    y = rv.loc[idx].values[1:]
    rv_today = rv.loc[idx].values[:-1]  # persistence baseline
    return X, y, rv_today, idx[:-1]


def run_fold(features_df, rv_series, tr_start, tr_end, te_start, te_end):
    tr = make_supervised(features_df, rv_series, tr_start, tr_end)
    te = make_supervised(features_df, rv_series, te_start, te_end)
    if tr is None or te is None:
        return None
    X_tr, y_tr, _, _ = tr
    X_te, y_te, rv_at_t, _ = te

    cut = int(len(X_tr) * 0.85)
    dtrain = lgb.Dataset(X_tr[:cut], y_tr[:cut])
    dval = lgb.Dataset(X_tr[cut:], y_tr[cut:])

    model = lgb.train(
        LGB_PARAMS,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    pred = np.maximum(model.predict(X_te, num_iteration=model.best_iteration), 0.0)

    mse_model = float(np.mean((pred - y_te) ** 2))
    mse_persist = float(np.mean((rv_at_t - y_te) ** 2))
    skill = 1.0 - mse_model / mse_persist if mse_persist > 0 else float("nan")
    ic = float(np.corrcoef(pred, y_te)[0, 1]) if pred.std() > 0 and y_te.std() > 0 else float("nan")
    return {
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "best_iter": model.best_iteration,
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
            raise SystemExit(f"missing {path}")

    t0 = time.time()
    print("LightGBM intraday-RV walk-forward (ETHUSDT)")
    print("target: RV[t+1] from 5-min returns, baseline: yesterday's RV")
    print(
        f"lgb params: lr={LGB_PARAMS['learning_rate']}, "
        f"num_leaves={LGB_PARAMS['num_leaves']}, "
        f"lambda_l2={LGB_PARAMS['lambda_l2']}\n"
    )

    intraday = pd.read_parquet(INTRADAY_PARQUET)
    rv = intraday_to_daily_rv(intraday)
    print(f"intraday: {len(intraday):,} bars -> {len(rv)} daily RV values")

    har = har_features_from_intraday(rv).dropna()
    daily_df = load_csv(str(DAILY_CSV), with_dates=True)
    tech = build_technical_features(daily_df).dropna()
    combined = tech.join(har, how="inner").dropna()
    print(f"features: {combined.shape[1]} columns, {len(combined)} days\n")

    print(
        f"{'fold':<8}{'n_tr':>7}{'n_te':>7}{'iter':>6}"
        f"{'IC':>9}{'skill':>10}{'MAE_m%':>9}{'MAE_p%':>9}"
    )
    print("-" * 64)

    rows = []
    for label, ts, te, vs, ve in FOLDS:
        r = run_fold(combined, rv, ts, te, vs, ve)
        if r is None:
            continue
        rows.append({"fold": label, **r})
        print(
            f"{label:<8}{r['n_train']:>7}{r['n_test']:>7}{r['best_iter']:>6}"
            f"{r['ic']:>+9.4f}{r['skill']:>+10.4f}"
            f"{r['mae_model'] * 100:>9.4f}{r['mae_persist'] * 100:>9.4f}"
        )

    if not rows:
        return

    skills = [r["skill"] for r in rows]
    ics = [r["ic"] for r in rows]
    print("-" * 64)
    print(f"{'mean':<22}{np.mean(ics):>+9.4f}{np.mean(skills):>+10.4f}")

    t = t_stat(skills)
    sig = (
        "  p<0.01"
        if abs(t) >= 4.03
        else "  p<0.05"
        if abs(t) >= 2.57
        else "  p<0.10"
        if abs(t) >= 2.02
        else "  NS"
    )
    print(f"\nskill mean = {np.mean(skills):+.4f}, t-stat = {t:+.2f}{sig}  (n={len(skills)} folds)")

    print()
    print("============= comparison vs CNN-MSE on same setup =============")
    print("  CNN MSE, 16 features, 6 folds (walk_forward_intraday_har.py): skill -0.76")
    print(
        f"  LightGBM, 16 features, 6 folds (this file):                    skill {np.mean(skills):+.4f}"
    )
    if np.mean(skills) > 0:
        print()
        print("  -> LightGBM beats persistence on intraday RV where CNN-MSE could not.")
        print("     Confirms the diagnosis: the failure was the model + loss combo,")
        print("     not the data. Tree ensembles handle this regime (smooth target,")
        print("     small n) much better than deep nets with squared loss.")
    else:
        print()
        print("  -> Even LightGBM struggles. The intraday RV target may simply have")
        print("     too much auto-correlation for any model to beat persistence")
        print("     materially without much longer history or higher-frequency data.")

    print(f"\ntotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
