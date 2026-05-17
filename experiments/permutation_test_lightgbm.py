"""Permutation test: is LightGBM's intraday-RV skill real, or data mining?

The LightGBM rematch reported skill = +0.12 across 6 walk-forward folds.
That's positive, but only one test on one model — the obvious follow-up
question is "did I just get lucky on the configuration I happened to
pick?".

Permutation testing answers that. The procedure:

  1. Take the real training/test setup. Compute the actual skill score.
  2. N times: shuffle the target column (y) randomly, keeping features
     fixed. Train the SAME model. Record its skill.
  3. The N shuffled-skill values form the *null distribution* — what
     skill scores look like when the target is genuinely random.
  4. Where does the actual skill fall in that distribution? If it's
     above ~95 % of the null, the model is finding real structure.
     If it's near the centre, the original positive number was just
     a lucky sample from the null.

Reference: Welch (1990), Good (2005), Permutation Tests in Statistics.
Used routinely in academic finance for data-mining-bias checks
(Sullivan/Timmermann/White "Reality Check" is the elaborate version).

Run:
    python experiments/permutation_test_lightgbm.py
    python experiments/permutation_test_lightgbm.py --n-permutations 200
"""

from __future__ import annotations

import argparse
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

# Use a single big train + test slice (not the 6-fold setup) because we
# need to do this N times and each fold costs ~1s. With one slice and
# N=100 permutations the total budget is ~2-3 minutes.
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-05-17"

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
}


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


def fit_predict_skill(X_tr, y_tr, X_te, y_te, baseline, seed: int) -> float:
    cut = int(len(X_tr) * 0.85)
    params = {**LGB_PARAMS, "seed": seed}
    dtrain = lgb.Dataset(X_tr[:cut], y_tr[:cut])
    dval = lgb.Dataset(X_tr[cut:], y_tr[cut:])
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    pred = np.maximum(model.predict(X_te, num_iteration=model.best_iteration), 0.0)
    mse_m = float(np.mean((pred - y_te) ** 2))
    mse_b = float(np.mean((baseline - y_te) ** 2))
    return 1.0 - mse_m / mse_b if mse_b > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--n-permutations",
        type=int,
        default=100,
        help="how many shuffled-target trials to run (default 100)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    print("permutation test: LightGBM on intraday-RV")
    print(f"N permutations: {args.n_permutations}\n")

    intraday = pd.read_parquet(INTRADAY_PARQUET)
    rv = intraday_to_daily_rv(intraday)
    har = har_features(rv).dropna()
    daily_df = load_csv(str(DAILY_CSV), with_dates=True)
    tech = build_technical_features(daily_df).dropna()
    combined = tech.join(har, how="inner").dropna()

    common = combined.index.intersection(rv.index)
    train_idx = common[common <= TRAIN_END]
    test_idx = common[(common >= TEST_START) & (common <= TEST_END)]

    # Tabular setup: X = features on day t, y = RV[t+1]
    X_tr = combined.loc[train_idx].values[:-1].astype(np.float32)
    y_tr = rv.loc[train_idx].values[1:].astype(np.float32)
    X_te = combined.loc[test_idx].values[:-1].astype(np.float32)
    y_te = rv.loc[test_idx].values[1:].astype(np.float32)
    baseline = rv.loc[test_idx].values[:-1].astype(np.float32)

    print(f"train: {len(X_tr)} rows ({train_idx[0].date()} .. {train_idx[-1].date()})")
    print(f"test:  {len(X_te)} rows ({test_idx[0].date()} .. {test_idx[-1].date()})\n")

    # ---- Real skill ----
    print("computing actual skill...")
    actual_skill = fit_predict_skill(X_tr, y_tr, X_te, y_te, baseline, seed=args.seed)
    print(f"  actual skill = {actual_skill:+.4f}\n")

    # ---- Null distribution via permutations ----
    print(f"running {args.n_permutations} permutations (shuffle y_tr each time)...")
    rng = np.random.default_rng(args.seed)
    null_skills = []
    for i in range(args.n_permutations):
        y_tr_shuf = rng.permutation(y_tr)
        s = fit_predict_skill(X_tr, y_tr_shuf, X_te, y_te, baseline, seed=args.seed + i)
        null_skills.append(s)
        if (i + 1) % 25 == 0:
            print(
                f"  ... {i + 1}/{args.n_permutations}  "
                f"null_mean so far = {np.mean(null_skills):+.4f}"
            )
    null_skills = np.array(null_skills)

    # ---- Analysis ----
    null_mean = float(null_skills.mean())
    null_std = float(null_skills.std(ddof=1))
    z_score = (actual_skill - null_mean) / null_std if null_std > 0 else float("nan")
    pct_above = float(np.mean(null_skills > actual_skill) * 100)
    pct_below = float(np.mean(null_skills < actual_skill) * 100)

    # One-sided p-value: probability null produces something >= actual
    p_value = float((np.sum(null_skills >= actual_skill) + 1) / (len(null_skills) + 1))

    print(f"\n{'-' * 60}")
    print("  PERMUTATION TEST RESULT")
    print(f"{'-' * 60}")
    print(f"  actual skill              = {actual_skill:+.4f}")
    print(f"  null distribution mean    = {null_mean:+.4f}")
    print(f"  null distribution stdev   = {null_std:.4f}")
    print(f"  null min / max            = {null_skills.min():+.4f} / {null_skills.max():+.4f}")
    print(f"  actual is at {pct_above:.0f}th percentile of null")
    print(f"  z-score vs null           = {z_score:+.2f}")
    print(f"  one-sided p-value         = {p_value:.4f}")

    print()
    if p_value < 0.01:
        print("  -> p < 0.01: actual skill is STRONGLY beyond what random shuffling")
        print("     produces. The signal is real, not data mining.")
    elif p_value < 0.05:
        print("  -> p < 0.05: actual skill is significantly beyond the null.")
        print("     Signal is likely real.")
    elif p_value < 0.10:
        print("  -> p < 0.10: marginal evidence. Real signal possible but not")
        print("     decisive; would benefit from more data or more permutations.")
    else:
        print(f"  -> p = {p_value:.3f}: actual skill is indistinguishable from what")
        print("     you'd get by chance. The +0.12 result was data mining luck.")

    # Histogram-ish ascii summary
    print("\n  null distribution histogram (skill on shuffled targets):")
    bins = np.linspace(
        min(null_skills.min(), actual_skill) - 0.05, max(null_skills.max(), actual_skill) + 0.05, 21
    )
    hist, _ = np.histogram(null_skills, bins=bins)
    max_h = hist.max()
    for lo, hi, cnt in zip(bins[:-1], bins[1:], hist, strict=False):
        bar = "#" * int(40 * cnt / max_h) if max_h > 0 else ""
        marker = " <- actual" if lo <= actual_skill < hi else ""
        print(f"    {lo:>+7.3f} .. {hi:>+7.3f}  {bar:<40} {cnt:>3d}{marker}")

    print(f"\ntotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
