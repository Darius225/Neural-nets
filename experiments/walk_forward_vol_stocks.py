"""Stock-side mirror of walk_forward_vol_eth.py — volatility prediction
across 10 S&P 500 tickers and 6 test years.

The crypto experiment (walk_forward_vol_eth.py) found skill = +0.34
across 6 ETH+BTC folds with t-stat +13.45 — the first real positive
result in the repo. This script confirms (or refutes) the finding on
a completely different asset class: 10 large-cap US equities
(JPM/BAC/C/MSFT/AAPL/IBM/JNJ/PG/GE/XOM) across the 2005-2010 window
covering calm / crisis / recovery.

Same v3 pipeline, same ES-evolved config, same six test years as
walk_forward_stocks.py — only the target changes to |log_return[t+1]|
with persistence baseline = |log_return[t]|.

If skill stays clearly positive across both asset classes, the finding
isn't a crypto-specific quirk — volatility clustering is universal and
the model genuinely captures it.

Run:
    python experiments/walk_forward_vol_stocks.py
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

CSV_DIR = "stock_market_data/sp500/csv"
TICKERS = ["JPM", "BAC", "C", "MSFT", "AAPL", "IBM", "JNJ", "PG", "GE", "XOM"]

WINDOW_SIZE = 30
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 64
SEED = 42

# Same ES-evolved config as the return walk-forward on stocks.
CONFIG = ReturnsCNNConfig(
    dropout=0.4,
    huber_delta=0.01,
    learning_rate=0.002,
)

TEST_YEARS = [2005, 2006, 2007, 2008, 2009, 2010]
TRAIN_HISTORY_START = "1990-01-01"


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def build_features_and_closes(df: pd.DataFrame):
    features = build_technical_features(df).dropna()
    closes = df.loc[features.index, "Close"].astype(np.float32)
    return features, closes


def build_vol_windows(features: np.ndarray, closes: np.ndarray, window_size: int):
    """Same as walk_forward_vol_eth.build_vol_windows."""
    n = len(features)
    if n < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} rows, got {n}")
    log_returns = np.log(closes[1:] / closes[:-1])
    abs_log_returns = np.abs(log_returns).astype(np.float32)

    n_windows = n - window_size
    X = np.empty((n_windows, window_size, features.shape[1]), dtype=np.float32)
    y_vol = np.empty(n_windows, dtype=np.float32)
    vol_at_t = np.empty(n_windows, dtype=np.float32)

    for i in range(n_windows):
        X[i] = _zscore_window(features[i : i + window_size])
        y_vol[i] = abs_log_returns[i + window_size - 1]
        vol_at_t[i] = abs_log_returns[i + window_size - 2]
    return X, y_vol, vol_at_t


def windows_for_range(features: pd.DataFrame, closes: pd.Series, start: str, end: str):
    mask = (features.index >= pd.to_datetime(start)) & (features.index <= pd.to_datetime(end))
    f = features.loc[mask].values.astype(np.float32)
    c = closes.loc[mask].values.astype(np.float32)
    if len(f) < WINDOW_SIZE + 1:
        return None
    return build_vol_windows(f, c, WINDOW_SIZE)


def skill_score(mse_model: float, mse_baseline: float) -> float:
    return 1.0 - mse_model / mse_baseline if mse_baseline > 0 else float("nan")


def run_fold(features, closes, train_end: str, test_year: int):
    train_pack = windows_for_range(features, closes, TRAIN_HISTORY_START, train_end)
    test_pack = windows_for_range(features, closes, f"{test_year}-01-01", f"{test_year}-12-31")
    if train_pack is None or test_pack is None:
        return None
    X_tr_all, y_tr_all, _ = train_pack
    X_te, y_te, vol_at_t = test_pack

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
    pred = np.maximum(model.predict(X_te, verbose=0).flatten(), 0.0)

    mse_model = float(np.mean((pred - y_te) ** 2))
    mse_persist = float(np.mean((vol_at_t - y_te) ** 2))
    ic = float(np.corrcoef(pred, y_te)[0, 1]) if pred.std() > 0 and y_te.std() > 0 else float("nan")

    return {
        "n_train": len(X_tr_all),
        "n_test": len(X_te),
        "epochs": len(history.history["loss"]),
        "ic": ic,
        "skill": skill_score(mse_model, mse_persist),
        "mae_model": float(np.mean(np.abs(pred - y_te))),
        "mae_persist": float(np.mean(np.abs(vol_at_t - y_te))),
    }


def t_stat(values):
    arr = np.array([x for x in values if not math.isnan(x)])
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    return float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr))))


def main() -> None:
    t0 = time.time()
    print("walk-forward VOLATILITY prediction across 10 S&P 500 tickers, 6 years")
    print("target: |log_return[t+1]|,  baseline: |log_return[t]|")
    print(f"config: {CONFIG.model_dump()}")
    print(f"tickers: {', '.join(TICKERS)}\n")

    grid = {t: {} for t in TICKERS}
    all_skills = []
    all_ics = []

    print(
        f"{'ticker':<7}{'year':>6}{'n_tr':>7}{'n_te':>6}{'eps':>5}"
        f"{'IC':>10}{'skill':>10}{'MAE_m':>9}{'MAE_p':>9}"
    )
    print("-" * 69)

    for ticker in TICKERS:
        df = load_csv(f"{CSV_DIR}/{ticker}.csv", with_dates=True)
        features, closes = build_features_and_closes(df)
        for year in TEST_YEARS:
            train_end = f"{year - 1}-12-31"
            r = run_fold(features, closes, train_end, year)
            if r is None:
                continue
            grid[ticker][year] = r
            all_skills.append(r["skill"])
            all_ics.append(r["ic"])
            print(
                f"{ticker:<7}{year:>6}{r['n_train']:>7}{r['n_test']:>6}{r['epochs']:>5}"
                f"{r['ic']:>+10.4f}{r['skill']:>+10.4f}"
                f"{r['mae_model']:>9.5f}{r['mae_persist']:>9.5f}"
            )

    # ---------- per-year aggregate ----------
    print("\nskill per year (mean across 10 tickers):")
    print(f"  {'year':<6}{'mean':>10}{'stdev':>10}{'n>0':>6}    notes")
    print(f"  {'-' * 6}{'-' * 10}{'-' * 10}{'-' * 6}")
    for year in TEST_YEARS:
        skills = [grid[t][year]["skill"] for t in TICKERS if year in grid[t]]
        if not skills:
            continue
        s_mean = np.mean(skills)
        s_std = np.std(skills, ddof=1) if len(skills) > 1 else float("nan")
        n_pos = sum(1 for x in skills if x > 0)
        note = ""
        if year == 2008:
            note = "<- crisis year"
        elif year == 2009:
            note = "<- recovery"
        print(f"  {year:<6}{s_mean:>+10.4f}{s_std:>10.4f}{n_pos:>4d}/{len(skills)}    {note}")

    # ---------- overall ----------
    print(f"\noverall ({len(all_skills)}-cell panel):")
    print(f"  mean skill  = {np.mean(all_skills):+.4f}")
    print(f"  stdev skill = {np.std(all_skills, ddof=1):.4f}")
    print(f"  n > 0       = {sum(1 for x in all_skills if x > 0)}/{len(all_skills)}")
    t = t_stat(all_skills)
    sig = ""
    if not math.isnan(t):
        sig = (
            "  p<0.001"
            if abs(t) >= 3.40
            else "  p<0.01"
            if abs(t) >= 2.66
            else "  p<0.05"
            if abs(t) >= 2.00
            else "  p<0.10"
            if abs(t) >= 1.67
            else "  NS"
        )
    print(f"  t-stat      = {t:+.2f}{sig}")
    print(f"  mean IC     = {np.mean(all_ics):+.4f}")

    print()
    if np.mean(all_skills) > 0.10 and t >= 3.0:
        print("  -> volatility prediction is robust across stocks AND crypto.")
        print("     Cross-asset confirmation of the GARCH-style effect. Real finding.")
    elif np.mean(all_skills) > 0.05 and t >= 2.0:
        print("  -> positive skill across the panel, statistically significant.")
        print("     Cross-asset confirmation of the volatility clustering effect.")
    else:
        print("  -> skill is weaker on stocks than on crypto.")
        print("     Volatility clustering exists but is more pronounced for crypto.")

    print(f"\ntotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
