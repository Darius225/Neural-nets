"""Walk-forward validation across 10 S&P 500 tickers, six test years.

The stock-side counterpart to walk_forward_eth.py. Same idea: retrain
the model on everything before each test year, evaluate on that year
alone, then look at the IC distribution across all ticker x year
cells. With 10 tickers x 6 years = **60 measurements** we get
considerable statistical power compared to the single-fold backtest
of crisis_2008_v3.py.

Test years span calm (2005-2006) through the 2008 crisis through the
2009 recovery, so the panel shows both regime-shift failure and
normal-market noise in one place.

The 2008 fold is the direct walk-forward analogue of the 2022 LUNA
fold on the crypto side — both are the "year you actually need the
model to work". Comparing the per-year mean IC across the two asset
classes is the cleanest cross-asset finding the repo can produce.

Run:
    python experiments/walk_forward_stocks.py
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

from src.configs import ReturnsCNNConfig
from src.data import load_csv
from src.data.splits import _build_windows
from src.features import build_technical_features
from src.metrics import compute_metrics
from src.models import build_returns_cnn

CSV_DIR = "stock_market_data/sp500/csv"
TICKERS = ["JPM", "BAC", "C", "MSFT", "AAPL", "IBM", "JNJ", "PG", "GE", "XOM"]

WINDOW_SIZE = 30
HORIZON = 1
EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 64
SEED = 42

# Same ES-evolved config as the crypto walk-forward — the test is
# signal robustness, not search-loop robustness.
CONFIG = ReturnsCNNConfig(
    dropout=0.4,
    huber_delta=0.01,
    learning_rate=0.002,
)

# Six years spanning calm + crisis + recovery, all with at least a few
# years of pre-fold history on every test ticker.
TEST_YEARS = [2005, 2006, 2007, 2008, 2009, 2010]
TRAIN_HISTORY_START = "1990-01-01"


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def build_features_and_closes(df: pd.DataFrame):
    features = build_technical_features(df).dropna()
    closes = df.loc[features.index, "Close"].astype(np.float32)
    return features, closes


def windows_in_range(features: pd.DataFrame, closes: pd.Series, start: str, end: str):
    mask = (features.index >= pd.to_datetime(start)) & (features.index <= pd.to_datetime(end))
    f = features.loc[mask].values.astype(np.float32)
    c = closes.loc[mask].values.astype(np.float32)
    if len(f) < WINDOW_SIZE + HORIZON:
        return None
    return _build_windows(f, c, WINDOW_SIZE, HORIZON)


def run_fold(features, closes, train_end: str, test_year: int):
    train_pack = windows_in_range(features, closes, TRAIN_HISTORY_START, train_end)
    test_pack = windows_in_range(features, closes, f"{test_year}-01-01", f"{test_year}-12-31")
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


def t_stat(values):
    arr = np.array([v for v in values if not math.isnan(v)])
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    return float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr))))


def main() -> None:
    t0 = time.time()
    print("walk-forward validation across 10 S&P 500 tickers, 6 test years")
    print(f"config: {CONFIG.model_dump()}")
    print(f"tickers: {', '.join(TICKERS)}")
    print(f"test years: {TEST_YEARS}\n")

    # results[ticker][year] = ic
    grid = {t: {} for t in TICKERS}
    all_rows = []

    print(
        f"{'ticker':<7}{'year':>6}{'n_tr':>7}{'n_te':>6}{'eps':>5}"
        f"{'IC':>10}{'skill':>10}{'DirA%':>8}{'std_r':>8}"
    )
    print("-" * 67)

    for ticker in TICKERS:
        df = load_csv(f"{CSV_DIR}/{ticker}.csv", with_dates=True)
        features, closes = build_features_and_closes(df)
        for year in TEST_YEARS:
            train_end = f"{year - 1}-12-31"
            r = run_fold(features, closes, train_end, year)
            if r is None:
                print(f"{ticker:<7}{year:>6}  [skip — insufficient rows]")
                continue
            grid[ticker][year] = r["ic"]
            all_rows.append({"ticker": ticker, "year": year, **r})
            print(
                f"{ticker:<7}{year:>6}{r['n_train']:>7}{r['n_test']:>6}"
                f"{r['epochs']:>5}{r['ic']:>+10.4f}{r['skill']:>+10.4f}"
                f"{r['dir_acc']:>8.2f}{r['pred_std_ratio']:>8.3f}"
            )

    # ---------------- per-ticker means ----------------
    print("\nIC per ticker (across years):")
    print(f"  {'ticker':<7}{'mean':>9}{'stdev':>9}{'n>0':>5}")
    print(f"  {'-' * 7}{'-' * 9}{'-' * 9}{'-' * 5}")
    ticker_means = []
    for ticker in TICKERS:
        ics_all = list(grid[ticker].values())
        ics = [x for x in ics_all if not math.isnan(x)]
        if not ics:
            continue
        m = np.mean(ics)
        s = np.std(ics, ddof=1) if len(ics) > 1 else float("nan")
        ticker_means.append(m)
        n_nan = len(ics_all) - len(ics)
        nan_note = f"  [{n_nan} nan]" if n_nan else ""
        print(
            f"  {ticker:<7}{m:>+9.4f}{s:>9.4f}"
            f"{sum(1 for x in ics if x > 0):>4d}/{len(ics)}{nan_note}"
        )

    # ---------------- per-year means ----------------
    print("\nIC per year (across tickers):")
    print(f"  {'year':<7}{'mean':>9}{'stdev':>9}{'n>0':>5}    notes")
    print(f"  {'-' * 7}{'-' * 9}{'-' * 9}{'-' * 5}")
    year_means = {}
    for year in TEST_YEARS:
        ics_all = [grid[t].get(year) for t in TICKERS if year in grid[t]]
        ics = [x for x in ics_all if x is not None and not math.isnan(x)]
        if not ics:
            continue
        m = np.mean(ics)
        s = np.std(ics, ddof=1) if len(ics) > 1 else float("nan")
        year_means[year] = m
        note = ""
        if year == 2007:
            note = "<- crisis approaches"
        elif year == 2008:
            note = "<- crisis year"
        elif year == 2009:
            note = "<- recovery"
        print(
            f"  {year:<7}{m:>+9.4f}{s:>9.4f}{sum(1 for x in ics if x > 0):>4d}/{len(ics)}    {note}"
        )

    # ---------------- overall summary ----------------
    all_ics = [row["ic"] for row in all_rows if not math.isnan(row["ic"])]
    all_skills = [row["skill"] for row in all_rows if row["skill"] is not None]
    print("\noverall (60-cell panel):")
    print(f"  mean IC          = {np.mean(all_ics):+.4f}")
    print(f"  stdev IC         = {np.std(all_ics, ddof=1):.4f}")
    print(f"  n > 0            = {sum(1 for x in all_ics if x > 0)}/{len(all_ics)}")
    t = t_stat(all_ics)
    sig = ""
    if not math.isnan(t):
        sig = (
            "  p<0.01"
            if abs(t) >= 2.66
            else "  p<0.05"
            if abs(t) >= 2.00
            else "  p<0.10"
            if abs(t) >= 1.67
            else "  NS"
        )
    print(f"  t-stat vs zero   = {t:+.2f}{sig}")
    print(
        f"  mean skill       = {np.mean(all_skills):+.4f}  "
        f"(n>0: {sum(1 for x in all_skills if x > 0)}/{len(all_skills)})"
    )

    print(f"\ntotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
