"""2008 financial-crisis regime-shift experiment.

For each ticker:
  1. Train the default CNN on data ending ``TRAIN_END``.
  2. Predict next-day close for ``TEST_START`` .. ``TEST_END``.
  3. Compare against the naive persistence baseline (pred = yesterday's close).
  4. Report MAE, RMSE, MAPE, R², directional accuracy, mean signed error,
     worst-day error, **skill vs persistence**, and **out-of-train-range %**.

The point of skill score: a CNN that can't beat "predict tomorrow = today"
is genuinely useless. The point of out-of-range: shows directly when the
MinMaxScaler-bounded model is being asked to predict prices it has never
seen — the standard regime-shift failure mode.

Run:
    python experiments/crisis_2008.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import load_csv, prepare_train_test_split
from src.metrics import compute_metrics, naive_persistence_forecast
from src.models import build_best_cnn


TICKERS = ["JPM", "BAC", "C", "MSFT", "AAPL", "IBM", "JNJ", "PG", "GE", "XOM"]
TRAIN_END = "2007-07-31"
TEST_START = "2007-08-01"
TEST_END = "2009-12-31"
EPOCHS = 60
BATCH_SIZE = 64
CSV_DIR = "stock_market_data/sp500/csv"


def evaluate_ticker(ticker: str) -> dict:
    """Train pre-crisis, predict the crisis window, return one results row."""
    df = load_csv(f"{CSV_DIR}/{ticker}.csv", with_dates=True)
    split = prepare_train_test_split(df, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END)

    tf.keras.backend.clear_session()
    model = build_best_cnn(split.input_shape)
    model.fit(
        split.X_train, split.y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
    )

    y_pred = model.predict(split.X_test, verbose=0).flatten()
    y_baseline = naive_persistence_forecast(split.y_test_prev)

    model_metrics = compute_metrics(
        split.y_test, y_pred, y_prev=split.y_test_prev,
        train_min=split.train_close_min, train_max=split.train_close_max,
    )
    baseline_metrics = compute_metrics(
        split.y_test, y_baseline, y_prev=split.y_test_prev,
    )

    return {
        "ticker": ticker,
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "train_min$": round(split.train_close_min, 2),
        "train_max$": round(split.train_close_max, 2),
        "test_min$": round(float(split.y_test.min()), 2),
        "test_max$": round(float(split.y_test.max()), 2),
        "model": model_metrics,
        "baseline": baseline_metrics,
        "predictions": pd.DataFrame(
            {"actual": split.y_test, "predicted": y_pred, "baseline": y_baseline},
            index=split.test_index,
        ),
    }


def save_prediction_plots(rows: list[dict], out_dir: str = "experiments/plots") -> None:
    """Save per-ticker actual-vs-predicted-vs-baseline plots, with a
    vertical line at the Lehman Brothers bankruptcy filing (2008-09-15)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    lehman = pd.Timestamp("2008-09-15")

    for r in rows:
        df = r["predictions"]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["actual"], label="actual", color="black", linewidth=1.2)
        ax.plot(df.index, df["predicted"], label="CNN", color="C3", alpha=0.8, linewidth=1)
        ax.plot(df.index, df["baseline"], label="persistence", color="C0", alpha=0.5, linewidth=0.8)
        ax.axvline(lehman, color="grey", linestyle="--", alpha=0.7, label="Lehman (2008-09-15)")
        ax.set_title(f"{r['ticker']} — actual vs CNN vs persistence  "
                     f"(MAE={r['model'].mae:.2f}, skill={r['model'].skill_vs_persistence:+.3f})")
        ax.set_ylabel("Close ($)")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = f"{out_dir}/{r['ticker']}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
    print(f"\nSaved {len(rows)} plots to {out_dir}/")


def print_per_ticker_table(rows: list[dict]) -> None:
    header = (
        f"{'ticker':<7}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}{'DirAcc%':>9}"
        f"{'Skill':>9}{'OutR%':>8}    vs persistence (MAE / DirAcc%)"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        m, b = r["model"], r["baseline"]
        skill = "n/a" if m.skill_vs_persistence is None else f"{m.skill_vs_persistence:+.3f}"
        oor = "n/a" if m.out_of_train_range_pct is None else f"{m.out_of_train_range_pct:.1f}"
        print(
            f"{r['ticker']:<7}{m.mae:>8.3f}{m.rmse:>8.3f}{m.mape:>8.2f}"
            f"{m.directional_accuracy:>9.2f}{skill:>9}{oor:>8}    "
            f"{b.mae:.3f} (baseline MAE)"
        )
    print("note: persistence has no directional accuracy by construction "
          "(sign(pred - prev) == 0 always)")


def print_regime_shift_context(rows: list[dict]) -> None:
    print("\nRegime-shift context (price range seen at train vs test):")
    print(f"{'ticker':<7}{'train range':>22}{'test range':>22}{'test outside?':>16}")
    for r in rows:
        train_rng = f"${r['train_min$']}–${r['train_max$']}"
        test_rng = f"${r['test_min$']}–${r['test_max$']}"
        outside = (
            r["test_min$"] < r["train_min$"] or r["test_max$"] > r["train_max$"]
        )
        flag = "YES — leak" if outside else "no"
        print(f"{r['ticker']:<7}{train_rng:>22}{test_rng:>22}{flag:>16}")


def aggregate_summary(rows: list[dict]) -> None:
    model_mae = np.mean([r["model"].mae for r in rows])
    base_mae = np.mean([r["baseline"].mae for r in rows])
    model_dir = np.mean([r["model"].directional_accuracy for r in rows])
    skills = [r["model"].skill_vs_persistence for r in rows]
    n_beat = sum(1 for s in skills if s is not None and s > 0)

    print("\nAggregate over", len(rows), "tickers")
    print("-" * 50)
    print(f"  mean MAE         : model = {model_mae:.3f}  |  persistence = {base_mae:.3f}")
    print(f"  mean DirAcc%     : model = {model_dir:.2f}   |  persistence = n/a (sign always 0)")
    print(f"  beats persistence: {n_beat}/{len(rows)} tickers (skill > 0)")
    print(f"  mean skill score : {np.mean([s for s in skills if s is not None]):+.4f}")


def main() -> None:
    print(f"Training pre-crisis (up to {TRAIN_END}), evaluating {TEST_START} .. {TEST_END}")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"Epochs={EPOCHS}, batch={BATCH_SIZE}\n")

    rows = []
    overall_start = time.time()
    for ticker in TICKERS:
        t0 = time.time()
        try:
            row = evaluate_ticker(ticker)
            rows.append(row)
            print(f"  [{ticker}] done in {time.time() - t0:.1f}s "
                  f"(train={row['n_train']}, test={row['n_test']})")
        except Exception as exc:
            print(f"  [{ticker}] FAILED: {exc}")
    print(f"\nTotal: {time.time() - overall_start:.1f}s\n")

    if not rows:
        print("No tickers evaluated.")
        return

    print_per_ticker_table(rows)
    print_regime_shift_context(rows)
    aggregate_summary(rows)
    save_prediction_plots(rows)


if __name__ == "__main__":
    main()
