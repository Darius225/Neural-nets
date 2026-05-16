"""2008 crisis experiment, v2 pipeline.

Same tickers, same period, same seeds as ``crisis_2008.py`` — but with
the four data/training changes we agreed on:

  1. Target = next-day *return* (not raw price).
  2. Input = 30-day sliding window of OHLCV (not a single day).
  3. Per-window z-score normalisation (no global MinMaxScaler).
  4. Internal val set (last 15% of pre-crisis) + early stopping.

Predictions are converted back to prices via
``pred_price = close[t] * (1 + pred_return)`` so all metrics are
directly comparable to v1 (price-space MAE/RMSE/MAPE/DirAcc/skill).

Run:
    python experiments/crisis_2008_v2.py
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
from tensorflow.keras.callbacks import EarlyStopping

from src.data import load_csv, prepare_windowed_returns_split
from src.metrics import compute_metrics, naive_persistence_forecast
from src.models import build_returns_cnn

TICKERS = ["JPM", "BAC", "C", "MSFT", "AAPL", "IBM", "JNJ", "PG", "GE", "XOM"]
TRAIN_END = "2007-07-31"
TEST_START = "2007-08-01"
TEST_END = "2009-12-31"
WINDOW_SIZE = 30
EPOCHS = 100
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 8
CSV_DIR = "stock_market_data/sp500/csv"
SEED = 42


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def evaluate_ticker(ticker: str) -> dict:
    """Train v2 pipeline pre-crisis, predict 2008-2009 in price space."""
    df = load_csv(f"{CSV_DIR}/{ticker}.csv", with_dates=True)
    split = prepare_windowed_returns_split(
        df,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        window_size=WINDOW_SIZE,
    )

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_returns_cnn(split.window_size, split.n_features)
    hist = model.fit(
        split.X_train,
        split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True
            )
        ],
    )
    epochs_used = len(hist.history["loss"])

    pred_returns = model.predict(split.X_test, verbose=0).flatten()
    pred_prices = split.close_at_t_test * (1 + pred_returns)
    actual_prices = split.actual_close_test
    baseline_prices = naive_persistence_forecast(split.close_at_t_test)  # = close[t]

    model_metrics = compute_metrics(
        actual_prices,
        pred_prices,
        y_prev=split.close_at_t_test,
        train_min=split.train_close_min,
        train_max=split.train_close_max,
    )
    baseline_metrics = compute_metrics(
        actual_prices,
        baseline_prices,
        y_prev=split.close_at_t_test,
    )

    return {
        "ticker": ticker,
        "n_train": len(split.X_train),
        "n_val": len(split.X_val),
        "n_test": len(split.X_test),
        "epochs_used": epochs_used,
        "train_min$": round(split.train_close_min, 2),
        "train_max$": round(split.train_close_max, 2),
        "model": model_metrics,
        "baseline": baseline_metrics,
        "predictions": pd.DataFrame(
            {
                "actual": actual_prices,
                "predicted": pred_prices,
                "baseline": baseline_prices,
                "pred_return": pred_returns,
            },
            index=split.test_index,
        ),
    }


def print_per_ticker_table(rows: list[dict]) -> None:
    header = (
        f"{'ticker':<7}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}{'DirAcc%':>9}"
        f"{'Skill':>9}{'epochs':>8}    vs persistence (MAE)"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        m, b = r["model"], r["baseline"]
        skill = "n/a" if m.skill_vs_persistence is None else f"{m.skill_vs_persistence:+.3f}"
        print(
            f"{r['ticker']:<7}{m.mae:>8.3f}{m.rmse:>8.3f}{m.mape:>8.2f}"
            f"{m.directional_accuracy:>9.2f}{skill:>9}{r['epochs_used']:>8}    "
            f"{b.mae:.3f}"
        )


def aggregate_summary(rows: list[dict]) -> None:
    skills = [
        r["model"].skill_vs_persistence for r in rows if r["model"].skill_vs_persistence is not None
    ]
    n_beat = sum(1 for s in skills if s > 0)
    print("\nAggregate over", len(rows), "tickers")
    print("-" * 50)
    print(
        f"  mean MAE         : model = {np.mean([r['model'].mae for r in rows]):.3f}  "
        f"|  persistence = {np.mean([r['baseline'].mae for r in rows]):.3f}"
    )
    print(
        f"  mean DirAcc%     : model = {np.mean([r['model'].directional_accuracy for r in rows]):.2f}"
    )
    print(f"  beats persistence: {n_beat}/{len(rows)} tickers (skill > 0)")
    print(f"  mean skill score : {np.mean(skills):+.4f}")
    print(
        f"  mean epochs used : {np.mean([r['epochs_used'] for r in rows]):.1f} "
        f"(of {EPOCHS} max, early-stopped)"
    )


def save_prediction_plots(rows: list[dict], out_dir: str = "experiments/plots_v2") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    lehman = pd.Timestamp("2008-09-15")
    for r in rows:
        df = r["predictions"]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["actual"], label="actual", color="black", linewidth=1.2)
        ax.plot(
            df.index, df["predicted"], label="v2 CNN (returns)", color="C2", alpha=0.8, linewidth=1
        )
        ax.plot(df.index, df["baseline"], label="persistence", color="C0", alpha=0.4, linewidth=0.8)
        ax.axvline(lehman, color="grey", linestyle="--", alpha=0.7, label="Lehman")
        ax.set_title(
            f"{r['ticker']} v2  MAE={r['model'].mae:.2f}  "
            f"skill={r['model'].skill_vs_persistence:+.3f}  "
            f"DirAcc={r['model'].directional_accuracy:.1f}%"
        )
        ax.set_ylabel("Close ($)")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{r['ticker']}.png", dpi=100)
        plt.close(fig)
    print(f"\nSaved {len(rows)} plots to {out_dir}/")


def main() -> None:
    print(f"v2 pipeline: window={WINDOW_SIZE}d, target=returns, per-window z-score, early stop")
    print(f"Train up to {TRAIN_END}, test {TEST_START}..{TEST_END}")
    print(f"Tickers: {', '.join(TICKERS)}\n")

    rows = []
    start = time.time()
    for ticker in TICKERS:
        t0 = time.time()
        try:
            row = evaluate_ticker(ticker)
            rows.append(row)
            print(
                f"  [{ticker}] {time.time() - t0:.1f}s  "
                f"(train={row['n_train']}, val={row['n_val']}, test={row['n_test']}, "
                f"epochs={row['epochs_used']})"
            )
        except Exception as exc:
            print(f"  [{ticker}] FAILED: {exc}")
    print(f"\nTotal: {time.time() - start:.1f}s\n")

    if not rows:
        return
    print_per_ticker_table(rows)
    aggregate_summary(rows)
    save_prediction_plots(rows)


if __name__ == "__main__":
    main()
