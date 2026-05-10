"""2008 crisis experiment, v3 pipeline.

Identical setup to v2 (windowed returns, per-window z-score, internal
val + early stopping), but the feature space is **10 technical
indicators derived from OHLCV** instead of raw OHLCV:

  log_return, vol_10, vol_20, momentum_10, close_over_sma20, rsi_14,
  bb_position, volume_z, hl_range_pct, co_gap_pct

All features are computed from the ticker's own history — no internet,
no cross-asset data. The question we're answering: can a richer
representation of past behaviour beat the persistence baseline on the
2008-2009 crisis window?

Run:
    python experiments/crisis_2008_v3.py
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
from src.features import build_technical_features
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
    df = load_csv(f"{CSV_DIR}/{ticker}.csv", with_dates=True)
    split = prepare_windowed_returns_split(
        df, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END,
        window_size=WINDOW_SIZE,
        feature_builder=build_technical_features,
    )

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_returns_cnn(split.window_size, split.n_features)
    hist = model.fit(
        split.X_train, split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss",
                                 patience=EARLY_STOP_PATIENCE,
                                 restore_best_weights=True)],
    )

    pred_returns = model.predict(split.X_test, verbose=0).flatten()
    pred_prices = split.close_at_t_test * (1 + pred_returns)
    actual_prices = split.actual_close_test
    baseline_prices = naive_persistence_forecast(split.close_at_t_test)

    model_metrics = compute_metrics(
        actual_prices, pred_prices, y_prev=split.close_at_t_test,
        train_min=split.train_close_min, train_max=split.train_close_max,
    )
    baseline_metrics = compute_metrics(
        actual_prices, baseline_prices, y_prev=split.close_at_t_test,
    )

    # Sanity diagnostic on return predictions.
    return_corr = float(np.corrcoef(pred_returns, split.y_test)[0, 1])
    return_pred_std = float(pred_returns.std())
    return_actual_std = float(split.y_test.std())

    return {
        "ticker": ticker,
        "n_train": len(split.X_train),
        "n_test": len(split.X_test),
        "epochs_used": len(hist.history["loss"]),
        "model": model_metrics,
        "baseline": baseline_metrics,
        "return_corr": return_corr,
        "return_pred_std_ratio": return_pred_std / return_actual_std,
        "predictions": pd.DataFrame(
            {"actual": actual_prices, "predicted": pred_prices, "baseline": baseline_prices,
             "pred_return": pred_returns, "actual_return": split.y_test},
            index=split.test_index,
        ),
    }


def print_per_ticker_table(rows: list[dict]) -> None:
    header = (
        f"{'ticker':<7}{'MAE':>8}{'RMSE':>8}{'MAPE%':>8}{'DirAcc%':>9}"
        f"{'Skill':>9}{'r(pred,act)':>13}{'std_ratio':>11}{'epochs':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        m = r["model"]
        skill = "n/a" if m.skill_vs_persistence is None else f"{m.skill_vs_persistence:+.3f}"
        print(
            f"{r['ticker']:<7}{m.mae:>8.3f}{m.rmse:>8.3f}{m.mape:>8.2f}"
            f"{m.directional_accuracy:>9.2f}{skill:>9}"
            f"{r['return_corr']:>+13.4f}{r['return_pred_std_ratio']:>11.3f}{r['epochs_used']:>8}"
        )


def aggregate_summary(rows: list[dict]) -> None:
    skills = [r["model"].skill_vs_persistence for r in rows if r["model"].skill_vs_persistence is not None]
    n_beat = sum(1 for s in skills if s > 0)
    corrs = [r["return_corr"] for r in rows]
    print("\nAggregate over", len(rows), "tickers")
    print("-" * 60)
    print(f"  mean MAE          : model = {np.mean([r['model'].mae for r in rows]):.3f}  "
          f"|  persistence = {np.mean([r['baseline'].mae for r in rows]):.3f}")
    print(f"  mean DirAcc%      : {np.mean([r['model'].directional_accuracy for r in rows]):.2f}")
    print(f"  beats persistence : {n_beat}/{len(rows)} tickers (skill > 0)")
    print(f"  mean skill score  : {np.mean(skills):+.4f}")
    print(f"  mean r(pred, act) : {np.mean(corrs):+.4f}   "
          f"(positive = model output correlated with actual returns)")
    print(f"  mean epochs used  : {np.mean([r['epochs_used'] for r in rows]):.1f} of {EPOCHS}")


def save_prediction_plots(rows: list[dict], out_dir: str = "experiments/plots_v3") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    lehman = pd.Timestamp("2008-09-15")
    for r in rows:
        df = r["predictions"]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["actual"], label="actual", color="black", linewidth=1.2)
        ax.plot(df.index, df["predicted"], label="v3 CNN (10 tech features)",
                color="C4", alpha=0.85, linewidth=1)
        ax.plot(df.index, df["baseline"], label="persistence", color="C0", alpha=0.4, linewidth=0.8)
        ax.axvline(lehman, color="grey", linestyle="--", alpha=0.7, label="Lehman")
        ax.set_title(f"{r['ticker']} v3  MAE={r['model'].mae:.2f}  "
                     f"skill={r['model'].skill_vs_persistence:+.3f}  "
                     f"corr={r['return_corr']:+.3f}")
        ax.set_ylabel("Close ($)")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{out_dir}/{r['ticker']}.png", dpi=100)
        plt.close(fig)
    print(f"\nSaved {len(rows)} plots to {out_dir}/")


def main() -> None:
    print(f"v3 pipeline: window={WINDOW_SIZE}d, 10 technical features, per-window z-score, early stop")
    print(f"Train up to {TRAIN_END}, test {TEST_START}..{TEST_END}")
    print(f"Tickers: {', '.join(TICKERS)}\n")

    rows = []
    start = time.time()
    for ticker in TICKERS:
        t0 = time.time()
        try:
            row = evaluate_ticker(ticker)
            rows.append(row)
            print(f"  [{ticker}] {time.time() - t0:.1f}s  "
                  f"(train={row['n_train']}, test={row['n_test']}, epochs={row['epochs_used']})")
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
