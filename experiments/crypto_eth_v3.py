"""ETH-USD regime-shift experiment, v3 pipeline.

Mirror of crisis_2008_v3.py but on crypto. Same pipeline (30-day window
of 10 technical features, per-window z-score, next-day return target,
Huber loss) — different asset class and different "crisis" calendar.

Test window covers two crypto-native shocks:
  - 2022-05-09  Terra/LUNA depeg + collapse
  - 2022-11-11  FTX bankruptcy filing

Train pre-shocks (2018-01-01 .. 2022-04-30), test through the chaos
(2022-05-01 .. 2024-01-01).

Data comes from yfinance — no local CSV needed for crypto. Run:

    python experiments/crypto_eth_v3.py
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

from src.configs import ReturnsCNNConfig
from src.data import load_csv, load_yfinance, prepare_windowed_returns_split
from src.features import build_technical_features
from src.metrics import compute_metrics, naive_persistence_forecast
from src.models import build_returns_cnn


TICKER = "ETH-USD"
BINANCE_SYMBOL = "ETHUSDT"
LOCAL_CSV = Path("stock_market_data/crypto/csv/ETHUSDT.csv")
TRAIN_END = "2022-04-30"
TEST_START = "2022-05-01"
TEST_END = "2024-01-01"
WINDOW_SIZE = 30
EPOCHS = 80
BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 8
SEED = 42
EVOLVED_CONFIG = ReturnsCNNConfig(dropout=0.4, huber_delta=0.01, learning_rate=2e-3)


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def save_plot(test_index, actual, pred, baseline, metrics, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    luna = pd.Timestamp("2022-05-09")
    ftx = pd.Timestamp("2022-11-11")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_index, actual, label="actual", color="black", linewidth=1.2)
    ax.plot(test_index, pred, label="CNN (evolved config)", color="C4", alpha=0.85, linewidth=1)
    ax.plot(test_index, baseline, label="persistence", color="C0", alpha=0.4, linewidth=0.8)
    ax.axvline(luna, color="orange", linestyle="--", alpha=0.7, label="LUNA collapse")
    ax.axvline(ftx, color="red", linestyle="--", alpha=0.7, label="FTX bankruptcy")
    ax.set_title(f"{TICKER}  MAE=${metrics.mae:.2f}  "
                 f"skill={metrics.skill_vs_persistence:+.4f}  "
                 f"DirAcc={metrics.directional_accuracy:.1f}%")
    ax.set_ylabel("Close ($)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    print(f"v3 pipeline on {TICKER}: train ending {TRAIN_END}, "
          f"test {TEST_START}..{TEST_END}")
    print(f"shocks in test window: LUNA collapse 2022-05-09, FTX 2022-11-11\n")

    t0 = time.time()

    # Prefer a locally cached Binance CSV; fall back to yfinance only when
    # the CSV isn't there. yfinance has been flaky for crypto in 2024-2025,
    # so the Binance public REST API (downloaded by scripts/fetch_binance.py)
    # is the more reliable source.
    if LOCAL_CSV.exists():
        print(f"loading {TICKER} from local Binance CSV ({LOCAL_CSV})...")
        df = load_csv(str(LOCAL_CSV), with_dates=True)
    else:
        print(f"local CSV not found at {LOCAL_CSV}; falling back to yfinance...")
        df = load_yfinance(TICKER)
        if len(df) == 0:
            raise SystemExit(
                f"yfinance returned 0 rows for {TICKER} and no local CSV exists.\n"
                f"Fix: run  python scripts/fetch_binance.py {BINANCE_SYMBOL} "
                f"--start 2018-01-01 --end 2024-01-01"
            )
    print(f"  {len(df)} rows, {df.index.min().date()} .. {df.index.max().date()}")

    split = prepare_windowed_returns_split(
        df, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END,
        window_size=WINDOW_SIZE, feature_builder=build_technical_features,
    )
    print(f"  X_train={split.X_train.shape}, X_val={split.X_val.shape}, "
          f"X_test={split.X_test.shape}")
    print(f"  y_test return distribution: mean={split.y_test.mean():+.4f} "
          f"std={split.y_test.std():.4f}")

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_returns_cnn(split.window_size, split.n_features, config=EVOLVED_CONFIG)
    hist = model.fit(
        split.X_train, split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE,
                                 restore_best_weights=True)],
    )
    print(f"  trained {len(hist.history['loss'])} epochs, "
          f"best val_loss={min(hist.history['val_loss']):.6f}")

    pred_returns = model.predict(split.X_test, verbose=0).flatten()
    pred_prices = split.close_at_t_test * (1 + pred_returns)
    baseline = naive_persistence_forecast(split.close_at_t_test)

    m = compute_metrics(split.actual_close_test, pred_prices, y_prev=split.close_at_t_test)
    b = compute_metrics(split.actual_close_test, baseline, y_prev=split.close_at_t_test)
    corr = float(np.corrcoef(pred_returns, split.y_test)[0, 1])

    print(f"\n{TICKER}  {TEST_START}..{TEST_END}")
    print("-" * 60)
    print(f"  model:       MAE=${m.mae:>8.3f}  RMSE=${m.rmse:>8.3f}  "
          f"DirAcc={m.directional_accuracy:5.2f}%  skill={m.skill_vs_persistence:+.4f}")
    print(f"  persistence: MAE=${b.mae:>8.3f}  RMSE=${b.rmse:>8.3f}")
    print(f"  corr(pred_returns, actual_returns) = {corr:+.4f}")
    print(f"  pred return std / actual return std = "
          f"{pred_returns.std() / split.y_test.std():.3f}")

    plot_path = "experiments/plots_crypto/ETH-USD.png"
    save_plot(split.test_index, split.actual_close_test, pred_prices, baseline, m, plot_path)
    print(f"\nplot: {plot_path}")
    print(f"total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
