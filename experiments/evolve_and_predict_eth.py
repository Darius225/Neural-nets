"""End-to-end: search the best ReturnsCNNConfig for ETH-USD via (1+1)-ES,
then train on ALL available history with that config and predict the
next trading day's return.

This is the demo loop a user actually wants to run: "find good hyperparams,
train, predict tomorrow". It closes the narrative of the repo — but the
honest reading of every prior experiment (skill score near zero on stocks
AND crypto) says **the prediction has near-zero predictive value beyond
the persistence baseline**. The output ends with a loud disclaimer to
discourage trading on it.

Workflow:
  Phase 1  search ES on ETHUSDT up to 2024-12-31, validate on 2025
  Phase 2  retrain best config on ALL data through yesterday
  Phase 3  z-score the last 30-day window, predict tomorrow's return
  Phase 4  print the result with disclaimer

Run:
    python experiments/evolve_and_predict_eth.py
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

from src.configs import EvolutionConfig, RETURNS_CNN_RANGES, ReturnsCNNConfig
from src.data import load_csv, prepare_windowed_returns_split
from src.data.splits import _zscore_window
from src.features import build_technical_features
from src.metrics import compute_metrics, naive_persistence_forecast
from src.models import build_returns_cnn
from src.search.evolution import one_plus_one_es


LOCAL_CSV = Path("stock_market_data/crypto/csv/ETHUSDT.csv")
WINDOW_SIZE = 30
SEARCH_EPOCHS = 20
SEARCH_PATIENCE = 4
FINAL_EPOCHS = 80
FINAL_PATIENCE = 8
BATCH_SIZE = 64
SEED = 42

# Hold out 2025 as the search-validation period so ES doesn't peek at
# the very-recent days we'll use to build the prediction window.
SEARCH_TRAIN_END = "2024-12-31"
SEARCH_TEST_START = "2025-01-01"
SEARCH_TEST_END = "2026-04-30"

ES = EvolutionConfig(max_iterations=20, mutation_probability=0.3, reset_threshold=8, seed=SEED)


def set_seed(s: int) -> None:
    np.random.seed(s); tf.random.set_seed(s)


def train_for_score(split, config: ReturnsCNNConfig, epochs: int, patience: int):
    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_returns_cnn(split.window_size, split.n_features, config=config)
    hist = model.fit(
        split.X_train, split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=epochs, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
                                 restore_best_weights=True)],
    )
    return model, hist.history


def phase1_search(df: pd.DataFrame) -> ReturnsCNNConfig:
    split = prepare_windowed_returns_split(
        df, train_end=SEARCH_TRAIN_END, test_start=SEARCH_TEST_START,
        test_end=SEARCH_TEST_END, window_size=WINDOW_SIZE,
        feature_builder=build_technical_features,
    )
    print(f"\nPHASE 1 — ES search (train up to {SEARCH_TRAIN_END}, "
          f"val on 2025+)")
    print(f"  X_train={split.X_train.shape}, X_val={split.X_val.shape}")

    def fitness(config: ReturnsCNNConfig) -> float:
        _, history = train_for_score(split, config, SEARCH_EPOCHS, SEARCH_PATIENCE)
        return float(min(history["val_loss"]))

    result = one_plus_one_es(
        ReturnsCNNConfig, RETURNS_CNN_RANGES, fitness, ES,
        initial=ReturnsCNNConfig(dropout=0.4, huber_delta=0.01, learning_rate=2e-3),
    )
    print(f"\n  ES done in {result.wall_time_s:.1f}s, "
          f"{result.evaluations} evals ({result.cache_hits} cache hits)")
    print(f"  best val_loss: {result.best_fitness:.6f}")
    print(f"  best config:   {result.best_config.model_dump()}")

    # Sanity check: score the best config against persistence on the held-out
    # 2025+ window so the user sees the (almost certainly near-zero) skill
    # before we use the same config for tomorrow's prediction.
    model, _ = train_for_score(split, result.best_config, SEARCH_EPOCHS, SEARCH_PATIENCE)
    pred_r = model.predict(split.X_test, verbose=0).flatten()
    pred_p = split.close_at_t_test * (1 + pred_r)
    m = compute_metrics(split.actual_close_test, pred_p, y_prev=split.close_at_t_test)
    corr = float(np.corrcoef(pred_r, split.y_test)[0, 1])
    print(f"\n  out-of-sample on 2025+ ({len(split.y_test)} days):")
    print(f"    MAE=${m.mae:.2f}  DirAcc={m.directional_accuracy:.1f}%  "
          f"skill={m.skill_vs_persistence:+.4f}  corr={corr:+.4f}")
    return result.best_config


def phase2_train_on_all(df: pd.DataFrame, config: ReturnsCNNConfig):
    """Retrain on ALL data so tomorrow's prediction uses every available
    observation. Internal-val on the last 15 % of windows for early stop;
    no held-out test (we already evaluated in Phase 1)."""
    from src.data.splits import _build_windows

    features = build_technical_features(df).dropna()
    closes = df.loc[features.index, "Close"].values.astype(np.float32)
    feat_arr = features.values.astype(np.float32)

    X_full, y_full, _ = _build_windows(feat_arr, closes, WINDOW_SIZE)
    cut = int(len(X_full) * 0.85)
    X_train, X_val = X_full[:cut], X_full[cut:]
    y_train, y_val = y_full[:cut], y_full[cut:]

    print(f"\nPHASE 2 — retrain on ALL data through {df.index[-1].date()} "
          f"({len(X_train)} train + {len(X_val)} val windows)")

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_returns_cnn(WINDOW_SIZE, feat_arr.shape[1], config=config)
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=FINAL_EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=FINAL_PATIENCE,
                                 restore_best_weights=True)],
    )
    print(f"  trained {len(history.history['loss'])} epochs, "
          f"best val_loss={min(history.history['val_loss']):.6f}")
    return model


def phase3_predict_tomorrow(model, df: pd.DataFrame) -> dict:
    """Build the last 30-day window from features, z-score it, predict."""
    features = build_technical_features(df).dropna()
    last_window = features.iloc[-WINDOW_SIZE:].values.astype(np.float32)
    X = _zscore_window(last_window)[np.newaxis, ...]
    pred_return = float(model.predict(X, verbose=0)[0, 0])
    last_close = float(df["Close"].iloc[-1])
    last_ts = df.index[-1]
    pred_price = last_close * (1 + pred_return)
    return {
        "last_date": last_ts.date(),
        "next_date": (last_ts + pd.Timedelta(days=1)).date(),
        "last_close": last_close,
        "pred_return": pred_return,
        "pred_price": pred_price,
    }


def banner(text: str, char: str = "=") -> None:
    print(f"\n{char * 70}\n  {text}\n{char * 70}")


def main() -> None:
    if not LOCAL_CSV.exists():
        raise SystemExit(
            f"missing {LOCAL_CSV}. Run:\n"
            f"  python scripts/fetch_binance.py ETHUSDT "
            f"--start 2018-01-01 --end 2026-05-17"
        )

    t0 = time.time()
    banner("ETH-USD: evolve config + predict tomorrow's return")
    df = load_csv(str(LOCAL_CSV), with_dates=True)
    print(f"  {len(df)} daily candles, {df.index[0].date()} .. {df.index[-1].date()}")

    best_config = phase1_search(df)
    model = phase2_train_on_all(df, best_config)
    pred = phase3_predict_tomorrow(model, df)

    banner("PREDICTION", "-")
    print(f"  last observed close: ${pred['last_close']:.2f}  ({pred['last_date']})")
    print(f"  predicted return:    {pred['pred_return']:+.4%}")
    print(f"  predicted price:     ${pred['pred_price']:.2f}  ({pred['next_date']})")

    banner("READ THIS BEFORE DOING ANYTHING WITH THE NUMBER ABOVE", "!")
    print("""
  The model's out-of-sample skill score vs the naive persistence
  baseline ("tomorrow == today") was essentially zero on every backtest
  we ran — across 10 S&P 500 tickers in the 2008 crisis AND across
  ETH-USD through the 2022 LUNA / FTX shocks.

  In plain English: the model is mathematically equivalent to "predict
  tomorrow will be the same as today, give or take a few basis points".
  ES tuning narrowed the gap to the baseline but did NOT cross it.

  Treat the number above as a methodology demo, not a trading signal.
  Do not size positions off it. The repo's value is in the honest
  pipeline + backtest discipline, not in the prediction itself.
""")
    print(f"  total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
