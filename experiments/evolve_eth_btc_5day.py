"""ETH-USD prediction with the two highest-probability levers stacked:

  1. Multi-day forecast horizon (5-day return instead of 1-day).
     Persistence is a much weaker baseline at H=5 — daily noise averages
     out and any actual signal compounds.
  2. BTC as an exogenous feature (10 features) on top of ETH's own 10.
     ETH-BTC daily return correlation is ~0.85; the v3 model never saw
     BTC because each ticker trained alone.

Combined: ``(window=30, n_features=20)`` input, target = 5-day cumulative
return on ETH. Same v3 z-score-per-window + Huber loss + ES-evolved
config.

Run:
    python experiments/evolve_eth_btc_5day.py
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
from src.data import load_csv
from src.data.splits import _build_windows, _zscore_window
from src.features import build_technical_features
from src.metrics import compute_metrics, naive_persistence_forecast
from src.models import build_returns_cnn
from src.search.evolution import one_plus_one_es


ETH_CSV = Path("stock_market_data/crypto/csv/ETHUSDT.csv")
BTC_CSV = Path("stock_market_data/crypto/csv/BTCUSDT.csv")
WINDOW_SIZE = 30
HORIZON = 5
SEARCH_EPOCHS = 20
SEARCH_PATIENCE = 4
FINAL_EPOCHS = 80
FINAL_PATIENCE = 8
BATCH_SIZE = 64
SEED = 42

SEARCH_TRAIN_END = "2024-12-31"
SEARCH_TEST_START = "2025-01-01"
SEARCH_TEST_END = "2026-04-30"

ES = EvolutionConfig(max_iterations=20, mutation_probability=0.3, reset_threshold=8, seed=SEED)


def set_seed(s: int) -> None:
    np.random.seed(s); tf.random.set_seed(s)


def build_combined_features(eth_df: pd.DataFrame, btc_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """ETH features + BTC features (prefixed) on the date intersection."""
    eth_f = build_technical_features(eth_df)
    btc_f = build_technical_features(btc_df)
    btc_f = btc_f.add_prefix("btc_")

    combined = eth_f.join(btc_f, how="inner").dropna()
    # Closes aligned to the same index (ETH closes — that's what we predict).
    closes = eth_df.loc[combined.index, "Close"].astype(np.float32)
    return combined, closes


def build_windows_for_range(features: pd.DataFrame, closes: pd.Series,
                            start: str, end: str):
    """Slice by date then build windows + targets."""
    mask = (features.index >= pd.to_datetime(start)) & (features.index <= pd.to_datetime(end))
    feat_arr = features.loc[mask].values.astype(np.float32)
    close_arr = closes.loc[mask].values.astype(np.float32)
    idx = features.loc[mask].index
    if len(feat_arr) < WINDOW_SIZE + HORIZON:
        raise ValueError(
            f"Not enough rows in {start}..{end} for window={WINDOW_SIZE}, "
            f"horizon={HORIZON}: got {len(feat_arr)}"
        )
    X, y, close_at_t = _build_windows(feat_arr, close_arr, WINDOW_SIZE, HORIZON)
    test_index = idx[WINDOW_SIZE + HORIZON - 1 : WINDOW_SIZE + HORIZON - 1 + len(X)]
    return X, y, close_at_t, test_index


def train(X_train, y_train, X_val, y_val, config: ReturnsCNNConfig,
          epochs: int, patience: int):
    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_returns_cnn(WINDOW_SIZE, X_train.shape[2], config=config)
    hist = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience,
                                 restore_best_weights=True)],
    )
    return model, hist.history


def evaluate(model, X, y, close_at_t):
    pred_r = model.predict(X, verbose=0).flatten()
    pred_p = close_at_t * (1 + pred_r)
    actual_p = close_at_t * (1 + y)
    metrics = compute_metrics(actual_p, pred_p, y_prev=close_at_t)
    corr = float(np.corrcoef(pred_r, y)[0, 1]) if y.std() > 0 else float("nan")
    return metrics, pred_r, corr


def banner(text: str, char: str = "=") -> None:
    print(f"\n{char * 70}\n  {text}\n{char * 70}")


def main() -> None:
    for path in (ETH_CSV, BTC_CSV):
        if not path.exists():
            symbol = path.stem
            raise SystemExit(
                f"missing {path}. Run:\n"
                f"  python scripts/fetch_binance.py {symbol} "
                f"--start 2018-01-01 --end 2026-05-17"
            )

    t0 = time.time()
    banner("ETH-USD + BTC exogenous, 5-day horizon — evolve + predict")
    eth_df = load_csv(str(ETH_CSV), with_dates=True)
    btc_df = load_csv(str(BTC_CSV), with_dates=True)
    features, closes = build_combined_features(eth_df, btc_df)
    print(f"  combined features: {features.shape[1]} cols, "
          f"{features.index[0].date()} .. {features.index[-1].date()}, "
          f"{len(features)} rows")

    # PHASE 1 — ES on train (up to 2024) with val on 2025+ ----------------
    X_full, y_full, c_full, _ = build_windows_for_range(
        features, closes, "1900-01-01", SEARCH_TRAIN_END,
    )
    val_cut = int(len(X_full) * 0.85)
    X_tr, X_val = X_full[:val_cut], X_full[val_cut:]
    y_tr, y_val = y_full[:val_cut], y_full[val_cut:]
    print(f"\nPHASE 1 — ES (train {X_tr.shape}, val {X_val.shape})")

    def fitness(config: ReturnsCNNConfig) -> float:
        _, h = train(X_tr, y_tr, X_val, y_val, config, SEARCH_EPOCHS, SEARCH_PATIENCE)
        return float(min(h["val_loss"]))

    result = one_plus_one_es(
        ReturnsCNNConfig, RETURNS_CNN_RANGES, fitness, ES,
        initial=ReturnsCNNConfig(dropout=0.4, huber_delta=0.01, learning_rate=2e-3),
    )
    print(f"\n  ES: {result.wall_time_s:.1f}s, {result.evaluations} evals, "
          f"best val_loss={result.best_fitness:.6f}")
    print(f"  best config: {result.best_config.model_dump()}")

    # Score on the held-out 2025+ window with the BEST config.
    X_test, y_test, c_test, test_idx = build_windows_for_range(
        features, closes, SEARCH_TEST_START, SEARCH_TEST_END,
    )
    model, _ = train(X_tr, y_tr, X_val, y_val, result.best_config,
                     SEARCH_EPOCHS, SEARCH_PATIENCE)
    m, pred_r, corr = evaluate(model, X_test, y_test, c_test)
    print(f"\n  OOS on {SEARCH_TEST_START}..{SEARCH_TEST_END} "
          f"({len(y_test)} {HORIZON}-day windows):")
    print(f"    MAE=${m.mae:.2f}  RMSE=${m.rmse:.2f}  DirAcc={m.directional_accuracy:.1f}%")
    print(f"    skill_vs_persistence={m.skill_vs_persistence:+.4f}  "
          f"corr(pred,act)={corr:+.4f}")
    print(f"    pred std / actual std = {pred_r.std()/y_test.std():.3f}")

    # PHASE 2 — retrain on ALL data ----------------------------------------
    X_all, y_all, _, _ = build_windows_for_range(
        features, closes, "1900-01-01", str(features.index[-1].date()),
    )
    cut = int(len(X_all) * 0.85)
    print(f"\nPHASE 2 — retrain on ALL data ({cut} train + {len(X_all) - cut} val)")
    final_model, h2 = train(X_all[:cut], y_all[:cut], X_all[cut:], y_all[cut:],
                            result.best_config, FINAL_EPOCHS, FINAL_PATIENCE)
    print(f"  trained {len(h2['loss'])} epochs, "
          f"best val_loss={min(h2['val_loss']):.6f}")

    # PHASE 3 — predict 5 days ahead ---------------------------------------
    last_window = features.iloc[-WINDOW_SIZE:].values.astype(np.float32)
    X_now = _zscore_window(last_window)[np.newaxis, ...]
    pred_return = float(final_model.predict(X_now, verbose=0)[0, 0])
    last_close = float(closes.iloc[-1])
    last_date = features.index[-1].date()
    pred_price = last_close * (1 + pred_return)
    target_date = (features.index[-1] + pd.Timedelta(days=HORIZON)).date()

    banner("PREDICTION (5 trading days ahead)", "-")
    print(f"  last observed close: ${last_close:.2f}  ({last_date})")
    print(f"  predicted 5-day ret: {pred_return:+.4%}")
    print(f"  predicted close:     ${pred_price:.2f}  (~{target_date})")

    banner("HOW TO READ THIS", "!")
    print(f"""
  Phase 1 OOS skill_vs_persistence = {m.skill_vs_persistence:+.4f}
  Phase 1 OOS corr(pred, actual)   = {corr:+.4f}

  Multi-day horizon (H=5) + BTC exogenous features were the two
  highest-probability levers per the prior analysis. Read the skill
  score above carefully — if it's still near zero, the model is
  effectively predicting a constant (~zero return), and the price
  number is just last_close * (1 + small_constant).

  This experiment's role is to test whether stacking the two biggest
  levers crosses the persistence floor on crypto data. If yes: real
  improvement. If no: confirms the OHLCV-only ceiling holds even with
  cross-asset info — strong evidence for the EMH conclusion.
""")
    print(f"  total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
