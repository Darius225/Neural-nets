"""Evolve the best ReturnsCNNConfig with (1+1)-ES, validate on the
2008-2009 crisis test set.

Two-phase experiment:

  Phase 1 — *search*: run ES on a single proxy ticker (JPM) with the v3
  technical-features pipeline. Each fitness call trains a fresh returns
  CNN from the candidate config and returns its internal validation
  loss. Cheap (~5s per eval), so 25 iterations finishes in ~2 minutes.

  Phase 2 — *validate*: take the best config found in Phase 1 and apply
  it to all 10 test tickers. Compare against the hand-picked default
  ``ReturnsCNNConfig()`` we've been using through v2/v3/v4.

We do NOT use the crisis test set during search — that would be
selection-on-test leakage and would inflate the apparent skill score.

Run:
    python experiments/evolve_returns_v5.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.configs import RETURNS_CNN_RANGES, EvolutionConfig, ReturnsCNNConfig
from src.data import load_csv, prepare_windowed_returns_split
from src.features import build_technical_features
from src.metrics import compute_metrics, naive_persistence_forecast
from src.models import build_returns_cnn
from src.search.evolution import one_plus_one_es

TEST_TICKERS = ["JPM", "BAC", "C", "MSFT", "AAPL", "IBM", "JNJ", "PG", "GE", "XOM"]
PROXY_TICKER = "JPM"  # used only for the search phase
TRAIN_END = "2007-07-31"
TEST_START = "2007-08-01"
TEST_END = "2009-12-31"
WINDOW_SIZE = 30
SEARCH_EPOCHS = 25  # short — we're scoring configs, not optimising final fit
SEARCH_PATIENCE = 4
VALIDATE_EPOCHS = 60
VALIDATE_PATIENCE = 8
BATCH_SIZE = 64
CSV_DIR = "stock_market_data/sp500/csv"
ES = EvolutionConfig(max_iterations=25, mutation_probability=0.3, reset_threshold=8, seed=42)


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def load_split(ticker: str):
    df = load_csv(f"{CSV_DIR}/{ticker}.csv", with_dates=True)
    return prepare_windowed_returns_split(
        df,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        window_size=WINDOW_SIZE,
        feature_builder=build_technical_features,
    )


def train_one(split, config: ReturnsCNNConfig, epochs: int, patience: int):
    tf.keras.backend.clear_session()
    set_seed(ES.seed)
    model = build_returns_cnn(split.window_size, split.n_features, config=config)
    hist = model.fit(
        split.X_train,
        split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)],
    )
    return model, hist.history


def evaluate_on_ticker(model, split) -> dict:
    pred_returns = model.predict(split.X_test, verbose=0).flatten()
    pred_prices = split.close_at_t_test * (1 + pred_returns)
    actual = split.actual_close_test
    baseline = naive_persistence_forecast(split.close_at_t_test)
    m = compute_metrics(actual, pred_prices, y_prev=split.close_at_t_test)
    b = compute_metrics(actual, baseline, y_prev=split.close_at_t_test)
    return {"model": m, "baseline": b}


def phase1_search(proxy_split) -> ReturnsCNNConfig:
    """Run ES on the proxy ticker. Fitness = final val_loss (Huber)."""
    print(f"\nPHASE 1 — searching with ES on {PROXY_TICKER}")
    print(f"  ES budget: {ES.max_iterations} iterations, mutation_p={ES.mutation_probability}")

    def fitness(config: ReturnsCNNConfig) -> float:
        _, history = train_one(proxy_split, config, SEARCH_EPOCHS, SEARCH_PATIENCE)
        return float(min(history["val_loss"]))

    result = one_plus_one_es(
        ReturnsCNNConfig,
        RETURNS_CNN_RANGES,
        fitness,
        ES,
        initial=ReturnsCNNConfig(),  # start from the v2/v3/v4 default
    )
    print(
        f"\n  search done in {result.wall_time_s:.1f}s, "
        f"{result.evaluations} evals ({result.cache_hits} cache hits)"
    )
    print(f"  best val_loss: {result.best_fitness:.6f}")
    print(f"  best config:   {result.best_config.model_dump()}")
    return result.best_config


def phase2_validate(best_config: ReturnsCNNConfig) -> None:
    """Compare best_config against the default on all test tickers."""
    print("\nPHASE 2 — validating on all 10 test tickers")
    print(f"  default config: {ReturnsCNNConfig().model_dump()}")
    print(f"  found config:   {best_config.model_dump()}\n")

    header = f"{'ticker':<7}{'default skill':>16}{'evolved skill':>16}{'diff':>10}"
    print(header)
    print("-" * len(header))

    default_skills, evolved_skills = [], []
    for ticker in TEST_TICKERS:
        split = load_split(ticker)
        default_model, _ = train_one(split, ReturnsCNNConfig(), VALIDATE_EPOCHS, VALIDATE_PATIENCE)
        evolved_model, _ = train_one(split, best_config, VALIDATE_EPOCHS, VALIDATE_PATIENCE)

        d = evaluate_on_ticker(default_model, split)
        e = evaluate_on_ticker(evolved_model, split)
        ds = d["model"].skill_vs_persistence or 0
        es = e["model"].skill_vs_persistence or 0
        default_skills.append(ds)
        evolved_skills.append(es)
        delta = es - ds
        print(f"{ticker:<7}{ds:>+16.4f}{es:>+16.4f}{delta:>+10.4f}")

    n_default_win = sum(1 for s in default_skills if s > 0)
    n_evolved_win = sum(1 for s in evolved_skills if s > 0)
    print("-" * len(header))
    print(
        f"{'mean':<7}{np.mean(default_skills):>+16.4f}{np.mean(evolved_skills):>+16.4f}"
        f"{np.mean(evolved_skills) - np.mean(default_skills):>+10.4f}"
    )
    print(f"\nbeats persistence: default {n_default_win}/10  |  evolved {n_evolved_win}/10")


def main() -> None:
    print("v5 evolution: searching ReturnsCNNConfig with (1+1)-ES")
    print(f"Train up to {TRAIN_END}, test {TEST_START}..{TEST_END}, window={WINDOW_SIZE}")

    proxy_split = load_split(PROXY_TICKER)
    print(
        f"\nproxy ticker {PROXY_TICKER}: "
        f"train={len(proxy_split.X_train)}, val={len(proxy_split.X_val)}"
    )

    overall_t0 = time.time()
    best_config = phase1_search(proxy_split)
    phase2_validate(best_config)
    print(f"\ntotal wall time: {time.time() - overall_t0:.1f}s")


if __name__ == "__main__":
    main()
