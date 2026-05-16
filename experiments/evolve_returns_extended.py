"""Extended-budget version of evolve_returns_v5.

Same pipeline (v3 windowed-returns + per-ticker tech features + Pydantic
ReturnsCNNConfig + (1+1)-ES), but the search budget is doubled to 50
iterations. The question: did v5's "ES finds a better config" result
plateau at 25 iters, or is there more juice with more samples?

Validation phase is identical to v5: train both the default and the
evolved config on each of the 10 test tickers and compare skill scores.

Run:
    python experiments/evolve_returns_extended.py
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
from src.search.evolution import one_plus_one_es
from src.features import build_technical_features
from src.metrics import compute_metrics, naive_persistence_forecast
from src.models import build_returns_cnn


TEST_TICKERS = ["JPM", "BAC", "C", "MSFT", "AAPL", "IBM", "JNJ", "PG", "GE", "XOM"]
PROXY_TICKER = "JPM"
TRAIN_END = "2007-07-31"
TEST_START = "2007-08-01"
TEST_END = "2009-12-31"
WINDOW_SIZE = 30
SEARCH_EPOCHS = 25
SEARCH_PATIENCE = 4
VALIDATE_EPOCHS = 60
VALIDATE_PATIENCE = 8
BATCH_SIZE = 64
CSV_DIR = "stock_market_data/sp500/csv"
ES = EvolutionConfig(max_iterations=50, mutation_probability=0.3, reset_threshold=12, seed=42)


def set_seed(s: int) -> None:
    np.random.seed(s); tf.random.set_seed(s)


def load_split(ticker: str):
    df = load_csv(f"{CSV_DIR}/{ticker}.csv", with_dates=True)
    return prepare_windowed_returns_split(
        df, train_end=TRAIN_END, test_start=TEST_START, test_end=TEST_END,
        window_size=WINDOW_SIZE, feature_builder=build_technical_features,
    )


def train_one(split, config: ReturnsCNNConfig, epochs: int, patience: int):
    tf.keras.backend.clear_session()
    set_seed(ES.seed)
    model = build_returns_cnn(split.window_size, split.n_features, config=config)
    hist = model.fit(
        split.X_train, split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=epochs, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)],
    )
    return model, hist.history


def evaluate_on_ticker(model, split) -> dict:
    pred_returns = model.predict(split.X_test, verbose=0).flatten()
    pred_prices = split.close_at_t_test * (1 + pred_returns)
    return compute_metrics(
        split.actual_close_test, pred_prices, y_prev=split.close_at_t_test,
    )


def main() -> None:
    print(f"extended-budget v5: ES.max_iterations={ES.max_iterations}, "
          f"reset={ES.reset_threshold}, seed={ES.seed}")
    print(f"proxy={PROXY_TICKER}, validate on {len(TEST_TICKERS)} tickers\n")

    proxy_split = load_split(PROXY_TICKER)

    def fitness(config: ReturnsCNNConfig) -> float:
        _, history = train_one(proxy_split, config, SEARCH_EPOCHS, SEARCH_PATIENCE)
        return float(min(history["val_loss"]))

    t0 = time.time()
    result = one_plus_one_es(
        ReturnsCNNConfig, RETURNS_CNN_RANGES, fitness, ES,
        initial=ReturnsCNNConfig(),
    )
    print(f"\nsearch: {result.wall_time_s:.1f}s, evals={result.evaluations}, "
          f"cache_hits={result.cache_hits}, best_val_loss={result.best_fitness:.6f}")
    print(f"best config: {result.best_config.model_dump()}")

    print("\nvalidating default vs evolved on all 10 test tickers")
    header = f"{'ticker':<7}{'default skill':>16}{'evolved skill':>16}{'diff':>10}"
    print(header); print("-" * len(header))

    d_skills, e_skills = [], []
    for ticker in TEST_TICKERS:
        split = load_split(ticker)
        d_model, _ = train_one(split, ReturnsCNNConfig(), VALIDATE_EPOCHS, VALIDATE_PATIENCE)
        e_model, _ = train_one(split, result.best_config, VALIDATE_EPOCHS, VALIDATE_PATIENCE)
        d = evaluate_on_ticker(d_model, split).skill_vs_persistence or 0
        e = evaluate_on_ticker(e_model, split).skill_vs_persistence or 0
        d_skills.append(d); e_skills.append(e)
        print(f"{ticker:<7}{d:>+16.4f}{e:>+16.4f}{e - d:>+10.4f}")

    print("-" * len(header))
    print(f"{'mean':<7}{np.mean(d_skills):>+16.4f}{np.mean(e_skills):>+16.4f}"
          f"{np.mean(e_skills) - np.mean(d_skills):>+10.4f}")
    print(f"\nbeats persistence: default {sum(1 for s in d_skills if s>0)}/10  |  "
          f"evolved {sum(1 for s in e_skills if s>0)}/10")
    print(f"total wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
