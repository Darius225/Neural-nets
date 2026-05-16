"""Head-to-head benchmark: naive search loop vs the optimised version.

Three scenarios, same synthetic data, same RNG seed, same N evaluations:

  A. NAIVE       — re-runs ``prepare_dataset`` per evaluation, no cache,
                   no ``clear_session``. (Original notebook pattern.)
  B. PREPARED    — dataset prepared once, no cache, no clear_session.
                   Isolates the cost of dataset prep.
  C. FULL        — prepared dataset + fitness cache + clear_session +
                   dedup mutation. (Current ``one_plus_one_es``.)

Run:
    python benchmarks/bench_search.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import Dataset, prepare_dataset
from src.search.hyperparam import _key, mutate, random_individual
from src.models import build_general_cnn
from src.training import train, train_on_prepared

# ---------- tiny synthetic problem so the benchmark runs in seconds ----------

def make_synthetic_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1, n)) + 100
    return pd.DataFrame(
        {
            "Open":   base + rng.normal(0, 0.5, n),
            "High":   base + rng.uniform(0, 1, n),
            "Low":    base - rng.uniform(0, 1, n),
            "Close":  base + rng.normal(0, 0.5, n),
            "Volume": rng.integers(1e6, 2e6, n),
        }
    )


N_EVALS = 20          # individuals per run
EPOCHS = 8            # keep each train short — we measure structure, not convergence
BATCH_SIZE = 32
MUTATION_P = 0.3
SEED = 42

# Tiny hyperparameter ranges so trainings finish in <1s each, but we keep
# the same shape of work (Conv1D + Dense). The point is *relative* timing.
TINY_RANGES = {
    "number_of_filters": [16, 24, 32],
    "kernel_size": [2, 3],
    "activation_in_convolution": ["relu", "tanh"],
    "activation_in_dense_layer": ["relu", "linear"],
    "nodes_in_dense_layer": [16, 32],
    "optimizer": ["adam"],
    "loss": ["mean_squared_error", "mean_absolute_error"],
}


def tiny_individual(rng: random.Random) -> Dict:
    return {k: rng.choice(v) for k, v in TINY_RANGES.items()}


def tiny_mutate(ind: Dict, rng: random.Random, force_change: bool = False) -> Dict:
    while True:
        mutated = {
            k: (rng.choice(v) if rng.random() < MUTATION_P else ind[k])
            for k, v in TINY_RANGES.items()
        }
        if not force_change or mutated != ind:
            return mutated


def generate_individuals(n: int, seed: int) -> List[Dict]:
    """Generate the same sequence of individuals every run so each
    scenario evaluates *identical* configs — only the machinery differs.
    """
    rng = random.Random(seed)
    seq = [tiny_individual(rng)]
    for _ in range(n - 1):
        seq.append(tiny_mutate(seq[-1], rng, force_change=False))
    return seq


# ----------------------------- scenarios -----------------------------------


def scenario_naive(df: pd.DataFrame, individuals: List[Dict]) -> Tuple[float, int, int]:
    tf.keras.backend.clear_session()  # fair startup state
    start = time.time()
    n_dup = 0
    seen = set()
    for ind in individuals:
        k = _key(ind)
        if k in seen:
            n_dup += 1
        seen.add(k)
        # The original pattern: prepare dataset inside every fitness eval.
        train(
            df,
            model_factory=build_general_cnn,
            params=ind,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
    return time.time() - start, len(individuals), n_dup


def scenario_prepared(dataset: Dataset, individuals: List[Dict]) -> Tuple[float, int, int]:
    tf.keras.backend.clear_session()  # fair startup state
    start = time.time()
    n_dup = 0
    seen = set()
    for ind in individuals:
        k = _key(ind)
        if k in seen:
            n_dup += 1
        seen.add(k)
        train_on_prepared(
            dataset,
            model_factory=build_general_cnn,
            params=ind,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
    return time.time() - start, len(individuals), n_dup


def scenario_full(dataset: Dataset, individuals: List[Dict]) -> Tuple[float, int, int]:
    tf.keras.backend.clear_session()  # fair startup state
    cache: Dict = {}
    start = time.time()
    cache_hits = 0
    for ind in individuals:
        k = _key(ind)
        if k in cache:
            cache_hits += 1
            continue
        tf.keras.backend.clear_session()
        result = train_on_prepared(
            dataset,
            model_factory=build_general_cnn,
            params=ind,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
        cache[k] = result.final_val_mape
    return time.time() - start, len(individuals) - cache_hits, cache_hits


# ----------------------------- driver --------------------------------------


def main() -> None:
    df = make_synthetic_df()
    dataset = prepare_dataset(df)
    individuals = generate_individuals(N_EVALS, SEED)

    print(f"{N_EVALS} individuals, {EPOCHS} epochs each, {len(df)} rows of synthetic OHLCV\n")

    # Warm-up: pay TF init / first-fit JIT cost up front so it doesn't
    # land on whichever scenario runs first.
    print("warming up TF...")
    train_on_prepared(
        dataset, model_factory=build_general_cnn,
        params=individuals[0], epochs=2, batch_size=BATCH_SIZE,
    )
    print("done. running scenarios...\n")

    t_naive, n_eval_a, n_dup_a = scenario_naive(df, individuals)
    t_prep, n_eval_b, n_dup_b = scenario_prepared(dataset, individuals)
    t_full, n_eval_c, n_hits_c = scenario_full(dataset, individuals)

    rows = [
        ("A. naive (prep-per-eval)", t_naive, n_eval_a, f"{n_dup_a} dups (retrained)"),
        ("B. prepared once",         t_prep,  n_eval_b, f"{n_dup_b} dups (retrained)"),
        ("C. prepared + cache",      t_full,  n_eval_c, f"{n_hits_c} cache hits (skipped)"),
    ]

    width = max(len(r[0]) for r in rows)
    print(f"{'scenario':<{width}}  {'total':>8}  {'evals':>6}  notes")
    print("-" * (width + 40))
    for name, t, n, notes in rows:
        print(f"{name:<{width}}  {t:>7.2f}s  {n:>6}  {notes}")

    print()
    print(f"B vs A speedup: {t_naive / t_prep:.2f}x  (dataset prep overhead removed)")
    print(f"C vs B speedup: {t_prep / t_full:.2f}x  (cache + clear_session)")
    print(f"C vs A speedup: {t_naive / t_full:.2f}x  (total)")


if __name__ == "__main__":
    main()
