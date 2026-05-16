"""Measure RAM growth during a long search loop, with and without
``tf.keras.backend.clear_session()`` between model builds.

Why it matters: each ``Sequential(...)`` + ``model.fit(...)`` leaves
state in the global TF graph and weak refs. Over hundreds of fitness
evaluations, RSS creeps up monotonically. On a long search (or on GPU)
this is the difference between finishing and OOM-killing.

Run:
    python benchmarks/bench_memory.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf

from src.data import prepare_dataset
from src.models import build_general_cnn
from src.training import train_on_prepared

N_ITER = 80
EPOCHS = 3
BATCH_SIZE = 32
PARAMS = {
    "number_of_filters": 32,
    "kernel_size": 3,
    "activation_in_convolution": "relu",
    "activation_in_dense_layer": "relu",
    "nodes_in_dense_layer": 32,
    "optimizer": "adam",
    "loss": "mean_squared_error",
}


def make_synthetic_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 1, n)) + 100
    return pd.DataFrame(
        {
            "Open": base + rng.normal(0, 0.5, n),
            "High": base + rng.uniform(0, 1, n),
            "Low": base - rng.uniform(0, 1, n),
            "Close": base + rng.normal(0, 0.5, n),
            "Volume": rng.integers(1e6, 2e6, n),
        }
    )


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_loop(clear_session: bool, dataset) -> list[float]:
    """Return RSS in MB after each iteration."""
    rss_trace = []
    for _ in range(N_ITER):
        if clear_session:
            tf.keras.backend.clear_session()
            gc.collect()
        train_on_prepared(
            dataset,
            model_factory=build_general_cnn,
            params=PARAMS,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
        rss_trace.append(rss_mb())
    return rss_trace


def summarise(label: str, trace: list[float]) -> None:
    start = trace[0]
    end = trace[-1]
    peak = max(trace)
    growth = end - start
    print(
        f"  {label:<24}  start={start:>7.1f}MB  end={end:>7.1f}MB  "
        f"peak={peak:>7.1f}MB  growth={growth:+.1f}MB"
    )


def main() -> None:
    df = make_synthetic_df()
    dataset = prepare_dataset(df)

    print(f"{N_ITER} iterations, {EPOCHS} epochs each, RSS measured per iter\n")
    print("warming up TF...")
    train_on_prepared(
        dataset, model_factory=build_general_cnn, params=PARAMS, epochs=2, batch_size=BATCH_SIZE
    )
    tf.keras.backend.clear_session()
    gc.collect()
    print("done.\n")

    t0 = time.time()
    trace_off = run_loop(clear_session=False, dataset=dataset)
    t1 = time.time()
    tf.keras.backend.clear_session()
    gc.collect()
    trace_on = run_loop(clear_session=True, dataset=dataset)
    t2 = time.time()

    summarise("WITHOUT clear_session", trace_off)
    summarise("WITH clear_session", trace_on)
    print(f"\n  time WITHOUT clear : {t1 - t0:.1f}s")
    print(f"  time WITH    clear : {t2 - t1:.1f}s")

    growth_off = trace_off[-1] - trace_off[0]
    growth_on = trace_on[-1] - trace_on[0]
    print(f"\n  memory leak reduced by {growth_off - growth_on:+.1f}MB over {N_ITER} iterations")
    if growth_off > 0:
        print(
            f"  (without clear_session, RSS grew "
            f"{growth_off / N_ITER:.2f}MB per iteration on average)"
        )


if __name__ == "__main__":
    main()
