"""(1+1) Evolution Strategy over the discrete CNN hyperparameter space.

Each individual is a config dict; each fitness evaluation trains a fresh
CNN from scratch on a *shared, pre-prepared* ``Dataset``. For
weight-level neuroevolution, see ``src.neuroevolution``.

Three speedups over the naive notebook version:

1. **Dataset prepared once** outside the loop (no re-scaling per eval).
2. **Fitness cache** keyed by the config — mutations that don't change
   any gene, or revisits after a restart, are free.
3. **clear_session()** before each model build — bounds TF memory growth
   over long searches.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

from .data import Dataset
from .models import HYPERPARAMETER_RANGES, build_general_cnn
from .training import train_on_prepared

Individual = Dict[str, Any]
_CacheKey = Tuple[Tuple[str, Any], ...]


def _key(individual: Individual) -> _CacheKey:
    return tuple(sorted(individual.items()))


def random_individual(rng: Optional[random.Random] = None) -> Individual:
    rng = rng if rng is not None else random.Random()
    return {key: rng.choice(values) for key, values in HYPERPARAMETER_RANGES.items()}


def mutate(
    individual: Individual,
    mutation_probability: float,
    rng: Optional[random.Random] = None,
    *,
    force_change: bool = True,
    max_tries: int = 16,
) -> Individual:
    """Return a mutated copy. If ``force_change`` (default), retry until
    at least one gene differs — avoids wasting a fitness evaluation on
    an identical individual."""
    rng = rng if rng is not None else random.Random()
    for _ in range(max_tries):
        mutated = {
            k: (rng.choice(v) if rng.random() < mutation_probability else individual[k])
            for k, v in HYPERPARAMETER_RANGES.items()
        }
        if not force_change or mutated != individual:
            return mutated
    # Couldn't differ after max_tries — flip one random gene unconditionally.
    forced = dict(individual)
    gene = rng.choice(list(HYPERPARAMETER_RANGES))
    choices = [v for v in HYPERPARAMETER_RANGES[gene] if v != individual[gene]]
    if choices:
        forced[gene] = rng.choice(choices)
    return forced


def evaluate(
    individual: Individual,
    dataset: Dataset,
    *,
    epochs: int,
    batch_size: int,
    early_stopping_patience: Optional[int],
    verbose: bool = True,
    cache: Optional[Dict[_CacheKey, float]] = None,
) -> float:
    """Train and return final validation MAPE (lower is better).

    ``cache`` short-circuits duplicate evaluations. Failed configurations
    (e.g. kernel larger than feature window) return ``inf`` so they're
    never selected.
    """
    key = _key(individual)
    if cache is not None and key in cache:
        if verbose:
            print(f"[cache] {cache[key]:.4f} MAPE  {individual}")
        return cache[key]

    tf.keras.backend.clear_session()
    start = time.time()
    try:
        result = train_on_prepared(
            dataset,
            model_factory=build_general_cnn,
            params=individual,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=early_stopping_patience,
        )
        fitness = result.final_val_mape
        if verbose:
            print(f"[ok] {fitness:.4f} MAPE in {time.time() - start:.1f}s  {individual}")
    except Exception as exc:
        if verbose:
            print(f"[fail] {exc}  {individual}")
        fitness = float("inf")

    if cache is not None:
        cache[key] = fitness
    return fitness


@dataclass
class SearchHistory:
    best_fitness_per_iteration: List[float] = field(default_factory=list)
    best_params: Optional[Individual] = None
    best_fitness: float = float("inf")
    cache_hits: int = 0
    evaluations: int = 0


def one_plus_one_es(
    dataset: Dataset,
    *,
    max_iterations: int = 200,
    reset_threshold: int = 15,
    mutation_probability: float = 0.3,
    initial: Optional[Individual] = None,
    epochs: int = 100,
    batch_size: int = 50,
    early_stopping_patience: Optional[int] = 15,
    verbose: bool = True,
    seed: Optional[int] = None,
    use_cache: bool = True,
) -> SearchHistory:
    """(1+1)-ES with restart on stagnation. Operates on a pre-prepared
    ``Dataset`` so scaling/splitting happens exactly once."""
    rng = random.Random(seed)
    cache: Optional[Dict[_CacheKey, float]] = {} if use_cache else None

    eval_fn = partial(
        evaluate,
        dataset=dataset,
        epochs=epochs,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
        cache=cache,
    )

    current = initial if initial is not None else random_individual(rng)
    current_fitness = eval_fn(current)
    history = SearchHistory(
        best_fitness_per_iteration=[current_fitness],
        best_params=dict(current),
        best_fitness=current_fitness,
        evaluations=1,
    )
    no_progress = 0

    for _ in range(max_iterations):
        candidate = mutate(current, mutation_probability, rng)
        if cache is not None and _key(candidate) in cache:
            history.cache_hits += 1
        candidate_fitness = eval_fn(candidate)
        history.evaluations += 1

        if candidate_fitness <= current_fitness:
            current, current_fitness = candidate, candidate_fitness
            no_progress = 0
            if candidate_fitness < history.best_fitness:
                history.best_fitness = candidate_fitness
                history.best_params = dict(candidate)
                if verbose:
                    print(f"  -> new best {candidate_fitness:.4f}")
        else:
            no_progress += 1

        if no_progress >= reset_threshold:
            if verbose:
                print("  -> stagnation, restarting")
            current = random_individual(rng)
            current_fitness = eval_fn(current)
            history.evaluations += 1
            no_progress = 0

        history.best_fitness_per_iteration.append(history.best_fitness)

    return history
