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
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf

from .data import Dataset
from .evolution import memoize_by
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
) -> float:
    """Train and return final validation MAPE (lower is better).

    Pure score-the-individual. Caching lives in :func:`memoize_by` and
    is applied by the caller; failed configurations return ``inf`` so
    they're never selected.
    """
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
        return fitness
    except Exception as exc:
        if verbose:
            print(f"[fail] {exc}  {individual}")
        return float("inf")


@dataclass
class SearchHistory:
    best_fitness_per_iteration: List[float] = field(default_factory=list)
    best_params: Optional[Individual] = None
    best_fitness: float = float("inf")
    cache_hits: int = 0
    evaluations: int = 0

    def consider(self, candidate: Individual, fitness: float, *, verbose: bool = False) -> bool:
        """Update best_* if ``fitness`` is the new global optimum.

        Returns ``True`` when a new optimum was recorded. The verbose
        toggle prints a short progress line — the caller doesn't have
        to wrap this in its own ``if verbose:``.
        """
        if fitness >= self.best_fitness:
            return False
        self.best_fitness = fitness
        self.best_params = dict(candidate)
        if verbose:
            print(f"  -> new best {fitness:.4f}")
        return True


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
    history = SearchHistory()

    @memoize_by(_key, enabled=use_cache)
    def score(individual: Individual) -> float:
        return evaluate(
            individual, dataset, epochs=epochs, batch_size=batch_size,
            early_stopping_patience=early_stopping_patience, verbose=verbose,
        )

    current = initial if initial is not None else random_individual(rng)
    current_fitness = score(current)
    history.consider(current, current_fitness, verbose=verbose)
    history.best_fitness_per_iteration.append(current_fitness)
    no_progress = 0

    for _ in range(max_iterations):
        candidate = mutate(current, mutation_probability, rng)
        candidate_fitness = score(candidate)

        if candidate_fitness <= current_fitness:
            current, current_fitness = candidate, candidate_fitness
            history.consider(candidate, candidate_fitness, verbose=verbose)
            no_progress = 0
        else:
            no_progress += 1

        if no_progress >= reset_threshold:
            if verbose:
                print("  -> stagnation, restarting")
            current = random_individual(rng)
            current_fitness = score(current)
            no_progress = 0

        history.best_fitness_per_iteration.append(history.best_fitness)

    history.evaluations = score.hits + score.misses
    history.cache_hits = score.hits
    return history
