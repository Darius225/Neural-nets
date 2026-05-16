"""Schema-driven (1+1) Evolution Strategy.

Generalises the (1+1)-ES from :mod:`src.hyperparam_search` to operate on
*any* Pydantic configuration class. Caller supplies:

  - the config class (e.g. :class:`src.configs.ReturnsCNNConfig`)
  - a ``ranges`` dict with one entry per field naming discrete choices
  - a ``fitness`` function ``(config) -> float`` (lower is better)

Mutation flips a subset of fields to fresh draws from ``ranges``;
restart-on-stagnation is the same idea as the original hyperparameter
search. Pydantic validates every candidate at construction, so a buggy
ranges entry surfaces as a loud error rather than a silent invalid
training run.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel

from .configs import EvolutionConfig

ConfigT = Type[BaseModel]
FitnessFn = Callable[[BaseModel], float]
_CacheKey = Tuple[Tuple[str, Any], ...]


def _key(cfg: BaseModel) -> _CacheKey:
    return tuple(sorted(cfg.model_dump().items()))


def random_config(schema: ConfigT, ranges: Dict[str, List[Any]], rng: random.Random) -> BaseModel:
    """Construct a config by picking each field from ``ranges``.

    Fields not present in ``ranges`` keep their default. Pydantic will
    validate the result; a bad combination becomes a ``ValidationError``
    that the caller can choose to penalise or skip.
    """
    values = {key: rng.choice(choices) for key, choices in ranges.items()}
    return schema(**values)


def mutate_config(
    cfg: BaseModel,
    ranges: Dict[str, List[Any]],
    mutation_probability: float,
    rng: random.Random,
    *,
    force_change: bool = True,
    max_tries: int = 16,
) -> BaseModel:
    """Flip each field in ``ranges`` to a fresh choice with probability
    ``mutation_probability``. With ``force_change`` (default), retry
    until at least one field differs — avoids spending fitness on an
    identical candidate."""
    schema = type(cfg)
    base = cfg.model_dump()
    for _ in range(max_tries):
        values = dict(base)
        for key, choices in ranges.items():
            if rng.random() < mutation_probability:
                values[key] = rng.choice(choices)
        if not force_change or values != base:
            return schema(**values)
    # Force one gene to flip.
    forced = dict(base)
    gene = rng.choice(list(ranges))
    alts = [v for v in ranges[gene] if v != base[gene]]
    if alts:
        forced[gene] = rng.choice(alts)
    return schema(**forced)


@dataclass
class EvolutionResult:
    best_config: Optional[BaseModel]
    best_fitness: float
    best_fitness_per_iter: List[float] = field(default_factory=list)
    evaluations: int = 0
    cache_hits: int = 0
    wall_time_s: float = 0.0

    def as_summary(self) -> Dict[str, Any]:
        return {
            "best_fitness": self.best_fitness,
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "wall_time_s": round(self.wall_time_s, 1),
            "best_config": self.best_config.model_dump() if self.best_config else None,
        }


def one_plus_one_es(
    schema: ConfigT,
    ranges: Dict[str, List[Any]],
    fitness: FitnessFn,
    es: Optional[EvolutionConfig] = None,
    *,
    initial: Optional[BaseModel] = None,
    use_cache: bool = True,
) -> EvolutionResult:
    """Run (1+1)-ES with stagnation-restart and an optional fitness cache.

    ``fitness`` is called as ``fitness(config)`` and must return a float
    where *lower* is better. Failed evaluations (training crashed, NaNs,
    etc.) should return ``float("inf")``.
    """
    es = es or EvolutionConfig()
    rng = random.Random(es.seed)
    cache: Optional[Dict[_CacheKey, float]] = {} if use_cache else None

    def evaluate(cfg: BaseModel) -> float:
        if cache is not None:
            k = _key(cfg)
            if k in cache:
                if es.verbose:
                    print(f"  [cache] {cache[k]:.5f}  {cfg.model_dump()}")
                return cache[k]
        try:
            score = float(fitness(cfg))
        except Exception as exc:
            if es.verbose:
                print(f"  [fail] {exc}  {cfg.model_dump()}")
            score = float("inf")
        if cache is not None:
            cache[_key(cfg)] = score
        return score

    start = time.time()
    current = initial if initial is not None else random_config(schema, ranges, rng)
    current_fit = evaluate(current)
    result = EvolutionResult(
        best_config=current.model_copy(),
        best_fitness=current_fit,
        best_fitness_per_iter=[current_fit],
        evaluations=1,
    )
    no_progress = 0

    for _ in range(es.max_iterations):
        candidate = mutate_config(current, ranges, es.mutation_probability, rng)
        if cache is not None and _key(candidate) in cache:
            result.cache_hits += 1
        candidate_fit = evaluate(candidate)
        result.evaluations += 1

        if candidate_fit <= current_fit:
            current, current_fit = candidate, candidate_fit
            no_progress = 0
            if candidate_fit < result.best_fitness:
                result.best_fitness = candidate_fit
                result.best_config = candidate.model_copy()
                if es.verbose:
                    print(f"  -> new best {candidate_fit:.5f}")
        else:
            no_progress += 1

        if no_progress >= es.reset_threshold:
            if es.verbose:
                print("  -> stagnation, restarting from random individual")
            current = random_config(schema, ranges, rng)
            current_fit = evaluate(current)
            result.evaluations += 1
            no_progress = 0

        result.best_fitness_per_iter.append(result.best_fitness)

    result.wall_time_s = time.time() - start
    return result
