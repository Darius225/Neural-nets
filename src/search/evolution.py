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

import functools
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from ..configs import EvolutionConfig

ConfigT = Type[BaseModel]
FitnessFn = Callable[[BaseModel], float]
_CacheKey = Tuple[Tuple[str, Any], ...]
T = TypeVar("T")


def _key(cfg: BaseModel) -> _CacheKey:
    return tuple(sorted(cfg.model_dump().items()))


def memoize_by(key_fn: Callable[[Any], Hashable], *, enabled: bool = True):
    """Cache function results, keying by ``key_fn(first_arg)``.

    The decorated function exposes ``.hits``, ``.misses``, and ``.cache``
    attributes for inspection. With ``enabled=False`` no caching happens
    and ``.misses`` simply counts every call — handy for benchmarking
    "with vs without cache" without changing call sites.

    Used here because the natural argument (a dict / Pydantic model) is
    not hashable, so :func:`functools.cache` can't help directly.
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(arg, *rest, **kwargs):
            if not enabled:
                wrapper.misses += 1
                return fn(arg, *rest, **kwargs)
            key = key_fn(arg)
            if key in wrapper.cache:
                wrapper.hits += 1
                return wrapper.cache[key]
            wrapper.misses += 1
            result = fn(arg, *rest, **kwargs)
            wrapper.cache[key] = result
            return result

        wrapper.cache = {}
        wrapper.hits = 0
        wrapper.misses = 0
        return wrapper
    return decorator


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

    def consider(self, candidate: BaseModel, fitness: float, *, verbose: bool = False) -> bool:
        """Update best_* if ``fitness`` is the new global optimum.

        Returns ``True`` when a new optimum was recorded.
        """
        if fitness >= self.best_fitness:
            return False
        self.best_fitness = fitness
        self.best_config = candidate.model_copy()
        if verbose:
            print(f"  -> new best {fitness:.5f}")
        return True

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
    result = EvolutionResult(best_config=None, best_fitness=float("inf"))

    @memoize_by(_key, enabled=use_cache)
    def evaluate(cfg: BaseModel) -> float:
        """Train the candidate and score it. No caching here — that lives
        in the decorator. Failures become +inf so they're never selected."""
        try:
            return float(fitness(cfg))
        except Exception as exc:
            if es.verbose:
                print(f"  [fail] {exc}  {cfg.model_dump()}")
            return float("inf")

    start = time.time()
    current = initial if initial is not None else random_config(schema, ranges, rng)
    current_fit = evaluate(current)
    result.consider(current, current_fit, verbose=es.verbose)
    result.best_fitness_per_iter.append(current_fit)
    no_progress = 0

    for _ in range(es.max_iterations):
        candidate = mutate_config(current, ranges, es.mutation_probability, rng)
        candidate_fit = evaluate(candidate)

        if candidate_fit <= current_fit:
            current, current_fit = candidate, candidate_fit
            result.consider(candidate, candidate_fit, verbose=es.verbose)
            no_progress = 0
        else:
            no_progress += 1

        if no_progress >= es.reset_threshold:
            if es.verbose:
                print("  -> stagnation, restarting from random individual")
            current = random_config(schema, ranges, rng)
            current_fit = evaluate(current)
            no_progress = 0

        result.best_fitness_per_iter.append(result.best_fitness)

    result.evaluations = evaluate.hits + evaluate.misses
    result.cache_hits = evaluate.hits
    result.wall_time_s = time.time() - start
    return result
