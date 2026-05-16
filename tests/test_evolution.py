"""Tests for the memoize_by decorator + ES result/history dataclass logic.

We deliberately don't run real model trainings here — those belong in
the experiments/ scripts. These tests pin the *plumbing* (cache
counting, consider() acceptance, mutate validity)."""

import pytest

from src.configs import RETURNS_CNN_RANGES, EvolutionConfig, ReturnsCNNConfig
from src.search.evolution import EvolutionResult, memoize_by, mutate_config, one_plus_one_es, random_config
from src.search.hyperparam import SearchHistory
from random import Random


class TestMemoizeBy:
    def test_caches_repeated_calls(self):
        @memoize_by(lambda x: x)
        def square(x):
            return x * x

        # Return values must match the underlying function on hit AND miss.
        assert square(2) == 4
        assert square(2) == 4
        assert square(3) == 9
        assert square(2) == 4
        assert square.hits == 2
        assert square.misses == 2
        assert square.cache == {2: 4, 3: 9}

    def test_disabled_counts_every_call_as_miss(self):
        call_count = 0

        @memoize_by(lambda x: x, enabled=False)
        def cube(x):
            nonlocal call_count
            call_count += 1
            return x * x * x

        # Disabled = pass-through. Same input still invokes the function each time.
        assert cube(2) == 8
        assert cube(2) == 8
        assert cube(3) == 27
        assert call_count == 3
        assert cube.hits == 0
        assert cube.misses == 3
        assert cube.cache == {}

    def test_key_fn_canonicalises_unhashable_args(self):
        @memoize_by(lambda d: tuple(sorted(d.items())))
        def f(d):
            return d["a"] + d["b"]

        # The exact use case in this repo: dict args projected to a hashable key.
        assert f({"a": 1, "b": 2}) == 3
        assert f({"b": 2, "a": 1}) == 3  # same key, different insertion order
        assert f.hits == 1
        assert f.misses == 1
        # Single cache entry — the two dict literals resolve to the same key.
        assert len(f.cache) == 1

    def test_preserves_name_and_doc_via_functools_wraps(self):
        @memoize_by(lambda x: x)
        def fancy_fn(x):
            """A docstring that must survive the decorator."""
            return x

        assert fancy_fn.__name__ == "fancy_fn"
        assert fancy_fn.__doc__ == "A docstring that must survive the decorator."

    def test_exception_is_not_cached(self):
        n_calls = 0

        @memoize_by(lambda x: x)
        def maybe_fail(x):
            nonlocal n_calls
            n_calls += 1
            if x < 0:
                raise ValueError("nope")
            return x * 2

        with pytest.raises(ValueError):
            maybe_fail(-1)
        with pytest.raises(ValueError):
            maybe_fail(-1)
        # If exceptions were cached, the second call would not invoke the body.
        assert n_calls == 2
        assert maybe_fail.cache == {}

    def test_forwards_extra_positional_and_keyword_args(self):
        @memoize_by(lambda x: x)
        def with_extras(x, multiplier, *, offset=0):
            return x * multiplier + offset

        # Cache is keyed only on first arg, but extras still flow through to fn
        # on the first (uncached) call.
        assert with_extras(2, 3, offset=10) == 16
        # Cached: subsequent calls return the SAME stored value, ignoring extras.
        assert with_extras(2, 999, offset=999) == 16
        assert with_extras.hits == 1
        assert with_extras.misses == 1


class TestSearchHistoryConsider:
    def test_accepts_better_fitness(self):
        h = SearchHistory(best_fitness=10.0)
        assert h.consider({"x": 1}, fitness=5.0) is True
        assert h.best_fitness == 5.0
        assert h.best_params == {"x": 1}

    def test_rejects_equal_or_worse(self):
        h = SearchHistory(best_fitness=5.0, best_params={"a": 1})
        assert h.consider({"x": 1}, fitness=5.0) is False
        assert h.consider({"x": 1}, fitness=6.0) is False
        assert h.best_params == {"a": 1}  # unchanged

    def test_stores_copy_not_reference(self):
        h = SearchHistory()
        individual = {"x": 1}
        h.consider(individual, fitness=1.0)
        individual["x"] = 999
        assert h.best_params == {"x": 1}  # snapshot, not aliased


class TestEvolutionResultConsider:
    def test_accepts_better(self):
        r = EvolutionResult(best_config=None, best_fitness=10.0)
        cfg = ReturnsCNNConfig(dropout=0.1)
        assert r.consider(cfg, fitness=5.0) is True
        assert r.best_fitness == 5.0
        assert r.best_config.dropout == 0.1

    def test_stores_pydantic_copy(self):
        r = EvolutionResult(best_config=None, best_fitness=float("inf"))
        cfg = ReturnsCNNConfig(dropout=0.1)
        r.consider(cfg, fitness=1.0)
        # mutating the original shouldn't affect the stored copy
        # (Pydantic model_copy returns a new instance)
        assert r.best_config is not cfg


class TestRandomMutate:
    def test_random_config_within_ranges(self):
        rng = Random(0)
        for _ in range(20):
            cfg = random_config(ReturnsCNNConfig, RETURNS_CNN_RANGES, rng)
            for field, choices in RETURNS_CNN_RANGES.items():
                assert getattr(cfg, field) in choices

    def test_mutate_force_change_actually_changes(self):
        rng = Random(0)
        cfg = random_config(ReturnsCNNConfig, RETURNS_CNN_RANGES, rng)
        for _ in range(10):
            mutated = mutate_config(cfg, RETURNS_CNN_RANGES, 0.3, rng, force_change=True)
            assert mutated.model_dump() != cfg.model_dump()

    def test_pydantic_rejects_out_of_bounds(self):
        with pytest.raises(Exception):  # ValidationError
            ReturnsCNNConfig(dropout=1.5)


class TestESDriver:
    def test_es_minimises_simple_objective(self):
        """ES on a synthetic landscape should converge close to the optimum."""
        # Optimum: dropout=0.1, conv1_filters=128 (both extreme values in ranges)
        def fitness(cfg: ReturnsCNNConfig) -> float:
            return cfg.dropout + 1.0 / cfg.conv1_filters

        result = one_plus_one_es(
            ReturnsCNNConfig, RETURNS_CNN_RANGES, fitness,
            EvolutionConfig(max_iterations=30, verbose=False, seed=0),
        )
        # Should at least beat what a random initial would give on this landscape
        # (best possible: 0.0 + 1/192 ≈ 0.0052; tolerate something modest)
        assert result.best_fitness < 0.1
        assert result.evaluations >= 31  # 1 initial + 30 iters (+ restarts)
        assert result.best_config is not None
