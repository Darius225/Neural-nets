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

        square(2); square(2); square(3); square(2)
        assert square.hits == 2
        assert square.misses == 2
        assert sorted(square.cache) == [2, 3]

    def test_disabled_counts_every_call_as_miss(self):
        @memoize_by(lambda x: x, enabled=False)
        def cube(x):
            return x * x * x

        cube(2); cube(2); cube(3)
        assert cube.hits == 0
        assert cube.misses == 3
        assert cube.cache == {}

    def test_key_fn_canonicalises_unhashable_args(self):
        @memoize_by(lambda d: tuple(sorted(d.items())))
        def f(d):
            return d["a"] + d["b"]

        f({"a": 1, "b": 2})
        f({"b": 2, "a": 1})  # same key, different insertion order
        assert f.hits == 1
        assert f.misses == 1


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
