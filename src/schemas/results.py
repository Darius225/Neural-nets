"""Schemas for what training / evaluation / search functions return.

Mixed Pydantic + @dataclass on purpose:

  - :class:`EvaluationResult` is Pydantic. It's the output a caller
    most often inspects, serialises, or asserts bounds on (mae / mape
    are non-negative by construction).

  - :class:`TrainingResult` / :class:`SearchHistory` /
    :class:`EvolutionResult` are plain dataclasses. They hold Keras
    models, Pydantic configs, history dicts, and per-iteration lists
    that are heavily mutated *during* a training or search loop.
    Pydantic models prefer immutability and validate every assignment;
    that's the wrong fit for mutable internal bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from tensorflow.keras import Model

from .splits import Dataset

# ----------------------------------- training ------------------------------


@dataclass
class TrainingResult:
    """Output of :func:`src.training.train` / ``train_on_prepared``.

    Holds the trained model, the dataset it was trained on (so callers
    can re-use the scaler / inspect the splits), and the full Keras
    history dict. Plain dataclass because every field is either a heavy
    object (Model, ndarray-backed Dataset) or a mutable mapping
    (history) — none of which Pydantic validates meaningfully.
    """

    model: Model
    dataset: Dataset
    history: dict[str, list]

    @property
    def final_val_mape(self) -> float:
        return float(self.history["val_mape"][-1])

    @property
    def final_val_mae(self) -> float:
        return float(self.history["val_mae"][-1])


# ----------------------------------- evaluation ----------------------------


class EvaluationResult(BaseModel):
    """Output of :func:`src.evaluation.predict_and_evaluate`.

    Pydantic with ``arbitrary_types_allowed`` so the prediction /
    actual ndarrays are accepted as-is, while ``mae`` and ``mape``
    still benefit from the ``Field(ge=0)`` non-negativity invariant.
    ``frozen=True`` matches the snapshot semantics — evaluation
    results don't get updated after the fact.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    predictions: np.ndarray
    actual: np.ndarray
    mae: float = Field(ge=0)
    mape: float = Field(ge=0)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "actual": self.actual,
                "predicted": self.predictions,
                "abs_error": np.abs(self.actual - self.predictions),
            }
        )


# ----------------------------------- search histories ----------------------

Individual = dict[str, Any]


@dataclass
class SearchHistory:
    """Mutable bookkeeping for :mod:`src.search.hyperparam`'s (1+1)-ES.

    Plain dataclass: the ES loop mutates ``best_*`` and the per-iter
    trajectory list dozens of times per run. Pydantic with assignment
    validation would just add overhead.
    """

    best_fitness_per_iteration: list[float] = field(default_factory=list)
    best_params: Individual | None = None
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


@dataclass
class EvolutionResult:
    """Mutable bookkeeping for the schema-driven ES in :mod:`src.search.evolution`.

    Same rationale as :class:`SearchHistory` — mutated heavily during
    the loop, no validation invariants worth enforcing on each write.
    """

    best_config: BaseModel | None
    best_fitness: float
    best_fitness_per_iter: list[float] = field(default_factory=list)
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

    def as_summary(self) -> dict[str, Any]:
        return {
            "best_fitness": self.best_fitness,
            "evaluations": self.evaluations,
            "cache_hits": self.cache_hits,
            "wall_time_s": round(self.wall_time_s, 1),
            "best_config": self.best_config.model_dump() if self.best_config else None,
        }
